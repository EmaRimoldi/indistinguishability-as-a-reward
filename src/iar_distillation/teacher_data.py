from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from tqdm import tqdm

from iar_cqa.scoring import score_response as score_cqa
from iar_gsm8k.prompts import TASK_PROMPT_VERSION as GSM8K_PROMPT_VERSION
from iar_gsm8k.prompts import task_prompt as gsm8k_task_prompt
from iar_gsm8k.scoring import score_response as score_gsm8k
from iar_phase0.env import load_env_file
from iar_phase0.evaluate import run_mbpp_tests
from iar_phase0.openrouter import OpenRouterClient, make_chat_payload, response_text


SCHEMA_VERSION = 1
DOMAINS = ("mbpp", "commonsenseqa", "gsm8k")
SCORER_VERSIONS = {
    "mbpp": "mbpp-public-tests-v1",
    "commonsenseqa": "commonsenseqa-answer-line-markdown-v2",
    "gsm8k": "gsm8k-answer-line-markdown-v2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def one_user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Expected exactly one prompt message")
    message = messages[0]
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("Expected one user message with string content")
    return str(message["content"])


def strip_code_fences(text: str) -> str:
    """Normalize MBPP code without importing the local Torch generator stack."""
    stripped = (text or "").strip()
    if "[BEGIN]" in stripped:
        stripped = stripped.split("[BEGIN]", 1)[1].strip()
    if "[DONE]" in stripped:
        stripped = stripped.split("[DONE]", 1)[0].strip()
    fenced = re.findall(r"```\s*([A-Za-z0-9_+.-]*)\s*\n?(.*?)```", stripped, re.DOTALL)
    if fenced:
        candidates = [
            {"language": language.lower(), "body": body.strip()}
            for language, body in fenced
            if body.strip()
        ]
        implementations = [
            candidate
            for candidate in candidates
            if re.search(
                r"(?m)^\s*(?:(?:async\s+)?def|class)\s+[A-Za-z_]\w*",
                candidate["body"],
            )
        ]
        pool = implementations or candidates
        if pool:
            return max(
                pool,
                key=lambda candidate: (
                    candidate["language"] in {"python", "py"},
                    len(candidate["body"]),
                ),
            )["body"]
    return stripped


def load_mbpp_sources(root: Path, domain: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    source_path = resolve(root, str(domain["source_file"]))
    training_rows = read_jsonl(source_path)
    dataset = load_dataset(
        str(domain["dataset_id"]),
        split=str(domain["dataset_split"]),
        cache_dir=str(resolve(root, str(domain["cache_dir"]))),
    )
    sources: list[dict[str, Any]] = []
    for order, row in enumerate(training_rows):
        index = int(row["dataset_index"])
        raw = dataset[index]
        task_id = str(row["task_id"])
        if str(raw.get("task_id")) != task_id:
            raise ValueError(f"MBPP task mismatch at dataset index {index}")
        tests = raw.get("test_list") or raw.get("tests") or []
        if isinstance(tests, str):
            tests = [tests]
        sources.append(
            {
                "id": f"mbpp-{row['split']}-{task_id}",
                "task_id": task_id,
                "dataset_index": index,
                "experiment_order": order,
                "prompt": one_user_prompt(row),
                "prompt_version": "mbpp-google-original-[BEGIN]",
                "gold": {
                    "tests": [str(test) for test in tests],
                    "test_setup_code": str(raw.get("test_setup_code") or ""),
                },
            }
        )
    return sources


def load_cqa_sources(root: Path, domain: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(resolve(root, str(domain["source_file"])))
    return [
        {
            "id": str(row["task_id"]),
            "task_id": str(row["task_id"]),
            "dataset_index": int(row["dataset_index"]),
            "experiment_order": int(row["experiment_order"]),
            "prompt": one_user_prompt(row),
            "prompt_version": "commonsenseqa-explain-answer-v1",
            "gold": {"answer_key": str(row["answer_key"])},
        }
        for row in rows
    ]


def load_gsm8k_sources(root: Path, domain: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(resolve(root, str(domain["source_file"])))
    return [
        {
            "id": str(row["id"]),
            "task_id": str(row["id"]),
            "dataset_index": int(row["source_index"]),
            "experiment_order": int(row["experiment_order"]),
            "prompt": gsm8k_task_prompt(str(row["question"])),
            "prompt_version": GSM8K_PROMPT_VERSION,
            "gold": {"answer_value": str(row["answer_value"])},
        }
        for row in rows
    ]


def load_sources(root: Path, name: str, domain: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "mbpp":
        rows = load_mbpp_sources(root, domain)
    elif name == "commonsenseqa":
        rows = load_cqa_sources(root, domain)
    elif name == "gsm8k":
        rows = load_gsm8k_sources(root, domain)
    else:
        raise ValueError(f"Unsupported domain: {name}")
    expected = int(domain["expected_count"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {name} rows, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {name} IDs")
    orders = [int(row["experiment_order"]) for row in rows]
    if orders != list(range(expected)):
        raise ValueError(f"{name} experiment order is not contiguous and frozen")
    return rows


def score_output(
    domain: str,
    raw_text: str,
    source: dict[str, Any],
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    if domain == "mbpp":
        normalized = strip_code_fences(raw_text)
        test_result = run_mbpp_tests(
            normalized,
            list(source["gold"]["tests"]),
            timeout,
            str(source["gold"]["test_setup_code"]),
        )
        return normalized, {
            "valid": bool(normalized.strip()),
            "correct": test_result.passed,
            "returncode": test_result.returncode,
            "stderr": test_result.stderr,
        }
    if domain == "commonsenseqa":
        return raw_text.strip(), score_cqa(raw_text, str(source["gold"]["answer_key"]))
    if domain == "gsm8k":
        return raw_text.strip(), score_gsm8k(raw_text, str(source["gold"]["answer_value"]))
    raise ValueError(domain)


def numeric_usage(rows: list[dict[str, Any]], field: str) -> float:
    values = [row.get("usage", {}).get(field) for row in rows]
    return float(sum(value for value in values if isinstance(value, (int, float))))


def build_summary(domain: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    valid = sum(bool(row["score"].get("valid")) for row in rows)
    correct = sum(bool(row["score"].get("correct")) for row in rows)
    stopped = sum(row.get("finish_reason") == "stop" for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "status": "complete",
        "n": n,
        "valid": valid,
        "valid_rate": valid / n if n else None,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "terminated": stopped,
        "termination_rate": stopped / n if n else None,
        "prompt_tokens": int(numeric_usage(rows, "prompt_tokens")),
        "completion_tokens": int(numeric_usage(rows, "completion_tokens")),
        "total_tokens": int(numeric_usage(rows, "total_tokens")),
        "openrouter_cost": numeric_usage(rows, "cost"),
        "models": sorted({str(row.get("raw_model")) for row in rows}),
        "providers": sorted({str(row.get("raw_provider")) for row in rows}),
        "scorer_versions": sorted(
            {str(row.get("scorer_version", SCORER_VERSIONS[domain])) for row in rows}
        ),
        "completed_at": utc_now(),
    }


def rescore_cached_responses(
    domain: str,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    timeout: int,
) -> list[dict[str, Any]]:
    """Apply the current lightweight scorer uniformly to cached text responses."""
    if domain not in {"commonsenseqa", "gsm8k"}:
        return rows
    sources_by_id = {str(row["id"]): row for row in source_rows}
    rescored: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row["id"])
        if source_id not in sources_by_id:
            raise ValueError(f"Cached response has no frozen source row: {source_id}")
        raw_text = str(row.get("raw_response", row.get("response", "")))
        normalized, score = score_output(
            domain,
            raw_text,
            sources_by_id[source_id],
            timeout,
        )
        updated = dict(row)
        updated["response"] = normalized
        updated["response_sha256"] = sha256_text(normalized)
        updated["score"] = score
        updated["scorer_version"] = SCORER_VERSIONS[domain]
        rescored.append(updated)
    return rescored


def export_sft(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    def project(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "domain": row["domain"],
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
            "teacher_correct": bool(row["score"].get("correct")),
            "teacher_valid": bool(row["score"].get("valid")),
            "source_response_sha256": row["response_sha256"],
        }

    write_jsonl(run_dir / "sft_all.jsonl", [project(row) for row in rows])
    write_jsonl(
        run_dir / "sft_correct_only.jsonl",
        [project(row) for row in rows if bool(row["score"].get("correct"))],
    )


def verified_chat_completion(
    *,
    client: OpenRouterClient,
    payload: dict[str, Any],
    source_id: str,
    expected_model: str,
    expected_provider: str,
    event_path: Path,
    max_attempts: int,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    """Retry transient 403s and semantically incomplete HTTP-200 responses."""
    last_detail: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        retryable = False
        try:
            raw = client.chat_completion(payload)
        except httpx.HTTPStatusError as exc:
            status_code = int(exc.response.status_code)
            retryable = status_code == 403
            last_detail = {
                "kind": "http_status",
                "status_code": status_code,
                "response_body": exc.response.text[:500],
            }
            if not retryable or attempt >= max_attempts:
                raise
        else:
            choices = raw.get("choices")
            content = None
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = message.get("content")
            observed_model = raw.get("model")
            observed_provider = raw.get("provider")
            valid = (
                observed_model == expected_model
                and observed_provider == expected_provider
                and isinstance(content, str)
                and bool(content.strip())
            )
            if valid:
                return raw
            retryable = True
            last_detail = {
                "kind": "incomplete_success_response",
                "response_keys": sorted(str(key) for key in raw),
                "observed_model": observed_model,
                "observed_provider": observed_provider,
                "error": raw.get("error"),
                "has_nonempty_content": isinstance(content, str) and bool(content.strip()),
            }
        event = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "source_id": source_id,
            "attempt": attempt,
            "will_retry": retryable and attempt < max_attempts,
            **last_detail,
        }
        events = read_jsonl(event_path) if event_path.exists() else []
        events.append(event)
        write_jsonl(event_path, events)
        if attempt < max_attempts:
            time.sleep(retry_wait_seconds * attempt)
    raise RuntimeError(
        f"No verified completion for {source_id} after {max_attempts} attempts: {last_detail}"
    )


def generate(config_path: Path, domain_name: str) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    domain = dict(config["domains"][domain_name])
    output_dir = resolve(root, str(config["paths"]["output_root"])) / domain_name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_sources(root, domain_name, domain)
    source_manifest_rows = [
        {
            key: row[key]
            for key in (
                "id",
                "task_id",
                "dataset_index",
                "experiment_order",
                "prompt",
                "prompt_version",
            )
        }
        for row in source_rows
    ]
    source_manifest_path = output_dir / "source_prompts.jsonl"
    write_jsonl(source_manifest_path, source_manifest_rows)

    model = config["model"]
    transport = config["transport"]
    load_env_file(resolve(root, str(config["paths"]["openrouter_env"])))
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    response_path = output_dir / "responses.jsonl"
    responses = read_jsonl(response_path) if response_path.exists() else []
    responses = rescore_cached_responses(
        domain_name,
        responses,
        source_rows,
        int(domain.get("unit_test_timeout_seconds", 5)),
    )
    if responses:
        write_jsonl(response_path, responses)
    by_id = {str(row["id"]): row for row in responses}
    if len(by_id) != len(responses):
        raise ValueError(f"Duplicate completed rows in {response_path}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "domain": domain_name,
        "expected_count": len(source_rows),
        "source_file": str(resolve(root, str(domain["source_file"]))),
        "source_file_sha256": sha256_file(resolve(root, str(domain["source_file"]))),
        "source_prompts_file": str(source_manifest_path),
        "source_prompts_sha256": sha256_file(source_manifest_path),
        "model_id": str(model["model_id"]),
        "expected_provider": str(model["expected_provider"]),
        "message_roles": ["user"],
        "system_prompt_present": False,
        "scorer_version": SCORER_VERSIONS[domain_name],
        "generation": {
            "temperature": float(model["temperature"]),
            "top_p": float(model["top_p"]),
            "max_tokens": int(domain["max_tokens"]),
            "reasoning_enabled": False,
        },
        "config_file": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "started_at": utc_now(),
        "completed_rows": len(responses),
    }
    write_json(output_dir / "manifest.json", manifest)

    progress = tqdm(
        total=len(source_rows),
        initial=len(responses),
        desc=f"DeepSeek teacher {domain_name}",
    )
    for source in source_rows:
        if source["id"] in by_id:
            existing = by_id[source["id"]]
            if existing.get("prompt_sha256") != sha256_text(source["prompt"]):
                raise ValueError(f"Prompt drift while resuming {source['id']}")
            continue
        payload = make_chat_payload(
            model_id=str(model["model_id"]),
            prompt=str(source["prompt"]),
            system_prompt=None,
            max_tokens=int(domain["max_tokens"]),
            temperature=float(model["temperature"]),
            provider_routing=dict(model["provider_routing"]),
            request_overrides={
                "top_p": float(model["top_p"]),
                "reasoning": {"enabled": False},
            },
        )
        started = time.time()
        raw = verified_chat_completion(
            client=client,
            payload=payload,
            source_id=str(source["id"]),
            expected_model=str(model["model_id"]),
            expected_provider=str(model["expected_provider"]),
            event_path=output_dir / "transport_events.jsonl",
            max_attempts=int(transport["logical_max_attempts"]),
            retry_wait_seconds=float(transport["logical_retry_wait_seconds"]),
        )
        raw_model = str(raw["model"])
        raw_provider = str(raw["provider"])
        raw_text = response_text(raw)
        normalized, score = score_output(
            domain_name,
            raw_text,
            source,
            int(domain.get("unit_test_timeout_seconds", 5)),
        )
        choice = raw["choices"][0]
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": source["id"],
            "task_id": source["task_id"],
            "domain": domain_name,
            "dataset_index": source["dataset_index"],
            "experiment_order": source["experiment_order"],
            "prompt": source["prompt"],
            "prompt_version": source["prompt_version"],
            "prompt_sha256": sha256_text(source["prompt"]),
            "message_roles": ["user"],
            "system_prompt_present": False,
            "source": "DeepSeek_teacher",
            "requested_model": str(model["model_id"]),
            "raw_model": raw_model,
            "raw_provider": raw_provider,
            "raw_response": raw_text,
            "response": normalized,
            "response_sha256": sha256_text(normalized),
            "score": score,
            "scorer_version": SCORER_VERSIONS[domain_name],
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "usage": raw.get("usage", {}),
            "generation": manifest["generation"],
            "latency_seconds": time.time() - started,
            "created_at": utc_now(),
        }
        responses.append(row)
        by_id[source["id"]] = row
        write_jsonl(response_path, responses)
        manifest["completed_rows"] = len(responses)
        manifest["updated_at"] = utc_now()
        write_json(output_dir / "manifest.json", manifest)
        progress.update(1)
        time.sleep(float(transport["inter_request_delay_seconds"]))
    progress.close()

    responses.sort(key=lambda row: int(row["experiment_order"]))
    write_jsonl(response_path, responses)
    export_sft(output_dir, responses)
    summary = build_summary(domain_name, responses)
    write_json(output_dir / "summary.json", summary)
    manifest.update(
        {
            "status": "complete",
            "completed_rows": len(responses),
            "responses_sha256": sha256_file(response_path),
            "sft_all_sha256": sha256_file(output_dir / "sft_all.jsonl"),
            "sft_correct_only_sha256": sha256_file(output_dir / "sft_correct_only.jsonl"),
            "completed_at": summary["completed_at"],
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    args = parser.parse_args()
    output = generate(args.config.resolve(), args.domain)
    print(json.dumps({"status": "complete", "domain": args.domain, "output": str(output)}))


if __name__ == "__main__":
    main()
