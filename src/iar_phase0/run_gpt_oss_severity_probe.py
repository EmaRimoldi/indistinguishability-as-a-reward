from __future__ import annotations

import argparse
import ast
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import ensure_dirs, load_config
from .data import load_mbpp, materialize_split, read_jsonl, write_jsonl
from .env import load_env_file
from .evaluate import run_mbpp_tests
from .openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from .parameters import initialize_parameter_record
from .prompts import (
    pointwise_self_attribution_prompt,
    pointwise_self_attribution_system_prompt,
)
from .run_pointwise_self_attribution import parse_binary_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    probe_cfg = config.raw["severity_probe"]
    run_id = datetime.now(timezone.utc).strftime(
        f"{probe_cfg['run_name']}_%Y%m%dT%H%M%SZ"
    )
    out_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    initialize_parameter_record(out_dir, resolved_config=config.raw, source_config=config.path)

    examples = materialize_split(
        load_mbpp(config),
        split=str(probe_cfg["split"]),
        limit=int(probe_cfg["scan_count"]),
    )
    source_runs = {
        name: config.resolve(path)
        for name, path in probe_cfg["source_runs"].items()
    }
    responses = {
        name: {int(row["index"]): row for row in read_jsonl(path / "responses.jsonl")}
        for name, path in source_runs.items()
    }

    candidates: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for index, example in enumerate(examples):
        own_code = str(responses["b_own"][index]["text"])
        plain_code = str(responses["plain"][index]["text"])
        variants = make_variants(own_code=own_code, plain_code=plain_code)
        tested: dict[str, dict[str, Any]] = {}
        all_pass = True
        for variant, code in variants.items():
            result = run_mbpp_tests(
                code,
                example.tests,
                timeout_seconds=int(config.raw["unit_tests"]["timeout_seconds"]),
                test_setup_code=example.test_setup_code,
            )
            tested[variant] = {
                "passed": result.passed,
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
            all_pass = all_pass and result.passed
        if not all_pass:
            continue

        selected_indices.append(index)
        for variant, code in variants.items():
            candidates.append({
                "index": index,
                "task_id": example.task_id,
                "variant": variant,
                "source_family": "B_own" if variant.startswith("B_") else "A0_plain",
                "code": code,
                "chars": len(code),
                "unit_tests": tested[variant],
            })
        if len(selected_indices) == int(probe_cfg["selected_task_count"]):
            break

    expected = int(probe_cfg["selected_task_count"])
    if len(selected_indices) != expected:
        raise RuntimeError(f"Only found {len(selected_indices)} fully valid tasks; expected {expected}.")

    write_jsonl(out_dir / "candidates.jsonl", candidates)
    (out_dir / "selected_indices.json").write_text(
        json.dumps(selected_indices, indent=2), encoding="utf-8"
    )

    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    client = OpenRouterClient.from_env()
    disc_cfg = config.raw["discriminator"]
    system_prompt = pointwise_self_attribution_system_prompt(
        str(probe_cfg["target_model_name"])
    )

    judgments: list[dict[str, Any]] = []
    workers = int(probe_cfg["workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                judge_candidate,
                candidate=candidate,
                example=examples[int(candidate["index"])],
                client=client,
                disc_cfg=disc_cfg,
                system_prompt=system_prompt,
            )
            for candidate in candidates
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"gpt_oss_severity_probe ({workers} workers)",
        ):
            judgments.append(future.result())
            write_jsonl(out_dir / "judgments.jsonl", judgments)

    summary = summarize(judgments)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.raw,
        "selected_indices": selected_indices,
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2))


def make_variants(*, own_code: str, plain_code: str) -> dict[str, str]:
    return {
        "B_original": own_code,
        "B_without_docstrings": transform_code(own_code, remove_docstrings=True),
        "B_canonical": transform_code(
            own_code,
            remove_docstrings=True,
            remove_annotations=True,
            remove_main_guard=True,
        ),
        "A_original": plain_code,
        "A_add_docstrings": transform_code(plain_code, add_docstrings=True),
        "A_add_annotations": transform_code(plain_code, add_annotations=True),
        "A_add_docstrings_annotations": transform_code(
            plain_code,
            add_docstrings=True,
            add_annotations=True,
        ),
    }


def transform_code(
    code: str,
    *,
    remove_docstrings: bool = False,
    remove_annotations: bool = False,
    remove_main_guard: bool = False,
    add_docstrings: bool = False,
    add_annotations: bool = False,
) -> str:
    tree = ast.parse(code)

    def is_docstring(node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    def is_main_guard(node: ast.stmt) -> bool:
        if not isinstance(node, ast.If):
            return False
        try:
            return ast.unparse(node.test) in {"__name__ == '__main__'", '"__main__" == __name__'}
        except Exception:
            return False

    class Transformer(ast.NodeTransformer):
        def visit_Module(self, node: ast.Module) -> ast.AST:
            self.generic_visit(node)
            if remove_docstrings and node.body and is_docstring(node.body[0]):
                node.body = node.body[1:]
            if remove_main_guard:
                node.body = [statement for statement in node.body if not is_main_guard(statement)]
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self.generic_visit(node)
            if remove_docstrings and node.body and is_docstring(node.body[0]):
                node.body = node.body[1:]
            if add_docstrings and (not node.body or not is_docstring(node.body[0])):
                node.body.insert(
                    0,
                    ast.Expr(value=ast.Constant(value="Solve the requested task and return the result.")),
                )
            if remove_annotations:
                clear_annotations(node)
            if add_annotations:
                add_generic_annotations(node)
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    transformed = Transformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed).strip()


def all_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    return [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *([] if node.args.vararg is None else [node.args.vararg]),
        *([] if node.args.kwarg is None else [node.args.kwarg]),
    ]


def clear_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    for argument in all_arguments(node):
        argument.annotation = None
    node.returns = None
    node.type_comment = None


def add_generic_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    for argument in all_arguments(node):
        if argument.arg not in {"self", "cls"} and argument.annotation is None:
            argument.annotation = ast.Name(id="object", ctx=ast.Load())
    if node.returns is None:
        node.returns = ast.Name(id="object", ctx=ast.Load())


def judge_candidate(
    *,
    candidate: dict[str, Any],
    example: Any,
    client: OpenRouterClient,
    disc_cfg: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    payload = make_chat_payload(
        model_id=disc_cfg["model_id"],
        system_prompt=system_prompt,
        prompt=pointwise_self_attribution_prompt(example, str(candidate["code"])),
        max_tokens=int(disc_cfg["judge_max_tokens"]),
        temperature=float(disc_cfg["temperature"]),
        logprobs=True,
        top_logprobs=int(disc_cfg["top_logprobs"]),
        provider_routing=disc_cfg["provider_routing"],
        request_overrides=disc_cfg.get("request_overrides"),
    )
    started = time.time()
    response = client.chat_completion(payload)
    text = response_text(response)
    parsed = binary_probability_from_logprobs(response)
    if bool(disc_cfg.get("strict_logprobs", True)) and not parsed["ok"]:
        raise RuntimeError(f"Could not extract binary log-probabilities: {parsed}")
    prediction = parse_binary_prediction(text, parsed)
    return {
        "index": candidate["index"],
        "task_id": candidate["task_id"],
        "variant": candidate["variant"],
        "source_family": candidate["source_family"],
        "chars": candidate["chars"],
        "unit_tests": candidate["unit_tests"],
        "prediction": prediction,
        "judge": {
            "text": text,
            "parsed": parsed,
            "raw_model": response.get("model"),
            "usage": response.get("usage"),
            "latency_seconds": time.time() - started,
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["variant"]), []).append(row)

    by_variant: dict[str, Any] = {}
    for variant, group in sorted(groups.items()):
        p_values = [float(row["judge"]["parsed"]["p1"]) for row in group]
        gaps = [
            float(row["judge"]["parsed"]["logp_1"])
            - float(row["judge"]["parsed"]["logp_0"])
            for row in group
        ]
        by_variant[variant] = {
            "n": len(group),
            "acceptance_emitted": sum(row["prediction"] == 1 for row in group) / len(group),
            "acceptance_argmax": sum(value > 0.5 for value in p_values) / len(group),
            "mean_p_own": sum(p_values) / len(p_values),
            "median_p_own": sorted(p_values)[len(p_values) // 2],
            "mean_logit_gap": sum(gaps) / len(gaps),
            "mean_chars": sum(int(row["chars"]) for row in group) / len(group),
            "pass_at_1": sum(bool(row["unit_tests"]["passed"]) for row in group) / len(group),
        }

    baselines = {"B": "B_original", "A": "A_original"}
    paired_changes: dict[str, Any] = {}
    for family, baseline_variant in baselines.items():
        baseline = {
            int(row["index"]): float(row["judge"]["parsed"]["p1"])
            for row in groups[baseline_variant]
        }
        for variant, group in groups.items():
            if not variant.startswith(family + "_") or variant == baseline_variant:
                continue
            deltas = [
                float(row["judge"]["parsed"]["p1"]) - baseline[int(row["index"])]
                for row in group
            ]
            paired_changes[f"{variant}_minus_{baseline_variant}"] = {
                "mean_delta_p_own": sum(deltas) / len(deltas),
                "increased": sum(delta > 0 for delta in deltas),
                "decreased": sum(delta < 0 for delta in deltas),
            }

    return {
        "n_judgments": len(rows),
        "by_variant": by_variant,
        "paired_changes": paired_changes,
        "emitted_argmax_consistency": sum(
            int(row["prediction"])
            == int(float(row["judge"]["parsed"]["p1"]) > 0.5)
            for row in rows
        )
        / len(rows),
        "total_api_cost": sum(
            float((row["judge"].get("usage") or {}).get("cost") or 0.0)
            for row in rows
        ),
    }


if __name__ == "__main__":
    main()
