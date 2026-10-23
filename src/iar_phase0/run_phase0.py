from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import ensure_dirs, load_config
from .data import MbppExample, fewshot_examples, load_mbpp, materialize_split, write_jsonl
from .discriminator import Discriminator
from .env import load_env_file
from .evaluate import run_mbpp_tests
from .generator import LocalGenerator
from .metrics import summarize
from .openrouter import OpenRouterClient
from .parameters import initialize_parameter_record
from .prompts import generator_prompt, sha256_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument("--skip-b", action="store_true", help="Skip OpenRouter B calls.")
    parser.add_argument("--skip-a", action="store_true", help="Skip local A generation.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    run_cfg = config.raw["run_modes"][args.run_mode]
    run_id = datetime.now(timezone.utc).strftime(f"{args.run_mode}_%Y%m%dT%H%M%SZ")
    out_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    initialize_parameter_record(
        out_dir,
        resolved_config=config.raw,
        source_config=config.path,
    )

    dataset = load_mbpp(config)
    fewshot = fewshot_examples(
        dataset,
        split=config.raw["dataset"]["fewshot_split"],
        count=int(config.raw["dataset"]["fewshot_count"]),
    )
    examples_by_split = {
        split: materialize_split(dataset, split=split, limit=limit)
        for split, limit in run_cfg["splits"].items()
    }

    generator = None
    if not args.skip_a:
        gen_cfg = config.raw["generator"]
        generator = LocalGenerator(
            model_id=gen_cfg["model_id"],
            cache_dir=config.resolve(config.raw["paths"]["hf_cache"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            local_files_only=bool(gen_cfg.get("local_files_only", False)),
            dtype=gen_cfg.get("dtype", "auto"),
            device_map=gen_cfg.get("device_map", "auto"),
        )

    discriminator = None
    if not args.skip_b:
        env_path = config.resolve(config.raw["paths"]["openrouter_env"])
        load_env_file(env_path)
        disc_cfg = config.raw["discriminator"]
        discriminator = Discriminator(
            client=OpenRouterClient.from_env(),
            model_id=disc_cfg["model_id"],
            own_answer_max_tokens=int(disc_cfg["own_answer_max_tokens"]),
            judge_max_tokens=int(disc_cfg["judge_max_tokens"]),
            temperature=float(disc_cfg["temperature"]),
            top_logprobs=int(disc_cfg["top_logprobs"]),
            provider_routing=disc_cfg["provider_routing"],
            request_overrides=disc_cfg.get("request_overrides"),
            strict_logprobs=bool(disc_cfg["strict_logprobs"]),
        )

    rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    workers = int(run_cfg.get("workers", 1))
    if workers < 1:
        raise ValueError("run_modes.<name>.workers must be at least 1")
    if workers > 1 and generator is not None:
        raise ValueError("Parallel Phase-0 execution is supported only for API-only runs (--skip-a).")

    for split, examples in examples_by_split.items():
        if workers == 1:
            completed = (
                process_example_isolated(
                    example=example,
                    variants=config.raw["generator"]["variants"],
                    fewshot=fewshot,
                    generator=generator,
                    discriminator=discriminator,
                    collect_b_own=bool(config.raw["discriminator"]["collect_own_answer"]),
                    judge_candidates=bool(config.raw["discriminator"].get("judge_candidates", True)),
                    timeout_seconds=int(config.raw["unit_tests"]["timeout_seconds"]),
                )
                for example in examples
            )
            iterator = tqdm(completed, total=len(examples), desc=f"{args.run_mode}:{split}")
            for result in iterator:
                extend_and_checkpoint(
                    result=result,
                    out_dir=out_dir,
                    rows=rows,
                    response_rows=response_rows,
                    judge_rows=judge_rows,
                    test_rows=test_rows,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        process_example_isolated,
                        example=example,
                        variants=config.raw["generator"]["variants"],
                        fewshot=fewshot,
                        generator=generator,
                        discriminator=discriminator,
                        collect_b_own=bool(config.raw["discriminator"]["collect_own_answer"]),
                        judge_candidates=bool(config.raw["discriminator"].get("judge_candidates", True)),
                        timeout_seconds=int(config.raw["unit_tests"]["timeout_seconds"]),
                    )
                    for example in examples
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{args.run_mode}:{split} ({workers} workers)",
                ):
                    extend_and_checkpoint(
                        result=future.result(),
                        out_dir=out_dir,
                        rows=rows,
                        response_rows=response_rows,
                        judge_rows=judge_rows,
                        test_rows=test_rows,
                    )

    summary = summarize(rows)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_mode": args.run_mode,
        "actor_model_id": config.raw["generator"]["model_id"],
        "discriminator_model_id": config.raw["discriminator"]["model_id"],
        "config": config.raw,
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))


def process_example_isolated(**kwargs: Any) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    response_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    rows = process_example(
        **kwargs,
        response_rows=response_rows,
        judge_rows=judge_rows,
        test_rows=test_rows,
    )
    return rows, response_rows, judge_rows, test_rows


def extend_and_checkpoint(
    *,
    result: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    out_dir: Path,
    rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    new_rows, new_responses, new_judgments, new_tests = result
    rows.extend(new_rows)
    response_rows.extend(new_responses)
    judge_rows.extend(new_judgments)
    test_rows.extend(new_tests)
    write_jsonl(out_dir / "rows.jsonl", rows)
    write_jsonl(out_dir / "responses.jsonl", response_rows)
    write_jsonl(out_dir / "judgments.jsonl", judge_rows)
    write_jsonl(out_dir / "unit_tests.jsonl", test_rows)


def process_example(
    *,
    example: MbppExample,
    variants: list[str],
    fewshot: list[MbppExample],
    generator: LocalGenerator | None,
    discriminator: Discriminator | None,
    collect_b_own: bool,
    judge_candidates: bool,
    timeout_seconds: int,
    response_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    if generator is not None:
        for variant in variants:
            prompt = generator_prompt(example, variant=variant, fewshot=fewshot)
            started = time.time()
            text = generator.generate(prompt)
            item = base_row(example, source="A0", variant=variant)
            item["text"] = text
            item["generator_prompt_sha256"] = sha256_text(prompt)
            item["latency_seconds"] = time.time() - started
            candidates.append(item)
            response_rows.append(public_response_row(item))

    if discriminator is not None and collect_b_own:
        started = time.time()
        own = discriminator.own_answer(example)
        item = base_row(example, source="B_own", variant="own_answer")
        item["text"] = own["text"]
        item["raw_model"] = own["raw_model"]
        item["raw_provider"] = own.get("raw_provider")
        item["user_prompt_sha256"] = own.get("user_prompt_sha256")
        item["usage"] = own["usage"]
        item["latency_seconds"] = time.time() - started
        candidates.append(item)
        response_rows.append(public_response_row(item))

    for item in candidates:
        test_result = run_mbpp_tests(
            item["text"],
            example.tests,
            timeout_seconds=timeout_seconds,
            test_setup_code=example.test_setup_code,
        )
        item["unit_tests"] = {
            "passed": test_result.passed,
            "returncode": test_result.returncode,
            "stdout": test_result.stdout,
            "stderr": test_result.stderr,
        }
        test_rows.append({
            **identity_fields(item),
            "unit_tests": item["unit_tests"],
        })

        if discriminator is not None and judge_candidates:
            started = time.time()
            judge = discriminator.judge(example, item["text"])
            item["judge"] = judge
            item["judge_latency_seconds"] = time.time() - started
            judge_rows.append({
                **identity_fields(item),
                "judge": judge,
            })
        rows.append(item)
    return rows


def base_row(example: MbppExample, *, source: str, variant: str) -> dict[str, Any]:
    return {
        "split": example.split,
        "index": example.index,
        "task_id": example.task_id,
        "source": source,
        "variant": variant,
    }


def identity_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": item["split"],
        "index": item["index"],
        "task_id": item["task_id"],
        "source": item["source"],
        "variant": item["variant"],
    }


def public_response_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **identity_fields(item),
        "text": item["text"],
        "raw_model": item.get("raw_model"),
        "raw_provider": item.get("raw_provider"),
        "generator_prompt_sha256": item.get("generator_prompt_sha256"),
        "user_prompt_sha256": item.get("user_prompt_sha256"),
        "usage": item.get("usage"),
        "latency_seconds": item.get("latency_seconds"),
    }


if __name__ == "__main__":
    main()
