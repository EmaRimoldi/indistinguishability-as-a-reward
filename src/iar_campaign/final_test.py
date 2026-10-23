from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from iar_campaign.four_domain_screen import _adapter_for_step
from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import sha256_file, utc_now, write_json, write_jsonl
from iar_campaign.score_baselines import (
    SPIDER_EXECUTION_LIMITS,
    _audit_spider_reference_one,
    _score_kodcode,
    _score_simple,
    _score_spider_one,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config

CAMPAIGN_ROOT = Path("/orcd/pool/008/erimoldi/iar/experiments/campaign_20260908_v2")
STEP_ROOT = CAMPAIGN_ROOT / "12_final_test_analysis_reconciliation"
PROTOCOL = STEP_ROOT / "configs/final_test_protocol_v1.json"
REGISTRY = STEP_ROOT / "manifests/final_condition_registry_attempt001.json"
REFERENCE_GATE = STEP_ROOT / "results/final_reference_gate_v1.json"
TEST_ROOT = STEP_ROOT / "sealed_inputs"

STEP1 = CAMPAIGN_ROOT / "01_data_protocol_runtime_gates"
STEP5 = CAMPAIGN_ROOT / "05_four_domain_screen_1p5b"
STEP7 = CAMPAIGN_ROOT / "07_baseline_suite"
STEP8 = CAMPAIGN_ROOT / "08_model_scale_3b"
STEP9 = CAMPAIGN_ROOT / "09_pairwise_generate_distinguish"
STEP10 = CAMPAIGN_ROOT / "10_discriminator_transfer"
STEP11 = CAMPAIGN_ROOT / "11_closed_weight_teacher_k10"

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
EXPECTED_TEST_COUNTS = {
    "commonsenseqa": 1221,
    "gsm8k": 1319,
    "kodcode": 1000,
    "spider": 1034,
}
SPIDER_SOURCE = Path(
    "/orcd/pool/008/erimoldi/iar/datasets/spider-release-1403EGqz/spider_data/dev.json"
)
SPIDER_TABLES = Path(
    "/orcd/pool/008/erimoldi/iar/datasets/"
    "spider-release-1403EGqz/spider_data/tables.json"
)
SPIDER_DATABASES = Path(
    "/orcd/pool/008/erimoldi/iar/datasets/spider-release-1403EGqz/spider_data/database"
)


class FinalProtocolError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    inference_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
    files = []
    for child in sorted(
        value
        for value in path.rglob("*")
        if value.is_file() and value.name in inference_names
    ):
        files.append((str(child.relative_to(path)), sha256_file(child)))
    if not files:
        raise FinalProtocolError(f"Empty adapter directory: {path}")
    return _sha_json(files)


def validate_protocol() -> dict[str, Any]:
    value = _read(PROTOCOL)
    if (
        value.get("protocol") != "campaign-final-test-analysis-v1"
        or value.get("official_test_accessed") is not False
        or value.get("mbpp") != "FORBIDDEN"
        or value.get("ifbench") != "DEFERRED"
        or {key: int(row["n"]) for key, row in (value.get("test_splits") or {}).items()}
        != EXPECTED_TEST_COUNTS
    ):
        raise FinalProtocolError("Step 12 protocol is missing or drifted")
    return value


def _complete(path: Path, protocol: str) -> dict[str, Any]:
    value = _read(path)
    if (
        value.get("status") != "complete"
        or value.get("protocol") != protocol
        or value.get("official_test_accessed") is not False
    ):
        raise FinalProtocolError(f"Incomplete upstream result: {path}")
    return value


def _verify_frozen_file(
    *, path: Path, expected_count: int, expected_sha256: str
) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise FinalProtocolError(f"Frozen final split hash mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != expected_count or len({str(row["id"]) for row in rows}) != len(
        rows
    ):
        raise FinalProtocolError(f"Frozen final split coverage mismatch: {path}")
    return rows


def _materialize_test_inputs(attempt: int) -> dict[str, Any]:
    if attempt <= 0:
        raise ValueError("Final freeze attempt must be positive")
    root = TEST_ROOT / f"freeze_attempt{attempt:03d}"
    root.mkdir(parents=True, exist_ok=False)
    inputs: dict[str, Any] = {}

    historical = {
        "commonsenseqa": (
            STEP1 / "datasets/commonsenseqa/frozen_historical_seed23",
            "test.jsonl",
        ),
        "gsm8k": (
            STEP1 / "datasets/gsm8k/frozen_historical_seed23",
            "test.jsonl",
        ),
    }
    for domain, (parent, filename) in historical.items():
        receipt = _read(parent / "campaign_copy_receipt.json")
        path = parent / filename
        expected = str(receipt["files"]["test"]["file_sha256"])
        rows = _verify_frozen_file(
            path=path,
            expected_count=EXPECTED_TEST_COUNTS[domain],
            expected_sha256=expected,
        )
        gold_field = "answer_key" if domain == "commonsenseqa" else "answer_value"
        if any(gold_field not in row for row in rows):
            raise FinalProtocolError(
                f"Missing final answer field: {domain}/{gold_field}"
            )
        inputs[domain] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "n": len(rows),
            "source_manifest": str(parent / "split_manifest.json"),
            "source_manifest_sha256": sha256_file(parent / "split_manifest.json"),
        }

    kod_root = STEP1 / "datasets/kodcode/frozen_seed23_v1"
    kod_manifest = _read(kod_root / "split_manifest.json")
    kod_spec = kod_manifest["splits"]["sealed_test1000"]
    kod_path = kod_root / str(kod_spec["file"])
    kod_rows = _verify_frozen_file(
        path=kod_path,
        expected_count=EXPECTED_TEST_COUNTS["kodcode"],
        expected_sha256=str(kod_spec["file_sha256"]),
    )
    if any(
        not str(row.get("reference_solution", "")).strip()
        or not str(row.get("hidden_tests", "")).strip()
        for row in kod_rows
    ):
        raise FinalProtocolError("KodCode final references are incomplete")
    inputs["kodcode"] = {
        "path": str(kod_path),
        "sha256": sha256_file(kod_path),
        "n": len(kod_rows),
        "source_manifest": str(kod_root / "split_manifest.json"),
        "source_manifest_sha256": sha256_file(kod_root / "split_manifest.json"),
    }

    if not SPIDER_SOURCE.is_file() or not SPIDER_TABLES.is_file():
        raise FileNotFoundError("Pinned Spider 1.0 test assets are missing")
    source_rows = json.loads(SPIDER_SOURCE.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or len(source_rows) != 1034:
        raise FinalProtocolError("Spider official validation must contain 1,034 rows")
    spider_rows = []
    for index, source in enumerate(source_rows):
        required = {"db_id", "question", "query"}
        if not isinstance(source, dict) or not required <= set(source):
            raise FinalProtocolError(f"Malformed Spider final row {index}")
        core = {key: str(source[key]).strip() for key in sorted(required)}
        if not all(core.values()):
            raise FinalProtocolError(f"Empty Spider final field at row {index}")
        spider_rows.append(
            {
                "id": f"spider-1.0-validation-{index:05d}",
                "source_split": "validation",
                "source_index": index,
                "db_id": core["db_id"],
                "question": core["question"],
                "reference_query": core["query"],
                "source_row_sha256": _sha_json(core),
            }
        )
    spider_path = root / "spider_official_test1034.jsonl"
    write_jsonl(spider_path, spider_rows)
    spider_path.chmod(0o600)
    used_databases = sorted({str(row["db_id"]) for row in spider_rows})
    database_hashes = {}
    for db_id in used_databases:
        database = SPIDER_DATABASES / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
        database_hashes[db_id] = sha256_file(database)
    inputs["spider"] = {
        "path": str(spider_path),
        "sha256": sha256_file(spider_path),
        "n": len(spider_rows),
        "source": str(SPIDER_SOURCE),
        "source_sha256": sha256_file(SPIDER_SOURCE),
        "tables_json": str(SPIDER_TABLES),
        "tables_json_sha256": sha256_file(SPIDER_TABLES),
        "database_root": str(SPIDER_DATABASES),
        "database_sha256": database_hashes,
    }
    return {"root": str(root), "inputs": inputs}


def _load_config(path: Path) -> GrpoTrialConfig:
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_grpo_config(path)


def _execution_from_config(
    config_path: Path,
    *,
    selected_step: int,
    final_step: int,
    run_id: str = "attempt001",
    checkpoint_root: Path | None = None,
    prompt_variant: str = "a0",
) -> dict[str, Any]:
    config = _load_config(config_path)
    root = checkpoint_root or Path(str(config.raw["paths"]["checkpoints"])) / run_id
    adapter = _adapter_for_step(root, selected_step, final_step)
    if selected_step and adapter is None:
        raise FinalProtocolError(f"Selected adapter cannot be resolved: {config_path}")
    execution = {
        "actor_source": str(config.actor_source()),
        "actor_model": str(config.raw["actor"]["model_id"]),
        "actor_revision": str(config.raw["actor"]["revision"]),
        "adapter": str(adapter) if adapter is not None else None,
        "adapter_sha256": _tree_sha256(adapter) if adapter is not None else None,
        "prompt_variant": prompt_variant,
        "batch_size": int(config.raw["evaluation"]["batch_size"]),
        "max_new_tokens": int(config.raw["evaluation"]["max_new_tokens"]),
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "selected_step": selected_step,
    }
    execution["execution_key"] = _sha_json(
        {
            key: execution[key]
            for key in (
                "actor_source",
                "adapter",
                "adapter_sha256",
                "prompt_variant",
                "batch_size",
                "max_new_tokens",
            )
        }
    )[:24]
    return execution


def _add_condition(
    conditions: dict[str, list[dict[str, Any]]],
    *,
    domain: str,
    condition_id: str,
    origin_step: int,
    execution: dict[str, Any],
    selection: Path,
    baseline_family: str,
    primary_baseline: str | None,
) -> None:
    if any(row["condition_id"] == condition_id for row in conditions[domain]):
        raise FinalProtocolError(f"Duplicate final condition: {domain}/{condition_id}")
    conditions[domain].append(
        {
            "condition_id": condition_id,
            "origin_step": origin_step,
            "selection": str(selection),
            "selection_sha256": sha256_file(selection),
            "baseline_family": baseline_family,
            "primary_baseline": primary_baseline,
            **execution,
        }
    )


def _replace_adapter(
    execution: dict[str, Any], *, adapter: Path | None, selected_step: int
) -> dict[str, Any]:
    value = dict(execution)
    value.update(
        {
            "adapter": str(adapter) if adapter is not None else None,
            "adapter_sha256": _tree_sha256(adapter) if adapter is not None else None,
            "selected_step": selected_step,
        }
    )
    value["execution_key"] = _sha_json(
        {
            key: value[key]
            for key in (
                "actor_source",
                "adapter",
                "adapter_sha256",
                "prompt_variant",
                "batch_size",
                "max_new_tokens",
            )
        }
    )[:24]
    return value


def prepare_final_registry(*, attempt: int = 1) -> dict[str, Any]:
    validate_protocol()
    upstream_paths = {
        "step5": STEP5 / "results/four_domain_screen_1p5b.json",
        "step7": STEP7 / "results/baseline_suite_validation_v1.json",
        "step8": STEP8 / "results/model_scale_3b_validation_v1.json",
        "step9": STEP9 / "results/pairwise_full_validation_v1.json",
        "step10": STEP10 / "results/discriminator_transfer_validation_v1.json",
        "step11": STEP11 / "results/closed_weight_validation_v1.json",
    }
    upstream = {
        "step5": _complete(
            upstream_paths["step5"], "campaign-four-domain-screen-1p5b-v1"
        ),
        "step7": _complete(
            upstream_paths["step7"], "campaign-selected-domain-baseline-suite-v1"
        ),
        "step8": _complete(
            upstream_paths["step8"], "campaign-selected-domain-model-scale-3b-v1"
        ),
        "step9": _complete(
            upstream_paths["step9"], "campaign-pairwise-generate-distinguish-full-v1"
        ),
        "step10": _complete(
            upstream_paths["step10"], "campaign-discriminator-transfer-final-v1"
        ),
        "step11": _complete(
            upstream_paths["step11"], "campaign-step11-closed-weight-final-v1"
        ),
    }
    selected = list(upstream["step7"].get("selected_domains") or [])
    if (
        len(selected) != 2
        or selected[0] != "commonsenseqa"
        or set(upstream["step8"].get("domains") or {}) != set(selected)
        or not set(upstream["step9"].get("passing_domains") or []) <= set(selected)
    ):
        raise FinalProtocolError("Upstream domain selections are inconsistent")

    frozen = _materialize_test_inputs(attempt)
    freeze_path = Path(str(frozen["root"])) / "split_freeze.json"
    write_json(
        freeze_path,
        {
            "schema_version": 1,
            "status": "frozen_before_generation",
            "frozen_at": utc_now(),
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "inputs": frozen["inputs"],
            "official_test_accessed": True,
            "model_generation_started": False,
            "mbpp": "FORBIDDEN",
            "ifbench": "DEFERRED",
        },
    )

    conditions: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    base_configs = {
        domain: STEP5 / f"domains/{domain}/grpo/configs/screen_prep001.yaml"
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        step5 = upstream["step5"]["domains"][domain]
        base = _execution_from_config(
            base_configs[domain], selected_step=0, final_step=2000
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id="qwen2.5_1.5b_a0",
            origin_step=5,
            execution=base,
            selection=upstream_paths["step5"],
            baseline_family="qwen2.5_1.5b",
            primary_baseline=None,
        )
        imitation = _execution_from_config(
            base_configs[domain],
            selected_step=0,
            final_step=2000,
            prompt_variant="prompt_imitation",
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id="qwen2.5_1.5b_prompt_imitation",
            origin_step=5,
            execution=imitation,
            selection=upstream_paths["step5"],
            baseline_family="qwen2.5_1.5b",
            primary_baseline="qwen2.5_1.5b_a0",
        )
        identity = _execution_from_config(
            base_configs[domain],
            selected_step=int(step5["selected_step"]),
            final_step=2000,
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id="step5_grpo_identity",
            origin_step=5,
            execution=identity,
            selection=upstream_paths["step5"],
            baseline_family="qwen2.5_1.5b",
            primary_baseline="qwen2.5_1.5b_a0",
        )

    step7_preparation = STEP7 / "manifests/baseline_suite_prepared_attempt001.json"
    step7_prepared = _read(step7_preparation)
    if step7_prepared.get("status") != "PREPARED_NOT_LAUNCHED":
        raise FinalProtocolError("Step 7 preparation is incomplete")
    for domain in selected:
        domain_result = upstream["step7"]["results"][domain]
        for method in ("sft_teacher", "grpo_quality", "grpo_oracle"):
            chosen = domain_result[method]
            selected_step = int(chosen["selected_step"])
            if method == "sft_teacher":
                adapter = _adapter_for_step(
                    STEP7 / f"domains/{domain}/sft_teacher/sft_checkpoints/attempt001",
                    selected_step,
                    500,
                )
                if selected_step and adapter is None:
                    raise FinalProtocolError(
                        f"Selected SFT adapter cannot be resolved: {domain}/{selected_step}"
                    )
                execution = _replace_adapter(
                    _execution_from_config(
                        base_configs[domain], selected_step=0, final_step=2000
                    ),
                    adapter=adapter,
                    selected_step=selected_step,
                )
            else:
                config_path = Path(
                    str(step7_prepared["configs"][domain][method]["path"])
                )
                execution = _execution_from_config(
                    config_path, selected_step=selected_step, final_step=2000
                )
            _add_condition(
                conditions,
                domain=domain,
                condition_id=f"step7_{method}",
                origin_step=7,
                execution=execution,
                selection=Path(str(chosen["selection"])),
                baseline_family="qwen2.5_1.5b",
                primary_baseline="qwen2.5_1.5b_a0",
            )

    step8_preparation = STEP8 / "manifests/model_scale_3b_prepared_attempt001.json"
    step8_prepared = _read(step8_preparation)
    for domain in selected:
        config_path = Path(str(step8_prepared["configs"][domain]["path"]))
        base3 = _execution_from_config(config_path, selected_step=0, final_step=2000)
        _add_condition(
            conditions,
            domain=domain,
            condition_id="qwen2.5_3b_a0",
            origin_step=8,
            execution=base3,
            selection=upstream_paths["step8"],
            baseline_family="qwen2.5_3b",
            primary_baseline=None,
        )
        chosen = upstream["step8"]["domains"][domain]
        grpo3 = _execution_from_config(
            config_path,
            selected_step=int(chosen["selected_step"]),
            final_step=2000,
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id="step8_grpo_identity_3b",
            origin_step=8,
            execution=grpo3,
            selection=upstream_paths["step8"],
            baseline_family="qwen2.5_3b",
            primary_baseline="qwen2.5_3b_a0",
        )

    step9_preparation = STEP9 / "manifests/pairwise_prepared_attempt001.json"
    step9_prepared = _read(step9_preparation)
    for domain in upstream["step9"].get("passing_domains") or []:
        chosen = upstream["step9"]["domains"][domain]
        config_path = Path(str(step9_prepared["configs"][domain]["path"]))
        execution = _execution_from_config(
            config_path,
            selected_step=int(chosen["selected_step"]),
            final_step=2000,
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id="step9_grpo_pairwise",
            origin_step=9,
            execution=execution,
            selection=upstream_paths["step9"],
            baseline_family="qwen2.5_1.5b",
            primary_baseline="qwen2.5_1.5b_a0",
        )

    step10_preparation = (
        STEP10 / "manifests/discriminator_transfer_prepared_attempt001.json"
    )
    step10_prepared = _read(step10_preparation)
    for row in upstream["step10"].get("rows") or []:
        pair_id = str(row["pair_id"])
        domain = str(row["domain"])
        config_path = Path(
            str(step10_prepared["pairs"][pair_id]["grpo_config"]["path"])
        )
        execution = _execution_from_config(
            config_path,
            selected_step=int(row["selected_step"]),
            final_step=2000,
        )
        _add_condition(
            conditions,
            domain=domain,
            condition_id=f"step10_transfer_{row['candidate_slug']}",
            origin_step=10,
            execution=execution,
            selection=Path(str(row["selection"])),
            baseline_family="qwen2.5_1.5b",
            primary_baseline="qwen2.5_1.5b_a0",
        )

    step11_preparation = (
        STEP11 / "manifests/closed_weight_full_prepared_attempt001.json"
    )
    if upstream["step11"].get("domains"):
        step11_prepared = _read(step11_preparation)
        for domain, chosen in upstream["step11"]["domains"].items():
            config_path = Path(str(step11_prepared["configs"][domain]["path"]))
            execution = _execution_from_config(
                config_path,
                selected_step=int(chosen["selected_step"]),
                final_step=2000,
            )
            _add_condition(
                conditions,
                domain=domain,
                condition_id="step11_closed_weight_k10",
                origin_step=11,
                execution=execution,
                selection=Path(str(chosen["selection"])),
                baseline_family="qwen2.5_1.5b",
                primary_baseline="qwen2.5_1.5b_a0",
            )

    executions: dict[str, list[dict[str, Any]]] = {}
    for domain, rows in conditions.items():
        canonical: dict[str, str] = {}
        unique = []
        for row in rows:
            key = str(row["execution_key"])
            row["canonical_condition_id"] = canonical.setdefault(
                key, str(row["condition_id"])
            )
            row["is_alias"] = row["canonical_condition_id"] != row["condition_id"]
            if not row["is_alias"]:
                unique.append(
                    {
                        "domain": domain,
                        **{
                            key: row[key]
                            for key in (
                                "execution_key",
                                "canonical_condition_id",
                                "actor_source",
                                "actor_model",
                                "actor_revision",
                                "adapter",
                                "adapter_sha256",
                                "prompt_variant",
                                "batch_size",
                                "max_new_tokens",
                            )
                        },
                    }
                )
        executions[domain] = unique

    result = {
        "schema_version": 1,
        "status": "FROZEN_READY_FOR_REFERENCE_GATES",
        "created_at": utc_now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "split_freeze": str(freeze_path),
        "split_freeze_sha256": sha256_file(freeze_path),
        "test_inputs": frozen["inputs"],
        "selected_domains": selected,
        "conditions": conditions,
        "executions": executions,
        "execution_count": sum(map(len, executions.values())),
        "condition_count": sum(map(len, conditions.values())),
        "upstream": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in upstream_paths.items()
        },
        "official_test_accessed": True,
        "model_generation_started": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    if REGISTRY.exists():
        raise FileExistsError(REGISTRY)
    write_json(REGISTRY, result)
    return result


def _validated_registry(path: Path = REGISTRY) -> dict[str, Any]:
    value = _read(path)
    if (
        value.get("status") != "FROZEN_READY_FOR_REFERENCE_GATES"
        or value.get("official_test_accessed") is not True
        or value.get("model_generation_started") is not False
        or value.get("protocol_sha256") != sha256_file(PROTOCOL)
    ):
        raise FinalProtocolError("Final condition registry is incomplete")
    if (
        set(value.get("test_inputs") or {}) != set(DOMAINS)
        or set(value.get("conditions") or {}) != set(DOMAINS)
        or set(value.get("executions") or {}) != set(DOMAINS)
        or any(not value["conditions"][domain] for domain in DOMAINS)
    ):
        raise FinalProtocolError("Final registry does not cover the four domains")
    for domain, item in value["test_inputs"].items():
        path = Path(str(item["path"]))
        if (
            domain not in DOMAINS
            or int(item["n"]) != EXPECTED_TEST_COUNTS[domain]
            or sha256_file(path) != str(item["sha256"])
        ):
            raise FinalProtocolError(f"Final test input drifted: {domain}")
    return value


def audit_kodcode_references(
    *, registry: Path = REGISTRY, attempt: int = 1, workers: int = 8, timeout: int = 60
) -> dict[str, Any]:
    frozen = _validated_registry(registry)
    rows = read_jsonl(Path(str(frozen["test_inputs"]["kodcode"]["path"])))
    output_dir = STEP_ROOT / f"reference_gates/kodcode/attempt{attempt:03d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = [
        (str(row["reference_solution"]), str(row["hidden_tests"]), timeout)
        for row in rows
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(_score_kodcode, payloads))
    audited = [
        {
            "id": str(row["id"]),
            "source_index": int(row["source_index"]),
            "valid_reference": bool(score["correct"]),
            "score": score,
        }
        for row, score in zip(rows, scores, strict=True)
    ]
    write_jsonl(output_dir / "rows.jsonl", audited)
    valid = [row["id"] for row in audited if row["valid_reference"]]
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": "kodcode",
        "n": len(rows),
        "valid_n": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "valid_ids": valid,
        "rows": str(output_dir / "rows.jsonl"),
        "rows_sha256": sha256_file(output_dir / "rows.jsonl"),
        "test_sha256": frozen["test_inputs"]["kodcode"]["sha256"],
        "workers": workers,
        "timeout_seconds": timeout,
        "official_test_accessed": True,
    }
    write_json(output_dir / "summary.json", result)
    return result


def audit_spider_references(
    *, registry: Path = REGISTRY, attempt: int = 1, workers: int = 8
) -> dict[str, Any]:
    frozen = _validated_registry(registry)
    spec = frozen["test_inputs"]["spider"]
    rows = read_jsonl(Path(str(spec["path"])))
    output_dir = STEP_ROOT / f"reference_gates/spider/attempt{attempt:03d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    database_root = Path(str(spec["database_root"]))
    for db_id, expected in spec["database_sha256"].items():
        path = database_root / db_id / f"{db_id}.sqlite"
        if sha256_file(path) != expected:
            raise FinalProtocolError(f"Spider database drifted: {db_id}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        audited = list(
            pool.map(
                _audit_spider_reference_one,
                [(row, database_root) for row in rows],
            )
        )
    write_jsonl(output_dir / "rows.jsonl", audited)
    valid = [str(row["id"]) for row in audited if bool(row["valid"])]
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": "spider",
        "n": len(rows),
        "valid_n": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "valid_ids": valid,
        "rows": str(output_dir / "rows.jsonl"),
        "rows_sha256": sha256_file(output_dir / "rows.jsonl"),
        "test_sha256": spec["sha256"],
        "database_root": str(database_root),
        "execution_limits": SPIDER_EXECUTION_LIMITS,
        "workers": workers,
        "official_test_accessed": True,
    }
    write_json(output_dir / "summary.json", result)
    return result


def collect_reference_gate(
    *, registry: Path = REGISTRY, kodcode_attempt: int = 1, spider_attempt: int = 1
) -> dict[str, Any]:
    frozen = _validated_registry(registry)
    minimum_valid = float(
        validate_protocol()["reference_gates"]["minimum_valid_fraction"]
    )
    sources = {
        "kodcode": STEP_ROOT
        / f"reference_gates/kodcode/attempt{kodcode_attempt:03d}/summary.json",
        "spider": STEP_ROOT
        / f"reference_gates/spider/attempt{spider_attempt:03d}/summary.json",
    }
    domains = {
        "commonsenseqa": {
            "n": 1221,
            "valid_n": 1221,
            "valid_ids": [
                str(row["id"])
                for row in read_jsonl(
                    Path(str(frozen["test_inputs"]["commonsenseqa"]["path"]))
                )
            ],
            "gate": "structural",
        },
        "gsm8k": {
            "n": 1319,
            "valid_n": 1319,
            "valid_ids": [
                str(row["id"])
                for row in read_jsonl(Path(str(frozen["test_inputs"]["gsm8k"]["path"])))
            ],
            "gate": "structural",
        },
    }
    passed = True
    for domain, path in sources.items():
        item = _read(path)
        if (
            item.get("status") != "complete"
            or item.get("domain") != domain
            or item.get("official_test_accessed") is not True
            or item.get("test_sha256") != frozen["test_inputs"][domain]["sha256"]
        ):
            raise FinalProtocolError(f"Incomplete final reference audit: {domain}")
        domain_passed = float(item["valid_fraction"]) >= minimum_valid
        passed &= domain_passed
        domains[domain] = {
            "n": int(item["n"]),
            "valid_n": int(item["valid_n"]),
            "valid_fraction": float(item["valid_fraction"]),
            "valid_ids": item["valid_ids"],
            "gate": "GO" if domain_passed else "NO_GO",
            "summary": str(path),
            "summary_sha256": sha256_file(path),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "decision": "GO" if passed else "NO_GO",
        "passed": passed,
        "minimum_reference_valid_fraction": minimum_valid,
        "domains": domains,
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "official_test_accessed": True,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    if REFERENCE_GATE.exists():
        raise FileExistsError(REFERENCE_GATE)
    write_json(REFERENCE_GATE, result)
    return result


def _execution(registry: dict[str, Any], domain: str, key: str) -> dict[str, Any]:
    matches = [
        row
        for row in registry["executions"][domain]
        if str(row["execution_key"]) == key
    ]
    if len(matches) != 1:
        raise FinalProtocolError(f"Unknown final execution: {domain}/{key}")
    return matches[0]


def generate_final_execution(
    *, domain: str, execution_key: str, registry: Path = REGISTRY, attempt: int = 1
) -> dict[str, Any]:
    if domain not in DOMAINS or attempt <= 0:
        raise ValueError("Invalid final generation target")
    frozen = _validated_registry(registry)
    gate = _read(REFERENCE_GATE)
    if gate.get("passed") is not True:
        raise FinalProtocolError("Final generation is blocked by reference gates")
    execution = _execution(frozen, domain, execution_key)
    split = frozen["test_inputs"][domain]
    examples = read_jsonl(Path(str(split["path"])))
    output = (
        STEP_ROOT
        / f"executions/{domain}/{execution_key}/generation_attempt{attempt:03d}"
    )
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    tables = (
        Path(str(frozen["test_inputs"]["spider"]["tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)[
        str(execution["prompt_variant"])
    ]
    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    source = Path(str(execution["actor_source"]))
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = None
    model = None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        adapter = execution.get("adapter")
        if adapter:
            adapter_path = Path(str(adapter))
            if _tree_sha256(adapter_path) != execution["adapter_sha256"]:
                raise FinalProtocolError("Final adapter hash drifted")
            model = PeftModel.from_pretrained(
                base_model, adapter_path, is_trainable=False
            )
        else:
            model = base_model
        model.eval()
        candidates = []
        batch_size = int(execution["batch_size"])
        max_tokens = int(execution["max_new_tokens"])
        for start in range(0, len(examples), batch_size):
            batch_rows = examples[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for example, output_ids in zip(batch_rows, generated, strict=True):
                completion = output_ids[prompt_length:]
                candidates.append(
                    {
                        "id": str(example["id"]),
                        "source_index": int(example["source_index"]),
                        "domain": domain,
                        "execution_key": execution_key,
                        "canonical_condition_id": execution["canonical_condition_id"],
                        "response": tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip(),
                        "completion_tokens": len(completion),
                        "max_new_tokens": max_tokens,
                        "hit_max_new_tokens": len(completion) >= max_tokens,
                    }
                )
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "execution_key": execution_key,
            "canonical_condition_id": execution["canonical_condition_id"],
            "n": len(candidates),
            "truncated": sum(row["hit_max_new_tokens"] for row in candidates),
            "candidate_file": str(output / "candidates.jsonl"),
            "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
            "test_file_sha256": split["sha256"],
            "registry_sha256": sha256_file(registry),
            "sdpa": sdpa,
            "official_test_accessed": True,
        }
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": True,
            },
        )
        raise
    finally:
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_final_execution(
    *,
    domain: str,
    execution_key: str,
    registry: Path = REGISTRY,
    generation_attempt: int = 1,
    scoring_attempt: int = 1,
    workers: int = 8,
) -> dict[str, Any]:
    frozen = _validated_registry(registry)
    gate = _read(REFERENCE_GATE)
    if gate.get("passed") is not True:
        raise FinalProtocolError("Final scoring is blocked by reference gates")
    _execution(frozen, domain, execution_key)
    split_path = Path(str(frozen["test_inputs"][domain]["path"]))
    examples = read_jsonl(split_path)
    generation = (
        STEP_ROOT
        / f"executions/{domain}/{execution_key}/generation_attempt{generation_attempt:03d}"
    )
    generation_summary = _read(generation / "summary.json")
    candidates = read_jsonl(generation / "candidates.jsonl")
    if (
        generation_summary.get("status") != "complete"
        or generation_summary.get("candidate_file_sha256")
        != sha256_file(generation / "candidates.jsonl")
        or [str(row["id"]) for row in candidates]
        != [str(row["id"]) for row in examples]
    ):
        raise FinalProtocolError("Final generation is incomplete or misaligned")
    valid_ids = set(map(str, gate["domains"][domain]["valid_ids"]))
    example_by_id = {str(row["id"]): row for row in examples}
    scored = [row for row in candidates if str(row["id"]) in valid_ids]
    if domain in {"commonsenseqa", "gsm8k"}:
        _score_simple(domain, examples, candidates)
        scored = candidates
    elif domain == "kodcode":
        payloads = [
            (
                str(row["response"]),
                str(example_by_id[str(row["id"])]["hidden_tests"]),
                60,
            )
            for row in scored
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_score_kodcode, payloads))
        for row, score in zip(scored, scores, strict=True):
            row["score"] = score
    elif domain == "spider":
        database_root = Path(str(frozen["test_inputs"]["spider"]["database_root"]))
        payloads = [
            (example_by_id[str(row["id"])], row, database_root) for row in scored
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_score_spider_one, payloads))
        for row, score in zip(scored, scores, strict=True):
            row.update(score)
    else:
        raise ValueError(domain)
    excluded = set(example_by_id) - valid_ids
    for row in candidates:
        if str(row["id"]) in excluded:
            row["score"] = {
                "correct": False,
                "valid": False,
                "excluded_reference_invalid": True,
            }
    output = (
        STEP_ROOT
        / f"executions/{domain}/{execution_key}/score_attempt{scoring_attempt:03d}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "candidates.jsonl", candidates)
    correct = sum(bool(row["score"]["correct"]) for row in scored)
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "execution_key": execution_key,
        "canonical_condition_id": generation_summary["canonical_condition_id"],
        "n": len(candidates),
        "scored_n": len(scored),
        "reference_invalid_excluded": len(candidates) - len(scored),
        "correct": correct,
        "task_utility": correct / len(scored),
        "valid_output_n": sum(bool(row["score"].get("valid", True)) for row in scored),
        "status_counts": dict(
            sorted(
                Counter(
                    str(row["score"].get("status", "scored")) for row in scored
                ).items()
            )
        ),
        "candidate_file": str(output / "candidates.jsonl"),
        "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
        "generation_summary": str(generation / "summary.json"),
        "generation_summary_sha256": sha256_file(generation / "summary.json"),
        "test_file_sha256": sha256_file(split_path),
        "official_test_accessed": True,
    }
    write_json(output / "summary.json", result)
    return result


def collect_final_scores(
    *, registry: Path = REGISTRY, scoring_attempt: int = 1
) -> dict[str, Any]:
    frozen = _validated_registry(registry)
    gate = _read(REFERENCE_GATE)
    if gate.get("passed") is not True:
        raise FinalProtocolError("Final score collection is blocked by reference gates")
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_rows = []
        for condition in frozen["conditions"][domain]:
            key = str(condition["execution_key"])
            summary_path = (
                STEP_ROOT
                / f"executions/{domain}/{key}/score_attempt{scoring_attempt:03d}/summary.json"
            )
            summary = _read(summary_path)
            candidate_path = Path(str(summary["candidate_file"]))
            if (
                summary.get("status") != "complete"
                or summary.get("domain") != domain
                or summary.get("execution_key") != key
                or summary.get("official_test_accessed") is not True
                or sha256_file(candidate_path) != summary.get("candidate_file_sha256")
            ):
                raise FinalProtocolError(
                    f"Incomplete final score: {domain}/{condition['condition_id']}"
                )
            domain_rows.append(
                {
                    **condition,
                    "task_utility": float(summary["task_utility"]),
                    "correct": int(summary["correct"]),
                    "scored_n": int(summary["scored_n"]),
                    "reference_invalid_excluded": int(
                        summary["reference_invalid_excluded"]
                    ),
                    "valid_output_n": int(summary["valid_output_n"]),
                    "score_summary": str(summary_path),
                    "score_summary_sha256": sha256_file(summary_path),
                    "candidate_file": str(candidate_path),
                    "candidate_file_sha256": sha256_file(candidate_path),
                }
            )
        domains[domain] = {
            "test_n": EXPECTED_TEST_COUNTS[domain],
            "scored_n": int(gate["domains"][domain]["valid_n"]),
            "conditions": domain_rows,
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-final-test-results-v1",
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "reference_gate": str(REFERENCE_GATE),
        "reference_gate_sha256": sha256_file(REFERENCE_GATE),
        "domains": domains,
        "official_test_accessed": True,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output = STEP_ROOT / "results/final_test_results_v1.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def _bootstrap_paired(
    left: list[float], right: list[float], *, seed: int, replicates: int = 10_000
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap inputs are misaligned")
    try:
        import numpy as np

        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        rng = np.random.default_rng(seed)
        estimates = []
        for start in range(0, replicates, 500):
            size = min(500, replicates - start)
            indexes = rng.integers(0, len(delta), size=(size, len(delta)))
            estimates.extend(np.mean(delta[indexes], axis=1).tolist())
    except ImportError:
        rng = random.Random(seed)
        delta = [a - b for a, b in zip(left, right, strict=True)]
        estimates = [
            statistics.fmean(delta[rng.randrange(len(delta))] for _ in delta)
            for _ in range(replicates)
        ]
    estimates.sort()
    return {
        "delta": statistics.fmean(a - b for a, b in zip(left, right, strict=True)),
        "ci95_low": estimates[math.floor(0.025 * replicates)],
        "ci95_high": estimates[math.ceil(0.975 * replicates) - 1],
        "replicates": replicates,
    }


def _mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    b = sum(a and not z for a, z in zip(left, right, strict=True))
    c = sum(not a and z for a, z in zip(left, right, strict=True))
    n = b + c
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(b, c) + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"candidate_only_correct": b, "baseline_only_correct": c, "p_exact": p}


def _pearson_optional(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    lm = statistics.fmean(left)
    rm = statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right, strict=True))
    ls = math.sqrt(sum((x - lm) ** 2 for x in left))
    rs = math.sqrt(sum((y - rm) ** 2 for y in right))
    return None if not ls or not rs else numerator / (ls * rs)


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["mcnemar"]["p_exact"]))
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * float(row["mcnemar"]["p_exact"]))
        running = max(running, adjusted)
        row["holm_p"] = running


def analyze_final_scores(*, result_path: Path | None = None) -> dict[str, Any]:
    result_path = result_path or STEP_ROOT / "results/final_test_results_v1.json"
    final = _read(result_path)
    if (
        final.get("status") != "complete"
        or final.get("official_test_accessed") is not True
    ):
        raise FinalProtocolError("Final test results are incomplete")
    output_domains = {}
    for domain, domain_data in final["domains"].items():
        conditions = {
            str(row["condition_id"]): row for row in domain_data["conditions"]
        }
        vectors = {}
        shortcut = {}
        for condition_id, condition in conditions.items():
            rows = [
                row
                for row in read_jsonl(Path(str(condition["candidate_file"])))
                if not bool(row["score"].get("excluded_reference_invalid", False))
            ]
            vectors[condition_id] = {
                str(row["id"]): bool(row["score"]["correct"]) for row in rows
            }
            lengths = [float(row["completion_tokens"]) for row in rows]
            correctness = [float(bool(row["score"]["correct"])) for row in rows]
            shortcut[condition_id] = {
                "mean_completion_tokens": statistics.fmean(lengths),
                "length_correctness_pearson": _pearson_optional(lengths, correctness),
                "truncated_n": sum(bool(row["hit_max_new_tokens"]) for row in rows),
                "invalid_output_n": sum(
                    not bool(row["score"].get("valid", True)) for row in rows
                ),
            }
        comparisons = []
        for condition_id, condition in conditions.items():
            baseline_id = condition.get("primary_baseline")
            if not baseline_id:
                continue
            if baseline_id not in vectors or set(vectors[condition_id]) != set(
                vectors[baseline_id]
            ):
                raise FinalProtocolError(
                    f"Paired final coverage mismatch: {domain}/{condition_id}"
                )
            ids = sorted(vectors[condition_id])
            candidate = [vectors[condition_id][row_id] for row_id in ids]
            baseline = [vectors[baseline_id][row_id] for row_id in ids]
            seed = int(
                hashlib.sha256(f"23|{domain}|{condition_id}".encode()).hexdigest()[:8],
                16,
            )
            comparisons.append(
                {
                    "condition_id": condition_id,
                    "baseline_id": baseline_id,
                    "baseline_family": condition["baseline_family"],
                    "n": len(ids),
                    "paired_bootstrap": _bootstrap_paired(
                        list(map(float, candidate)),
                        list(map(float, baseline)),
                        seed=seed,
                    ),
                    "mcnemar": _mcnemar(candidate, baseline),
                    "continuous_permutation": "not_applicable_binary_task_metric",
                }
            )
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in comparisons:
            by_family[str(row["baseline_family"])].append(row)
        for rows in by_family.values():
            _holm(rows)
        output_domains[domain] = {
            "conditions": {
                key: {
                    "task_utility": float(value["task_utility"]),
                    "correct": int(value["correct"]),
                    "scored_n": int(value["scored_n"]),
                }
                for key, value in conditions.items()
            },
            "comparisons": comparisons,
            "shortcut_audit": shortcut,
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-final-statistical-analysis-v1",
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 23,
        "mcnemar": "exact_two_sided",
        "multiplicity": "Holm within domain and baseline family",
        "domains": output_domains,
        "official_test_accessed": True,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output = STEP_ROOT / "results/final_statistical_analysis_v1.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def _collect_job_ids(value: Any, *, parent_key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found |= _collect_job_ids(item, parent_key=str(key))
    elif isinstance(value, list):
        for item in value:
            found |= _collect_job_ids(item, parent_key=parent_key)
    elif (
        isinstance(value, int)
        and not isinstance(value, bool)
        and "job" in parent_key.lower()
        and value >= 1_000_000
    ):
        found.add(value)
    return found


def _api_reconciliation() -> dict[str, Any]:
    receipts: dict[str, dict[str, Any]] = {}
    scanned_files = 0
    ledger_files: set[str] = set()
    malformed_rows = 0
    for path in sorted(CAMPAIGN_ROOT.rglob("*.jsonl")):
        scanned_files += 1
        step = str(path.relative_to(CAMPAIGN_ROOT).parts[0])
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue
                if not isinstance(row, dict):
                    continue
                raw = row.get("raw_response")
                raw = raw if isinstance(raw, dict) else {}
                usage = row.get("usage")
                usage = usage if isinstance(usage, dict) else raw.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                generation_id = (
                    row.get("generation_id")
                    or row.get("openrouter_response_id")
                    or row.get("response_id")
                    or raw.get("id")
                )
                row_id = row.get("id")
                if (
                    not generation_id
                    and isinstance(row_id, str)
                    and row_id.startswith("gen-")
                ):
                    generation_id = row_id
                cost = usage.get("cost")
                has_cost = (
                    not isinstance(cost, bool)
                    and isinstance(cost, (int, float))
                    and math.isfinite(float(cost))
                    and float(cost) >= 0
                )
                if not generation_id and not has_cost:
                    continue
                ledger_files.add(str(path.relative_to(CAMPAIGN_ROOT)))
                receipt_key = (
                    str(generation_id)
                    if generation_id
                    else f"{path}:{line_number}:cost-without-id"
                )
                value = {
                    "step": step,
                    "cost_usd": float(cost) if has_cost else None,
                    "has_generation_id": bool(generation_id),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
                prior = receipts.get(receipt_key)
                if prior is not None:
                    if (
                        prior["cost_usd"] is not None
                        and value["cost_usd"] is not None
                        and not math.isclose(prior["cost_usd"], value["cost_usd"])
                    ):
                        raise FinalProtocolError(
                            f"Conflicting API cost receipts: {receipt_key}"
                        )
                    for key in (
                        "cost_usd",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ):
                        if prior.get(key) is None and value.get(key) is not None:
                            prior[key] = value[key]
                    continue
                receipts[receipt_key] = value
    by_step: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "receipts": 0,
            "receipts_with_cost": 0,
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    for row in receipts.values():
        item = by_step[row["step"]]
        item["receipts"] += 1
        if row["cost_usd"] is not None:
            item["receipts_with_cost"] += 1
            item["cost_usd"] += float(row["cost_usd"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item[key] += int(value)
    return {
        "scanned_jsonl_files": scanned_files,
        "identified_ledger_files": len(ledger_files),
        "malformed_jsonl_rows": malformed_rows,
        "unique_api_receipts": len(receipts),
        "receipts_with_generation_id": sum(
            row["has_generation_id"] for row in receipts.values()
        ),
        "receipts_with_cost": sum(
            row["cost_usd"] is not None for row in receipts.values()
        ),
        "receipts_without_generation_id": sum(
            not row["has_generation_id"] for row in receipts.values()
        ),
        "observed_cost_usd": sum(
            float(row["cost_usd"] or 0.0) for row in receipts.values()
        ),
        "by_step": dict(sorted(by_step.items())),
        "scope": "All campaign JSONL rows carrying an explicit provider generation/response ID or a finite non-negative usage.cost",
        "deduplication": "Provider generation/response ID; cost-bearing rows without one use their immutable file and line",
    }


def reconcile_resources() -> dict[str, Any]:
    job_ids: set[int] = set()
    sources = []
    for path in sorted(CAMPAIGN_ROOT.glob("*/manifests/**/*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        try:
            values = (
                [json.loads(line) for line in path.read_text().splitlines() if line]
                if path.suffix == ".jsonl"
                else [_read(path)]
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
        before = len(job_ids)
        for value in values:
            job_ids |= _collect_job_ids(value)
        if len(job_ids) > before:
            sources.append(str(path))
    status_path = CAMPAIGN_ROOT / "00_campaign_control/status/STATUS.json"
    if status_path.is_file():
        active_jobs = _read(status_path).get("active_jobs") or {}
        active_job_values = (
            [*active_jobs.keys(), *active_jobs.values()]
            if isinstance(active_jobs, dict)
            else active_jobs
        )
        job_ids |= {int(value) for value in active_job_values if str(value).isdigit()}
    for path in CAMPAIGN_ROOT.rglob("*.jobid"):
        try:
            value = path.read_text(encoding="utf-8").strip().split(";", 1)[0]
            if value.isdigit():
                job_ids.add(int(value))
        except OSError:
            continue
    records = []
    for start in range(0, len(job_ids), 200):
        batch = sorted(job_ids)[start : start + 200]
        command = [
            "sacct",
            "-n",
            "-X",
            "-j",
            ",".join(map(str, batch)),
            "-o",
            "JobIDRaw,State,ElapsedRaw,AllocCPUS,ReqTRES,AllocTRES",
            "-P",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        for line in completed.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 6 or not parts[0].isdigit():
                continue
            records.append(
                {
                    "job_id": int(parts[0]),
                    "state": parts[1],
                    "elapsed_seconds": int(parts[2] or 0),
                    "allocated_cpus": int(parts[3] or 0),
                    "requested_tres": parts[4],
                    "allocated_tres": parts[5],
                }
            )

    def gpu_count(record: dict[str, Any]) -> int:
        text = record["allocated_tres"] + "," + record["requested_tres"]
        for field in text.split(","):
            if "gpu" in field and "=" in field:
                try:
                    return int(field.rsplit("=", 1)[1])
                except ValueError:
                    pass
        return 0

    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "discovered_job_ids": len(job_ids),
        "sacct_records": len(records),
        "state_counts": dict(sorted(Counter(row["state"] for row in records).items())),
        "allocated_cpu_hours": sum(
            row["elapsed_seconds"] * row["allocated_cpus"] for row in records
        )
        / 3600,
        "allocated_gpu_hours": sum(
            row["elapsed_seconds"] * gpu_count(row) for row in records
        )
        / 3600,
        "jobs": records,
        "manifest_sources": sources,
        "api_reconciliation": _api_reconciliation(),
        "credential_policy": "No API-key values or key fingerprints are copied into this report.",
        "official_test_accessed": True,
    }
    output = STEP_ROOT / "results/resource_reconciliation_v1.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def collect_final_report() -> dict[str, Any]:
    sources = {
        "test_results": STEP_ROOT / "results/final_test_results_v1.json",
        "analysis": STEP_ROOT / "results/final_statistical_analysis_v1.json",
        "resources": STEP_ROOT / "results/resource_reconciliation_v1.json",
    }
    for path in sources.values():
        value = _read(path)
        if value.get("status") != "complete":
            raise FinalProtocolError(f"Incomplete Step 12 output: {path}")
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step12-final-v1",
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in sources.items()
        },
        "official_test_accessed": True,
        "official_test_access_policy": "opened only after every upstream validation result was terminal and the final condition registry was frozen",
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output = STEP_ROOT / "results/FINAL_REPORT.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result
