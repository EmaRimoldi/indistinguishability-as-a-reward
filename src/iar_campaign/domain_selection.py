from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMPAIGN_ROOT = Path(
    "/orcd/pool/008/erimoldi/iar/experiments/campaign_20260908_v2"
)
STEP5_ROOT = CAMPAIGN_ROOT / "05_four_domain_screen_1p5b"
STEP6_ROOT = CAMPAIGN_ROOT / "06_two_domain_selection"
PROTOCOL_PATH = STEP6_ROOT / "configs/domain_selection_protocol_v1.json"
SCREEN_DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
MANDATORY_DOMAIN = "commonsenseqa"
# Frozen before observing Step 5 outcomes. Earlier entries win only after an exact
# tie in validation gain and selected validation utility.
SECOND_DOMAIN_TIEBREAK = ("gsm8k", "kodcode", "spider")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_two_domains(
    *,
    screen_result: Path,
    output: Path,
) -> dict[str, Any]:
    screen_result = screen_result.resolve()
    output = output.resolve()
    expected_screen = (STEP5_ROOT / "results/four_domain_screen_1p5b.json").resolve()
    expected_output = (STEP6_ROOT / "results/two_domain_selection_v1.json").resolve()
    if screen_result != expected_screen:
        raise ValueError(f"Unexpected Step 5 result path: {screen_result}")
    if output != expected_output:
        raise ValueError(f"Unexpected Step 6 output path: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {output}")

    protocol = _read_json(PROTOCOL_PATH)
    if protocol != {
        "schema_version": 1,
        "protocol": "campaign-two-domain-selection-v1",
        "mandatory_domain": MANDATORY_DOMAIN,
        "second_domain_candidates_in_exact_tie_order": list(SECOND_DOMAIN_TIEBREAK),
        "selection_metric": "validation_gain_vs_a0",
        "tie_break": [
            "higher selected validation task utility",
            "fixed pre-result candidate order",
        ],
        "roles": {
            "commonsenseqa": "mandatory_primary_domain",
            "selected_second_domain": "validation_gain_selected_domain",
        },
        "source_result": str(STEP5_ROOT / "results/four_domain_screen_1p5b.json"),
        "downstream_train_split": "train_core1000",
        "downstream_validation_split": "validation300",
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }:
        raise RuntimeError("Frozen Step 6 selection protocol is missing or drifted")

    screen = _read_json(screen_result)
    if (
        screen.get("status") != "complete"
        or screen.get("protocol") != "campaign-four-domain-screen-1p5b-v1"
        or screen.get("official_test_accessed") is not False
        or screen.get("mbpp") != "FORBIDDEN"
    ):
        raise RuntimeError("Step 5 result is incomplete or violates the campaign contract")
    domains = screen.get("domains")
    if not isinstance(domains, dict) or tuple(domains) != SCREEN_DOMAINS:
        raise RuntimeError("Step 5 does not contain the four frozen domains in order")

    ranking_rows: list[dict[str, Any]] = []
    for domain in SECOND_DOMAIN_TIEBREAK:
        row = domains.get(domain)
        if not isinstance(row, dict):
            raise TypeError(f"Missing Step 5 domain: {domain}")
        gain = float(row["validation_gain_vs_a0"])
        utility = float(row["selected_task_utility"])
        selected_step = int(row["selected_step"])
        if selected_step not in {0, 500, 1000, 1500, 2000}:
            raise RuntimeError(f"Invalid selected checkpoint for {domain}: {selected_step}")
        if abs((utility - float(row["a0_task_utility"])) - gain) > 1e-12:
            raise RuntimeError(f"Inconsistent validation gain for {domain}")
        ranking_rows.append(
            {
                "domain": domain,
                "validation_gain_vs_a0": gain,
                "selected_task_utility": utility,
                "selected_step": selected_step,
                "fixed_tiebreak_rank": SECOND_DOMAIN_TIEBREAK.index(domain),
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            -float(row["validation_gain_vs_a0"]),
            -float(row["selected_task_utility"]),
            int(row["fixed_tiebreak_rank"]),
        )
    )
    selected_second = str(ranking_rows[0]["domain"])
    cqa = domains[MANDATORY_DOMAIN]
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": _utc_now(),
        "protocol": "campaign-two-domain-selection-v1",
        "selection_protocol": str(PROTOCOL_PATH),
        "selection_protocol_sha256": _sha256(PROTOCOL_PATH),
        "source_screen": str(screen_result),
        "source_screen_sha256": _sha256(screen_result),
        "mandatory_domain": MANDATORY_DOMAIN,
        "selected_second_domain": selected_second,
        "selected_domains": [MANDATORY_DOMAIN, selected_second],
        "roles": {
            MANDATORY_DOMAIN: "mandatory_primary_domain",
            selected_second: "validation_gain_selected_domain",
        },
        "selection_metric": "validation_gain_vs_a0",
        "tie_break": [
            "higher selected validation task utility",
            "fixed pre-result order: gsm8k, kodcode, spider",
        ],
        "mandatory_domain_evidence": {
            "selected_step": int(cqa["selected_step"]),
            "selected_task_utility": float(cqa["selected_task_utility"]),
            "validation_gain_vs_a0": float(cqa["validation_gain_vs_a0"]),
        },
        "second_domain_ranking": ranking_rows,
        "downstream_split_contract": "reuse frozen train_core1000 and validation300",
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result
