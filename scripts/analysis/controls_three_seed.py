"""Calibration and within-prompt controls for Identity seeds 42 and 7.

Re-runs the seed-23 reward-alignment analysis (reward_alignment.py of the v79
snapshot) on the other two Identity seeds. Correctness comes from the stored
three-seed rollout labels instead of re-executing the evaluators; features,
models and metrics are computed by the original code. Seed 23 is read from the
archived analysis. Writes one JSON with the five table columns per seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path("/path/to/iar_code")
SNAPSHOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("/path/to/runs")
ARCHIVE = (CAMPAIGN / "evaluation_audits_and_reproducibility/12_final_test_analysis_reconciliation/"
           "correlations/nonofficial_reward_alignment_attempt001")
LABELS = ROOT / "results/revision_20260922/seed_rollout_labels"
sys.path.insert(0, str(SNAPSHOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import iar_campaign.reward_alignment as ra  # noqa: E402
from rollout_signal import reward_log  # noqa: E402

logger = logging.getLogger(__name__)
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
COLUMNS = {"cal_gap": ("metrics", "calibration_in_the_large"),
           "ece": ("metrics", "ece_equal_frequency_10"),
           "partial_r": ("models", "within_prompt_partial_corr_reward_correct_given_style"),
           "delta_r2_correctness": ("models", "correctness_delta_r2_over_style"),
           "delta_r2_features": ("models", "style_delta_r2_over_correctness")}


def labels_as_scores(domain: str, seed: str):
    by_call = {int(r["call_id"]): r for r in ra.read_jsonl(LABELS / f"rows_{domain}_{seed}.jsonl")}

    def score_rows(*, reward_rows, **_):
        return [{"correct": bool(by_call[int(r["call_id"])]["correct"]),
                 "valid": bool(by_call[int(r["call_id"])].get("evaluator_valid", 1)),
                 "status": "scored"} for r in reward_rows]
    return score_rows


def pick(domain: str, metrics: list, models: list) -> dict[str, float]:
    m = next(r for r in metrics if r["domain"] == domain and r["signal"] == "p_own")
    i = next(r for r in models if r["domain"] == domain and r["signal"] == "p_own")
    return {name: float((m if src == "metrics" else i)[key]) for name, (src, key) in COLUMNS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    archived_metrics = json.loads((ARCHIVE / "metrics.json").read_text())
    archived_models = json.loads((ARCHIVE / "incremental_models.json").read_text())
    result = {d: {"seed23": pick(d, archived_metrics, archived_models)} for d in DOMAINS}
    for seed in ("seed42", "seed7"):
        out_dir = args.work_dir / seed
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for domain in DOMAINS:
            calls = Path(reward_log("grpo_identity", domain, seed)).resolve()
            spec = ra.RunSpec(domain=domain, signal="p_own", run_id=f"identity_{seed}",
                              reward_calls=calls, train_examples=calls.parent / "train_examples.jsonl",
                              dataset=ra._dataset_for_domain(CAMPAIGN / "evaluation_audits_and_reproducibility", domain))
            target = out_dir / f"rows_{domain}_p_own.jsonl"
            if not target.exists():
                ra._score_rows = labels_as_scores(domain, seed)
                ra.extract_run(spec, campaign_root=CAMPAIGN, output_dir=out_dir,
                               tokenizer_cache=Path("/path/to/cache/huggingface"))
            rows.extend(ra.read_jsonl(target))
            logger.info("%s %s extracted", seed, domain)
        metrics = ra.analyze_metrics(rows, bootstrap_replicates=args.bootstrap, seed=23)
        models = ra.analyze_incremental_models(rows, folds=5, bootstrap_replicates=args.bootstrap, seed=23)
        for domain in DOMAINS:
            result[domain][seed] = pick(domain, metrics, models)
            logger.info("%s %s %s", seed, domain, result[domain][seed])
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
