"""Group-level reward-signal analyses from saved training logs (A6-A9).

* A7: frequency of constant-reward groups under the binary verifier (Oracle runs'
  logged ``frac_reward_zero_std``; verifier labels on seed-23 Identity rollouts).
* A8: CommonsenseQA Oracle collapse: response length, zero-contrast groups and
  entropy by 100-step window, against the Identity run.
* A9: binarized-Identity counterfactual: share of groups whose binary own-label
  1{p_own > 1/2} is constant (zero advantage) versus log p_own, 3 seeds x 4 domains.
* A6: KodCode groups whose four rollouts all fail the hidden tests: does p_own
  increase with the fraction of hidden-test assertions passed?
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code/experiments/main")
AUDITS = Path("/path/to/runs/"
              "evaluation_audits_and_reproducibility")
ALIGN = AUDITS / "12_final_test_analysis_reconciliation/correlations/nonofficial_reward_alignment_attempt001"
PARTIAL = AUDITS / "14_kodcode_partial_quality/results/task_results.partial.jsonl"
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
GROUP = 4
RNG_SEED = 23


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reward_log(condition: str, domain: str, seed: str) -> Path:
    """Latest reward log with 8,000 successful calls for this run."""
    candidates = sorted(glob.glob(str(ROOT / condition / domain / seed / "runs" / "*" / "reward_calls.jsonl")))
    for path in reversed(candidates):
        ok = [r for r in read_jsonl(path) if r.get("status") == "ok"]
        if len(ok) == 8000:
            return Path(path)
    raise FileNotFoundError(f"No complete reward log for {condition}/{domain}/{seed}")


def groups_from_calls(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split successful calls into four-response prompt groups by call order."""
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["call_id"]):
        by_prompt[int(row["dataset_index"])].append(row)
    groups = []
    for calls in by_prompt.values():
        for start in range(0, len(calls), GROUP):
            chunk = calls[start:start + GROUP]
            if len(chunk) == GROUP and sorted(int(c["rollout_index"]) for c in chunk) == [0, 1, 2, 3]:
                groups.append(chunk)
    return groups


def binarized_counterfactual() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in DOMAINS:
        per_seed = {}
        for seed in SEEDS:
            path = reward_log("grpo_identity", domain, seed)
            ok = [r for r in read_jsonl(path) if r.get("status") == "ok"]
            groups = groups_from_calls(ok)
            binary_const = [len({float(c["p_own"]) > 0.5 for c in g}) == 1 for g in groups]
            cont_const = [np.std([float(c["log_p_own"]) for c in g]) == 0.0 for g in groups]
            per_seed[seed] = {"log": str(path), "groups": len(groups),
                              "binary_constant_pct": 100 * float(np.mean(binary_const)),
                              "logp_constant_pct": 100 * float(np.mean(cont_const))}
        vals = [per_seed[s]["binary_constant_pct"] for s in SEEDS]
        out[domain] = {"per_seed": per_seed, "binary_constant_mean": float(np.mean(vals)),
                       "binary_constant_sd": float(np.std(vals, ddof=1))}
        logger.info("A9 %s binary-constant %.1f±%.1f%%", domain, np.mean(vals), np.std(vals, ddof=1))
    return out


def labelled_groups(domain: str) -> list[list[dict[str, Any]]]:
    rows = read_jsonl(ALIGN / f"rows_{domain}_p_own.jsonl")
    calls = {r["call_id"]: r for r in read_jsonl(reward_log("grpo_identity", domain, "seed23"))
             if r.get("status") == "ok"}
    merged = []
    for row in rows:
        call = calls[row["call_id"]]
        merged.append({**call, "correct": int(row["correct"])})
    return groups_from_calls(merged)


def verifier_signal() -> dict[str, Any]:
    """Seed-23 Identity rollouts: constant correctness vs constant binary own-label."""
    out: dict[str, Any] = {}
    for domain in DOMAINS:
        groups = labelled_groups(domain)
        n = len(groups)
        corr_const = [len({c["correct"] for c in g}) == 1 for g in groups]
        all_wrong = [all(c["correct"] == 0 for c in g) for g in groups]
        bin_const = [len({float(c["p_own"]) > 0.5 for c in g}) == 1 for g in groups]
        mixed_bin_const = [b for b, cc in zip(bin_const, corr_const, strict=True) if not cc]
        pairs = []
        for g, b, cc in zip(groups, bin_const, corr_const, strict=True):
            if cc or not b:
                continue
            for x in g:
                for y in g:
                    if x["correct"] == 1 and y["correct"] == 0:
                        diff = float(x["log_p_own"]) - float(y["log_p_own"])
                        pairs.append(1.0 if diff > 0 else 0.5 if diff == 0 else 0.0)
        out[domain] = {
            "groups": n,
            "oracle_constant_pct": 100 * float(np.mean(corr_const)),
            "all_incorrect_pct": 100 * float(np.mean(all_wrong)),
            "binary_constant_pct": 100 * float(np.mean(bin_const)),
            "binary_constant_within_mixed_pct": 100 * float(np.mean(mixed_bin_const)),
            "concordance_where_binary_blind": float(np.mean(pairs)) if pairs else None,
            "pairs_where_binary_blind": len(pairs),
        }
        logger.info("A7 %s oracle-constant %.1f%% binary-constant %.1f%%", domain,
                    out[domain]["oracle_constant_pct"], out[domain]["binary_constant_pct"])
    return out


def trainer_log(condition: str, domain: str) -> list[dict[str, Any]]:
    path = sorted(glob.glob(str(ROOT / condition / domain / "seed23" / "runs" / "*" / "trainer_log_history.jsonl")))[0]
    return [r for r in read_jsonl(path) if "frac_reward_zero_std" in r]


def oracle_dynamics() -> dict[str, Any]:
    out: dict[str, Any] = {"oracle_zero_contrast": {}, "cqa_windows": {}}
    for domain in DOMAINS:
        rows = trainer_log("grpo_oracle", domain)
        out["oracle_zero_contrast"][domain] = {
            "all_steps_pct": 100 * float(np.mean([r["frac_reward_zero_std"] for r in rows])),
            "first_100_logged_pct": 100 * float(np.mean([r["frac_reward_zero_std"] for r in rows[:100]])),
            "last_100_logged_pct": 100 * float(np.mean([r["frac_reward_zero_std"] for r in rows[-100:]])),
        }
    # Response length per 100-step window, for every domain and both conditions.
    out["length_windows"] = {}
    for condition in ("grpo_oracle", "grpo_identity"):
        out["length_windows"][condition] = {}
        for domain in DOMAINS:
            rows = trainer_log(condition, domain)
            out["length_windows"][condition][domain] = [
                {"end_step": start + 100,
                 "mean_length": float(np.mean([r["completions/mean_length"] for r in seg]))}
                for start in range(0, 2000, 100)
                if (seg := [r for r in rows if start < int(r["step"]) <= start + 100])]
    for condition in ("grpo_oracle", "grpo_identity"):
        rows = trainer_log(condition, "commonsenseqa")
        windows = []
        for start in range(0, 2000, 100):
            seg = [r for r in rows if start < int(r["step"]) <= start + 100]
            if not seg:
                continue
            windows.append({"end_step": start + 100,
                            "mean_length": float(np.mean([r["completions/mean_length"] for r in seg])),
                            "zero_contrast_pct": 100 * float(np.mean([r["frac_reward_zero_std"] for r in seg]))})
        first_short = next((int(r["step"]) for r in rows if r["completions/mean_length"] <= 5), None)
        out["cqa_windows"][condition] = {"windows": windows, "first_step_length_le5": first_short,
                                         "initial_length": float(rows[0]["completions/mean_length"])}
    return out


def kodcode_partial() -> dict[str, Any]:
    """A6: p_own vs fraction of hidden-test assertions passed in all-incorrect groups."""
    correct = {r["call_id"]: int(r["correct"]) for r in read_jsonl(ALIGN / "rows_kodcode_p_own.jsonl")}
    rows = [row for task in read_jsonl(PARTIAL) for row in task["rows"]]
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partial_total"):
            groups[(int(row["dataset_index"]), int(row["occurrence"]))].append(
                {**row, "correct": correct[row["call_id"]]})
    all_wrong = [g for g in groups.values() if len(g) == GROUP and all(r["correct"] == 0 for r in g)]
    informative = [g for g in all_wrong if len({r["q_partial"] for r in g}) > 1]
    xs, ys, conc = [], [], []
    for g in informative:
        q = np.array([r["q_partial"] for r in g])
        lp = np.array([r["log_p_own"] for r in g])
        xs.extend(q - q.mean())
        ys.extend(lp - lp.mean())
        for i in range(GROUP):
            for j in range(GROUP):
                if q[i] > q[j]:
                    conc.append(1.0 if lp[i] > lp[j] else 0.5 if lp[i] == lp[j] else 0.0)
    xs, ys = np.array(xs), np.array(ys)
    slope = float(np.sum(xs * ys) / np.sum(xs * xs))
    rng = np.random.default_rng(RNG_SEED)
    boot_s, boot_c = [], []
    for _ in range(2000):
        pick = rng.integers(0, len(informative), len(informative))
        bx, by, bc = [], [], []
        for k in pick:
            g = informative[k]
            q = np.array([r["q_partial"] for r in g]); lp = np.array([r["log_p_own"] for r in g])
            bx.extend(q - q.mean()); by.extend(lp - lp.mean())
            bc.extend(1.0 if lp[i] > lp[j] else 0.5 if lp[i] == lp[j] else 0.0
                      for i in range(GROUP) for j in range(GROUP) if q[i] > q[j])
        bx, by = np.array(bx), np.array(by)
        boot_s.append(np.sum(bx * by) / np.sum(bx * bx)); boot_c.append(np.mean(bc))
    bins = np.linspace(0, 1, 6)
    raw_q = np.array([r["q_partial"] for g in all_wrong for r in g])
    raw_p = np.array([r["p_own"] for g in all_wrong for r in g])
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (raw_q >= lo) & ((raw_q < hi) if hi < 1 else (raw_q <= hi))
        binned.append({"bin": [float(lo), float(hi)], "n": int(mask.sum()),
                       "mean_p_own": float(raw_p[mask].mean()) if mask.any() else None})
    res = {"groups_total": len(groups), "all_incorrect_groups": len(all_wrong),
           "informative_groups": len(informative), "pairs": len(conc),
           "within_group_corr": float(np.corrcoef(xs, ys)[0, 1]),
           "slope_logp_per_unit_q": slope, "slope_ci95": [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
           "concordance": float(np.mean(conc)),
           "concordance_ci95": [float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))],
           "binned_raw": binned}
    logger.info("A6 all-wrong groups %d informative %d concordance %.3f", len(all_wrong), len(informative), res["concordance"])
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = {"A9_binarized": binarized_counterfactual(), "A7_verifier": verifier_signal(),
              "A7_A8_oracle": oracle_dynamics(), "A6_kodcode_partial": kodcode_partial()}
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
