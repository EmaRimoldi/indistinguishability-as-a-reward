"""Recompute short-pilot and pairwise-probe summaries from recorded outputs."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .paths import ROOT, DOMAINS
from .records import materialize, result_path


def read(relative):
    p = materialize(relative)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def run(output: Path) -> dict:
    pilot_ref = json.loads(
        (result_path("results/reward_transform_selection.json")).read_text()
    )
    pilot = {}
    accuracy = lambda rows: (
        sum(
            bool(r["score"]["correct"])
            for r in rows
            if not r["score"].get("excluded_reference_invalid")
        )
        / sum(not r["score"].get("excluded_reference_invalid", False) for r in rows)
    )
    for domain, forms in pilot_ref["domains"].items():
        baseline = accuracy(read(f"results/pilot/{domain}/a0/candidates.jsonl.gz"))
        pilot[domain] = {}
        for form, expected in forms.items():
            steps = {
                step: accuracy(
                    read(
                        f"results/pilot/{domain}/{form}/step_{step}/candidates.jsonl.gz"
                    )
                )
                for step in (187, 374)
            }
            best = max(steps, key=lambda step: (steps[step], -step))
            gain = steps[best] - baseline
            if best != expected["selected_step"] or not math.isclose(
                gain, expected["validation_gain"], abs_tol=1e-12
            ):
                raise ValueError(f"Pilot result mismatch: {domain}/{form}")
            pilot[domain][form] = {
                "baseline_accuracy": baseline,
                "validation_accuracy": steps,
                "selected_step": best,
                "gain_pp": 100 * gain,
            }
    pairwise = {}
    # The manuscript reports mean m/K over 100 updates = 400 rollouts, not
    # the 1,000-rollout auxiliary windows in the older held-out summary.
    k10 = {}
    for domain in DOMAINS:
        rows = read(f"results/pairwise/{domain}/judgments.jsonl.gz")
        pairs = defaultdict(dict)
        for row in rows:
            if row["order"] in pairs[row["candidate_key"]]:
                raise ValueError("Duplicate pairwise order")
            pairs[row["candidate_key"]][row["order"]] = float(row["p_actor"])
        if (
            len(rows) != 800
            or len(pairs) != 400
            or any(set(v) != {"actor_first", "actor_second"} for v in pairs.values())
        ):
            raise ValueError(f"Pairwise coverage mismatch: {domain}")
        first = [v["actor_first"] for v in pairs.values()]
        second = [v["actor_second"] for v in pairs.values()]
        mean_pairs = [(a + b) / 2 for a, b in zip(first, second, strict=True)]
        pairwise[domain] = {
            "n_pairs": len(pairs),
            "judge_calls": len(rows),
            "mean_p_pair": statistics.mean(mean_pairs),
            "sample_sd_p_pair": statistics.stdev(mean_pairs),
            "p_actor_first": statistics.mean(first),
            "p_actor_second": statistics.mean(second),
            "mean_order_effect": statistics.mean(
                a - b for a, b in zip(first, second, strict=True)
            ),
            "correctness_strata": "Recovered in results/analysis_reference/pairwise; not rescored here (KodCode environment and Spider databases required).",
        }
        groups = read(f"results/k10_groups/{domain}.jsonl.gz")
        groups = sorted(groups, key=lambda r: int(r["logical_reward_id"]))
        if [int(r["logical_reward_id"]) for r in groups] != list(range(1, 8001)):
            raise ValueError("K10 reward-group coverage mismatch")
        if any(
            r["status"] != "ok"
            or r["k"] != 10
            or len(r["labels"]) != 10
            or sum(r["labels"]) != r["m"]
            for r in groups
        ):
            raise ValueError("K10 label receipt mismatch")
        k10[domain] = {
            "updates_per_window": 100,
            "rollouts_per_window": 400,
            "start_p_ind_pct": 100
            * statistics.mean(r["m"] / r["k"] for r in groups[:400]),
            "end_p_ind_pct": 100
            * statistics.mean(r["m"] / r["k"] for r in groups[-400:]),
        }
    screen = read("results/teacher_screen/judge_sanity_gate_v1.json.gz")["domains"]
    screen["commonsenseqa"] = read(
        "results/teacher_screen/judge_sanity_cqa_confirm_gate_v4.json.gz"
    )["domains"]["commonsenseqa"]
    from iar_campaign.judge_sanity import auc

    teacher_screen = {}
    for domain in DOMAINS:
        run = (
            "judgments_cqa_confirm_attempt004"
            if domain == "commonsenseqa"
            else "judgments_attempt001"
        )
        rows = [
            r
            for lane in range(3)
            for r in read(f"results/teacher_screen/{run}/lane_{lane}.jsonl.gz")
            if r["domain"] == domain
        ]
        groups = {
            variant: {
                str(r["id"]): float(r["p_own"]) for r in rows if r["variant"] == variant
            }
            for variant in ("own", "a0", "prompt_imitation")
        }
        if (
            len(rows) != 300
            or any(len(group) != 100 for group in groups.values())
            or groups["own"].keys() != groups["a0"].keys()
        ):
            raise ValueError("Teacher-screen coverage mismatch")
        means = {
            variant: statistics.mean(values.values())
            for variant, values in groups.items()
        }
        score = auc(list(groups["own"].values()), list(groups["a0"].values()))
        gap = statistics.mean(
            groups["own"][key] - groups["a0"][key] for key in groups["own"]
        )
        ref = screen[domain]
        if not math.isclose(
            score, ref["auc_own_vs_a0"], abs_tol=1e-12
        ) or not math.isclose(
            gap, ref["paired_own_minus_a0"]["mean_delta"], abs_tol=1e-12
        ):
            raise ValueError("Teacher-screen statistics mismatch")
        if any(
            not math.isclose(value, ref["mean_p_own"][variant], abs_tol=1e-12)
            for variant, value in means.items()
        ):
            raise ValueError("Teacher-screen probability means mismatch")
        teacher_screen[domain] = {
            "auc_own_vs_a0": score,
            "mean_p_own": means,
            "paired_gap": gap,
            "status": "recomputed from original per-response judgments; no new API calls",
        }
    report = {
        "status": "passed",
        "model_calls": 0,
        "full_experiments_rerun": False,
        "pilot": pilot,
        "pairwise": pairwise,
        "k10_training_windows": k10,
        "teacher_screen": teacher_screen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
