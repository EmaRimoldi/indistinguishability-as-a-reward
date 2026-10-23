from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    own_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["source"] == "B_own":
            own_rows.append(row)
        else:
            by_variant[row["variant"]].append(row)

    summary: dict[str, Any] = {"n_rows": len(rows), "variants": {}}
    for variant, variant_rows in sorted(by_variant.items()):
        pass_values = [float(r["unit_tests"]["passed"]) for r in variant_rows if r.get("unit_tests")]
        reward_values = [
            float(r["judge"]["parsed"]["p1"])
            for r in variant_rows
            if r.get("judge") and r["judge"]["parsed"].get("ok")
        ]
        summary["variants"][variant] = {
            "n": len(variant_rows),
            "pass@1": mean(pass_values),
            "S_A": mean(reward_values),
        }

    own_pass_values = [float(r["unit_tests"]["passed"]) for r in own_rows if r.get("unit_tests")]
    own_rewards = [
        float(r["judge"]["parsed"]["p1"])
        for r in own_rows
        if r.get("judge") and r["judge"]["parsed"].get("ok")
    ]
    summary["B_own"] = {
        "n": len(own_rows),
        "pass@1": mean(own_pass_values),
        "S_B": mean(own_rewards),
    }
    summary["S_B"] = summary["B_own"]["S_B"]
    if summary["S_B"] is not None:
        for variant, values in summary["variants"].items():
            if values["S_A"] is not None:
                values["G"] = summary["S_B"] - values["S_A"]
            else:
                values["G"] = None
    return summary


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
