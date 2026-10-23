"""Reanalyze the recorded 2x2 intervention panel without API calls."""

from .records import manifest, materialize


def run(*, bootstrap: int = 5000) -> dict:
    from iar_campaign.causal_reward_audit import analyze_responses

    root = None
    for item in manifest():
        if item["local"].startswith("results/causal/"):
            path = materialize(item["local"])
            if path.name == "candidates.jsonl":
                root = path.parent
    if root is None:
        raise FileNotFoundError("Recorded intervention panel missing")
    return analyze_responses(
        candidate_path=root / "candidates.jsonl",
        lane_root=root / "judge",
        output_dir=root / "recomputed",
        campaign_root=root,
        bootstrap_replicates=bootstrap,
        seed=23,
    )
