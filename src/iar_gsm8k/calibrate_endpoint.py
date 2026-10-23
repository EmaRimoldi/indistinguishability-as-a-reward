from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from iar_phase0.env import load_env_file
from iar_phase0.openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)

from .data import read_jsonl, write_json


SETTINGS: tuple[dict[str, Any], ...] = (
    {"name": "primary_exact", "temperature": 0.0},
    {"name": "explicit_top_p_1", "temperature": 0.0, "top_p": 1.0},
    {"name": "nucleus_top_token", "temperature": 0.0, "top_p": 0.01},
    {"name": "epsilon_temperature", "temperature": 0.000001, "top_p": 1.0},
    {"name": "top_k_1", "temperature": 0.0, "top_k": 1},
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit emitted-token/log-prob consistency")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--only", choices=tuple(setting["name"] for setting in SETTINGS))
    parser.add_argument("--output-name", default="endpoint_temperature_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    run_dir = args.run_dir.resolve()
    rows = read_jsonl(run_dir / "probe_judgments.jsonl")
    mismatch = next(row for row in rows if not bool(row["emitted_argmax_consistent"]))
    prompt = str(mismatch["judge_prompt"])
    disc = config["discriminator"]
    env_path = Path(config["paths"]["openrouter_env"])
    load_env_file(env_path if env_path.is_absolute() else root / env_path)
    client = OpenRouterClient.from_env(max_attempts=1)
    records: list[dict[str, Any]] = []
    selected_settings = tuple(
        setting for setting in SETTINGS if args.only is None or setting["name"] == args.only
    )
    for setting in selected_settings:
        for repetition in range(args.repetitions):
            overrides = dict(disc.get("request_overrides", {}))
            for key in ("top_p", "top_k"):
                if key in setting:
                    overrides[key] = setting[key]
            payload = make_chat_payload(
                model_id=str(disc["model_id"]),
                prompt=prompt,
                system_prompt=None,
                max_tokens=1,
                temperature=float(setting["temperature"]),
                logprobs=True,
                top_logprobs=int(disc["judgment"]["top_logprobs"]),
                provider_routing=dict(disc["provider_routing"]),
                request_overrides=overrides,
            )
            started = time.time()
            try:
                raw = client.chat_completion(payload)
                parsed = binary_probability_from_logprobs(raw)
                emitted = response_text(raw).strip()
                argmax = None
                if parsed.get("ok"):
                    argmax = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
                records.append(
                    {
                        "setting": setting,
                        "repetition": repetition,
                        "status": "ok",
                        "emitted": emitted,
                        "argmax_01": argmax,
                        "consistent": emitted in {"0", "1"} and emitted == argmax,
                        "parsed": parsed,
                        "raw_model": raw.get("model"),
                        "raw_provider": raw.get("provider"),
                        "usage": raw.get("usage", {}),
                        "raw_response": raw,
                        "latency_seconds": time.time() - started,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "setting": setting,
                        "repetition": repetition,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "latency_seconds": time.time() - started,
                    }
                )
    summaries: dict[str, Any] = {}
    for setting in selected_settings:
        selected = [row for row in records if row["setting"]["name"] == setting["name"]]
        ok = [row for row in selected if row["status"] == "ok"]
        summaries[setting["name"]] = {
            "n": len(selected),
            "n_ok": len(ok),
            "consistency_rate": sum(bool(row["consistent"]) for row in ok) / len(ok) if ok else None,
            "emitted_tokens": {token: sum(row["emitted"] == token for row in ok) for token in ("0", "1")},
            "mean_logp_0": sum(float(row["parsed"]["logp_0"]) for row in ok) / len(ok) if ok else None,
            "mean_logp_1": sum(float(row["parsed"]["logp_1"]) for row in ok) / len(ok) if ok else None,
            "minimum_binary_label_mass": min(float(row["parsed"]["binary_label_mass"]) for row in ok)
            if ok else None,
        }
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Diagnose emitted-token versus binary-log-prob argmax mismatch before training.",
        "source_candidate": {
            "id": mismatch["id"],
            "rollout_index": mismatch["rollout_index"],
            "source_judgment_order": mismatch["judgment_order"],
        },
        "message_roles": ["user"],
        "system_prompt_present": False,
        "summaries": summaries,
        "records": records,
    }
    write_json(run_dir / args.output_name, result)
    print(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
