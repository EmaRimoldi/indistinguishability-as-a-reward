from __future__ import annotations

import argparse
import json

from .config import load_config
from .env import load_env_file
from .openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)

    disc = config.raw["discriminator"]
    client = OpenRouterClient.from_env()
    payload = make_chat_payload(
        model_id=disc["model_id"],
        prompt=(
            "Answer with exactly one token.\n"
            "1 = yes\n"
            "0 = no\n\n"
            "Question: Is Python a programming language?\n"
            "Answer:"
        ),
        max_tokens=int(disc["judge_max_tokens"]),
        temperature=0,
        logprobs=True,
        top_logprobs=disc["top_logprobs"],
        provider_routing=disc["provider_routing"],
        request_overrides=disc.get("request_overrides"),
    )
    response = client.chat_completion(payload)
    parsed = binary_probability_from_logprobs(response)
    result = {
        "model_requested": disc["model_id"],
        "model_returned": response.get("model"),
        "text": response_text(response),
        "parsed": parsed,
        "usage": response.get("usage"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not parsed["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
