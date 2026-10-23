# Reproduction guide

Run commands from the repository root after the [README installation](../README.md#install-and-verify). The audited environment uses Python 3.14.3, PyTorch 2.13.0, Transformers 4.57.6, TRL 0.27.2, PEFT 0.19.1 and Accelerate 1.13.0. [Portable requirements](../reproduction/environment/requirements-portable.txt) are the tested install route; [historical CUDA requirements](../reproduction/environment/requirements-grpo.txt) record the original Linux stack.

## External assets and local paths

| Asset | Requirement |
|---|---|
| Actor | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, cached locally |
| GPU | Compatible NVIDIA/CUDA environment; one A100 80GB or H100 per training job |
| Teacher / Quality | DeepSeek V4 Flash via OpenRouter, StreamLake only, no fallbacks; frozen sources use `deepseek/deepseek-v4-flash` |
| K=10 judge | `openai/gpt-5-mini`, OpenAI Flex routing; 10 judgments per rollout |
| Credentials | `OPENROUTER_API_KEY` for single-key conditions; screening uses `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_2` and `OPENROUTER_API_KEY_3` as specified by its config |
| Spider | Official `spider_release` containing `spider_data/tables.json` and database binaries |
| KodCode | An isolated execution environment with task-specific libraries; do not run untrusted generated code on a shared host |

```bash
python scripts/setup_paths.py --cache /path/to/hf/hub --spider-root /path/to/spider_release --runs /path/to/local/runs
python -m iar_repro verify --external
```

The Hugging Face cache must contain `models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Download those weights through your usual Hugging Face tooling. `IAR_HF_CACHE`, `IAR_SPIDER_ROOT` and `IAR_RUNS_ROOT` override the ignored `.reproduction-paths.json`. No credentials belong in version control; screening can read its three named keys from the environment or an ignored `secrets.env` inside the runs directory.

Fixed splits and training corpora are included. Do not repartition them: split seed 23 is shared by all training seeds. [Data documentation](../data/README.md) lists their origins and sizes. Gold answers, hidden tests and reference SQL are evaluator inputs; actor prompts exclude them.

## Main experiment grid

Domains: `commonsenseqa`, `gsm8k`, `kodcode`, `spider`. Conditions: `indist`, `quality`, `oracle`, `sft`. Training seeds: `23`, `42`, `7`.

```bash
export OPENROUTER_API_KEY=...
python -m iar_repro configure --domain gsm8k --condition indist --seed 23
python -m iar_repro train --domain gsm8k --condition indist --seed 23
python -m iar_repro evaluate --domain gsm8k --condition indist --seed 23
```

Repeat the three commands for each domain/condition/seed. `configure` and `verify` make no model calls. `train` performs real training and can incur API charges. Each default attempt is `attempt001`; use a new `--run-id` for another attempt. Each process should handle one domain. GRPO uses 2,000 updates, four generations, two passes over 1,000 prompts, learning rate `3e-5`, Dr.GRPO loss, KL coefficient `0.003`, LoRA rank/alpha 32 and dropout 0.05. Indistinguishability reward is `log(p_own)` with epsilon `1e-6`, derived from normalized binary-label log probabilities.

SFT uses the frozen 1,000-response corpus per domain, completion-only loss, no shuffling, two passes/500 updates and learning rate `1e-4`. Exact sequence lengths and arguments are in [SFT metadata](../reproduction/provenance/sft/). Training messages are unchanged; correctness labels were removed from these training inputs. CPU/CUDA seeds are set before model and adapter construction.

Evaluate checkpoints with `--step 0`, `500`, `1000`, `1500`, `2000` for GRPO; SFT supports `0`, `125`, `250`, `375`, `500` and resolves the final adapter. The final checkpoint is the paper comparison; held-out evaluation does not select training checkpoints.

```bash
python -m iar_repro evaluate --domain gsm8k --condition a0 --seed 23
python -m iar_repro train --domain gsm8k --condition closed_weight_k10 --seed 23
python -m iar_repro evaluate --domain gsm8k --condition closed_weight_k10 --seed 23
```

A0 is untrained. K=10 is reported only for seed 23: reward `log((m+1)/12)`, 8,000 reward groups and 80,000 logical judgments per domain. Transient requests allow eight attempts; invalid labels terminate. Fresh K=10 runs are supported. Receipt-preserving continuation requires the original external receipts/checkpoints and frozen campaign driver. Standard `train --resume /path/to/checkpoint` is exposed for supported non-K=10 GRPO conditions; SFT, Oracle and pilot continuation are not exposed.

## Reward-form pilot

Run both `gsm8k` and `kodcode`, each with `log_p_own`, `p_own`, `logit_p_own`, `binary_own`:

```bash
python -m iar_repro pilot configure --domain gsm8k --form log_p_own
python -m iar_repro pilot train --domain gsm8k --form log_p_own
python -m iar_repro pilot evaluate --domain gsm8k --form log_p_own --step 0
python -m iar_repro pilot evaluate --domain gsm8k --form log_p_own --step 187
python -m iar_repro pilot evaluate --domain gsm8k --form log_p_own --step 374
```

Each arm uses the nested 374-task training subset, 374 updates and 1,496 rewards, with a maximum 7,480 HTTP attempts. Compare the best validation checkpoint per arm with its fresh step-0 baseline and apply `iar_campaign.reward_pilot.select_reward_with_logp_tie_rule`. Original model outputs and selection statistics are external; the selected transform and frozen settings remain as metadata.

## Teacher corpus, screening and pairwise probe

The bundled SFT corpus avoids regeneration costs. To generate a new corpus using the frozen Step 7 / Step 15 protocol:

```bash
python -m iar_repro teacher-corpus --domain gsm8k
python -m iar_repro teacher-corpus --domain gsm8k --run-id teacher_new --execute
```

Screening uses the exact 100-task splits in [data/screen](../data/screen/). Actor and teacher responses must be generated; no recorded responses or judgments are bundled.

```bash
python -m iar_repro teacher-screen configure --domain gsm8k
python -m iar_repro teacher-screen actors --domain gsm8k
python -m iar_repro teacher-screen teacher --domain gsm8k --shard-index 0
python -m iar_repro teacher-screen teacher --domain gsm8k --shard-index 1
python -m iar_repro teacher-screen teacher --domain gsm8k --shard-index 2
python -m iar_repro teacher-screen collect-teacher --domain gsm8k
python -m iar_repro teacher-screen judge --domain gsm8k --shard-index 0
python -m iar_repro teacher-screen judge --domain gsm8k --shard-index 1
python -m iar_repro teacher-screen judge --domain gsm8k --shard-index 2
python -m iar_repro teacher-screen collect-judge --domain gsm8k
```

Repeat for all four domains. CommonsenseQA uses the confirmatory v4 binary-contract template. Actor generation uses A0 and prompt imitation; the teacher then provides 100 own responses, followed by 300 judgments per domain.

```bash
python -m iar_repro pairwise configure --domain gsm8k
python -m iar_repro pairwise generate --domain gsm8k
python -m iar_repro pairwise judge --domain gsm8k
python -m iar_repro pairwise collect --domain gsm8k
```

The pairwise probe uses the first 100 frozen training tasks, four actor rollouts each and both candidate orders (800 judgments). Teacher responses are derived from the SFT training messages. `judge` requires a completed local generation; it never substitutes a stored paper rollout.

## External results and figures

The repository and source archive contain no experimental outputs. Fresh outputs go to `IAR_RUNS_ROOT` (default `runs/`). Keep result collections and model/checkpoint files outside Git. Existing paper records remain on the original Engaging project; the repository does not automatically fetch or publish them.

To analyze an existing result collection, set `IAR_RESULTS_ROOT` to its directory. The expected collection has `manifest.json` (entries with `local: "results/<path>"` and `package_sha256` for compressed records) plus the corresponding `heldout/`, `training_logs/`, `alignment/`, `reanalysis/`, `causal/`, `pilot/`, `pairwise/`, `teacher_screen/`, `k10_groups/`, `analysis_reference/`, `partial/` and `figure_data/` files. SHA-256 verification is retained before decompression. The index and files are external assets, not generated by `verify`; use an exported collection in this schema. Missing assets produce an explicit error, never fabricated numbers.

```bash
export IAR_RESULTS_ROOT=/path/to/local/results
python -m iar_repro tables
python -m iar_repro probes
python -m iar_repro analyze --extended
python -m iar_repro causal --bootstrap 5000
python -m iar_repro figures
```

These commands analyze supplied outputs; they perform no training or hosted requests. `tables` checks aligned per-item outcomes and reference summaries; `probes` checks pilot, screening, pairwise and the paper’s K=10 first/last 100-update windows. `analyze` runs rollout and diagnostic analyses; `--extended` includes bootstrap calibration/logistic controls. Spearman uses average tied ranks. `causal` analyzes the 2×2 intervention panel. `figures` produces all nine active manuscript PDFs, including the method diagram, into ignored local output directories.

The core analysis and figure scripts are under [scripts/analysis](../scripts/analysis/) and [scripts/figures](../scripts/figures/); their CLI arguments also accept individual inputs. New experiment output must be collected into the documented schemas before aggregate analysis. MirrorDuel policy training is proposed in the manuscript; it is not a reported executed experiment.

## Maintenance checks

```bash
python scripts/tools/check_links.py
python scripts/tools/build_reproduction_package.py
```

The builder excludes results, run state, credentials, external assets and environments, and verifies every ZIP member against its generated checksum manifest. Only maintainers intentionally updating source/input identities should run `scripts/tools/update_integrity_manifest.py`, after reviewing the diff. It is never called to suppress a verification failure during training.
