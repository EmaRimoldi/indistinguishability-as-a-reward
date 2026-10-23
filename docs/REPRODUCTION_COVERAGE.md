# Paper coverage

This package implements the experiments and framework for manuscript revision `929ec888e05dccb5b483ca0f658f95970c5c91d3`. Commands below are described in the [reproduction guide](REPRODUCIBILITY.md). Experimental outputs are external.

| Paper component | Implementation / command | Required inputs |
|---|---|---|
| A0 and main four-domain comparison | `train`, `evaluate`; `iar_grpo`, `iar_campaign.baseline_rewards` | Frozen splits, pinned weights, GPU; API for Indist./Quality |
| Three seeds, Indist./Quality/Oracle/SFT | `--condition indist/quality/oracle/sft --seed 23/42/7` | SFT messages included; Spider assets external |
| Closed-weight K=10 | `closed_weight_k10`, seed 23; frozen K13/multidomain retry logic | GPT-5 Mini/OpenAI Flex credentials and GPU |
| Reward-form pilot | `pilot configure/train/evaluate` | Two nested 374-task splits and validation sets |
| Teacher SFT corpus | `teacher-corpus` | Cached training messages included; API needed to regenerate |
| Teacher screening | `teacher-screen` | Fixed probe100 tasks, fresh actor/teacher responses and hosted judgments |
| Pairwise order effects | `pairwise` | Prefix100 tasks, SFT teacher messages, fresh four-rollout generation and hosted judgments |
| Main statistics and paired tests | `tables`; `scripts/analysis/heldout_stats.py` | External per-item outcomes and reference summaries |
| Reward, correctness, length, calibration and style diagnostics | `analyze --extended` | External rollout labels, reward logs and feature rows |
| Counterfactual 2×2 intervention | `iar_campaign.causal_reward_audit`; `causal` for analysis | Fresh API judgments or external panel/judge records |
| K=10 training windows, pilot and probe summaries | `probes` | External logs/judgments |
| All nine active manuscript figures | `figures`; `scripts/figures/` | External numerical figure inputs and trainer histories |
| MirrorDuel binary pilot / policy training | Algorithm described by the paper | Proposed, not a reported executed run |

The nine figure outputs are `turing_reward_overview`, `learning_to_fool`, `reward_signal`, `reward_distributions_by_correctness`, `grpo_loss`, `training_diagnostics_4x4`, `within_prompt_correctness_gap`, `judge_vs_reward`, and `response_length_by_reward` (PDF).

See [verification scope](REPRODUCTION_AUDIT.md) for the distinction between offline checks, synthetic training, external-data analysis and full experiments.
