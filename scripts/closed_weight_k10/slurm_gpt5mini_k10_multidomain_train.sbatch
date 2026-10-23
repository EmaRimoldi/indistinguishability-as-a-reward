#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
set -euo pipefail
set +x
: "${IAR_CAMPAIGN_SOURCE:?}"
: "${IAR_GPT5MINI_LOG_DIR:?}"
: "${IAR_OPENROUTER_KEY_NAME:?}"
: "${IAR_GPT5MINI_DOMAIN:?}"
POOL_ROOT=/path/to/storage
PYTHON=${PYTHON:-python}
KEY_POOL=/path/to/iar_code/keys/key_pool.env
LOCK_DIR="$POOL_ROOT/experiments/campaign_20260908_v2/00_campaign_control/locks/openrouter"
export HF_HOME="$POOL_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$POOL_ROOT/cache/datasets"
export TRITON_CACHE_DIR="$POOL_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$POOL_ROOT/cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$POOL_ROOT/cache/torch_extensions"
export XDG_CACHE_HOME="$POOL_ROOT/cache/xdg"
export TMPDIR="$POOL_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$IAR_CAMPAIGN_SOURCE/src:$IAR_CAMPAIGN_SOURCE"
if [[ "$IAR_GPT5MINI_DOMAIN" == spider ]]; then
  export PYTHONPATH="$POOL_ROOT/envs/spider-eval-v1/site-packages:$PYTHONPATH"
fi
mkdir -p "$IAR_GPT5MINI_LOG_DIR" "$TMPDIR"
exec >"$IAR_GPT5MINI_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
exec 2>"$IAR_GPT5MINI_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
cd "$IAR_CAMPAIGN_SOURCE"
"$PYTHON" -B scripts/campaign_gpt5mini_k10_multidomain.py train \
  --key-pool "$KEY_POOL" --key-name "$IAR_OPENROUTER_KEY_NAME" \
  --lock-dir "$LOCK_DIR" --run-id "${IAR_GPT5MINI_RUN_ID:-attempt001}"
