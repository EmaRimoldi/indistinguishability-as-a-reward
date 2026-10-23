# Indistinguishability as a Reward

Experiment and framework code for **Uncovering the Turing Test’s Latent Signal: Indistinguishability as a Reward** (Rimoldi, Neumarker, Poggio, and Mitropolsky).

Qwen2.5-1.5B-Instruct learns from the log-probability that a frozen teacher attributes a response to itself. The package implements the paper’s four-domain, three-seed comparison with Quality, Oracle and SFT, the GPT-5 Mini K=10 condition, reward-form selection and diagnostic analyses.

## Install and verify

Python **3.14** was tested. Install `uv` and Cairo first (`brew install cairo` on macOS; `libcairo2` on Ubuntu). Run from the repository root:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r reproduction/environment/requirements-portable.txt
source .venv/bin/activate
python -m iar_repro verify
python -m pytest -q
python -m iar_repro smoke
```

These checks require no model download, GPU or API calls. The smoke test runs two GRPO and two SFT updates with tiny random weights and synthetic rewards; choose a new `--output-dir` when repeating it.

## Run experiments

Obtain the pinned actor weights and official Spider assets, then configure their locations:

```bash
python scripts/setup_paths.py --cache /path/to/hf/hub --spider-root /path/to/spider_release
export OPENROUTER_API_KEY=...
python -m iar_repro train --domain gsm8k --condition indist --seed 23
python -m iar_repro evaluate --domain gsm8k --condition indist --seed 23
```

Training uses one A100 80GB or H100 per job. Hosted conditions require the exact model/provider routes. The [reproduction guide](docs/REPRODUCIBILITY.md) covers every condition, seed, dataset, pilot, probe and analysis; the [coverage matrix](docs/REPRODUCTION_COVERAGE.md) maps them to the paper.

## Package

| Directory | Contents |
|---|---|
| [src/](src/) | Framework, training, rewards, scorers and reproduction CLI |
| [configs/](configs/) | Experiment settings and frozen protocols |
| [data/](data/) | Fixed splits, reference-valid IDs and SFT training corpora |
| [scripts/](scripts/) | Analysis, figure generation and maintenance tools |
| [reproduction/](reproduction/) | Dependencies, code/input provenance and upstream notices |
| [tests/](tests/) | Offline protocol, numerical and workflow checks |
| [docs/](docs/) | Reproduction instructions and verification scope |

**Experimental results are not distributed in this repository or its current reproduction archive.** Training logs, evaluation outputs, figure data, generated figures and checkpoints belong in ignored local directories or external storage. Analysis commands accept an external result collection through `IAR_RESULTS_ROOT`; see [external results](docs/REPRODUCIBILITY.md#external-results-and-figures).

Code and input identities match the latest audited manuscript revision `929ec888e05dccb5b483ca0f658f95970c5c91d3`. See [provenance](reproduction/provenance/README.md), [data terms](data/README.md), [upstream notices](reproduction/third_party/README.md) and [verification limits](docs/REPRODUCTION_AUDIT.md).
