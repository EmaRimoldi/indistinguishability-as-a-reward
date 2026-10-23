# Verification scope

The package was reconciled with the untouched Desktop supplement, the Engaging project and its frozen source families, and Overleaf revision `929ec888e05dccb5b483ca0f658f95970c5c91d3`. Source/input identities and intentional settings-only extracts are recorded in [provenance](../reproduction/provenance/README.md).

The maintained package fixes portable path resolution, dependency declarations, K=10 transient retries, the pilot HTTP budget, evaluation checkpoint handling, SFT initialization seeding and Spearman tied ranks. Training/evaluation inputs are hash checked; settings-only protocol extracts have new package hashes while retaining their original source identities.

The source-only layout groups environment, provenance and notices under `reproduction/`, and tools under `scripts/`. Exploratory cluster launchers, unrelated mechanistic analyses, duplicate source archives and experimental outputs are excluded. Cached SFT messages remain training inputs; their correctness/validity labels were removed. No generated figures, per-item evaluation outcomes, trainer logs, statistical summaries or historical result archives are distributed.

## Checks

The offline CI installs a clean environment, validates the complete configuration grid, runs the test suite and tiny synthetic training, checks documentation links, and builds/verifies the source archive. Analysis/figure commands are tested separately against an external result collection; a fresh checkout explicitly reports missing external records.

A smoke test performs two CPU GRPO and two completion-only SFT updates with random tiny weights, checks finite/nonzero gradients and saved adapters, and exercises positive/negative KodCode scoring. It makes no hosted requests and downloads no model. These are implementation checks, not paper experiments.

## External verification

Full 2,000-update GRPO, 500-update SFT, 80,000-judgment K=10 runs and full-model inference have not been rerun. Current hosted endpoint availability, full GPU stability, official Spider assets and all problem-specific KodCode dependencies need verification in the target execution environment. Full KodCode/Spider pairwise correctness rescoring also requires those external assets.

Receipt-preserving K=10 continuation requires the original external HTTP receipts and checkpoint state. The portable CLI supports fresh K=10 attempts. The guide identifies paid commands and external requirements.
