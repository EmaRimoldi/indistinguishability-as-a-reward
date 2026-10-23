# Code and input provenance

- `reproduction.json` identifies the manuscript, Engaging checkout and six recovered source families.
- `sources/*.inventory.json` retains original source-file hashes; duplicate source archives are excluded.
- `source-map.json` maps maintained Python files to exact frozen-source matches where available.
- `desktop-original-sha256.json` retains original supplement identities, including fixed dataset split hashes.
- `input-manifest.json` maps historical protocol paths and original hashes to deduplicated settings-only extracts in `protocols/`. Measurements were removed; package hashes explicitly identify the extracts rather than claiming original byte identity.
- `sft-manifest.json` binds the training corpora and actual trainer arguments. Evaluation labels were removed; the original and transformed corpus identities are both recorded, and training messages are unchanged.
- `package-integrity.json` protects runnable source, settings and inputs after the reorganization.

These records contain source/input provenance and frozen settings. Experimental outputs, audit measurements, numerical result manifests and packed historical results are external. [Reproduction instructions](../../docs/REPRODUCIBILITY.md) explain configuration and optional external analyses.
