# Frozen paper inputs

These are the prepared, fixed splits used by the experiments. Their original hashes are recorded in `reproduction/provenance/desktop-original-sha256.json`; original split/scoring manifests are under `reproduction/provenance/protocols/`. Reproduction from these files requires no new sampling. Rebuilding from upstream sources is implemented in `iar_campaign.data_protocol`, `iar_cqa.data`, `iar_gsm8k.data`, and `iar_spider.data`; use the pinned source revisions/hashes in those modules and manifests rather than current dataset heads.

| Input | Upstream source | Prepared sizes |
|---|---|---|
| CommonsenseQA with CoS-E | `tau/commonsense_qa`, `Salesforce/cos_e` | train 1,000, validation 300, test 1,221 |
| GSM8K | official GSM8K train/test | train 1,000, validation 300, test 1,319, pilot 374 |
| KodCode | `KodCode/KodCode-Light-RL-10K` | train 1,000, validation 300, test 1,000, pilot 374 |
| Spider 1.0 | `xlangai/spider` and official Spider release | train 1,000, validation 300, test 1,034 |
| Teacher corpora | Recorded DeepSeek/StreamLake generation | 1,000 responses per domain |

Dataset ownership and upstream terms are unchanged. Consult the authors’ dataset releases before redistribution or use (in particular KodCode’s noncommercial terms and Spider’s release terms). This repository does not relicense upstream data. Cached teacher outputs also remain subject to the originating service’s terms.

The gate retains 992 KodCode test items and all 1,034 Spider test items. KodCode pilot validation retains 299/300. Gold answers, references and hidden tests are evaluator inputs; prompt builders exclude them. Spider database binaries and schema release files must be obtained separately. Teacher corpora are frozen inputs for SFT, not newly generated audit results.

`screen/` contains the exact four probe100 task splits used by screening. Cached SFT corpora retain training messages and source response identities; per-item teacher correctness/validity labels are excluded.
