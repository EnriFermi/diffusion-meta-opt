---
pretty_name: WeightCLIP ResNet18Slim operator bank
tags:
  - weight-compression
  - model-weights
  - activations
  - gptq
---

# WeightCLIP ResNet18Slim operator bank

This private dataset repository stores the exact preprocessed operator dataset
used by the current WeightCLIP production training run. It contains the active
weight-tile bank, the matching activation-context bank, and the metadata needed
to reproduce production sampling.

- Hub: `https://huggingface.co/datasets/EnriFermi/weightclip-resnet18slim-operator-bank`
- Pinned dataset revision: `79d9b8c97362a3892d1f8475f71face6f9859b72`
- Completed payload revision: `3bc8cabb959f7dfd5682e9e36c6d69a51a00f444`

The active payload is 41,724,270,793 bytes (38.859 GiB) across 3,622 source
files:

- `weight_tile_bank-9020c363ee2005e8`: 135,100 weight records, 793 files,
  11,181,671,316 bytes;
- `context_bank-234e31b2ec93af4f`: 90,300 activation-context records, 2,825
  files, 30,517,934,053 bytes;
- `operator_dataset-c478efe716506765.json`, coverage metadata, and both
  permutation-view files.

This snapshot deliberately excludes older bank generations, raw source model
checkpoints, and cached experiment outputs. The authoritative machine-readable
contract is `dataset_contract.json` in the Hub repository and
`conf/weightclip_benchmark/operator_dataset_remote_hf.json` in the Git source.

## Download on another machine

The repository is private. Authenticate with a Hugging Face token that has read
access, then run from the Git checkout:

```bash
export HF_TOKEN=...
python projects/weight-vae/workspace/scripts/download_weightclip_operator_dataset_from_hf.py \
  --output-root /data/weightclip_resnet18slim_operator_dataset
```

The downloader pins the revision recorded in the Git contract, validates the
small immutable manifests and expected file inventory, and writes a resolved
pair manifest whose runtime paths point into the downloaded snapshot. It prints
the resolved manifest path and its SHA-256; use those values for
`operator_bank.pair_manifest` and `operator_bank.pair_manifest_sha256` in a
local production config.

The original pair manifest is retained byte-for-byte. It contains absolute
paths from the machine that built the dataset. Only its top-level runtime paths
need relocation; absolute paths nested under `contract` are historical
provenance and are not required to read the preprocessed banks.

## Source contract

- Pair manifest SHA-256:
  `f3a26fc9772d43feabe138ef9fa5be5baf2f76323fb4f3079c12c5f65275f347`
- Dataset contract SHA-256:
  `c478efe716506765af35236942e128d416472690d6dfc2cb5045bdfc359b562e`
- Weight-bank content SHA-256:
  `9020c363ee2005e88c5a9bc746e97c8620cf62c6fac94f4ed865b08bb7d9a19a`
- Context-bank content SHA-256:
  `234e31b2ec93af4f3bce294a5c09c202ae5c560d0faf2f5515c00ddfc3104bdd`

The bank content hashes above were sealed when the banks were built. The
download helper uses the small manifest hashes plus file inventory and does not
rehash all 38.9 GiB. The current production `OperatorDatasetBank` loader is
stricter: when it opens the materialized banks, it SHA-256-validates every
payload file. Plan for one full sequential read during first dataset open on a
new machine.
