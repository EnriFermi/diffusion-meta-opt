# Repository layout

## Compatibility policy

The repository contains several projects whose configs, launchers, notebooks,
and stored results refer to paths relative to the repository root. Independent
projects live under `projects/`; relative symbolic links preserve their former
paths. Tightly coupled monorepo components stay at the root and are exposed in
the organized view through links.

The rules are:

1. Preserve every established path as either its original directory or a
   compatibility link.
2. Add independent projects physically to their research domain (`weight-vae`,
   `cfm`, `lsdl`, or `unclassified`); add tightly coupled components as
   relative symbolic links.
3. Put reusable implementation in the existing shared roots (`big_vae/`,
   `dataset/`, `training/`, `post_train_research/`) rather than under
   `projects/` itself.
4. Keep third-party Git checkouts under the domain that uses them. Preserve an
   old path with a relative compatibility link when stored configs depend on it.
5. Keep generated and heavyweight content under `data/`, `artifacts/`, `logs/`,
   or `outputs/`; do not mix it into source directories.
6. Keep superseded but reproducibility-relevant code under `legacy/`. Nothing
   in `legacy/` should be removed merely as part of cleanup.
7. Do not commit secrets. Files such as local `*.env` files are runtime state,
   not project documentation.

## Research ownership map

### Weight VAE

Canonical view: `projects/weight-vae/`.

- `big_vae/`: public BigVAE package boundary.
- `dataset/`: dataset collection, virtualization, caching, and streaming.
- `training/`: shared training implementation.
- `post_train_research/`: evaluation and post-training research pipelines.
- `benchmark/celo_bench/`: CELO benchmark implementation.

- `experiments/`: experiment entry points and dataset-building programs.
- `docs/reparam_preconditioning_experiments/`: Variant A and related
  reparameterization/preconditioning research records.

### CFM

- `projects/cfm/source/`: main CFM text project.
- `projects/cfm/semicat/`: working and dated Semicat checkouts.
- `projects/cfm/baselines/`: BD3LMS upstream baseline.
- `projects/cfm/paper/`: TinyStories paper, measurements, and exports.
- `projects/cfm/artifacts/`: filtered Semicat and TinyStories evidence.

### LSDL

- `projects/lsdl/source/`: LSDL/LFQ notebook and local data.
- `projects/lsdl/artifacts`: LFQ loss-fix evidence.
- `projects/lsdl/lfq-notebook-loss-test.py`: notebook regression test.

### Unclassified

- `projects/unclassified/guided-diffusion/`: independent upstream checkout.
- `projects/unclassified/zo-muon-sst2/`: standalone optimizer experiment.
- `projects/unclassified/adhocs/` and `loose-root/`: one-off work.
- Sparse benchmarks, rotation metrics, and producer-unknown artifacts remain
  here until evidence establishes a more specific owner.

### Shared infrastructure

- `conf/`: Hydra and runtime configuration.
- `scripts/`: launchers, audits, reviews, and analysis utilities.
- `tests/`: regression and protocol tests.
- `docs/`: documentation and research records.

### Storage and history

- `data/`: downloaded data and caches.
- `artifacts/`: experiment artifacts and checkpoints.
- `logs/` and `outputs/`: runtime products.
- `legacy/`: historical and quarantined implementations.
- `adhocs/`: one-off work that has not become a maintained project.

## Root policy

The repository root is reserved for repository-level entry points and package
manager metadata: `README.md`, `AGENTS.md`, `.gitignore`, `Pipfile`, and
`Pipfile.lock`. Standalone documents, scripts, logs, archives, datasets,
experiment outputs, environment files, and notes must live in their matching
owned directory rather than accumulating at the root.

## Adding a project

Create an independent project directly in its research domain, or keep a
tightly coupled shared component in its established root and expose it with a
relative link. Add the corresponding manifest entry, then run:

```bash
python -m unittest tests.repository_layout_test
```

The check rejects missing projects, dangling links, views that resolve to the
wrong source, and loose files added back to the repository root.
