# Weight VAE

This domain owns the BigVAE/weight-distribution pipeline and the research built
on its learned latent parameterization.

- `workspace/`: canonical BigVAE monorepo tree, preserving the historical
  relative layout of configs, scripts, tests, docs, and implementation.
- `components/`: convenient links into that workspace.
- `research/`: experiment entry points and reparameterization/preconditioning
  causal-debug records.
- `docs/`: BigVAE reports and dataset/model inventory.
- `artifacts/`: domain-filtered views of BigVAE, CELO, latent-landscape, Kron,
  and training artifacts.
- `data`: shared training data used by this pipeline.
- `runtime/`: Jupyter virtual documents produced from Weight VAE notebooks.

Former root paths are compatibility links into `workspace/`. Keeping all
path-sensitive components together preserves their relative filesystem
semantics after `Path(__file__).resolve()`.
