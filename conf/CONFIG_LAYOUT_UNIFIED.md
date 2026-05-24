# Unified Config Layout (No Backward Compatibility)

## Entry points
- `conf/big_vae/train/default.yaml` -> `run_profiles/train_big_vae`
- `conf/big_vae/train/v2.yaml` -> `run_profiles/train_big_vae_v2`
- `conf/config.yaml` -> compatibility wrapper for `big_vae/train/default`
- `conf/config_big_vae_v2.yaml` -> compatibility wrapper for `big_vae/train/v2`

## Main groups
- `shared_runtime_environment/`
  - global app env (`hf`, `hydra`, `logging`, model registry)
  - BigVAE artifact/checkpoint path conventions

- `data_collection_runtime/`
  - `data_profiles/`: what datasets/profile to use (e.g. `data_profile_for_big_vae_training`, `data_profile_for_hf_assets_test`)
  - `collector_profiles/`: collector runtime behavior
  - `streaming_profiles/`: chunk transport/storage runtime

- `shared_training_parameters/`
  - trainer execution common params shared by both stages
  - model backbone shared by both stages (`distribution` + `mini_vae`)

- `big_vae_experiment/`
  - big-specific trainer params
  - big-specific model params

- `big_vae/train/`
  - one-file BigVAE train entrypoints; add a new run here instead of touching
    several root configs

- `run_profiles/`
  - final composition per BigVAE run

## Notes
- MiniVAE train configs and launchers live under `legacy/mini_vae/`.
- New active BigVAE runs write under `BIG_VAE_ARTIFACT_ROOT` when set, otherwise
  `./artifacts/big_vae`.
- Use direct key overrides (e.g. `train.max_steps=...`).
