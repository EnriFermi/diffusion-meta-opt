# Unified Config Layout (No Backward Compatibility)

## Entry points
- `conf/config.yaml` -> `run_profiles/train_big_vae`
- `conf/mini_vae_train.yaml` -> `run_profiles/train_mini_vae`

## Main groups
- `shared_runtime_environment/`
  - global app env (`hf`, `hydra`, `logging`, model registry)
  - artifact/checkpoint path conventions (mini -> big)

- `data_collection_runtime/`
  - `data_profiles/`: what datasets/profile to use
  - `collector_profiles/`: collector runtime behavior
  - `streaming_profiles/`: chunk transport/storage runtime

- `shared_training_parameters/`
  - trainer execution common params shared by both stages
  - model backbone shared by both stages (`distribution` + `mini_vae`)

- `mini_vae_experiment/`
  - mini-specific trainer params (loss/telemetry/patch sampling)
  - mini-specific model params

- `big_vae_experiment/`
  - big-specific trainer params
  - big-specific model params

- `run_profiles/`
  - final composition per run (`train_big_vae`, `train_mini_vae`)

## Notes
- Legacy groups (`train/`, `mini_train/`, `model/`, `mini_model/`, `data/collector/`, `data/streaming/`) were removed.
- Use direct key overrides (e.g. `mini_train.max_steps=...`, `train.max_steps=...`).
