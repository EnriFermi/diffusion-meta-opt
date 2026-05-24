# MiniVAE Legacy Area

MiniVAE training is no longer the active path for new work. The code remains in
place because BigVAE configs, checkpoints, and compatibility tests still refer
to MiniVAE-shaped config fields.

Use `legacy/mini_vae/scripts/train_mini_vae.sh` only for old checkpoint
reproduction or migration checks. New training launchers should go through
`scripts/launchers/big_vae/`.
