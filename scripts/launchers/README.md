# Launchers

Shell entrypoints are grouped here by workflow.

- `big_vae/`: active BigVAE training, dataset build, latent diffusion, and diagnostics.
- `post_train/`: post-training evaluation and analysis launchers.

Root-level BigVAE `.sh` files are compatibility wrappers. Prefer calling the
scripts in this directory for new runs.

BigVAE training config selection:

```bash
scripts/launchers/big_vae/train.sh
BIG_VAE_CONFIG=big_vae/train/v2 scripts/launchers/big_vae/train.sh
```
