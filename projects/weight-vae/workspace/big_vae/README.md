# BigVAE Package Boundary

`big_vae/` is the active public boundary for BigVAE code:

```text
big_vae/
  entrypoints/   # script/module entrypoints used by launchers
  datasets/      # BigVAE dataset APIs
  eval/          # evaluation and post-train experiment packages
  models/        # active BigVAE model implementation
  runtime/       # artifact/logging helpers shared by train and eval
```

Active BigVAE code should import through this package boundary. Legacy
compatibility modules were removed; old root `models/` imports are intentionally
unsupported.
