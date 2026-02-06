import argparse
import math
import sys

import torch
from omegaconf import OmegaConf


def default_cfg():
    return OmegaConf.create({
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "downstream": {
            # Which downstream problem the meta-optimizer is trained/evaluated on.
            # Options: "regression" (toy tasks) | "cifar10"
            "name": "regression",
        },
        # Toy regression task-mixture (used when downstream.name == "regression")
        "tasks": {
            "strategy": "weighted_random",
            "defaults": {
                "n_train": 32,
                "n_val": 32,
                "x_min": -math.pi,
                "x_max": math.pi,
                "noise_std": 0.0,
            },
            "mix": [
                {"name": "sine", "weight": 1.0, "params": {"x_coef": 0.1}},
            ],
        },
        "cifar10": {
            "data_dir": "./data",
            "download": True,
            # Number of samples per episode (train/val subsets sampled from CIFAR-10)
            "n_train": 64,
            "n_val": 64,
            # Lightweight augments to keep the inner-loop fast
            "augment": True,
            "normalize": True,
            # Keep CIFAR tensors on GPU between episodes (faster, uses VRAM)
            "cache_on_gpu": False,
        },
        "model": {
            # Regression model (TwoLayerMLP)
            "hidden": 32,
            # CIFAR-10 model (tiny ResNet)
            "resnet": {
                "base_channels": 16,
                "blocks": [2, 2, 2],
                "num_classes": 10,
            },
        },
        "meta": {
            # Outer/meta training algorithm: "sac" | "velo"
            "algorithm": "sac",
            "inner_steps": 25,
            "mom_beta": 0.9,
            "adam_beta2": 0.999,  # squared-grad EMA like Adam
        },
        "velo": {
            # VeLO-style ES training of the actor (black-box gradient estimate)
            "population_size": 8,   # number of antithetic noise pairs per update
            "sigma": 0.01,          # parameter noise std
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "episodes_per_eval": 1,  # average this many episodes per +/-
            "normalize_diffs": True,
            "grad_clip_norm": 1.0,
        },
        "rl": {
            "algorithm": "sac",
            "gamma": 0.99,
            "tau": 0.005,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "batch_size": 64,
            "buffer_size": 50000,
            "train_episodes": 10000,
            "warmup_steps": 500,
            "updates_per_step": 1,
            "max_action": 0.05,
            "log_std_min": -10.0,
            "log_std_max": 3.0,
        },
        "baseline": {
            "episodes": 128,
            "adam_lr": 1e-2,
            "sgd_lr": 5e-2,
        },
        "checkpoint": {
            "path": "micro_meta_sac.pt",
            "save_every_episodes": 100,
        },
        # Performance knobs. Defaults are conservative and CPU-safe.
        "perf": {
            "compile": bool(torch.cuda.is_available()),
            # AMP: "off" | "auto" | "bf16" | "fp16"
            "amp": "auto" if torch.cuda.is_available() else "off",
            # torch.compile backend; "inductor" is default, but keep configurable
            "compile_backend": "inductor",
            # Dataloader parallelism (used by CIFAR-10 task)
            "num_workers": 0 if sys.platform in ("darwin", "win32") else 4,
            "pin_memory": True,
            "persistent_workers": True if sys.platform not in ("darwin", "win32") else False,
            # CUDA perf
            "cudnn_benchmark": True,
            "tf32": True,
            "channels_last": True,
        },
    })


def get_cfg(argv=None):
    """
    Build runtime config as:
      default_cfg()
        <- (optional) --config/-c YAML
        <- (optional) OmegaConf dotlist overrides: key=value
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args(argv)

    cfg = default_cfg()
    if args.config:
        file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


EVAL_EVERY = 100  # episodes
