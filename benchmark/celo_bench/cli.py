from __future__ import annotations

import argparse
from typing import Any

from .config import CeloBenchConfig
from .runner import run_or_load


def _patch_argparse_lazy_help_for_hydra_py314() -> None:
    if getattr(argparse.ArgumentParser, "_hydra_lazy_help_py314_patch", False):
        return
    original_check_help = getattr(argparse.ArgumentParser, "_check_help", None)
    if original_check_help is None:
        return

    def patched_check_help(self: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        original_check_help(self, action)

    argparse.ArgumentParser._check_help = patched_check_help  # type: ignore[method-assign]
    argparse.ArgumentParser._hydra_lazy_help_py314_patch = True  # type: ignore[attr-defined]


_patch_argparse_lazy_help_for_hydra_py314()


def _plain_hydra_cfg(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved Celo benchmark config must be a mapping")
    return payload


try:
    import hydra
except Exception as exc:  # pragma: no cover - only hit in incomplete runtime envs.
    hydra = None
    _HYDRA_IMPORT_ERROR = exc
else:
    _HYDRA_IMPORT_ERROR = None


if hydra is not None:

    @hydra.main(version_base=None, config_path="../../conf/celo_bench", config_name="config")
    def main(cfg: Any) -> None:
        bench_cfg = CeloBenchConfig.from_mapping(_plain_hydra_cfg(cfg))
        tables = run_or_load(bench_cfg)
        print(f"Celo benchmark summary: {tables.output_dir / 'summary.json'}")

else:

    def main(_cfg: Any = None) -> None:
        raise RuntimeError(f"hydra-core is required to run benchmark.celo_bench.cli: {_HYDRA_IMPORT_ERROR}")


if __name__ == "__main__":
    main()

