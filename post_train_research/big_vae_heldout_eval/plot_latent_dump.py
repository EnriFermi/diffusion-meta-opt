from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluate_big_vae_heldout import plot_latent_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild PCA/t-SNE plots from a held-out BigVAE latent dump.")
    parser.add_argument(
        "dump_path",
        nargs="?",
        default=os.environ.get("EVAL_LATENT_DUMP_PATH", ""),
        help="Path to latent_dump.pt. Defaults to EVAL_LATENT_DUMP_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EVAL_LATENT_PLOT_DIR", ""),
        help="Directory for PNG/CSV outputs. Defaults to <dump_dir>/latent_plots.",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", "42")))
    parser.add_argument("--no-tsne", action="store_true", help="Only rebuild PCA plots.")
    args = parser.parse_args()

    if not str(args.dump_path).strip():
        raise SystemExit("Pass dump_path or set EVAL_LATENT_DUMP_PATH=/path/to/latent_dump.pt")

    output_dir = Path(args.output_dir).expanduser() if str(args.output_dir).strip() else None
    info = plot_latent_dump(
        args.dump_path,
        output_dir=output_dir,
        seed=int(args.seed),
        run_tsne=not bool(args.no_tsne),
    )
    print(f"wrote latent plots to {info.get('output_dir', '')}")


if __name__ == "__main__":
    main()
