#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from big_vae.flow_matching.sample import seal_flow_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a flow only after immutable E4 decoded validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--decoded-validation", type=Path, required=True)
    parser.add_argument("--global-e4-selection", type=Path, required=True)
    parser.add_argument("--codec", choices=("ours", "weightclip"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal_flow_checkpoint(
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer,
        decoded_validation_path=args.decoded_validation,
        global_selection_path=args.global_e4_selection,
        seal_path=args.output,
        codec=args.codec,
    )
    print(f"[flow-seal] output={args.output.resolve()}")


if __name__ == "__main__":
    main()
