from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    import transformers

    print(f"[preflight] torch={torch.__version__}")
    print(f"[preflight] cuda_available={torch.cuda.is_available()}")
    print(f"[preflight] transformers={transformers.__version__}")

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("[preflight] CUDA is required but torch.cuda.is_available() is False.")

    try:
        from transformers import (  # noqa: F401
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
    except ImportError as exc:
        raise SystemExit(
            "[preflight] Qwen3-VL dense/MoE classes are not available in the installed transformers build."
        ) from exc

    print("[preflight] imports_ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        raise
