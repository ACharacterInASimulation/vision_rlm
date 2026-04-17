from __future__ import annotations

import argparse
import importlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-transformers", default="4.50.0")
    parser.add_argument("--expected-gptqmodel", default="2.2.0")
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

    if transformers.__version__ != args.expected_transformers:
        raise SystemExit(
            "[preflight] Unexpected transformers version "
            f"{transformers.__version__}; expected {args.expected_transformers}."
        )

    try:
        from transformers.modeling_utils import no_init_weights  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[preflight] transformers.modeling_utils.no_init_weights is missing; "
            "the installed transformers build is not compatible with the AWQ loader."
        ) from exc

    gptqmodel = importlib.import_module("gptqmodel")
    gptqmodel_version = getattr(gptqmodel, "__version__", "unknown")
    print(f"[preflight] gptqmodel={gptqmodel_version}")
    if gptqmodel_version != args.expected_gptqmodel:
        raise SystemExit(
            f"[preflight] Unexpected gptqmodel version {gptqmodel_version}; "
            f"expected {args.expected_gptqmodel}."
        )

    print("[preflight] imports_ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        raise
