from __future__ import annotations

import argparse
import importlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-transformers", default="4.50.0")
    parser.add_argument("--expected-gptqmodel", default="2.2.0")
    parser.add_argument("--min-autoawq", default="0.1.8")
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

    try:
        autoawq = importlib.import_module("awq")
    except Exception as exc:
        raise SystemExit(
            "[preflight] Failed to import autoawq/awq cleanly. "
            "This usually means Colab still has an incompatible autoawq build installed."
        ) from exc
    autoawq_version = getattr(autoawq, "__version__", "unknown")
    print(f"[preflight] autoawq={autoawq_version}")
    if autoawq_version == "unknown":
        raise SystemExit("[preflight] autoawq is installed but did not report a version.")

    def _version_tuple(value: str) -> tuple[int, ...]:
        cleaned = value.replace(".post", ".").split(".")
        parts: list[int] = []
        for chunk in cleaned:
            if chunk.isdigit():
                parts.append(int(chunk))
            else:
                digits = "".join(ch for ch in chunk if ch.isdigit())
                if digits:
                    parts.append(int(digits))
        return tuple(parts)

    if _version_tuple(autoawq_version) < _version_tuple(args.min_autoawq):
        raise SystemExit(
            f"[preflight] autoawq version {autoawq_version} is too old; "
            f"expected >= {args.min_autoawq}."
        )

    print("[preflight] imports_ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        raise
