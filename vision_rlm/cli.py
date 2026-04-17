from __future__ import annotations

import argparse
import importlib.util
import json
import shutil

try:
    import torch
except Exception:  # pragma: no cover - best effort environment report
    torch = None

from vision_rlm.paths import build_project_paths


def _cmd_show_paths() -> int:
    paths = build_project_paths()
    print(json.dumps(paths.as_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_bootstrap_dirs() -> int:
    paths = build_project_paths()
    paths.ensure()
    print("Created or confirmed Vision-RLM directories:")
    print(json.dumps(paths.as_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_doctor() -> int:
    package_names = [
        "transformers",
        "accelerate",
        "datasets",
        "peft",
        "pymupdf",
        "rank_bm25",
        "faiss",
        "paddleocr",
        "bitsandbytes",
        "trl",
    ]
    binary_names = ["tmux", "nohup", "systemd-run", "sbatch", "srun", "nvidia-smi"]
    report = {
        "paths": build_project_paths().as_dict(),
        "python": {
            "packages": {
                name: bool(importlib.util.find_spec(name)) for name in package_names
            }
        },
        "binaries": {name: shutil.which(name) for name in binary_names},
    }
    if torch is not None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": gpu_count,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        }
    else:
        report["torch"] = {
            "version": None,
            "cuda_version": None,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision-RLM pilot utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-paths", help="Print canonical repo and storage paths")
    subparsers.add_parser(
        "bootstrap-dirs", help="Create the canonical large-file directories"
    )
    subparsers.add_parser(
        "doctor", help="Print an environment report for GPUs, binaries, and packages"
    )
    args = parser.parse_args()

    if args.command == "show-paths":
        return _cmd_show_paths()
    if args.command == "bootstrap-dirs":
        return _cmd_bootstrap_dirs()
    if args.command == "doctor":
        return _cmd_doctor()
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
