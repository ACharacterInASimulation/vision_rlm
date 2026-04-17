from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    large_root: Path
    data_root: Path
    raw_data_root: Path
    processed_data_root: Path
    manifests_root: Path
    artifacts_root: Path
    checkpoints_root: Path
    logs_root: Path
    cache_root: Path

    def as_dict(self) -> dict[str, str]:
        return {key: value.as_posix() for key, value in asdict(self).items()}

    def ensure(self) -> None:
        for path in (
            self.large_root,
            self.data_root,
            self.raw_data_root,
            self.processed_data_root,
            self.manifests_root,
            self.artifacts_root,
            self.checkpoints_root,
            self.logs_root,
            self.cache_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_project_paths() -> ProjectPaths:
    repo_root = Path(__file__).resolve().parents[1]
    default_large_root = Path("/l/users/badrinath.chandana/vision_rlm")
    large_root = Path(
        os.environ.get("VISION_RLM_LARGE_ROOT", default_large_root.as_posix())
    )
    data_root = Path(
        os.environ.get("VISION_RLM_DATA_ROOT", (large_root / "data").as_posix())
    )
    raw_data_root = Path(
        os.environ.get(
            "VISION_RLM_RAW_DATA_ROOT", (data_root / "raw").as_posix()
        )
    )
    processed_data_root = Path(
        os.environ.get(
            "VISION_RLM_PROCESSED_DATA_ROOT", (data_root / "processed").as_posix()
        )
    )
    manifests_root = Path(
        os.environ.get(
            "VISION_RLM_MANIFESTS_ROOT", (data_root / "manifests").as_posix()
        )
    )
    artifacts_root = Path(
        os.environ.get(
            "VISION_RLM_ARTIFACT_ROOT", (large_root / "artifacts").as_posix()
        )
    )
    checkpoints_root = Path(
        os.environ.get(
            "VISION_RLM_CHECKPOINT_ROOT", (large_root / "checkpoints").as_posix()
        )
    )
    logs_root = Path(
        os.environ.get("VISION_RLM_LOG_ROOT", (large_root / "logs").as_posix())
    )
    cache_root = Path(
        os.environ.get("VISION_RLM_CACHE_ROOT", (large_root / "cache").as_posix())
    )
    return ProjectPaths(
        repo_root=repo_root,
        large_root=large_root,
        data_root=data_root,
        raw_data_root=raw_data_root,
        processed_data_root=processed_data_root,
        manifests_root=manifests_root,
        artifacts_root=artifacts_root,
        checkpoints_root=checkpoints_root,
        logs_root=logs_root,
        cache_root=cache_root,
    )
