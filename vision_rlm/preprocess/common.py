from __future__ import annotations

from hashlib import sha1
import json
import math
from pathlib import Path
import re
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "item"


def stable_doc_id(path: Path) -> str:
    stem = sanitize_name(path.stem)
    digest = sha1(path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def stable_string_id(name: str, namespace: str = "") -> str:
    stem = sanitize_name(name)
    basis = f"{namespace}::{name}" if namespace else name
    digest = sha1(basis.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_pdfs(input_path: Path, limit_docs: int | None = None) -> list[Path]:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
    if limit_docs is not None:
        return paths[:limit_docs]
    return paths


def scale_bbox(
    bbox: list[float], source_width: float, source_height: float, target_width: int, target_height: int
) -> list[int]:
    x0, y0, x1, y1 = bbox
    scale_x = target_width / max(source_width, 1.0)
    scale_y = target_height / max(source_height, 1.0)
    return [
        max(0, int(round(x0 * scale_x))),
        max(0, int(round(y0 * scale_y))),
        max(0, int(round(x1 * scale_x))),
        max(0, int(round(y1 * scale_y))),
    ]


def bbox_area(bbox: list[int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_iou(left: list[int], right: list[int]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    intersection = bbox_area([ix0, iy0, ix1, iy1])
    union = bbox_area(left) + bbox_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def merge_bboxes(boxes: Iterable[list[int]]) -> list[int]:
    xs0: list[int] = []
    ys0: list[int] = []
    xs1: list[int] = []
    ys1: list[int] = []
    for x0, y0, x1, y1 in boxes:
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return [0, 0, 0, 0]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def token_count(text: str) -> int:
    return len(text.split())


def almost_same_line(box_a: list[int], box_b: list[int], tolerance: float = 0.03) -> bool:
    ay = (box_a[1] + box_a[3]) / 2
    by = (box_b[1] + box_b[3]) / 2
    height = max(box_a[3] - box_a[1], box_b[3] - box_b[1], 1)
    return abs(ay - by) <= max(2.0, height * tolerance)


def ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)
