from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from vision_rlm.preprocess.common import (
    bbox_iou,
    merge_bboxes,
    read_jsonl,
    token_count,
    write_json,
    write_jsonl,
)


def _fallback_regions(page_id: str, page_width: int, page_height: int) -> list[dict]:
    mid_x = page_width // 2
    mid_y = page_height // 2
    top_band = max(1, page_height // 5)
    center_y0 = page_height // 4
    center_y1 = page_height - center_y0
    return [
        {"region_type": "top_band", "bbox": [0, 0, page_width, top_band], "region_text": ""},
        {"region_type": "bottom_band", "bbox": [0, page_height - top_band, page_width, page_height], "region_text": ""},
        {"region_type": "left_half", "bbox": [0, 0, mid_x, page_height], "region_text": ""},
        {"region_type": "right_half", "bbox": [mid_x, 0, page_width, page_height], "region_text": ""},
        {"region_type": "center_band", "bbox": [0, center_y0, page_width, center_y1], "region_text": ""},
        {"region_type": "full_page", "bbox": [0, 0, page_width, page_height], "region_text": ""},
    ]


def _merge_text_blocks(blocks: list[dict], min_tokens: int = 40, max_tokens: int = 120) -> list[dict]:
    merged: list[dict] = []
    current_blocks: list[dict] = []
    current_tokens = 0
    current_text_parts: list[str] = []

    def flush() -> None:
        nonlocal current_blocks, current_tokens, current_text_parts
        if not current_blocks:
            return
        merged.append(
            {
                "region_type": "text_region",
                "bbox": merge_bboxes([block["bbox"] for block in current_blocks]),
                "region_text": " ".join(current_text_parts).strip(),
                "source_block_ids": [block["block_id"] for block in current_blocks],
            }
        )
        current_blocks = []
        current_tokens = 0
        current_text_parts = []

    for block in sorted(blocks, key=lambda item: item.get("reading_order", 0)):
        block_tokens = int(block.get("ocr_tokens", token_count(block.get("block_text", ""))))
        if current_blocks and current_tokens + block_tokens > max_tokens:
            flush()
        current_blocks.append(block)
        current_text_parts.append(block.get("block_text", ""))
        current_tokens += block_tokens
        if current_tokens >= min_tokens:
            flush()
    flush()
    return merged


def _dedupe_regions(regions: list[dict], iou_threshold: float = 0.70) -> list[dict]:
    deduped: list[dict] = []
    for region in regions:
        if any(bbox_iou(region["bbox"], existing["bbox"]) > iou_threshold for existing in deduped):
            continue
        deduped.append(region)
    return deduped


def build_regions(
    layout_manifest: Path,
    output_name: str,
    max_regions_per_page: int,
    *,
    verbose: bool,
) -> dict[str, str]:
    layout_rows = read_jsonl(layout_manifest)
    region_root = layout_manifest.parent.parent.parent / "regions" / output_name
    region_root.mkdir(parents=True, exist_ok=True)
    page_region_root = region_root / "pages"
    page_region_root.mkdir(parents=True, exist_ok=True)

    region_rows: list[dict] = []
    parquet_rows: list[dict] = []
    total_pages = len(layout_rows)

    if verbose:
        print(f"[build_regions] start output={output_name} pages={total_pages}")

    for index, row in enumerate(layout_rows, start=1):
        page_payload = Path(row["page_json_path"]).read_text(encoding="utf-8")
        page_data = json.loads(page_payload)
        blocks = page_data.get("blocks", [])
        text_blocks = [block for block in blocks if block.get("block_text")]
        regions = _merge_text_blocks(text_blocks)
        regions.extend(
            _fallback_regions(
                page_id=page_data["page_id"],
                page_width=int(page_data["hires_width"]),
                page_height=int(page_data["hires_height"]),
            )
        )
        regions = _dedupe_regions(regions)[:max_regions_per_page]

        page_regions: list[dict] = []
        for index, region in enumerate(regions):
            region_id = f'{page_data["page_id"]}_r{index:03d}'
            payload = {
                "region_id": region_id,
                "page_id": page_data["page_id"],
                "doc_id": page_data["doc_id"],
                "page_num": page_data["page_num"],
                "region_type": region["region_type"],
                "bbox": region["bbox"],
                "region_text": region.get("region_text", ""),
                "source_block_ids": region.get("source_block_ids", []),
            }
            page_regions.append(payload)
            parquet_rows.append(payload)

        page_region_path = page_region_root / f'{page_data["page_id"]}.json'
        write_json(
            page_region_path,
            {
                "page_id": page_data["page_id"],
                "doc_id": page_data["doc_id"],
                "page_num": page_data["page_num"],
                "regions": page_regions,
            },
        )
        region_rows.append(
            {
                "page_id": page_data["page_id"],
                "doc_id": page_data["doc_id"],
                "page_num": page_data["page_num"],
                "num_regions": len(page_regions),
                "page_region_path": page_region_path.as_posix(),
            }
        )

        if verbose and (index == total_pages or index % 25 == 0):
            print(
                f"[build_regions] processed {index}/{total_pages} pages "
                f"latest_page={page_data['page_id']} regions={len(page_regions)}"
            )

    write_jsonl(region_root / "region_manifest.jsonl", region_rows)
    try:
        import pandas as pd

        pd.DataFrame(parquet_rows).to_parquet(region_root / "regions.parquet", index=False)
    except Exception:
        pass
    write_json(
        region_root / "run_manifest.json",
        {
            "output_name": output_name,
            "layout_manifest": layout_manifest.as_posix(),
            "page_count": len(region_rows),
            "max_regions_per_page": max_regions_per_page,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if verbose:
        print(
            f"[build_regions] done output={output_name} page_count={len(region_rows)} "
            f"region_manifest={(region_root / 'region_manifest.jsonl').as_posix()}"
        )
    return {
        "region_root": region_root.as_posix(),
        "region_manifest": (region_root / "region_manifest.jsonl").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build page-level region inventories from parsed layout")
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--max-regions-per-page", type=int, default=24)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    outputs = build_regions(
        layout_manifest=Path(args.layout_manifest).expanduser().resolve(),
        output_name=args.output_name,
        max_regions_per_page=args.max_regions_per_page,
        verbose=args.verbose,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
