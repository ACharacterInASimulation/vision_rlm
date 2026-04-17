from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from vision_rlm.preprocess.common import ensure_dir, read_jsonl, token_count, write_json, write_jsonl


def _load_page_payload(page_json_path: str) -> dict:
    return json.loads(Path(page_json_path).read_text(encoding="utf-8"))


def _estimate_title(text_lines: list[str]) -> str:
    for line in text_lines[:5]:
        count = token_count(line)
        if 1 <= count <= 14:
            return line
    return text_lines[0] if text_lines else ""


def _build_page_row(layout_row: dict, regions_by_page: dict[str, list[dict]]) -> dict:
    page_data = _load_page_payload(layout_row["page_json_path"])
    text_lines = [
        str(block.get("block_text", "")).strip()
        for block in sorted(page_data.get("blocks", []), key=lambda item: item.get("reading_order", 0))
        if str(block.get("block_text", "")).strip()
    ]
    layout_tags = sorted({str(tag) for tag in page_data.get("layout_tags", []) if str(tag)})
    title = _estimate_title(text_lines)
    ocr_head = text_lines[:3]
    page_text = " ".join(text_lines).strip()
    sketch_parts = [title, " ".join(ocr_head), " ".join(layout_tags), page_text]
    sketch_text = " ".join(part for part in sketch_parts if part).strip()
    return {
        "page_id": page_data["page_id"],
        "doc_id": page_data["doc_id"],
        "page_num": int(page_data["page_num"]),
        "parser_used": page_data.get("parser_used", layout_row.get("parser_used", "")),
        "title": title,
        "ocr_head": ocr_head,
        "layout_tags": layout_tags,
        "page_text": page_text,
        "sketch_text": sketch_text,
        "num_blocks": len(page_data.get("blocks", [])),
        "num_regions": len(regions_by_page.get(page_data["page_id"], [])),
        "thumbnail_path": page_data.get("thumbnail_path", ""),
        "hires_path": page_data.get("hires_path", ""),
    }


def _build_region_rows(region_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in region_rows:
        page_regions = json.loads(Path(row["page_region_path"]).read_text(encoding="utf-8"))
        for region in page_regions.get("regions", []):
            region_text = str(region.get("region_text", "")).strip()
            sketch_text = " ".join(
                part
                for part in [
                    str(region.get("region_type", "")).strip(),
                    region_text,
                ]
                if part
            ).strip()
            rows.append(
                {
                    "region_id": region["region_id"],
                    "page_id": region["page_id"],
                    "doc_id": region["doc_id"],
                    "page_num": int(region["page_num"]),
                    "region_type": region["region_type"],
                    "bbox": region["bbox"],
                    "region_text": region_text,
                    "sketch_text": sketch_text,
                    "source_block_ids": region.get("source_block_ids", []),
                }
            )
    return rows


def build_indices(
    layout_manifest: Path,
    output_name: str,
    region_manifest: Path | None,
    *,
    verbose: bool,
) -> dict[str, str]:
    layout_rows = read_jsonl(layout_manifest)
    region_rows = read_jsonl(region_manifest) if region_manifest else []
    index_root = ensure_dir(layout_manifest.parent.parent.parent / "indices" / output_name)

    if verbose:
        print(
            f"[build_indices] start output={output_name} "
            f"pages={len(layout_rows)} region_pages={len(region_rows)}"
        )

    regions_by_page: dict[str, list[dict]] = {}
    for row in region_rows:
        page_regions = json.loads(Path(row["page_region_path"]).read_text(encoding="utf-8"))
        regions_by_page[row["page_id"]] = list(page_regions.get("regions", []))

    page_index_rows = [_build_page_row(row, regions_by_page) for row in layout_rows]
    page_index_rows.sort(key=lambda item: (item["doc_id"], item["page_num"]))

    doc_page_map: dict[str, list[dict]] = {}
    for row in page_index_rows:
        doc_page_map.setdefault(row["doc_id"], []).append(
            {
                "page_id": row["page_id"],
                "page_num": row["page_num"],
                "title": row["title"],
                "ocr_head": row["ocr_head"],
                "layout_tags": row["layout_tags"],
            }
        )

    region_index_rows = _build_region_rows(region_rows) if region_rows else []
    region_index_rows.sort(key=lambda item: (item["doc_id"], item["page_num"], item["region_id"]))

    page_index_path = index_root / "page_index.jsonl"
    region_index_path = index_root / "region_index.jsonl"
    write_jsonl(page_index_path, page_index_rows)
    if region_index_rows:
        write_jsonl(region_index_path, region_index_rows)

    write_json(index_root / "doc_page_map.json", doc_page_map)
    write_json(
        index_root / "run_manifest.json",
        {
            "output_name": output_name,
            "layout_manifest": layout_manifest.as_posix(),
            "region_manifest": region_manifest.as_posix() if region_manifest else None,
            "page_count": len(page_index_rows),
            "region_count": len(region_index_rows),
            "document_count": len(doc_page_map),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if verbose:
        print(
            f"[build_indices] done output={output_name} page_count={len(page_index_rows)} "
            f"region_count={len(region_index_rows)} index_root={index_root.as_posix()}"
        )

    return {
        "index_root": index_root.as_posix(),
        "page_index": page_index_path.as_posix(),
        "region_index": region_index_path.as_posix() if region_index_rows else "",
        "doc_page_map": (index_root / "doc_page_map.json").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build text retrieval indices from parsed layout and regions")
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--region-manifest", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    outputs = build_indices(
        layout_manifest=Path(args.layout_manifest).expanduser().resolve(),
        output_name=args.output_name,
        region_manifest=Path(args.region_manifest).expanduser().resolve() if args.region_manifest else None,
        verbose=args.verbose,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
