from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path

import fitz

from vision_rlm.preprocess.common import (
    ensure_dir,
    read_jsonl,
    scale_bbox,
    token_count,
    write_json,
    write_jsonl,
)

_PADDLE_OCR_CACHE: dict[tuple[object, ...], object] = {}


def _get_paddleocr(
    *,
    device: str,
    enable_mkldnn: bool,
    cpu_threads: int | None,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
):
    cache_key = (
        device,
        enable_mkldnn,
        cpu_threads,
        use_doc_orientation_classify,
        use_doc_unwarping,
    )
    if cache_key not in _PADDLE_OCR_CACHE:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        signature = inspect.signature(PaddleOCR)
        if "device" in signature.parameters:
            kwargs: dict[str, object] = {
                "use_textline_orientation": False,
                "device": device,
                "enable_mkldnn": enable_mkldnn,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
            }
            if cpu_threads is not None:
                kwargs["cpu_threads"] = cpu_threads
            _PADDLE_OCR_CACHE[cache_key] = PaddleOCR(
                lang="en",
                **kwargs,
            )
        else:
            # PaddleOCR 2.x uses use_gpu/use_angle_cls and ocr() rather than predict().
            _PADDLE_OCR_CACHE[cache_key] = PaddleOCR(
                lang="en",
                use_gpu=device.startswith("gpu"),
                use_angle_cls=use_doc_orientation_classify,
                show_log=False,
            )
    return _PADDLE_OCR_CACHE[cache_key]


def _blocks_from_predict_result(page_row: dict, page_result: dict) -> list[dict]:
    blocks: list[dict] = []
    rec_texts = list(page_result.get("rec_texts", []))
    rec_boxes = list(page_result.get("rec_boxes", []))

    for reading_order, (text, box) in enumerate(zip(rec_texts, rec_boxes)):
        text = str(text).strip()
        if not text:
            continue
        if hasattr(box, "tolist"):
            box = box.tolist()
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        blocks.append(
            {
                "block_id": f'{page_row["page_id"]}_b{reading_order:04d}',
                "block_type": "ocr_line",
                "bbox": [x0, y0, x1, y1],
                "reading_order": reading_order,
                "block_text": text,
                "ocr_tokens": token_count(text),
            }
        )
    return blocks


def _blocks_from_legacy_ocr_result(page_row: dict, results: list) -> list[dict]:
    blocks: list[dict] = []
    if not results:
        return blocks

    line_results = results[0] if isinstance(results, list) and results else []
    if line_results is None:
        return blocks
    for reading_order, line in enumerate(line_results):
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box_points, rec = line[0], line[1]
        if not isinstance(box_points, (list, tuple)) or len(box_points) < 4:
            continue
        text = ""
        if isinstance(rec, (list, tuple)) and rec:
            text = str(rec[0]).strip()
        else:
            text = str(rec).strip()
        if not text:
            continue
        xs = [float(point[0]) for point in box_points if len(point) >= 2]
        ys = [float(point[1]) for point in box_points if len(point) >= 2]
        if not xs or not ys:
            continue
        blocks.append(
            {
                "block_id": f'{page_row["page_id"]}_b{reading_order:04d}',
                "block_type": "ocr_line",
                "bbox": [
                    int(round(min(xs))),
                    int(round(min(ys))),
                    int(round(max(xs))),
                    int(round(max(ys))),
                ],
                "reading_order": reading_order,
                "block_text": text,
                "ocr_tokens": token_count(text),
            }
        )
    return blocks


def _extract_pdf_blocks(page: fitz.Page, page_row: dict) -> list[dict]:
    blocks: list[dict] = []
    page_dict = page.get_text("dict")
    block_index = 0
    reading_order = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = scale_bbox(
                bbox=[float(value) for value in line.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                source_width=float(page_row["original_width"]),
                source_height=float(page_row["original_height"]),
                target_width=int(page_row["hires_width"]),
                target_height=int(page_row["hires_height"]),
            )
            blocks.append(
                {
                    "block_id": f'{page_row["page_id"]}_b{block_index:04d}',
                    "block_type": "text_block",
                    "bbox": bbox,
                    "reading_order": reading_order,
                    "block_text": text,
                    "ocr_tokens": token_count(text),
                }
            )
            block_index += 1
            reading_order += 1
    return blocks


def _extract_paddleocr_blocks(
    page_row: dict,
    *,
    device: str,
    enable_mkldnn: bool,
    cpu_threads: int | None,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    text_det_limit_side_len: int | None,
) -> list[dict]:
    ocr = _get_paddleocr(
        device=device,
        enable_mkldnn=enable_mkldnn,
        cpu_threads=cpu_threads,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
    )
    predict_kwargs: dict[str, object] = {
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
    }
    if text_det_limit_side_len is not None:
        predict_kwargs["text_det_limit_side_len"] = text_det_limit_side_len
    if hasattr(ocr, "predict"):
        results = ocr.predict(page_row["hires_path"], **predict_kwargs)
        if not results:
            return []
        return _blocks_from_predict_result(page_row, results[0])

    results = ocr.ocr(page_row["hires_path"], cls=use_doc_orientation_classify)
    if results is None:
        return []
    return _blocks_from_legacy_ocr_result(page_row, results)


def parse_layout(
    render_manifest: Path,
    output_name: str,
    parser_name: str,
    limit_pages: int | None,
    *,
    shard_index: int,
    num_shards: int,
    ocr_device: str,
    cpu_threads: int | None,
    enable_mkldnn: bool,
    disable_doc_preprocess: bool,
    text_det_limit_side_len: int | None,
) -> dict[str, str]:
    page_rows = read_jsonl(render_manifest)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    if num_shards > 1:
        page_rows = [row for idx, row in enumerate(page_rows) if idx % num_shards == shard_index]
    if limit_pages is not None:
        page_rows = page_rows[:limit_pages]
    layout_root = ensure_dir(render_manifest.parent.parent.parent / "layout" / output_name)
    page_json_root = ensure_dir(layout_root / "pages")

    page_rows_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in page_rows:
        page_rows_by_source[row["source_path"]].append(row)

    layout_rows: list[dict] = []
    parser_counts: dict[str, int] = defaultdict(int)

    use_doc_preprocess = not disable_doc_preprocess
    for source_path, rows in page_rows_by_source.items():
        is_pdf_source = source_path.lower().endswith(".pdf")
        if is_pdf_source:
            with fitz.open(source_path) as pdf:
                for row in rows:
                    blocks: list[dict] = []
                    parser_used = parser_name
                    if parser_name in {"auto", "pdf_text"}:
                        blocks = _extract_pdf_blocks(pdf.load_page(int(row["page_num"]) - 1), row)
                        parser_used = "pdf_text"
                    if parser_name == "paddleocr" or (parser_name == "auto" and not blocks):
                        blocks = _extract_paddleocr_blocks(
                            row,
                            device=ocr_device,
                            enable_mkldnn=enable_mkldnn,
                            cpu_threads=cpu_threads,
                            use_doc_orientation_classify=use_doc_preprocess,
                            use_doc_unwarping=use_doc_preprocess,
                            text_det_limit_side_len=text_det_limit_side_len,
                        )
                        parser_used = "paddleocr"

                    page_payload = {
                        "page_id": row["page_id"],
                        "doc_id": row["doc_id"],
                        "source_path": row["source_path"],
                        "page_num": row["page_num"],
                        "hires_path": row["hires_path"],
                        "hires_width": row["hires_width"],
                        "hires_height": row["hires_height"],
                        "parser_used": parser_used,
                        "blocks": blocks,
                        "layout_tags": sorted({block["block_type"] for block in blocks}),
                    }
                    page_json_path = page_json_root / f'{row["page_id"]}.json'
                    write_json(page_json_path, page_payload)
                    layout_rows.append(
                        {
                            "page_id": row["page_id"],
                            "doc_id": row["doc_id"],
                            "page_num": row["page_num"],
                            "parser_used": parser_used,
                            "num_blocks": len(blocks),
                            "page_json_path": page_json_path.as_posix(),
                        }
                    )
                    parser_counts[parser_used] += 1
        else:
            for row in rows:
                parser_used = "paddleocr" if parser_name in {"auto", "pdf_text", "paddleocr"} else parser_name
                blocks = _extract_paddleocr_blocks(
                    row,
                    device=ocr_device,
                    enable_mkldnn=enable_mkldnn,
                    cpu_threads=cpu_threads,
                    use_doc_orientation_classify=use_doc_preprocess,
                    use_doc_unwarping=use_doc_preprocess,
                    text_det_limit_side_len=text_det_limit_side_len,
                )
                page_payload = {
                    "page_id": row["page_id"],
                    "doc_id": row["doc_id"],
                    "source_path": row["source_path"],
                    "page_num": row["page_num"],
                    "hires_path": row["hires_path"],
                    "hires_width": row["hires_width"],
                    "hires_height": row["hires_height"],
                    "parser_used": parser_used,
                    "blocks": blocks,
                    "layout_tags": sorted({block["block_type"] for block in blocks}),
                }
                page_json_path = page_json_root / f'{row["page_id"]}.json'
                write_json(page_json_path, page_payload)
                layout_rows.append(
                    {
                        "page_id": row["page_id"],
                        "doc_id": row["doc_id"],
                        "page_num": row["page_num"],
                        "parser_used": parser_used,
                        "num_blocks": len(blocks),
                        "page_json_path": page_json_path.as_posix(),
                    }
                )
                parser_counts[parser_used] += 1

    write_jsonl(layout_root / "layout_manifest.jsonl", layout_rows)
    write_json(
        layout_root / "run_manifest.json",
        {
            "output_name": output_name,
            "input_page_manifest": render_manifest.as_posix(),
            "page_count": len(layout_rows),
            "parser_name": parser_name,
            "parser_counts": dict(parser_counts),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "ocr_device": ocr_device,
            "cpu_threads": cpu_threads,
            "enable_mkldnn": enable_mkldnn,
            "disable_doc_preprocess": disable_doc_preprocess,
            "text_det_limit_side_len": text_det_limit_side_len,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "layout_root": layout_root.as_posix(),
        "layout_manifest": (layout_root / "layout_manifest.jsonl").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse OCR and layout information for rendered pages")
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--parser", choices=["auto", "pdf_text", "paddleocr"], default="auto")
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--disable-doc-preprocess", action="store_true")
    parser.add_argument("--text-det-limit-side-len", type=int, default=1024)
    args = parser.parse_args()

    outputs = parse_layout(
        render_manifest=Path(args.render_manifest).expanduser().resolve(),
        output_name=args.output_name,
        parser_name=args.parser,
        limit_pages=args.limit_pages,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        ocr_device=args.ocr_device,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.enable_mkldnn,
        disable_doc_preprocess=args.disable_doc_preprocess,
        text_det_limit_side_len=args.text_det_limit_side_len,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
