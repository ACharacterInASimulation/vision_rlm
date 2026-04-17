from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, stable_string_id, write_json, write_jsonl


PAGE_COLUMNS = [f"page_{index}" for index in range(1, 21)]


def _resize_and_save(image: Image.Image, output_path: Path, long_side: int) -> tuple[int, int]:
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) == 0:
        raise ValueError("Image has invalid dimensions")
    scale = long_side / max(width, height)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path, format="PNG")
    return resized.size


def extract_slidevqa_split(
    dataset_root: Path,
    split: str,
    output_name: str,
    limit_questions: int | None,
    limit_decks: int | None,
    thumbnail_long_side: int,
    hires_long_side: int,
) -> dict[str, str]:
    project_paths = build_project_paths()
    render_root = ensure_dir(project_paths.processed_data_root / "rendered" / output_name)
    thumbnails_root = ensure_dir(render_root / "thumbnails")
    hires_root = ensure_dir(render_root / "hires")
    questions_root = ensure_dir(render_root / "questions")

    parquet_paths = sorted((dataset_root / "data").glob(f"{split}-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for split '{split}' under {dataset_root}")

    document_rows: dict[str, dict] = {}
    page_rows: dict[str, dict] = {}
    question_rows: list[dict] = []
    seen_decks: set[str] = set()
    questions_seen = 0

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            if limit_questions is not None and questions_seen >= limit_questions:
                break

            deck_name = str(row["deck_name"])
            deck_url = str(row.get("deck_url", ""))
            doc_id = stable_string_id(deck_name, namespace=deck_url)
            is_new_deck = doc_id not in seen_decks
            if is_new_deck and limit_decks is not None and len(seen_decks) >= limit_decks:
                continue
            seen_decks.add(doc_id)

            page_count = 0
            for page_index, page_column in enumerate(PAGE_COLUMNS, start=1):
                page_payload = row.get(page_column)
                if not page_payload or not page_payload.get("bytes"):
                    continue
                page_count = page_index
                page_id = f"{doc_id}_p{page_index:04d}"
                if page_id in page_rows:
                    continue
                image = Image.open(BytesIO(page_payload["bytes"]))
                original_width, original_height = image.size
                thumb_path = thumbnails_root / doc_id / f"{page_id}.png"
                hires_path = hires_root / doc_id / f"{page_id}.png"
                thumb_width, thumb_height = _resize_and_save(image, thumb_path, thumbnail_long_side)
                hires_width, hires_height = _resize_and_save(image, hires_path, hires_long_side)
                page_rows[page_id] = {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "source_path": parquet_path.as_posix(),
                    "source_page_path": str(page_payload.get("path", "")),
                    "page_num": page_index,
                    "original_width": original_width,
                    "original_height": original_height,
                    "thumbnail_path": thumb_path.as_posix(),
                    "thumbnail_width": thumb_width,
                    "thumbnail_height": thumb_height,
                    "hires_path": hires_path.as_posix(),
                    "hires_width": hires_width,
                    "hires_height": hires_height,
                }

            document_rows.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "split": split,
                    "deck_name": deck_name,
                    "deck_url": deck_url,
                    "source_path": parquet_path.as_posix(),
                    "num_pages": page_count,
                },
            )

            question_row = {
                "split": split,
                "qa_id": int(row["qa_id"]),
                "doc_id": doc_id,
                "deck_name": deck_name,
                "deck_url": deck_url,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "arithmetic_expression": str(row.get("arithmetic_expression")),
                "evidence_pages": list(row.get("evidence_pages") or []),
            }
            question_rows.append(question_row)
            questions_seen += 1

            question_path = questions_root / f"qa_{int(row['qa_id']):07d}.json"
            write_json(question_path, question_row)

        if limit_questions is not None and questions_seen >= limit_questions:
            break

    write_jsonl(render_root / "document_manifest.jsonl", document_rows.values())
    write_jsonl(render_root / "page_manifest.jsonl", page_rows.values())
    write_jsonl(render_root / "question_manifest.jsonl", question_rows)
    write_json(
        render_root / "run_manifest.json",
        {
            "output_name": output_name,
            "dataset_root": dataset_root.as_posix(),
            "split": split,
            "question_count": len(question_rows),
            "document_count": len(document_rows),
            "page_count": len(page_rows),
            "thumbnail_long_side": thumbnail_long_side,
            "hires_long_side": hires_long_side,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "render_root": render_root.as_posix(),
        "document_manifest": (render_root / "document_manifest.jsonl").as_posix(),
        "page_manifest": (render_root / "page_manifest.jsonl").as_posix(),
        "question_manifest": (render_root / "question_manifest.jsonl").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SlideVQA page images from parquet shards")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--limit-decks", type=int, default=None)
    parser.add_argument("--thumbnail-long-side", type=int, default=896)
    parser.add_argument("--hires-long-side", type=int, default=2048)
    args = parser.parse_args()

    outputs = extract_slidevqa_split(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        split=args.split,
        output_name=args.output_name,
        limit_questions=args.limit_questions,
        limit_decks=args.limit_decks,
        thumbnail_long_side=args.thumbnail_long_side,
        hires_long_side=args.hires_long_side,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
