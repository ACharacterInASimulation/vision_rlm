from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import fitz

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import (
    discover_pdfs,
    ensure_dir,
    stable_doc_id,
    write_json,
    write_jsonl,
)


def _render_page(page: fitz.Page, output_path: Path, long_side: int) -> tuple[int, int]:
    rect = page.rect
    scale = long_side / max(rect.width, rect.height)
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(output_path.as_posix())
    return pixmap.width, pixmap.height


def render_documents(
    input_path: Path,
    output_name: str,
    limit_docs: int | None,
    thumbnail_long_side: int,
    hires_long_side: int,
) -> dict[str, str]:
    project_paths = build_project_paths()
    render_root = ensure_dir(project_paths.processed_data_root / "rendered" / output_name)
    thumbnails_root = ensure_dir(render_root / "thumbnails")
    hires_root = ensure_dir(render_root / "hires")

    pdf_paths = discover_pdfs(input_path=input_path, limit_docs=limit_docs)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {input_path}")

    document_rows: list[dict] = []
    page_rows: list[dict] = []

    for pdf_path in pdf_paths:
        doc_id = stable_doc_id(pdf_path)
        with fitz.open(pdf_path) as pdf:
            document_rows.append(
                {
                    "doc_id": doc_id,
                    "source_path": pdf_path.as_posix(),
                    "num_pages": pdf.page_count,
                }
            )
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                page_num = page_index + 1
                page_id = f"{doc_id}_p{page_num:04d}"
                thumb_path = thumbnails_root / doc_id / f"{page_id}.png"
                hires_path = hires_root / doc_id / f"{page_id}.png"
                thumb_width, thumb_height = _render_page(
                    page=page,
                    output_path=thumb_path,
                    long_side=thumbnail_long_side,
                )
                hires_width, hires_height = _render_page(
                    page=page,
                    output_path=hires_path,
                    long_side=hires_long_side,
                )
                page_rows.append(
                    {
                        "page_id": page_id,
                        "doc_id": doc_id,
                        "source_path": pdf_path.as_posix(),
                        "page_num": page_num,
                        "original_width": page.rect.width,
                        "original_height": page.rect.height,
                        "thumbnail_path": thumb_path.as_posix(),
                        "thumbnail_width": thumb_width,
                        "thumbnail_height": thumb_height,
                        "hires_path": hires_path.as_posix(),
                        "hires_width": hires_width,
                        "hires_height": hires_height,
                    }
                )

    write_jsonl(render_root / "document_manifest.jsonl", document_rows)
    write_jsonl(render_root / "page_manifest.jsonl", page_rows)
    write_json(
        render_root / "run_manifest.json",
        {
            "output_name": output_name,
            "input_path": input_path.as_posix(),
            "document_count": len(document_rows),
            "page_count": len(page_rows),
            "thumbnail_long_side": thumbnail_long_side,
            "hires_long_side": hires_long_side,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "render_root": render_root.as_posix(),
        "page_manifest": (render_root / "page_manifest.jsonl").as_posix(),
        "document_manifest": (render_root / "document_manifest.jsonl").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render PDF pages for Vision-RLM preprocessing")
    parser.add_argument("--input-path", required=True, help="PDF file or directory containing PDFs")
    parser.add_argument("--output-name", required=True, help="Name for this render run")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--thumbnail-long-side", type=int, default=896)
    parser.add_argument("--hires-long-side", type=int, default=2048)
    args = parser.parse_args()

    outputs = render_documents(
        input_path=Path(args.input_path).expanduser().resolve(),
        output_name=args.output_name,
        limit_docs=args.limit_docs,
        thumbnail_long_side=args.thumbnail_long_side,
        hires_long_side=args.hires_long_side,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
