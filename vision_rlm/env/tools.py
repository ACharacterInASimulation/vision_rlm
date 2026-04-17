from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any

from rank_bm25 import BM25Okapi

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, read_jsonl

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _estimate_visual_tokens(width: int, height: int) -> int:
    patches = max(1, round((width * height) / float(28 * 28)))
    return min(16384, patches)


def _safe_eval_expression(expr: str) -> float | int:
    node = ast.parse(expr, mode="eval")

    def _eval(item: ast.AST) -> float | int:
        if isinstance(item, ast.Expression):
            return _eval(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return item.value
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = _eval(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            left = _eval(item.left)
            right = _eval(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            if isinstance(item.op, ast.Div):
                return left / right
            if isinstance(item.op, ast.Mod):
                return left % right
            return left**right
        raise ValueError(f"Unsupported expression: {expr}")

    return _eval(node)


@dataclass
class EnvCostTracker:
    visual_tokens: int = 0
    pages_opened: set[str] = field(default_factory=set)
    regions_inspected: set[str] = field(default_factory=set)
    tool_steps: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "visual_tokens": self.visual_tokens,
            "pages_opened": len(self.pages_opened),
            "regions_inspected": len(self.regions_inspected),
            "tool_steps": self.tool_steps,
        }


class DocumentEnvironment:
    def __init__(
        self,
        *,
        page_index_path: Path,
        region_manifest_path: Path | None = None,
        crop_cache_name: str = "default",
    ) -> None:
        self.paths = build_project_paths()
        self.page_rows = read_jsonl(page_index_path)
        self.page_rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.page_rows_by_id: dict[str, dict[str, Any]] = {}
        for row in self.page_rows:
            self.page_rows_by_doc[str(row["doc_id"])].append(row)
            self.page_rows_by_id[str(row["page_id"])] = row
        for rows in self.page_rows_by_doc.values():
            rows.sort(key=lambda item: int(item["page_num"]))

        self.page_bm25_by_doc: dict[str, BM25Okapi] = {}
        for doc_id, rows in self.page_rows_by_doc.items():
            self.page_bm25_by_doc[doc_id] = BM25Okapi([_tokenize(str(row.get("sketch_text", ""))) for row in rows])

        self.page_region_paths: dict[str, Path] = {}
        self.region_rows_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.region_rows_by_id: dict[str, dict[str, Any]] = {}
        self.region_bm25_by_page: dict[str, BM25Okapi] = {}

        if region_manifest_path and region_manifest_path.exists():
            for row in read_jsonl(region_manifest_path):
                page_id = str(row["page_id"])
                self.page_region_paths[page_id] = Path(row["page_region_path"])
                payload = json.loads(Path(row["page_region_path"]).read_text(encoding="utf-8"))
                regions = list(payload.get("regions", []))
                self.region_rows_by_page[page_id] = regions
                for region in regions:
                    self.region_rows_by_id[str(region["region_id"])] = region
                self.region_bm25_by_page[page_id] = BM25Okapi(
                    [_tokenize(f"{region.get('region_type', '')} {region.get('region_text', '')}") for region in regions]
                )

        self.crop_root = ensure_dir(self.paths.cache_root / "region_crops" / crop_cache_name)
        self.page_payload_cache: dict[str, dict[str, Any]] = {}
        self.cost = EnvCostTracker()

    def _load_page_payload(self, page_id: str) -> dict[str, Any]:
        if page_id not in self.page_payload_cache:
            page_row = self.page_rows_by_id[page_id]
            layout_root = self.paths.processed_data_root / "layout"
            candidates = sorted(layout_root.rglob(f"{page_id}.json"))
            if not candidates:
                raise FileNotFoundError(f"Could not locate page payload for {page_id}")
            self.page_payload_cache[page_id] = json.loads(candidates[0].read_text(encoding="utf-8"))
        return self.page_payload_cache[page_id]

    def _crop_region(self, page_id: str, region_id: str) -> str | None:
        if Image is None:
            return None
        page_row = self.page_rows_by_id[page_id]
        region = self.region_rows_by_id[region_id]
        crop_path = self.crop_root / f"{region_id}.png"
        if not crop_path.exists():
            image = Image.open(page_row["hires_path"])
            x0, y0, x1, y1 = region["bbox"]
            image.crop((x0, y0, x1, y1)).save(crop_path)
        return crop_path.as_posix()

    def retrieve_pages(self, *, doc_id: str, query: str, k: int) -> dict[str, Any]:
        rows = self.page_rows_by_doc[doc_id]
        bm25 = self.page_bm25_by_doc[doc_id]
        query_tokens = _tokenize(query)
        scored = sorted(zip(rows, bm25.get_scores(query_tokens)), key=lambda item: float(item[1]), reverse=True)[:k]
        self.cost.tool_steps += 1
        return {
            "doc_id": doc_id,
            "query": query,
            "top_pages": [
                {
                    "page_id": row["page_id"],
                    "page_num": row["page_num"],
                    "score": float(score),
                    "title": row.get("title", ""),
                    "ocr_head": row.get("ocr_head", []),
                    "layout_tags": row.get("layout_tags", []),
                }
                for row, score in scored
            ],
        }

    def open_page(self, *, page_id: str) -> dict[str, Any]:
        page_row = self.page_rows_by_id[page_id]
        payload = self._load_page_payload(page_id)
        self.cost.tool_steps += 1
        self.cost.pages_opened.add(page_id)
        self.cost.visual_tokens += _estimate_visual_tokens(int(page_row.get("num_blocks", 1)) * 64, 1024)
        return {
            "page_id": page_id,
            "page_num": page_row["page_num"],
            "title": page_row.get("title", ""),
            "hires_path": page_row.get("hires_path", ""),
            "ocr_head": page_row.get("ocr_head", []),
            "layout_tags": payload.get("layout_tags", []),
            "num_regions": len(self.region_rows_by_page.get(page_id, [])),
        }

    def rank_regions(self, *, page_id: str, query: str, m: int) -> dict[str, Any]:
        regions = self.region_rows_by_page.get(page_id, [])
        bm25 = self.region_bm25_by_page.get(page_id)
        if not regions or bm25 is None:
            return {"page_id": page_id, "query": query, "top_regions": []}
        scored = sorted(zip(regions, bm25.get_scores(_tokenize(query))), key=lambda item: float(item[1]), reverse=True)[:m]
        self.cost.tool_steps += 1
        return {
            "page_id": page_id,
            "query": query,
            "top_regions": [
                {
                    "region_id": region["region_id"],
                    "bbox": region["bbox"],
                    "region_type": region["region_type"],
                    "region_text": region.get("region_text", ""),
                    "score": float(score),
                }
                for region, score in scored
            ],
        }

    def inspect_region(self, *, page_id: str, region_id: str) -> dict[str, Any]:
        region = self.region_rows_by_id[region_id]
        crop_path = self._crop_region(page_id, region_id)
        self.cost.tool_steps += 1
        self.cost.regions_inspected.add(region_id)
        bbox = region["bbox"]
        self.cost.visual_tokens += _estimate_visual_tokens(max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))
        return {
            "page_id": page_id,
            "region_id": region_id,
            "bbox": bbox,
            "region_type": region["region_type"],
            "region_text": region.get("region_text", ""),
            "crop_path": crop_path,
        }

    def compute(self, *, expression: str) -> dict[str, Any]:
        self.cost.tool_steps += 1
        result = _safe_eval_expression(expression)
        return {"expression": expression, "result": result}

    def write_note(self, *, fact: str, evidence_ref: dict[str, Any], evidence_type: str = "text_fact", confidence: float = 0.8) -> dict[str, Any]:
        self.cost.tool_steps += 1
        note_id = f"note_{stable_note_suffix(fact, evidence_ref)}"
        return {
            "note_id": note_id,
            "fact": fact,
            "evidence_ref": evidence_ref,
            "evidence_type": evidence_type,
            "confidence": confidence,
        }

    def answer(self, *, answer: str, evidence_refs: list[dict[str, Any]], confidence: float = 0.5) -> dict[str, Any]:
        self.cost.tool_steps += 1
        return {
            "answer": answer,
            "evidence_refs": evidence_refs,
            "confidence": confidence,
            "cost": self.cost.as_dict(),
        }

    def abstain(self, *, reason: str) -> dict[str, Any]:
        self.cost.tool_steps += 1
        return {"abstain": True, "reason": reason, "cost": self.cost.as_dict()}

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action_type", "")).upper()
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if action_type == "RETRIEVE_PAGES":
            return self.retrieve_pages(
                doc_id=str(payload["doc_id"]),
                query=str(payload["query"]),
                k=int(payload.get("k", 8)),
            )
        if action_type == "OPEN_PAGE":
            return self.open_page(page_id=str(payload["page_id"]))
        if action_type == "RANK_REGIONS":
            return self.rank_regions(
                page_id=str(payload["page_id"]),
                query=str(payload["query"]),
                m=int(payload.get("m", 8)),
            )
        if action_type == "INSPECT_REGION":
            return self.inspect_region(page_id=str(payload["page_id"]), region_id=str(payload["region_id"]))
        if action_type == "COMPUTE":
            return self.compute(expression=str(payload["expression"]))
        if action_type == "WRITE_NOTE":
            return self.write_note(
                fact=str(payload["fact"]),
                evidence_ref=dict(payload.get("evidence_ref", {})),
                evidence_type=str(payload.get("evidence_type", "text_fact")),
                confidence=float(payload.get("confidence", 0.8)),
            )
        if action_type == "ANSWER":
            return self.answer(
                answer=str(payload["answer"]),
                evidence_refs=list(payload.get("evidence_refs", [])),
                confidence=float(payload.get("confidence", 0.5)),
            )
        if action_type == "ABSTAIN":
            return self.abstain(reason=str(payload.get("reason", "")))
        raise ValueError(f"Unsupported action_type: {action_type}")


def stable_note_suffix(fact: str, evidence_ref: dict[str, Any]) -> str:
    page_id = evidence_ref.get("page_id", "")
    region_id = evidence_ref.get("region_id", "")
    basis = f"{fact}::{page_id}::{region_id}"
    return re.sub(r"[^a-z0-9]+", "", basis.lower())[:16] or "note"
