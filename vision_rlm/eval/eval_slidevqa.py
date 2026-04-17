from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, read_jsonl, write_json, write_jsonl


BUDGET_TO_K = {
    "small": 4,
    "medium": 8,
    "large": 12,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_page_index(page_index_path: Path) -> dict[str, list[dict]]:
    page_rows = read_jsonl(page_index_path)
    rows_by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in page_rows:
        rows_by_doc[row["doc_id"]].append(row)
    for doc_id in rows_by_doc:
        rows_by_doc[doc_id].sort(key=lambda item: int(item["page_num"]))
    return rows_by_doc


def _load_questions(question_manifest_path: Path, limit_questions: int | None) -> list[dict]:
    rows = read_jsonl(question_manifest_path)
    filtered = [row for row in rows if row.get("evidence_pages")]
    if limit_questions is not None:
        filtered = filtered[:limit_questions]
    return filtered


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_page_retrieval_bm25(
    question_manifest_path: Path,
    page_index_path: Path,
    run_name: str,
    budgets: list[str],
    limit_questions: int | None,
    artifact_subdir: str,
    require_gold_page_coverage: bool,
    verbose: bool,
) -> dict[str, str]:
    project_paths = build_project_paths()
    artifact_root = ensure_dir(project_paths.artifacts_root / "E10_page_retrieval_bm25" / artifact_subdir / run_name)

    questions = _load_questions(question_manifest_path, limit_questions)
    pages_by_doc = _load_page_index(page_index_path)

    if verbose:
        print(
            f"[eval_slidevqa] start mode=page_retrieval_bm25 run={run_name} "
            f"questions={len(questions)} docs={len(pages_by_doc)} budgets={budgets}"
        )

    doc_bm25: dict[str, tuple[BM25Okapi, list[dict]]] = {}
    for doc_id, page_rows in pages_by_doc.items():
        corpus_tokens = [_tokenize(str(row.get("sketch_text", ""))) for row in page_rows]
        doc_bm25[doc_id] = (BM25Okapi(corpus_tokens), page_rows)

    ranking_rows: list[dict] = []
    gold_ranks: list[int] = []
    reciprocal_ranks: list[float] = []
    budget_hits: dict[str, list[int]] = {budget: [] for budget in budgets}
    skipped_for_coverage = 0

    total_questions = len(questions)
    for index, question_row in enumerate(questions, start=1):
        doc_id = question_row["doc_id"]
        evidence_pages = sorted({int(page) for page in question_row.get("evidence_pages", [])})
        if not evidence_pages or doc_id not in doc_bm25:
            continue

        bm25, page_rows = doc_bm25[doc_id]
        indexed_page_nums = {int(row["page_num"]) for row in page_rows}
        if require_gold_page_coverage and not set(evidence_pages).issubset(indexed_page_nums):
            skipped_for_coverage += 1
            continue

        query_tokens = _tokenize(str(question_row["question"]))
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(page_rows, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        ranked_page_nums = [int(row["page_num"]) for row, _ in ranked]
        ranked_page_ids = [str(row["page_id"]) for row, _ in ranked]
        ranked_scores = [float(score) for _, score in ranked]
        best_rank = None
        for rank_index, page_num in enumerate(ranked_page_nums, start=1):
            if page_num in evidence_pages:
                best_rank = rank_index
                break

        if best_rank is None:
            gold_ranks.append(len(ranked_page_nums) + 1)
            reciprocal_ranks.append(0.0)
        else:
            gold_ranks.append(best_rank)
            reciprocal_ranks.append(1.0 / best_rank)

        for budget in budgets:
            k = BUDGET_TO_K[budget]
            hit = int(any(page_num in evidence_pages for page_num in ranked_page_nums[:k]))
            budget_hits[budget].append(hit)

        ranking_rows.append(
            {
                "qa_id": question_row["qa_id"],
                "doc_id": doc_id,
                "question": question_row["question"],
                "gold_answer": question_row["answer"],
                "evidence_pages": evidence_pages,
                "best_gold_rank": best_rank,
                "top_page_nums": ranked_page_nums[:12],
                "top_page_ids": ranked_page_ids[:12],
                "top_scores": ranked_scores[:12],
            }
        )

        if verbose and (index == total_questions or index % 25 == 0):
            print(
                f"[eval_slidevqa] processed {index}/{total_questions} questions "
                f"scored={len(ranking_rows)} skipped_for_coverage={skipped_for_coverage}"
            )

    metrics = {
        "run_name": run_name,
        "question_manifest": question_manifest_path.as_posix(),
        "page_index": page_index_path.as_posix(),
        "question_count": len(ranking_rows),
        "questions_seen": len(questions),
        "skipped_for_coverage": skipped_for_coverage,
        "coverage_rate": (len(ranking_rows) / len(questions)) if questions else 0.0,
        "budgets": budgets,
        "mean_reciprocal_rank": _mean(reciprocal_ranks),
        "mean_best_gold_rank": _mean(gold_ranks),
        "hits": {
            budget: {
                "k": BUDGET_TO_K[budget],
                "hit_count": int(sum(values)),
                "recall": _mean(values),
            }
            for budget, values in budget_hits.items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(artifact_root / "metrics.json", metrics)
    write_jsonl(artifact_root / "rankings.jsonl", ranking_rows)
    write_json(
        artifact_root / "config.json",
        {
            "run_name": run_name,
            "budgets": budgets,
            "artifact_root": artifact_root.as_posix(),
            "limit_questions": limit_questions,
            "require_gold_page_coverage": require_gold_page_coverage,
        },
    )
    if verbose:
        print(
            f"[eval_slidevqa] done run={run_name} scored_questions={len(ranking_rows)} "
            f"mrr={metrics['mean_reciprocal_rank']:.4f} metrics_path={(artifact_root / 'metrics.json').as_posix()}"
        )
    return {
        "artifact_root": artifact_root.as_posix(),
        "metrics_path": (artifact_root / "metrics.json").as_posix(),
        "rankings_path": (artifact_root / "rankings.jsonl").as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SlideVQA baselines under explicit budget regimes")
    parser.add_argument("--mode", choices=["page_retrieval_bm25"], required=True)
    parser.add_argument("--question-manifest", required=True)
    parser.add_argument("--page-index", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--budgets", nargs="+", default=["small", "medium"])
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--artifact-subdir", default="debug")
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    invalid_budgets = [budget for budget in args.budgets if budget not in BUDGET_TO_K]
    if invalid_budgets:
        raise ValueError(f"Unsupported budgets: {invalid_budgets}")

    outputs = evaluate_page_retrieval_bm25(
        question_manifest_path=Path(args.question_manifest).expanduser().resolve(),
        page_index_path=Path(args.page_index).expanduser().resolve(),
        run_name=args.run_name,
        budgets=args.budgets,
        limit_questions=args.limit_questions,
        artifact_subdir=args.artifact_subdir,
        require_gold_page_coverage=not args.allow_partial_coverage,
        verbose=args.verbose,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
