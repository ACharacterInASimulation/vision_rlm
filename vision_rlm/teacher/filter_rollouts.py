from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Iterable

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, read_jsonl, stable_string_id, write_json, write_jsonl


def _coalesce(mapping: dict, *keys: str, default: object = None) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.%/$+-]+", " ", text)
    return text.strip()


def _answer_score(prediction: str, gold: str) -> float:
    pred = _normalize_text(prediction)
    ref = _normalize_text(gold)
    if not pred or not ref:
        return 0.0
    if pred == ref:
        return 1.0
    return SequenceMatcher(None, pred, ref).ratio()


def _normalize_action(action: object) -> dict[str, object]:
    if isinstance(action, dict):
        if "action_type" in action:
            payload = action.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"value": payload}
            return {"action_type": str(action["action_type"]), "payload": payload}
        if "action" in action and isinstance(action["action"], str):
            payload = action.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"value": payload}
            return {"action_type": action["action"], "payload": payload}
        if len(action) == 1:
            action_type, payload = next(iter(action.items()))
            return {
                "action_type": str(action_type),
                "payload": payload if isinstance(payload, dict) else {"value": payload},
            }
    raise ValueError(f"Unsupported action format: {action!r}")


def _question_key(row: dict) -> str:
    if row.get("qa_id") is not None:
        return f"qa_id:{row['qa_id']}"
    doc_id = _coalesce(row, "doc_id", "document_id", default="")
    question = _normalize_text(_coalesce(row, "question", default=""))
    return f"{doc_id}::{question}"


def _build_question_lookup(question_manifest: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in read_jsonl(question_manifest):
        lookup[_question_key(row)] = row
    return lookup


def _extract_steps(trace: dict) -> list[dict]:
    steps = _as_list(_coalesce(trace, "steps", "trajectory", "events", "turns", default=[]))
    normalized_steps: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = _coalesce(step, "action", default=None)
        if action is None and "action_type" in step:
            action = {"action_type": step["action_type"], "payload": step.get("payload", {})}
        if action is None:
            continue
        try:
            normalized_action = _normalize_action(action)
        except ValueError:
            continue
        normalized_steps.append(
            {
                "action": normalized_action,
                "observation": step.get("observation"),
                "step_index": step.get("step_index", len(normalized_steps)),
            }
        )
    return normalized_steps


def _extract_final_answer(trace: dict, steps: list[dict]) -> str:
    for key in ("predicted_answer", "final_answer", "answer"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = trace.get("result")
    if isinstance(result, dict):
        for key in ("answer", "final_answer"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for step in reversed(steps):
        action = step["action"]
        if action["action_type"] == "ANSWER":
            value = _coalesce(action["payload"], "answer", default="")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_cited_pages(trace: dict, steps: list[dict]) -> list[int]:
    refs: list[object] = []
    for candidate in (
        trace.get("evidence_refs"),
        trace.get("citations"),
        _coalesce(trace.get("result", {}) if isinstance(trace.get("result"), dict) else {}, "evidence_refs", "citations", default=None),
    ):
        if candidate:
            refs.extend(_as_list(candidate))
    for step in reversed(steps):
        action = step["action"]
        if action["action_type"] == "ANSWER":
            refs.extend(_as_list(_coalesce(action["payload"], "evidence_refs", "citations", default=[])))
            break

    page_numbers: list[int] = []
    for ref in refs:
        if isinstance(ref, dict):
            page_value = _coalesce(ref, "page_num", "page", "evidence_page")
            page_id = ref.get("page_id")
            if page_value is not None:
                try:
                    page_numbers.append(int(page_value))
                    continue
                except (TypeError, ValueError):
                    pass
            if isinstance(page_id, str):
                match = re.search(r"_p(\d+)$", page_id)
                if match:
                    page_numbers.append(int(match.group(1)))
        elif isinstance(ref, int):
            page_numbers.append(ref)
    return sorted(set(page_numbers))


def _compute_cost(trace: dict, steps: list[dict], defaults: dict[str, int]) -> dict[str, float]:
    unique_pages: set[str] = set()
    unique_regions: set[tuple[str | None, str | None]] = set()
    visual_tokens = float(_coalesce(trace, "visual_tokens", default=0) or 0)

    for step in steps:
        action = step["action"]
        payload = action["payload"]
        if action["action_type"] == "OPEN_PAGE":
            page_id = payload.get("page_id")
            if page_id is not None:
                unique_pages.add(str(page_id))
        if action["action_type"] == "INSPECT_REGION":
            unique_regions.add((payload.get("page_id"), payload.get("region_id")))
        observation = step.get("observation")
        if isinstance(observation, dict):
            visual_tokens += float(_coalesce(observation, "visual_tokens", default=0) or 0)

    pages_opened = len(unique_pages)
    regions_inspected = len(unique_regions)
    tool_steps = len(steps)

    max_visual_tokens = max(int(_coalesce(trace, "max_visual_tokens", default=defaults["visual_tokens"]) or defaults["visual_tokens"]), 1)
    max_pages = max(int(_coalesce(trace, "max_pages_opened", default=defaults["pages_opened"]) or defaults["pages_opened"]), 1)
    max_regions = max(int(_coalesce(trace, "max_regions_inspected", default=defaults["regions_inspected"]) or defaults["regions_inspected"]), 1)
    max_steps = max(int(_coalesce(trace, "max_tool_steps", default=defaults["tool_steps"]) or defaults["tool_steps"]), 1)

    normalized_cost = (
        (visual_tokens / max_visual_tokens)
        + (pages_opened / max_pages)
        + (regions_inspected / max_regions)
        + (tool_steps / max_steps)
    ) / 4.0
    return {
        "visual_tokens": visual_tokens,
        "pages_opened": float(pages_opened),
        "regions_inspected": float(regions_inspected),
        "tool_steps": float(tool_steps),
        "normalized_cost": normalized_cost,
        "limits": {
            "visual_tokens": max_visual_tokens,
            "pages_opened": max_pages,
            "regions_inspected": max_regions,
            "tool_steps": max_steps,
        },
    }


def _trace_signature(steps: list[dict]) -> str:
    compact = [
        {
            "action_type": step["action"]["action_type"],
            "payload": step["action"]["payload"],
        }
        for step in steps
    ]
    return json.dumps(compact, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _evaluate_trace(
    trace: dict,
    question_lookup: dict[str, dict],
    *,
    min_answer_score: float,
    lambda_grounding: float,
    lambda_cost: float,
    defaults: dict[str, int],
) -> dict[str, object]:
    steps = _extract_steps(trace)
    key = _question_key(trace)
    question_row = question_lookup.get(key)

    rejection_reasons: list[str] = []
    executability = bool(steps) and not bool(_coalesce(trace, "execution_error", "runtime_error", default=False))
    if not executability:
        rejection_reasons.append("not_executable")

    final_answer = _extract_final_answer(trace, steps)
    gold_answer = str(_coalesce(question_row or {}, "answer", default=""))
    answer_score = _answer_score(final_answer, gold_answer)
    if answer_score < min_answer_score:
        rejection_reasons.append("answer_mismatch")

    cited_pages = _extract_cited_pages(trace, steps)
    gold_pages = [int(page) for page in _as_list(_coalesce(question_row or {}, "evidence_pages", default=[])) if str(page).strip()]
    grounding_score = 0.0
    if gold_pages:
        grounding_score = 1.0 if set(cited_pages) & set(gold_pages) else 0.0
        if grounding_score <= 0.0:
            rejection_reasons.append("grounding_mismatch")

    cost = _compute_cost(trace, steps, defaults=defaults)
    limits = cost["limits"]
    if cost["tool_steps"] > limits["tool_steps"]:
        rejection_reasons.append("too_many_steps")
    if cost["pages_opened"] > limits["pages_opened"]:
        rejection_reasons.append("too_many_pages")
    if cost["regions_inspected"] > limits["regions_inspected"]:
        rejection_reasons.append("too_many_regions")
    if cost["visual_tokens"] > limits["visual_tokens"]:
        rejection_reasons.append("too_many_visual_tokens")

    utility = answer_score + lambda_grounding * grounding_score - lambda_cost * cost["normalized_cost"]
    accepted = not rejection_reasons

    return {
        "trace": trace,
        "trace_id": str(_coalesce(trace, "trace_id", "rollout_id", default=stable_string_id(key, namespace="trace"))),
        "question_key": key,
        "question_row": question_row,
        "steps": steps,
        "final_answer": final_answer,
        "gold_answer": gold_answer,
        "answer_score": answer_score,
        "grounding_score": grounding_score,
        "cited_pages": cited_pages,
        "gold_pages": gold_pages,
        "cost": cost,
        "utility": utility,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "signature": _trace_signature(steps),
    }


def _select_pareto_like_traces(
    evaluations: list[dict[str, object]],
    *,
    keep_cheaper_alternative: bool,
    cheaper_delta: float,
) -> list[dict[str, object]]:
    by_question: dict[str, list[dict[str, object]]] = defaultdict(list)
    for evaluation in evaluations:
        if evaluation["accepted"]:
            by_question[evaluation["question_key"]].append(evaluation)

    selected: list[dict[str, object]] = []
    for candidates in by_question.values():
        candidates = sorted(candidates, key=lambda item: (-float(item["utility"]), float(item["cost"]["normalized_cost"])))
        best = candidates[0]
        selected.append(best)
        if not keep_cheaper_alternative:
            continue
        for candidate in candidates[1:]:
            if candidate["signature"] == best["signature"]:
                continue
            if float(candidate["cost"]["normalized_cost"]) <= float(best["cost"]["normalized_cost"]) - cheaper_delta:
                selected.append(candidate)
                break
    return selected


def filter_rollouts(
    raw_traces_path: Path,
    question_manifest: Path,
    accepted_output_path: Path,
    *,
    min_answer_score: float,
    lambda_grounding: float,
    lambda_cost: float,
    max_visual_tokens: int,
    max_pages_opened: int,
    max_regions_inspected: int,
    max_tool_steps: int,
    keep_cheaper_alternative: bool,
    cheaper_delta: float,
) -> dict[str, object]:
    traces = read_jsonl(raw_traces_path)
    question_lookup = _build_question_lookup(question_manifest)
    defaults = {
        "visual_tokens": max_visual_tokens,
        "pages_opened": max_pages_opened,
        "regions_inspected": max_regions_inspected,
        "tool_steps": max_tool_steps,
    }

    evaluations = [
        _evaluate_trace(
            trace,
            question_lookup,
            min_answer_score=min_answer_score,
            lambda_grounding=lambda_grounding,
            lambda_cost=lambda_cost,
            defaults=defaults,
        )
        for trace in traces
    ]
    selected = _select_pareto_like_traces(
        evaluations,
        keep_cheaper_alternative=keep_cheaper_alternative,
        cheaper_delta=cheaper_delta,
    )

    ensure_dir(accepted_output_path.parent)
    accepted_rows: list[dict[str, object]] = []
    for evaluation in selected:
        trace = dict(evaluation["trace"])
        trace["filter_metrics"] = {
            "trace_id": evaluation["trace_id"],
            "question_key": evaluation["question_key"],
            "answer_score": evaluation["answer_score"],
            "grounding_score": evaluation["grounding_score"],
            "utility": evaluation["utility"],
            "cost": evaluation["cost"],
            "final_answer": evaluation["final_answer"],
            "gold_answer": evaluation["gold_answer"],
            "cited_pages": evaluation["cited_pages"],
            "gold_pages": evaluation["gold_pages"],
        }
        accepted_rows.append(trace)
    write_jsonl(accepted_output_path, accepted_rows)

    rejection_rows = [
        {
            "trace_id": evaluation["trace_id"],
            "question_key": evaluation["question_key"],
            "rejection_reasons": evaluation["rejection_reasons"],
            "answer_score": evaluation["answer_score"],
            "grounding_score": evaluation["grounding_score"],
            "utility": evaluation["utility"],
            "cost": evaluation["cost"],
        }
        for evaluation in evaluations
        if not evaluation["accepted"]
    ]
    rejection_path = accepted_output_path.with_name(accepted_output_path.stem + "_rejections.jsonl")
    write_jsonl(rejection_path, rejection_rows)

    reason_counts: Counter[str] = Counter()
    for row in rejection_rows:
        reason_counts.update(row["rejection_reasons"])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_traces_path": raw_traces_path.as_posix(),
        "question_manifest": question_manifest.as_posix(),
        "accepted_output_path": accepted_output_path.as_posix(),
        "rejection_output_path": rejection_path.as_posix(),
        "trace_count": len(traces),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejection_rows),
        "accepted_question_count": len({row["filter_metrics"]["question_key"] for row in accepted_rows}),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "min_answer_score": min_answer_score,
        "lambda_grounding": lambda_grounding,
        "lambda_cost": lambda_cost,
        "defaults": defaults,
    }
    write_json(accepted_output_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    paths = build_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-traces", type=Path, required=True, help="Raw rollout JSONL path.")
    parser.add_argument("--question-manifest", type=Path, required=True, help="Question manifest JSONL with gold answers.")
    parser.add_argument(
        "--accepted-output-path",
        type=Path,
        default=paths.large_root / "data" / "accepted_traces.jsonl",
        help="Accepted trace JSONL output path.",
    )
    parser.add_argument("--min-answer-score", type=float, default=0.9, help="Minimum normalized answer score to keep.")
    parser.add_argument("--lambda-grounding", type=float, default=0.30, help="Grounding weight in utility.")
    parser.add_argument("--lambda-cost", type=float, default=0.20, help="Cost penalty weight in utility.")
    parser.add_argument("--max-visual-tokens", type=int, default=1536)
    parser.add_argument("--max-pages-opened", type=int, default=8)
    parser.add_argument("--max-regions-inspected", type=int, default=16)
    parser.add_argument("--max-tool-steps", type=int, default=10)
    parser.add_argument(
        "--keep-cheaper-alternative",
        action="store_true",
        help="Keep up to one additional cheaper accepted trace per question when the sequence differs.",
    )
    parser.add_argument(
        "--cheaper-delta",
        type=float,
        default=0.10,
        help="Required normalized-cost margin for the optional cheaper alternative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = filter_rollouts(
        raw_traces_path=args.raw_traces,
        question_manifest=args.question_manifest,
        accepted_output_path=args.accepted_output_path,
        min_answer_score=args.min_answer_score,
        lambda_grounding=args.lambda_grounding,
        lambda_cost=args.lambda_cost,
        max_visual_tokens=args.max_visual_tokens,
        max_pages_opened=args.max_pages_opened,
        max_regions_inspected=args.max_regions_inspected,
        max_tool_steps=args.max_tool_steps,
        keep_cheaper_alternative=args.keep_cheaper_alternative,
        cheaper_delta=args.cheaper_delta,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
