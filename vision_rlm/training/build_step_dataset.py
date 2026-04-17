from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, read_jsonl, stable_string_id, write_json, write_jsonl


DEFAULT_SYSTEM_PROMPT = (
    "You are Vision-RLM. Given the current planner state for long-document question answering, "
    "emit exactly one valid next JSON action using the available tool vocabulary."
)


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coalesce(mapping: dict, *keys: str, default: object = None) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalize_action(action: object) -> dict[str, object]:
    if isinstance(action, dict):
        if "action_type" in action:
            payload = action.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"value": payload}
            return {
                "action_type": str(action["action_type"]),
                "payload": payload,
            }
        if "action" in action and isinstance(action["action"], str):
            payload = action.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"value": payload}
            return {
                "action_type": action["action"],
                "payload": payload,
            }
        if len(action) == 1:
            action_type, payload = next(iter(action.items()))
            if isinstance(payload, dict):
                return {
                    "action_type": str(action_type),
                    "payload": payload,
                }
    raise ValueError(f"Unsupported action format: {action!r}")


def _normalize_budget(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _extract_latest_image_path(observation: object) -> str | None:
    if not isinstance(observation, dict):
        return None
    for key in ("image_path", "crop_path", "page_image_path", "thumbnail_path", "hires_path"):
        path = observation.get(key)
        if isinstance(path, str) and path:
            return path
    return None


def _summarize_history_entry(entry: object) -> dict[str, object]:
    if not isinstance(entry, dict):
        return {"value": entry}
    summary: dict[str, object] = {}
    for key in ("action", "action_type", "payload", "observation", "step_index"):
        if key in entry:
            summary[key] = entry[key]
    return summary or entry


def _state_to_user_text(state: dict[str, object]) -> str:
    sections = [
        f"Question:\n{state['question']}",
        "Remaining budget:\n" + _compact_json(state.get("remaining_budget", {})),
        "Page sketches:\n" + _compact_json(state.get("page_sketches", [])),
        "Candidate pages or regions:\n" + _compact_json(state.get("page_candidates", [])),
        "Memory notes:\n" + _compact_json(state.get("memory", [])),
        "Latest observation:\n" + _compact_json(state.get("latest_observation")),
        "Recent history:\n" + _compact_json(state.get("recent_history", [])),
        (
            "Available actions:\n"
            '["RETRIEVE_PAGES","OPEN_PAGE","RANK_REGIONS","INSPECT_REGION","COMPUTE","WRITE_NOTE","ANSWER","ABSTAIN"]'
        ),
        "Return only the next JSON action.",
    ]
    return "\n\n".join(sections)


def _derive_stateful_examples(trace: dict, history_window: int) -> list[dict[str, object]]:
    steps = _as_list(_coalesce(trace, "steps", "trajectory", default=[]))
    if not steps:
        return []

    trace_id = str(_coalesce(trace, "trace_id", "rollout_id", "example_id", default=stable_string_id(trace.get("question", "trace"))))
    question = str(_coalesce(trace, "question", default="")).strip()
    if not question:
        return []

    doc_id = _coalesce(trace, "doc_id", "document_id")
    page_sketches = _as_list(_coalesce(trace, "page_sketches", "initial_page_sketches", default=[]))
    static_budget = _normalize_budget(_coalesce(trace, "remaining_budget", "budget", default={}))
    memory = list(_as_list(_coalesce(trace, "memory", "initial_memory", default=[])))
    page_candidates = list(_as_list(_coalesce(trace, "page_candidates", "candidate_set", default=[])))
    latest_observation = _coalesce(trace, "latest_observation")
    recent_history: list[dict[str, object]] = []

    examples: list[dict[str, object]] = []
    for step_index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        state = raw_step.get("state")
        action = raw_step.get("action")
        observation = raw_step.get("observation")

        if isinstance(state, dict):
            step_memory = list(_as_list(_coalesce(state, "memory", default=memory)))
            step_budget = _normalize_budget(_coalesce(state, "remaining_budget", "budget", default=static_budget))
            step_candidates = list(_as_list(_coalesce(state, "page_candidates", "candidate_set", default=page_candidates)))
            step_latest_observation = _coalesce(state, "latest_observation", default=latest_observation)
            step_history = list(_as_list(_coalesce(state, "recent_history", default=recent_history)))[-history_window:]
            step_page_sketches = list(_as_list(_coalesce(state, "page_sketches", default=page_sketches)))
        else:
            step_memory = list(memory)
            step_budget = dict(static_budget)
            step_candidates = list(page_candidates)
            step_latest_observation = latest_observation
            step_history = list(recent_history)[-history_window:]
            step_page_sketches = list(page_sketches)

        if action is None:
            continue
        normalized_action = _normalize_action(action)
        image_path = _extract_latest_image_path(step_latest_observation)

        planner_input = {
            "question": question,
            "page_sketches": step_page_sketches,
            "page_candidates": step_candidates,
            "memory": step_memory,
            "remaining_budget": step_budget,
            "latest_observation": step_latest_observation,
            "recent_history": step_history,
        }
        example_id = stable_string_id(f"{trace_id}_step_{step_index:04d}", namespace="step")
        row = {
            "example_id": example_id,
            "trace_id": trace_id,
            "step_index": step_index,
            "doc_id": doc_id,
            "question": question,
            "gold_answer": _coalesce(trace, "gold_answer", "answer"),
            "planner_input": planner_input,
            "target_action": normalized_action,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": _state_to_user_text(planner_input)},
                {"role": "assistant", "content": _compact_json(normalized_action)},
            ],
            "latest_image_path": image_path,
            "metadata": {
                "source_trace_has_explicit_state": isinstance(state, dict),
                "parser_version": "build_step_dataset_v1",
            },
        }
        examples.append(row)

        if isinstance(state, dict):
            memory = list(_as_list(_coalesce(raw_step, "memory_after", default=step_memory)))
            page_candidates = list(_as_list(_coalesce(raw_step, "page_candidates_after", default=step_candidates)))
            latest_observation = _coalesce(raw_step, "observation", default=step_latest_observation)
            recent_history = list(_as_list(_coalesce(raw_step, "recent_history_after", default=step_history)))
        else:
            if normalized_action["action_type"] == "WRITE_NOTE":
                note = _coalesce(
                    observation if isinstance(observation, dict) else {},
                    "note",
                    "note_record",
                    default=None,
                )
                if note is None:
                    note = normalized_action["payload"]
                memory.append(note)
            latest_observation = observation
            recent_history.append(
                {
                    "step_index": step_index,
                    "action": normalized_action,
                    "observation": observation,
                }
            )
    return examples


def build_step_dataset(traces_path: Path, output_path: Path, history_window: int) -> dict[str, object]:
    traces = read_jsonl(traces_path)
    all_examples: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    skipped = 0

    for trace in traces:
        examples = _derive_stateful_examples(trace, history_window=history_window)
        if not examples:
            skipped += 1
            continue
        all_examples.extend(examples)
        for example in examples:
            action_counts[example["target_action"]["action_type"]] += 1

    ensure_dir(output_path.parent)
    write_jsonl(output_path, all_examples)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_traces_path": traces_path.as_posix(),
        "output_path": output_path.as_posix(),
        "trace_count": len(traces),
        "skipped_traces": skipped,
        "example_count": len(all_examples),
        "history_window": history_window,
        "action_counts": dict(sorted(action_counts.items())),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    paths = build_project_paths()
    default_output = paths.processed_data_root / "step_datasets" / "slidevqa_pilot_steps.jsonl"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True, help="Accepted trace JSONL path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output,
        help=f"Output step-dataset JSONL. Defaults to {default_output}",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=4,
        help="Number of previous action-observation turns to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_step_dataset(
        traces_path=args.traces,
        output_path=args.output_path,
        history_window=args.history_window,
    )
    print(_compact_json(summary))


if __name__ == "__main__":
    main()
