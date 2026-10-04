"""Evaluators.

Each evaluator has two faces:
  * a plain core `_xxx(outputs, inputs, expected) -> (score, comment)` usable
    offline (verify-only runs) and directly testable;
  * a `@run_evaluator` wrapper for LangSmith `evaluate()`.

Judge-based cores are async (they call the OpenRouter LLM-as-judge).
"""

import json
import re

from langsmith.evaluation.evaluator import run_evaluator

from evals.judge import judge


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).strip()


def _load_json(content: str):
    try:
        return json.loads(_strip_fences(content))
    except Exception:
        return None


# ── Prompt / structured-output validity ───────────────────────────────────────

def _check_shape(value, shape: dict, checks: list) -> None:
    kind = shape.get("type", "text")
    if kind == "object":
        if not isinstance(value, dict):
            checks.append((0.0, f"expected JSON object, got {type(value).__name__}"))
            return
        for key in shape.get("keys", []):
            if key not in value:
                checks.append((0.0, f"missing key '{key}'"))
        for key, lo in (shape.get("numeric_min") or {}).items():
            v = value.get(key)
            if not isinstance(v, (int, float)) or v < lo:
                checks.append((0.0, f"'{key}' should be >= {lo}, got {v!r}"))
        for akey, spec in (shape.get("arrays") or {}).items():
            arr = value.get(akey)
            if not isinstance(arr, list):
                checks.append((0.0, f"'{akey}' should be a list"))
                continue
            lo = spec.get("min_items", 0)
            hi = spec.get("max_items")
            if len(arr) < lo:
                checks.append((0.0, f"'{akey}' has {len(arr)} items, min {lo}"))
            elif hi is not None and len(arr) > hi:
                checks.append((0.0, f"'{akey}' has {len(arr)} items, max {hi}"))
            for item_keys in [spec.get("item_keys", [])]:
                if not item_keys:
                    continue
                for i, item in enumerate(arr):
                    if not isinstance(item, dict):
                        checks.append((0.0, f"'{akey}'[{i}] not an object"))
                    else:
                        for ik in item_keys:
                            if not item.get(ik):
                                checks.append((0.0, f"'{akey}'[{i}] missing non-empty '{ik}'"))
        sr = shape.get("score_range")
        if sr:
            v = value.get("score")
            if not isinstance(v, (int, float)) or not (sr[0] <= v <= sr[1]):
                checks.append((0.0, f"'score' should be in {sr}, got {v!r}"))
    elif kind == "list":
        if not isinstance(value, list):
            checks.append((0.0, "expected JSON array"))
            return
        lo, hi = shape.get("min_items", 0), shape.get("max_items")
        if len(value) < lo:
            checks.append((0.0, f"array has {len(value)} items, min {lo}"))
        elif hi is not None and len(value) > hi:
            checks.append((0.0, f"array has {len(value)} items, max {hi}"))
        vt = shape.get("value_type")
        for i, item in enumerate(value):
            if vt == "str" and not isinstance(item, str):
                checks.append((0.0, f"element {i} not a string"))
            if vt == "int" and not isinstance(item, int):
                checks.append((0.0, f"element {i} not an integer"))
    elif kind == "text":
        value = str(value)
        if shape.get("min_char") and len(value) < shape["min_char"]:
            checks.append((0.0, f"text too short ({len(value)} chars)"))
    else:
        checks.append((0.0, f"unknown shape type '{kind}'"))


def _format_validity(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    if outputs.get("error"):
        return 0.0, f"prediction error: {outputs['error']}"
    parsed = _load_json(outputs.get("content", ""))
    if parsed is None:
        return 0.0, "output is not valid JSON"
    checks: list = [(1.0, "valid JSON")]
    _check_shape(parsed, expected.get("expected_shape", {}), checks)
    score = sum(s for s, _ in checks) / len(checks)
    comments = "; ".join(c for _, c in checks if c.startswith("'") or "expected" in c)
    return round(score, 3), (comments or "shape OK")


# ── LLM-as-judge: prompt output quality ──────────────────────────────────────

async def _prompt_quality(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    if outputs.get("error"):
        return 0.0, f"prediction error: {outputs['error']}"
    result = await judge(
        criterion=expected.get("criterion", "matches the expected output"),
        output=outputs.get("content", ""),
        reference=expected.get("golden", ""),
    )
    return result["score"], result["reason"]


# ── End-to-end graph evaluators ──────────────────────────────────────────────

def _graph_completed(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    if outputs.get("status") != "done":
        return 0.0, f"graph errored: {outputs.get('errors')}"
    if not outputs.get("final_report"):
        return 0.0, "graph finished but final_report is empty"
    return 1.0, "graph completed with a final report"


def _tool_dispatch(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    called = {t for t in outputs.get("tools_called", [])}
    expected_tools = {f"search_{s}" for s in expected.get("expected_sites", [])}
    if not expected_tools:
        return 1.0, "no expected tools declared"
    hit = called & expected_tools
    missing = expected_tools - called
    score = len(hit) / len(expected_tools)
    extra = called - expected_tools
    detail = f"called={sorted(called)}"
    if missing:
        detail += f" | missing={sorted(missing)}"
    if extra:
        detail += f" | unexpected={sorted(extra)}"
    return round(score, 3), detail


def _jobs_found(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    min_jobs = int(expected.get("min_jobs", 1))
    got = int(outputs.get("num_jobs", 0))
    return (1.0 if got >= min_jobs else 0.0,
            f"{got} jobs found (minimum {min_jobs})")


def _node_flow(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    nodes = outputs.get("nodes_called", [])
    required = ["planner", "browser", "extractor", "ranker", "summary"]
    # require presence + relative order
    positions = []
    for node in required:
        try:
            positions.append(nodes.index(node))
        except ValueError:
            return 0.0, f"node '{node}' never executed (flow={nodes})"
    if positions != sorted(positions):
        return 0.0, f"nodes out of order (flow={nodes})"
    return 1.0, f"correct node flow (extra nodes: {[n for n in nodes if n not in required] or 'none'})"


def _report_keywords(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    report = (outputs.get("final_report") or "").lower()
    keywords = [k.lower() for k in expected.get("golden_keywords", [])]
    if not keywords:
        return 1.0, "no keywords declared"
    missing = [k for k in keywords if k not in report]
    score = 1.0 if not missing else max(0.0, 1.0 - len(missing) / len(keywords))
    return round(score, 3), (f"missing keywords: {missing}" if missing else "all keywords present")


async def _report_relevance(outputs: dict, inputs: dict, expected: dict) -> tuple[float, str]:
    report = outputs.get("final_report", "")
    if not report:
        return 0.0, "empty report"
    result = await judge(
        criterion=(
            "The report is relevant to the user's job search: it accurately reflects "
            "the found jobs, their relevance to the user's query/skills, and gives a "
            "correct, useful summary with matched and missing skills."
        ),
        output=report,
        reference=expected.get("golden_report", ""),
    )
    return result["score"], result["reason"]


# ── LangSmith run_evaluator wrappers ─────────────────────────────────────────

@run_evaluator
def prompt_format_validity(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _format_validity(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "prompt_format_validity", "score": score, "comment": comment}


@run_evaluator
async def prompt_quality_judge(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = await _prompt_quality(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "prompt_quality_judge", "score": score, "comment": comment}


@run_evaluator
def graph_completed(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _graph_completed(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "graph_completed", "score": score, "comment": comment}


@run_evaluator
def tool_dispatch(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _tool_dispatch(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "tool_dispatch", "score": score, "comment": comment}


@run_evaluator
def jobs_found(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _jobs_found(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "jobs_found", "score": score, "comment": comment}


@run_evaluator
def node_flow(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _node_flow(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "node_flow", "score": score, "comment": comment}


@run_evaluator
def report_keywords(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = _report_keywords(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "report_keywords", "score": score, "comment": comment}


@run_evaluator
async def report_relevance_judge(run, example):
    expected = (example.outputs or {}) if example else {}
    score, comment = await _report_relevance(run.outputs or {}, run.inputs or {}, expected)
    return {"key": "report_relevance_judge", "score": score, "comment": comment}


# Specs consumed by the offline verify-runner (name, core, is_async resolved at call)
PROMPT_SPECS = [("prompt_format_validity", _format_validity),
                ("prompt_quality_judge", _prompt_quality)]

AGENT_SPECS = [("graph_completed", _graph_completed),
               ("tool_dispatch", _tool_dispatch),
               ("jobs_found", _jobs_found),
               ("node_flow", _node_flow),
               ("report_keywords", _report_keywords),
               ("report_relevance_judge", _report_relevance)]

PROMPT_EVALUATORS = [prompt_format_validity, prompt_quality_judge]
AGENT_EVALUATORS = [graph_completed, tool_dispatch, jobs_found,
                    node_flow, report_keywords, report_relevance_judge]