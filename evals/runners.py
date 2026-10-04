"""Eval experiment runners.

Two execution modes:
  * LangSmith path — persists a dataset, runs `langsmith.evaluate()` with the
    `@run_evaluator` evaluators, and returns the experiment report.
  * verify-only path — offline, no LangSmith: runs predictions + core evaluator
    functions locally and prints a table. Used for fast iteration / CI smoke.

Core evaluators are async when judge-based; the local runner awaits them.
"""

import asyncio
import inspect
from dataclasses import dataclass
from loguru import logger
from langsmith import Client, evaluate

from evals.data import prompt_examples, graph_examples
from evals.predictions import prompt_predict, graph_predict
from evals.evaluators import (
    PROMPT_EVALUATORS, AGENT_EVALUATORS, PROMPT_SPECS, AGENT_SPECS,
)
from evals.trace import init_tracing

PROMPT_DATASET = "uphired-prompt-evals"
AGENT_DATASET = "uphired-agent-evals"


def _ensure_dataset(client: Client, name: str, examples: list[dict], refresh: bool) -> None:
    if refresh and client.has_dataset(dataset_name=name):
        client.delete_dataset(dataset_name=name)
        logger.info(f"Deleted existing dataset '{name}' (refresh)")
    if client.has_dataset(dataset_name=name):
        logger.info(f"Using existing dataset '{name}'")
        return
    client.create_dataset(dataset_name=name)
    client.create_examples(
        dataset_name=name,
        examples=[{"inputs": e["inputs"], "outputs": e["outputs"]} for e in examples],
    )
    logger.info(f"Created dataset '{name}' with {len(examples)} examples")


def _print_results(results) -> None:
    df = results.to_pandas()
    with_evals = [c for c in df.columns if c not in ("input", "output", "error")]
    print("\n---- Experiment summary ------------------------------")
    print(f"experiment: {results.experiment_name}")
    print(f"url: {results.url or 'n/a'}")
    if df.empty:
        print("no results")
        return
    for col in with_evals:
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(f"  {col:<28} mean={vals.mean():.3f}  min={vals.min():.3f}  n={len(vals)}")
    print("--------------------------------------")


@dataclass
class _Row:
    label: str
    key: str
    score: float
    comment: str


async def _run_local(predict, examples, specs) -> list[_Row]:
    rows: list[_Row] = []
    for i, ex in enumerate(examples, start=1):
        inputs = ex["inputs"]
        expected = ex.get("outputs", {}) or {}
        label = inputs.get("agent") or inputs.get("user_query") or f"case {i}"
        try:
            outputs = await predict(inputs)
        except Exception as e:
            logger.error(f"[local] prediction failed for '{label}': {e}")
            outputs = {"error": str(e)}
        for name, core in specs:
            try:
                result = core(outputs, inputs, expected)
                if inspect.isawaitable(result):
                    result = await result
                score, comment = result
            except Exception as e:
                score, comment = 0.0, f"evaluator error: {e}"
            rows.append(_Row(label=f"{i}:{label}", key=name, score=score, comment=comment))
    return rows


def _print_local(rows: list[_Row]) -> None:
    print("\n---- Offline verify report ----------------------------")
    current = None
    for r in rows:
        if r.label != current:
            current = r.label
            print(f"\n  [{current}]")
        flag = "PASS" if r.score >= 1.0 else ("WARN" if r.score >= 0.5 else "FAIL")
        print(f"    {r.key:<28} {flag}  {r.score:.2f}  {r.comment}")
    print("--------------------------------------")


def _local_exit_code(rows: list[_Row], min_score: float) -> int:
    if not rows:
        return 1
    from collections import defaultdict
    agg: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        agg[r.key].append(r.score)
    bad = [k for k, v in agg.items() if (sum(v) / len(v)) < min_score]
    if bad:
        print(f"threshold {min_score} not met for: {bad}")
        return 1
    print(f"all averages >= {min_score} - OK")
    return 0


def run_prompt_evals(
    agents: list[str] | None = None,
    verify_only: bool = False,
    refresh: bool = False,
    prefix: str = "prompts",
    inline: bool = False,
    max_concurrency: int = 2,
    min_score: float = 0.5,
) -> int:
    examples = [e for e in prompt_examples()
                if not agents or e["inputs"]["agent"] in agents]
    if not examples:
        logger.error("no matching prompt cases (agents: %s)", agents)
        return 1

    if verify_only:
        rows = asyncio.run(_run_local(prompt_predict, examples, PROMPT_SPECS))
        _print_local(rows)
        return _local_exit_code(rows, min_score)

    init_tracing()
    client = Client()
    name = PROMPT_DATASET if not agents else f"{PROMPT_DATASET}-{'|'.join(sorted(agents))}"
    if not inline:
        _ensure_dataset(client, name, examples, refresh)
    data: object = name if not inline else examples
    results = evaluate(
        prompt_predict,
        data=data,
        evaluators=PROMPT_EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=max_concurrency,
    )
    _print_results(results)
    return 0


def run_agent_evals(
    verify_only: bool = False,
    refresh: bool = False,
    prefix: str = "agents",
    inline: bool = False,
    max_concurrency: int = 2,
    min_score: float = 0.5,
) -> int:
    examples = graph_examples()
    if verify_only:
        rows = asyncio.run(_run_local(graph_predict, examples, AGENT_SPECS))
        _print_local(rows)
        return _local_exit_code(rows, min_score)

    init_tracing()
    client = Client()
    if not inline:
        _ensure_dataset(client, AGENT_DATASET, examples, refresh)
    data: object = AGENT_DATASET if not inline else examples
    results = evaluate(
        graph_predict,
        data=data,
        evaluators=AGENT_EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=max_concurrency,
    )
    _print_results(results)
    return 0