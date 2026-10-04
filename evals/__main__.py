"""Evals CLI.

Usage:
    python -m evals list
    python -m evals --type prompt --verify-only            # offline, no LangSmith
    python -m evals --type agent                           # LangSmith experiment
    python -m evals --type all --refresh --prefix v2        # full run, rebuild datasets
    python -m evals --type prompt --agents planner,summary
"""

import argparse
import sys

from evals.data import PROMPT_CASES, GRAPH_CASES
from evals.runners import run_prompt_evals, run_agent_evals

AGENTS = sorted({c["agent"] for c in PROMPT_CASES})


def _list_cases() -> None:
    print("Prompt eval cases:")
    for c in PROMPT_CASES:
        print(f"  - {c['agent']:<18} shape={c['expected_shape']}")
    print("\nEnd-to-end graph eval cases:")
    for i, c in enumerate(GRAPH_CASES, start=1):
        print(f"  - case {i}: sites={c['sites']} | min_jobs={c['min_jobs']} | {c['user_query']}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="evals", description="Uphired eval harness")
    parser.add_argument("--type", choices=["prompt", "agent", "all", "list", "validate"],
                        default="list", help="which evals to run")
    parser.add_argument("--agents", default="",
                        help="comma-separated prompt agents to evaluate (e.g. planner,summary)")
    parser.add_argument("--verify-only", action="store_true",
                        help="run offline without LangSmith (fast, no dataset/experiment)")
    parser.add_argument("--inline", action="store_true",
                        help="pass examples inline instead of persisting a dataset")
    parser.add_argument("--refresh", action="store_true",
                        help="recreate the LangSmith dataset before running")
    parser.add_argument("--prefix", default=None, help="experiment name prefix")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="verify-only pass threshold for average per metric")
    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    if args.type == "list":
        _list_cases()
        return 0

    if args.type == "validate":
        from evals.evaluators import _format_validity
        errs = []
        for c in PROMPT_CASES:
            shape = c["expected_shape"]
            if shape["type"] == "object" and shape.get("arrays"):
                pass
            if shape["type"] not in ("object", "list", "text"):
                errs.append(f"{c['agent']}: bad type {shape['type']}")
        print("dataset shapes OK" if not errs else "\n".join(errs))
        return 0 if not errs else 1

    code = 0
    if args.type in ("prompt", "all"):
        code |= run_prompt_evals(
            agents=agents or None,
            verify_only=args.verify_only,
            refresh=args.refresh,
            prefix=args.prefix or "prompts",
            inline=args.inline,
            max_concurrency=args.max_concurrency,
            min_score=args.min_score,
        )
    if args.type in ("agent", "all"):
        code |= run_agent_evals(
            verify_only=args.verify_only,
            refresh=args.refresh,
            prefix=args.prefix or "agents",
            inline=args.inline,
            max_concurrency=args.max_concurrency,
            min_score=args.min_score,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())