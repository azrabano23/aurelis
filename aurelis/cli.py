"""Command-line entry point.

    aurelis grade  configs/soap.yaml        # grade a task's notes, record, print metrics
    aurelis report <run_id>                  # render a saved run as a feedback report
    aurelis list                             # list past runs

Config (YAML):

    task: soap                 # soap
    grader: llm                # llm | checklist
    provider: anthropic        # anthropic | mock
    model: claude-opus-4-8
    max_tokens: 1024
    effort: high               # optional
    thinking: true             # optional
    limit: null                # optional sample cap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aurelis import tasks  # noqa: F401  (registers built-in tasks)
from aurelis.grading import get_grader
from aurelis.providers import get_provider
from aurelis.report import to_markdown
from aurelis.runner import run as run_task
from aurelis.store import RunStore
from aurelis.tasks.base import get_task
from aurelis.types import GenerationParams


def _load_config(path: str) -> dict:
    text = Path(path).read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        cfg: dict = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = (p.strip() for p in line.split(":", 1))
            if v in ("null", ""):
                cfg[k] = None
            elif v in ("true", "false"):
                cfg[k] = v == "true"
            elif v.isdigit():
                cfg[k] = int(v)
            else:
                cfg[k] = v
        return cfg


def _cmd_grade(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    params = GenerationParams(
        model=cfg.get("model", "claude-opus-4-8"),
        max_tokens=cfg.get("max_tokens", 1024),
        effort=cfg.get("effort"),
        thinking=bool(cfg.get("thinking", False)),
    )
    record = run_task(
        get_task(cfg["task"]),
        get_grader(cfg.get("grader", "llm")),
        get_provider(cfg.get("provider", "anthropic")),
        params,
        store=RunStore(),
        limit=cfg.get("limit"),
    )
    print(f"run_id: {record.run_id}")
    print(json.dumps({"metrics": record.metrics, "validation": record.validation}, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    print(to_markdown(RunStore().load(args.run_id)))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    for r in RunStore().list_runs():
        qwk = r.get("validation", {}).get("qwk")
        print(f"{r['run_id']:34s} {r['task']:8s} {r.get('model','')}  qwk={qwk}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aurelis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_grade = sub.add_parser("grade", help="grade a task's notes from a YAML config")
    p_grade.add_argument("config")
    p_grade.set_defaults(func=_cmd_grade)

    p_report = sub.add_parser("report", help="render a saved run as a feedback report")
    p_report.add_argument("run_id")
    p_report.set_defaults(func=_cmd_report)

    p_list = sub.add_parser("list", help="list past runs")
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
