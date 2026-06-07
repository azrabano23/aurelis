"""Render a grading run as Markdown — a student-facing feedback report plus the
grader-validation summary."""
from __future__ import annotations

from aurelis.store import RunRecord


def to_markdown(record: RunRecord) -> str:
    lines = [
        f"# Aurelis grading run `{record.run_id}`",
        "",
        f"- **task**: {record.task}   **grader**: {record.grader}",
        f"- **provider**: {record.provider}  (`{record.params.get('model')}`)",
        f"- **created**: {record.created_at}   **git**: {record.git_sha or 'n/a'}",
        f"- **cache**: {record.cache_hits} hits / {record.cache_misses} misses",
        "",
        "## Cohort metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for k, v in record.metrics.items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Grader validation (vs. human faculty)", ""]
    if record.validation.get("validated_notes"):
        lines += ["| metric | value |", "| --- | --- |"]
        for k, v in record.validation.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_no human gold scores in this dataset_")

    lines += ["", "## Per-note feedback", ""]
    for a in record.assessments:
        lines.append(f"### {a['note_id']}  —  {a['total']}/{a['max_total']} ({a['percent']}%)")
        for s in a["scores"]:
            lines.append(f"- **{s['dimension']}** {s['points']}/{s['max_points']} — {s['feedback']}")
            if s["missing"]:
                lines.append(f"  - missing: {', '.join(s['missing'])}")
        lines.append("")
    return "\n".join(lines) + "\n"
