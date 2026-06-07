# Aurelis

[![tests](https://github.com/azrabano23/aurelis/actions/workflows/tests.yml/badge.svg)](https://github.com/azrabano23/aurelis/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Aurelis grades medical students' clinical notes the way an attending would — against a rubric, with specific feedback — and proves it agrees with human faculty before any grade reaches a student.**

It pairs a reproducible LLM-as-judge grader with the statistics that make an automated grade defensible: quadratic-weighted kappa and Pearson correlation against faculty gold scores. Same note in, same grade out, every time.

> Built at the **Rutgers Health Hack**, where it placed **3rd out of 200+**. I taught myself to code that weekend, pitched my way onto a team of MS/PhD students, and owned the engineering end-to-end. The interesting constraint wasn't the model — it was shipping something a clinician would trust, in 48 hours, with no prior production experience. This repo is that prototype, rebuilt properly.

---

## The problem

Clinical documentation is a graded, high-stakes skill, and the feedback loop for it is broken:

- **It's central to training and to patient safety.** A SOAP note (Subjective / Objective / Assessment / Plan) is how a clinician reasons on paper. Bad notes drive diagnostic errors, billing failures, and miscommunication at handoff — documentation issues are a recurring factor in malpractice claims.
- **Feedback doesn't scale.** A clerkship director might have 40 students each writing dozens of notes. Detailed, dimension-by-dimension feedback on every note is the single most useful thing a student can get, and the single thing faculty have no time to give. Most students get a letter grade and a sentence.
- **Naïve automation is untrustworthy.** You can't put an opaque AI grade in front of a student. If the model is miscalibrated, inconsistent, or hallucinates a "missing" finding that's actually present, you've made the problem worse. The bar is *agreement with expert humans*, demonstrated, not assumed.

## The solution

Aurelis grades a note against an explicit, faculty-style rubric and returns, per dimension, a score **plus the evidence it cited from the note and the elements it found missing** — so feedback is actionable and every grade is auditable rather than a black box. Critically, it ships the machinery to **validate the grader against human scores** (`qwk`, `pearson_r`, `mae`) so a program can verify the grader is trustworthy *on their own rubric and their own notes* before relying on it.

On **real clinical notes from ACI-Bench** (20 expert encounter notes, no PHI), a deterministic keyword baseline detects *missing sections* well but is **completely blind to an injected clinical contradiction** — exactly the reasoning failure that requires a model. That gap *is* the product thesis: deterministic checks get you presence-of-facts; judging clinical reasoning needs an LLM (see [Validation](#validation-on-real-clinical-notes-aci-bench)).

---

## How it works

```
ClinicalCase + StudentNote + Rubric  ──Grader──►  NoteAssessment (per-dimension: score, feedback, missing, evidence)
        │                                                │
        └──────── runner: response cache + experiment store ────────┘
                                  │
                    Task.validate(...)  ──►  agreement vs. human faculty (QWK, Pearson, MAE)
```

Four decoupled seams, each independently testable:

- **`Provider`** — the only thing that touches a model SDK. `AnthropicProvider` uses the official SDK correctly for current models (adaptive thinking, `effort` under `output_config`, streaming above 16k tokens, no removed sampling params). `MockProvider` is scriptable and deterministic, so the entire harness — grader, cache, metrics — runs in CI with no API key.
- **`Grader`** — `LLMGrader` judges one rubric dimension per model call (isolating each dimension keeps the model's attention narrow and each grade independently cacheable); `ChecklistGrader` is a deterministic baseline and a fast sanity check.
- **`Rubric` / `Task`** — rubrics are *data*: the SOAP rubric is five 0–4 axes with explicit criteria. Define a new rubric (OSCE, discharge summary, SBAR handoff) and every grader works against it unchanged. A `Task` binds a rubric to a dataset and knows how to aggregate and validate.
- **`runner`** — thin: it owns only the cache and the store. Everything domain-specific lives in the Grader and Task.

Adding a rubric or a note type is a data/one-file change, not a rewrite.

## Reproducibility & auditability

Grades that affect students have to be reproducible and defensible, so this is treated as a first-class requirement, not a nicety:

- **Content-addressed cache.** Every grader call is keyed by a SHA-256 over the canonical `(messages, params)`. The same note graded twice returns a byte-identical grade; reruns issue zero model calls. (LLMs expose no usable seed and `temperature` is gone on current models — so determinism comes from caching the grader's output, which is the honest way to get it.)
- **Experiment store.** Each run is written with its model, parameters, grader, git SHA, timestamp, and cache stats — enough to reproduce or contest any grade months later.
- **Human-validation built in.** Where faculty gold scores exist, every run reports QWK / Pearson / MAE overall and per dimension. You don't take the grader on faith; you measure it.

## Validation — the part that earns trust

The headline question isn't "what grade did the AI give" but "**does the AI grader agree with faculty?**" Aurelis answers it with the standard inter-rater statistics for ordinal grades:

| metric | what it tells you |
|--------|-------------------|
| **Quadratic-weighted kappa** | agreement on the 0–4 scale, chance-corrected, penalizing larger disagreements quadratically |
| **Pearson r** | does the grader rank notes in the same order as faculty? |
| **MAE (points)** | average absolute gap in points per dimension |

This framing — *use a model to evaluate expert work, then validate the evaluator against human judgment* — is applied **scalable oversight**, the same question AI alignment research asks about supervising capable models. Aurelis is a concrete, measurable instance of it in a domain where the ground truth (a faculty grade) actually exists.

## Validation on real clinical notes (ACI-Bench)

Faculty-graded note corpora are scarce and PHI-protected, so to validate the grader on *real* clinical text I use a **perturbation method** that needs no human labels: take expert reference notes from the public [ACI-Bench](https://github.com/wyim/aci-bench) benchmark, systematically damage one section, and check that the grader's score drops on the dimension you damaged (**sensitivity**) and *only* that dimension (**specificity**). The known damage is the objective ground truth.

Real results on 20 ACI-Bench notes with the deterministic checklist grader (`python scripts/validate_acibench.py --grader checklist`):

| perturbation | targeted-dim score drop | off-target drop | detection rate |
|---|---:|---:|---:|
| drop Subjective sections | **3.0** | 0.0 | 0.90 |
| drop Objective sections | **1.8** | 0.0 | 0.80 |
| drop Assessment+Plan | **1.7** | 0.0 | 0.85 |
| inject a clinical contradiction | **0.0** | 0.0 | **0.00** |

Two things to read off this. First, **perfect specificity** (0.0 off-target drop everywhere): the grader never penalizes the wrong dimension. Second, the deterministic grader catches omissions but **cannot see a contradiction** (0.00) — it has no model of clinical reasoning. That last row is the empirical argument for the LLM grader, and the harness is grader-agnostic: swap in `--grader llm` (needs `ANTHROPIC_API_KEY`) and the same table measures whether the model closes that gap.

---

## Technical skills this demonstrates

Clean layered architecture with hard interface boundaries; LLM-as-judge eval design with per-dimension isolation; correct, current Anthropic SDK usage; reproducibility engineering (content-addressed caching, provenance-tracked experiment store); applied statistics (quadratic-weighted kappa, correlation, Wilson intervals) implemented from scratch and unit-tested; a deterministic mock that makes the whole system CI-testable without network or keys; 16 tests, packaged (`pyproject`, console script), CI on 3.10/3.12.

## Business model & go-to-market

- **Who pays:** US allopathic/osteopathic medical schools (~200), large residency programs, and USMLE/COMLEX/OSCE prep companies. Secondary: nursing and PA programs (same documentation skill), and EHR vendors wanting a documentation-coaching layer.
- **Wedge:** a single clerkship (e.g. Internal Medicine) at one school — the highest-volume note-writing rotation. Land as a faculty *time-saver*, not a grade-replacer: Aurelis drafts dimension-level feedback, the attending reviews and signs off. Trust compounds as the validation dashboard shows rising QWK on the program's own notes.
- **Pricing:** per-student/per-year SaaS seat ($X/student/yr at a few thousand students per school), plus a higher-margin OSCE/exam-prep B2C tier. Compute is cents per note; gross margin is high.
- **Moat:** the validated, rubric-specific grader plus the longitudinal student-progress data — once a program's rubric is calibrated and its students' trajectories are tracked, switching cost is real. The dataset of (note, faculty grade) pairs is itself defensible and improves the grader.
- **Why now:** frontier models finally grade clinical reasoning at near-faculty quality, and the validation layer is what lets a risk-averse institution actually adopt it.

## Roadmap & honest limitations

Two validation paths ship today: human-agreement (QWK/Pearson/MAE) on a small faculty-scored set, and perturbation-based construct validity on real ACI-Bench notes (above). Neither certifies a production grader on its own — real deployment requires calibrating QWK on a program's *own* faculty-scored corpus, and the perturbation set should grow to cover subtler reasoning errors than a single injected contradiction. LLM-as-judge inherits the judge's biases; the deterministic checklist is the guardrail, and rubrics are kept narrow and explicit. Next: a held-out human-graded benchmark, inter-faculty agreement as the ceiling to compare against, bias audits across note styles, and OSCE/SBAR rubrics.

---

## Usage

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...

aurelis grade  configs/soap.yaml     # grade the SOAP dataset, record the run, print metrics + validation
aurelis report <run_id>              # render a per-note feedback report (Markdown)
aurelis list                         # list past runs with their QWK
pytest -q                            # 20 tests, fully offline against the mock grader

# validate the grader on real ACI-Bench notes via section perturbation:
python scripts/validate_acibench.py --grader checklist --limit 20   # offline, no key
python scripts/validate_acibench.py --grader llm --provider anthropic  # needs ANTHROPIC_API_KEY
```

```python
from aurelis.tasks import SOAPTask
from aurelis.grading import LLMGrader
from aurelis.providers import get_provider
from aurelis.runner import run
from aurelis.types import GenerationParams

rec = run(SOAPTask(), LLMGrader(), get_provider("anthropic"), GenerationParams(model="claude-opus-4-8"))
print(rec.metrics, rec.validation)
```

## License

MIT — see [LICENSE](LICENSE).
