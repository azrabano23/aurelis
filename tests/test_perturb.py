"""Offline tests for the perturbation harness and grader validation (no network).

A fixture note with real ACI-style section headers stands in for the dataset so
these run in CI without hitting GitHub.
"""
from aurelis.datasets.acibench import split_sections
from aurelis.grading import ChecklistGrader
from aurelis.perturb import PERTURBATIONS
from aurelis.rubric import SOAP_RUBRIC
from aurelis.types import ClinicalCase, StudentNote
from aurelis.validation import run_perturbation_validation

FIXTURE = """CHIEF COMPLAINT
Chest pain.

HISTORY OF PRESENT ILLNESS
58yo M with substernal chest pain radiating to the left arm, with diaphoresis.

PHYSICAL EXAMINATION
BP 158/94, HR 98. Heart regular. Lungs clear.

RESULTS
ECG shows ST depressions. Troponin pending.

ASSESSMENT AND PLAN
Acute coronary syndrome. Start aspirin and heparin, cardiology consult, serial troponins."""


def _pairs():
    case = ClinicalCase(id="fx", specialty="EM", vignette="...")
    note = StudentNote(case_id="fx", note_id="fx", text=FIXTURE)
    return [(case, note)]


def test_split_sections_finds_headers():
    secs = split_sections(FIXTURE)
    assert "CHIEF COMPLAINT" in secs
    assert "PHYSICAL EXAMINATION" in secs
    assert "ASSESSMENT AND PLAN" in secs


def test_drop_subjective_removes_only_subjective_sections():
    drop_subj = next(p for p in PERTURBATIONS if p.name == "drop_subjective")
    out = drop_subj.apply(FIXTURE)
    assert "HISTORY OF PRESENT ILLNESS" not in out
    assert "PHYSICAL EXAMINATION" in out          # objective untouched


def test_validation_sensitive_to_structure_specific_off_target():
    summary = run_perturbation_validation(_pairs(), ChecklistGrader(), SOAP_RUBRIC)
    subj = summary["drop_subjective"]
    assert subj["mean_targeted_drop"] > 0      # dropping subjective lowers subjective
    assert subj["mean_offtarget_drop"] == 0.0  # leaves other dimensions alone


def test_checklist_is_blind_to_contradiction():
    # The motivating finding: a keyword grader cannot detect a logical
    # contradiction — exactly the reasoning failure the LLM grader exists for.
    summary = run_perturbation_validation(_pairs(), ChecklistGrader(), SOAP_RUBRIC)
    assert summary["inject_contradiction"]["detection_rate"] == 0.0
