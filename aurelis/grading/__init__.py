from aurelis.grading.base import Grader
from aurelis.grading.checklist import ChecklistGrader
from aurelis.grading.llm_grader import LLMGrader

__all__ = ["Grader", "ChecklistGrader", "LLMGrader", "get_grader"]


def get_grader(name: str) -> Grader:
    graders = {"llm": LLMGrader, "checklist": ChecklistGrader}
    if name not in graders:
        raise ValueError(f"unknown grader: {name!r}. Known: {sorted(graders)}")
    return graders[name]()
