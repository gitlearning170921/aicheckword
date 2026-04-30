from dataclasses import dataclass
from typing import Any, Dict, List


EXAM_TRACKS: Dict[str, str] = {
    "cn": "国内体考",
    "iso13485": "13485体考",
    "mdsap": "MDSAP体考",
}


QUESTION_TYPES: List[str] = [
    "single_choice",
    "multiple_choice",
    "true_false",
    "case_analysis",
]


@dataclass
class Question:
    question_type: str
    stem: str
    options: List[str]
    answer: Any
    explanation: str
    category: str = ""
    difficulty: str = "medium"
    evidence: List[Dict[str, Any]] = None

