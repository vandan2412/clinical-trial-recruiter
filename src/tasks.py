"""
tasks.py - Task definitions for ClinicalTrialRecruiter environment.
Three tasks: easy → medium → hard.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    display_name: str
    description: str
    difficulty: str  # "easy" | "medium" | "hard"
    trial_id: str
    max_steps: int
    target_score: float
    grader_description: str
    hints: List[str] = field(default_factory=list)


TASKS: Dict[str, TaskConfig] = {
    "easy_single_criterion": TaskConfig(
        name="easy_single_criterion",
        display_name="Single-Criterion Patient Screening",
        description=(
            "Screen patients for an ICMR TB trial using a single criterion: "
            "active TB diagnosis. Identify eligible patients and send recruitment invites. "
            "Graded by Jaccard similarity between your selected cohort and the "
            "ground-truth eligible set."
        ),
        difficulty="easy",
        trial_id="ICMR-TB-2024-001",
        max_steps=25,
        target_score=0.85,
        grader_description="Jaccard similarity of enrolled vs all eligible patients (0.0-1.0)",
        hints=[
            "Focus only on patients with 'tb_active' in their comorbidities.",
            "screen_eligible checks if the current patient meets the criterion.",
            "Use draft_invite to send invites to eligible patients.",
        ]
    ),

    "medium_comorbidities": TaskConfig(
        name="medium_comorbidities",
        display_name="Multi-Comorbidity Cohort Building",
        description=(
            "Build a diabetes trial cohort requiring patients with diabetes AND "
            "without heart disease. Also consider HbA1c lab values (7.0-9.5 range). "
            "Partial credit is awarded for each correctly enrolled patient. "
            "Penalties apply for enrolling ineligible patients."
        ),
        difficulty="medium",
        trial_id="ICMR-DIAB-2024-003",
        max_steps=35,
        target_score=0.65,
        grader_description=(
            "Precision × Recall F1 on enrolled cohort vs ground truth, "
            "penalized by ineligible enrollments (-0.3 each)"
        ),
        hints=[
            "Check both required AND excluded comorbidities.",
            "Verify HbA1c lab values are in the 7.0-9.5 range.",
            "mark_optout on patients who have already declined.",
            "prioritize_next skips the current patient and moves to the next.",
        ]
    ),

    "hard_diversity": TaskConfig(
        name="hard_diversity",
        display_name="Diversity-Optimised Cohort Recruitment",
        description=(
            "Build a dengue antiviral trial cohort meeting strict diversity targets: "
            ">= 30% female, >= 20% rural, age spread across all groups. "
            "Respect patient opt-outs (privacy). Penalized for contacting opted-out "
            "patients, cohort imbalance, and repeated actions. "
            "Requires strategic prioritisation and personalised Tamil-English invites."
        ),
        difficulty="hard",
        trial_id="ICMR-DENGUE-2024-002",
        max_steps=40,
        target_score=0.45,
        grader_description=(
            "Composite: eligibility_score × diversity_score × privacy_score. "
            "Each component 0-1. Penalties: -0.5/privacy violation, "
            "-0.3/diversity imbalance."
        ),
        hints=[
            "Check opted_out flag before any contact actions.",
            "Balance female and rural patients actively.",
            "draft_invite[<Tamil-English message>] for personalised outreach.",
            "prioritize_next to skip patients who don't improve diversity.",
            "Age spread: aim for patients in each decade bracket.",
        ]
    ),
}


def get_task(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_name]


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "name": t.name,
            "display_name": t.display_name,
            "difficulty": t.difficulty,
            "description": t.description,
            "target_score": t.target_score,
        }
        for t in TASKS.values()
    ]
