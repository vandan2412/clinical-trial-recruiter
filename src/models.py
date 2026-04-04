"""
models.py - Typed Pydantic models for ClinicalTrialRecruiter OpenEnv environment.
All observation, action, reward, and state types are defined here.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Patient / Trial data models
# ---------------------------------------------------------------------------

class LabValues(BaseModel):
    hba1c: Optional[float] = None
    wbc: Optional[float] = None
    hemoglobin: Optional[float] = None


class Patient(BaseModel):
    id: str
    age: int
    gender: str  # "male" | "female"
    location: str  # "urban" | "rural"
    ethnicity: str
    language: str
    comorbidities: List[str]
    labs: LabValues
    opted_out: bool = False
    contacted: bool = False
    enrolled: bool = False


class DiversityTargets(BaseModel):
    female_min_pct: float = 0.30
    rural_min_pct: float = 0.20
    age_spread_required: bool = True


class TrialCriteria(BaseModel):
    age_min: int
    age_max: int
    required_comorbidities: List[str]
    excluded_comorbidities: List[str]
    required_labs: Dict[str, float] = Field(default_factory=dict)
    gender_requirement: Optional[str] = None


class Trial(BaseModel):
    id: str
    name: str
    disease: str
    sponsor: str
    phase: str
    target_cohort_size: int
    criteria: TrialCriteria
    diversity_targets: DiversityTargets
    description: str
    status: str
    window_days: int


# ---------------------------------------------------------------------------
# Cohort statistics (computed dynamically)
# ---------------------------------------------------------------------------

class CohortStats(BaseModel):
    total_enrolled: int = 0
    female_count: int = 0
    rural_count: int = 0
    age_distribution: Dict[str, int] = Field(default_factory=lambda: {
        "18-30": 0, "31-45": 0, "46-60": 0, "61+": 0
    })
    female_pct: float = 0.0
    rural_pct: float = 0.0
    privacy_violations: int = 0
    diversity_score: float = 0.0


# ---------------------------------------------------------------------------
# Observation model (what the agent sees each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Observation returned to the agent each step."""
    # Trial context
    trial: Dict[str, Any]

    # Current patient under consideration (the "focus" patient)
    current_patient: Optional[Dict[str, Any]] = None
    patient_index: int = 0
    total_patients: int = 0

    # Cohort state
    cohort_stats: CohortStats = Field(default_factory=CohortStats)
    target_cohort_size: int = 20

    # History (last 5 actions + outcomes)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Response tracking
    invites_sent: int = 0
    follow_ups_sent: int = 0
    opt_outs_recorded: int = 0

    # Episode meta
    steps_remaining: int = 20
    done: bool = False
    task_name: str = ""


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "screen_eligible",
    "draft_invite",
    "follow_up",
    "mark_optout",
    "prioritize_next",
]


class Action(BaseModel):
    """Parsed action the agent takes each step."""
    action_type: str  # one of VALID_ACTIONS
    personalized_text: Optional[str] = None  # for draft_invite[...]

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """Parse raw action string like 'draft_invite[Hello patient...]'."""
        action_str = action_str.strip()

        if action_str.startswith("draft_invite[") and action_str.endswith("]"):
            text = action_str[len("draft_invite["):-1]
            return cls(action_type="draft_invite", personalized_text=text)

        # Handle partial matches (LLM might add extra words)
        for valid in VALID_ACTIONS:
            if valid in action_str:
                return cls(action_type=valid)

        return cls(action_type="screen_eligible")  # safe default


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Returned from env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment state (full internal state)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Complete environment state, returned by state()."""
    task_name: str
    trial: Trial
    patients: List[Patient]
    current_patient_idx: int
    enrolled_patient_ids: List[str]
    contacted_patient_ids: List[str]
    opted_out_patient_ids: List[str]
    privacy_violations: int
    total_reward: float
    step_count: int
    max_steps: int
    done: bool
    cohort_stats: CohortStats
    action_history: List[Dict[str, Any]]
