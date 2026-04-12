"""
graders.py - Deterministic, reproducible graders for ClinicalTrialRecruiter.
Each grader produces a score in [0.0, 1.0].
"""

from __future__ import annotations
from typing import Any, Dict, List, Set
from src.models import CohortStats, Patient, Trial


# ---------------------------------------------------------------------------
# Eligibility helpers (shared)
# ---------------------------------------------------------------------------

def _check_basic_eligibility(patient: Patient, trial: Trial) -> bool:
    """Hard eligibility check (required comorbidities, exclusions, age, labs)."""
    c = trial.criteria

    # Age range
    if not (c.age_min <= patient.age <= c.age_max):
        return False

    # Opt-out
    if patient.opted_out:
        return False

    # Required comorbidities
    if c.required_comorbidities:
        if not any(rc in patient.comorbidities for rc in c.required_comorbidities):
            return False

    # Excluded comorbidities
    for exc in c.excluded_comorbidities:
        if exc in patient.comorbidities:
            return False

    # Lab requirements
    if c.required_labs:
        if "hba1c_min" in c.required_labs:
            if patient.labs.hba1c is None:
                return False
            if patient.labs.hba1c < c.required_labs["hba1c_min"]:
                return False
        if "hba1c_max" in c.required_labs:
            if patient.labs.hba1c is None:
                return False
            if patient.labs.hba1c > c.required_labs["hba1c_max"]:
                return False
        if "wbc_max" in c.required_labs:
            if patient.labs.wbc is None:
                return False
            if patient.labs.wbc > c.required_labs["wbc_max"]:
                return False

    return True


def get_ground_truth_eligible(patients: List[Patient], trial: Trial) -> Set[str]:
    """Return set of patient IDs that are truly eligible for a trial."""
    return {p.id for p in patients if _check_basic_eligibility(p, trial)}


# ---------------------------------------------------------------------------
# EASY grader – Jaccard similarity
# ---------------------------------------------------------------------------

def grade_easy_single_criterion(
    enrolled_ids: List[str],
    patients: List[Patient],
    trial: Trial,
    privacy_violations: int,
) -> float:
    """
    Jaccard similarity between enrolled set and ground-truth eligible set.
    Penalty for privacy violations.
    Score in [0.0, 1.0].
    """
    enrolled_set = set(enrolled_ids)
    eligible_set = get_ground_truth_eligible(patients, trial)

    if not eligible_set:
        return 0.0

    intersection = enrolled_set & eligible_set
    union = enrolled_set | eligible_set

    jaccard = len(intersection) / len(union) if union else 0.0

    # Privacy penalty
    penalty = min(privacy_violations * 0.05, 0.25)
    score = max(0.0, jaccard - penalty)

    return round(score, 4)


# ---------------------------------------------------------------------------
# MEDIUM grader – F1 with eligibility precision/recall
# ---------------------------------------------------------------------------

def grade_medium_comorbidities(
    enrolled_ids: List[str],
    patients: List[Patient],
    trial: Trial,
    privacy_violations: int,
) -> float:
    """
    F1 score: precision × recall balance on enrolled vs eligible.
    Ineligible enrollments add -0.3 penalty each (capped at 0.5 total).
    Score in [0.0, 1.0].
    """
    enrolled_set = set(enrolled_ids)
    eligible_set = get_ground_truth_eligible(patients, trial)
    target = trial.target_cohort_size

    if not enrolled_set:
        return 0.0

    tp = len(enrolled_set & eligible_set)
    fp = len(enrolled_set - eligible_set)  # ineligible enrolled

    precision = tp / len(enrolled_set) if enrolled_set else 0.0
    recall = tp / min(len(eligible_set), target) if eligible_set else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Ineligible enrollment penalty
    ineligible_penalty = min(fp * 0.08, 0.40)
    privacy_penalty = min(privacy_violations * 0.05, 0.15)

    score = max(0.0, f1 - ineligible_penalty - privacy_penalty)
    return round(score, 4)


# ---------------------------------------------------------------------------
# HARD grader – Composite: eligibility × diversity × privacy
# ---------------------------------------------------------------------------

def _compute_diversity_score(
    cohort_stats: CohortStats,
    diversity_targets: Any,
    enrolled_ids: List[str],
    patients: List[Patient],
) -> float:
    """
    Compute diversity score [0, 1] based on:
    - Female % >= target
    - Rural % >= target
    - Age spread across all 4 brackets
    """
    if not enrolled_ids:
        return 0.0

    total = cohort_stats.total_enrolled
    if total == 0:
        return 0.0

    # Female score (linear up to target)
    female_target = diversity_targets.female_min_pct
    female_score = min(cohort_stats.female_pct / female_target, 1.0) if female_target > 0 else 1.0

    # Rural score
    rural_target = diversity_targets.rural_min_pct
    rural_score = min(cohort_stats.rural_pct / rural_target, 1.0) if rural_target > 0 else 1.0

    # Age spread score: how many of 4 brackets are represented
    age_dist = cohort_stats.age_distribution
    filled_brackets = sum(1 for v in age_dist.values() if v > 0)
    age_spread_score = filled_brackets / 4.0

    diversity = (female_score * 0.35) + (rural_score * 0.30) + (age_spread_score * 0.35)
    return round(diversity, 4)


def grade_hard_diversity(
    enrolled_ids: List[str],
    patients: List[Patient],
    trial: Trial,
    cohort_stats: CohortStats,
    privacy_violations: int,
) -> float:
    """
    Composite score: eligibility_score × diversity_score × privacy_score.
    Hard penalties for privacy violations and diversity imbalance.
    Score in [0.0, 1.0].
    """
    enrolled_set = set(enrolled_ids)
    eligible_set = get_ground_truth_eligible(patients, trial)
    target = trial.target_cohort_size

    # 1. Eligibility component
    if not enrolled_set:
        return 0.0

    tp = len(enrolled_set & eligible_set)
    eligibility_score = min(tp / target, 1.0)

    # 2. Diversity component
    diversity_score = _compute_diversity_score(
        cohort_stats, trial.diversity_targets, enrolled_ids, patients
    )

    # 3. Privacy component (severe penalty for opt-out violations)
    privacy_score = max(0.0, 1.0 - (privacy_violations * 0.20))

    # Composite
    composite = eligibility_score * 0.45 + diversity_score * 0.35 + privacy_score * 0.20

    # Hard imbalance penalty if diversity targets not met at all
    if cohort_stats.total_enrolled >= 5:
        if cohort_stats.female_pct < trial.diversity_targets.female_min_pct * 0.5:
            composite = max(0.0, composite - 0.15)
        if cohort_stats.rural_pct < trial.diversity_targets.rural_min_pct * 0.5:
            composite = max(0.0, composite - 0.10)

    return round(composite, 4)


# ---------------------------------------------------------------------------
# Unified grader interface
# ---------------------------------------------------------------------------

class Grader:
    """Unified grader that routes to the correct task grader."""

    def __init__(
        self,
        patients: List[Patient],
        trial: Trial,
    ):
        self.patients = patients
        self.trial = trial

    def score(
        self,
        task_name: str,
        enrolled_ids: List[str],
        cohort_stats: CohortStats,
        privacy_violations: int,
    ) -> float:
        if task_name == "easy_single_criterion":
            raw_score = grade_easy_single_criterion(
                enrolled_ids, self.patients, self.trial, privacy_violations
            )
        elif task_name == "medium_comorbidities":
            raw_score = grade_medium_comorbidities(
                enrolled_ids, self.patients, self.trial, privacy_violations
            )
        elif task_name == "hard_diversity":
            raw_score = grade_hard_diversity(
                enrolled_ids, self.patients, self.trial, cohort_stats, privacy_violations
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        return max(0.0001, min(0.9999, raw_score))
