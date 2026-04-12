"""
env.py - ClinicalTrialRecruiterEnv: Core OpenEnv environment.
Implements step() / reset() / state() interface.
"""

from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional

from src.models import (
    Action,
    CohortStats,
    EnvState,
    Observation,
    Patient,
    StepResult,
    Trial,
)
from src.tasks import TaskConfig, get_task
from src.graders import Grader, _check_basic_eligibility

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_HERE, "data")


def _load_patients() -> List[Patient]:
    path = os.path.join(_DATA_DIR, "patients.json")
    with open(path) as f:
        raw = json.load(f)
    patients = []
    for p in raw:
        p["labs"] = p.get("labs", {})
        patients.append(Patient(**p))
    return patients


def _load_trials() -> Dict[str, Trial]:
    path = os.path.join(_DATA_DIR, "trials.json")
    with open(path) as f:
        raw = json.load(f)
    trials = {}
    for t in raw:
        trial = Trial(**t)
        trials[trial.id] = trial
    return trials


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class ClinicalTrialRecruiterEnv:
    """
    OpenEnv environment for AI-driven clinical trial recruitment.
    
    Simulates an AI agent that screens anonymised patient EHRs,
    drafts personalised invitations, handles responses, and builds
    diverse, ethically compliant cohorts for ICMR clinical trials.
    
    Tasks:
      - easy_single_criterion: single-criterion TB screening
      - medium_comorbidities: multi-comorbidity diabetes cohort
      - hard_diversity: diversity-optimised dengue cohort
    """

    metadata = {
        "name": "ClinicalTrialRecruiter",
        "version": "1.0.0",
        "tasks": ["easy_single_criterion", "medium_comorbidities", "hard_diversity"],
        "render_modes": [],
    }

    # Reward constants
    R_ENROLL = 0.9
    R_DIVERSITY_PROGRESS = 0.4
    R_ELIGIBILITY_MATCH = 0.2
    R_PRIVACY_VIOLATION = -0.5
    R_IMBALANCE = -0.3
    R_INVALID_ACTION = -0.05
    R_REPEAT_ACTION = -0.05

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)

        # Load static data
        self._all_patients: List[Patient] = _load_patients()
        self._all_trials: Dict[str, Trial] = _load_trials()

        # Runtime state (set in reset())
        self._task: Optional[TaskConfig] = None
        self._trial: Optional[Trial] = None
        self._patients: List[Patient] = []
        self._enrolled_ids: List[str] = []
        self._contacted_ids: List[str] = []
        self._opted_out_ids: List[str] = []
        self._current_idx: int = 0
        self._step_count: int = 0
        self._max_steps: int = 30
        self._done: bool = True
        self._privacy_violations: int = 0
        self._total_reward: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._cohort_stats: CohortStats = CohortStats()
        self._grader: Optional[Grader] = None
        self._last_action_str: str = ""

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        """Reset the environment for a new episode. Returns initial observation."""
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)

        if task is None:
            task = "easy_single_criterion"

        self._task = get_task(task)
        self._trial = self._all_trials[self._task.trial_id]
        self._max_steps = self._task.max_steps

        # Shuffle patients with fixed seed for reproducibility
        patient_pool = copy.deepcopy(self._all_patients)
        self._rng.shuffle(patient_pool)
        self._patients = patient_pool

        # Reset state
        self._enrolled_ids = []
        self._contacted_ids = []
        self._opted_out_ids = []
        self._current_idx = 0
        self._step_count = 0
        self._done = False
        self._privacy_violations = 0
        self._total_reward = 0.0
        self._action_history = []
        self._cohort_stats = CohortStats()
        self._last_action_str = ""

        self._grader = Grader(patients=self._patients, trial=self._trial)

        return self._make_observation()

    def step(self, action_str: str) -> StepResult:
        """
        Execute one action in the environment.
        
        Args:
            action_str: Raw action string, e.g. 'screen_eligible',
                       'draft_invite[Hello patient, ...]', 'follow_up',
                       'mark_optout', 'prioritize_next'
        
        Returns:
            StepResult with (observation, reward, done, info)
        """
        if self._done:
            obs = self._make_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "Episode already done."})

        self._step_count += 1
        action = Action.from_string(action_str)
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action.action_type, "error": None}

        # Repeat action penalty (except prioritize_next)
        if action_str == self._last_action_str and action.action_type != "prioritize_next":
            reward += self.R_REPEAT_ACTION
            info["repeat_penalty"] = True
        self._last_action_str = action_str

        current_patient = self._get_current_patient()

        if current_patient is None:
            # No more patients — end episode
            self._done = True
        else:
            reward += self._execute_action(action, current_patient, info)

        # Episode termination conditions
        if self._cohort_stats.total_enrolled >= self._trial.target_cohort_size:
            self._done = True
            info["reason"] = "cohort_full"

        if self._step_count >= self._max_steps:
            self._done = True
            info["reason"] = "max_steps"

        self._total_reward += reward
        self._action_history.append({
            "step": self._step_count,
            "action": action_str,
            "reward": round(reward, 3),
            "patient_id": current_patient.id if current_patient else None,
        })
        # Keep only last 8 in obs (full history in state())
        if len(self._action_history) > 50:
            self._action_history = self._action_history[-50:]

        obs = self._make_observation()
        return StepResult(observation=obs, reward=round(reward, 4), done=self._done, info=info)

    def state(self) -> EnvState:
        """Return complete internal environment state."""
        return EnvState(
            task_name=self._task.name if self._task else "",
            trial=self._trial,
            patients=self._patients,
            current_patient_idx=self._current_idx,
            enrolled_patient_ids=list(self._enrolled_ids),
            contacted_patient_ids=list(self._contacted_ids),
            opted_out_patient_ids=list(self._opted_out_ids),
            privacy_violations=self._privacy_violations,
            total_reward=round(self._total_reward, 4),
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            cohort_stats=self._cohort_stats,
            action_history=list(self._action_history),
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    def grader_score(self, task_name: Optional[str] = None) -> float:
        """Run the grader and return final score for the episode."""
        if self._grader is None:
            return 0.0
        tn = task_name or (self._task.name if self._task else "easy_single_criterion")
        return self._grader.score(
            task_name=tn,
            enrolled_ids=self._enrolled_ids,
            cohort_stats=self._cohort_stats,
            privacy_violations=self._privacy_violations,
        )

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: Action, patient: Patient, info: Dict[str, Any]
    ) -> float:
        """Execute action on current patient and return reward."""
        reward = 0.0

        if action.action_type == "screen_eligible":
            reward = self._action_screen_eligible(patient, info)

        elif action.action_type == "draft_invite":
            reward = self._action_draft_invite(patient, action.personalized_text, info)

        elif action.action_type == "follow_up":
            reward = self._action_follow_up(patient, info)

        elif action.action_type == "mark_optout":
            reward = self._action_mark_optout(patient, info)

        elif action.action_type == "prioritize_next":
            reward = self._action_prioritize_next(patient, info)

        else:
            reward += self.R_INVALID_ACTION
            info["error"] = f"Unknown action: {action.action_type}"

        return reward

    def _action_screen_eligible(self, patient: Patient, info: Dict[str, Any]) -> float:
        """Screen current patient against trial criteria."""
        reward = 0.0
        eligible = _check_basic_eligibility(patient, self._trial)

        if eligible:
            reward += self.R_ELIGIBILITY_MATCH
            info["eligible"] = True
            info["patient_id"] = patient.id
        else:
            info["eligible"] = False
            # Small positive signal for correctly identifying ineligibility
            reward += 0.02

        self._advance_patient()
        return reward

    def _action_draft_invite(
        self, patient: Patient, text: Optional[str], info: Dict[str, Any]
    ) -> float:
        """Draft and send a recruitment invitation to current patient."""
        reward = 0.0

        # Privacy check: contacting opted-out patient
        if patient.opted_out:
            self._privacy_violations += 1
            reward += self.R_PRIVACY_VIOLATION
            info["error"] = "Privacy violation: patient opted out."
            self._advance_patient()
            return reward

        eligible = _check_basic_eligibility(patient, self._trial)
        if not eligible:
            reward += -0.15  # penalty for inviting ineligible patient
            info["error"] = "Invited ineligible patient."
            self._advance_patient()
            return reward

        # Already contacted?
        if patient.id in self._contacted_ids:
            reward += self.R_REPEAT_ACTION
            info["error"] = "Patient already contacted."
            self._advance_patient()
            return reward

        # Personalization bonus (for hard task)
        personalization_bonus = 0.0
        if text and len(text) > 20:
            # Reward for bilingual/personalised content
            has_tamil = any(w in text.lower() for w in ["vanakkam", "nandri", "ungal", "tamil", "icmr"])
            has_personalisation = patient.language.lower() in text.lower() or str(patient.age) in text
            if has_tamil or has_personalisation:
                personalization_bonus = 0.05

        # Mark as contacted and enroll (simplified: invite → enroll)
        patient.contacted = True
        patient.enrolled = True
        self._contacted_ids.append(patient.id)
        self._enrolled_ids.append(patient.id)

        reward += self.R_ENROLL + personalization_bonus
        reward += self._compute_diversity_reward(patient)

        self._update_cohort_stats(patient)
        info["enrolled"] = patient.id
        info["cohort_size"] = self._cohort_stats.total_enrolled

        self._advance_patient()
        return reward

    def _action_follow_up(self, patient: Patient, info: Dict[str, Any]) -> float:
        """Send follow-up to a previously contacted patient."""
        reward = 0.0

        # Privacy check
        if patient.opted_out:
            self._privacy_violations += 1
            reward += self.R_PRIVACY_VIOLATION
            info["error"] = "Privacy violation: patient opted out."
            self._advance_patient()
            return reward

        if patient.id in self._contacted_ids:
            reward += 0.05  # small signal for persistence
            info["follow_up"] = patient.id
        else:
            reward += self.R_INVALID_ACTION
            info["error"] = "Cannot follow up with non-contacted patient."

        self._advance_patient()
        return reward

    def _action_mark_optout(self, patient: Patient, info: Dict[str, Any]) -> float:
        """Record that this patient has opted out."""
        reward = 0.0

        if patient.opted_out and patient.id not in self._opted_out_ids:
            # Correctly recording known opt-out
            self._opted_out_ids.append(patient.id)
            reward += 0.05
            info["opted_out"] = patient.id
        elif patient.id not in self._opted_out_ids:
            # Marking a non-opted-out patient: small reward for defensive privacy
            self._opted_out_ids.append(patient.id)
            reward += 0.01
        else:
            reward += self.R_REPEAT_ACTION

        self._advance_patient()
        return reward

    def _action_prioritize_next(self, patient: Patient, info: Dict[str, Any]) -> float:
        """Skip current patient and advance to next (strategic skip)."""
        # Small cost for skipping (opportunity cost)
        reward = -0.01
        info["skipped"] = patient.id
        self._advance_patient()
        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_diversity_reward(self, patient: Patient) -> float:
        """Compute diversity progress reward for enrolling this patient."""
        reward = 0.0
        stats = self._cohort_stats
        targets = self._trial.diversity_targets
        total = max(stats.total_enrolled, 1)

        # Female progress
        if patient.gender == "female":
            current_female_pct = stats.female_count / total
            if current_female_pct < targets.female_min_pct:
                reward += self.R_DIVERSITY_PROGRESS * 0.5

        # Rural progress
        if patient.location == "rural":
            current_rural_pct = stats.rural_count / total
            if current_rural_pct < targets.rural_min_pct:
                reward += self.R_DIVERSITY_PROGRESS * 0.5

        return reward

    def _update_cohort_stats(self, patient: Patient) -> None:
        """Update cohort statistics after enrolling a patient."""
        s = self._cohort_stats
        s.total_enrolled += 1

        if patient.gender == "female":
            s.female_count += 1
        if patient.location == "rural":
            s.rural_count += 1

        # Age bracket
        if patient.age <= 30:
            s.age_distribution["18-30"] += 1
        elif patient.age <= 45:
            s.age_distribution["31-45"] += 1
        elif patient.age <= 60:
            s.age_distribution["46-60"] += 1
        else:
            s.age_distribution["61+"] += 1

        total = s.total_enrolled
        s.female_pct = round(s.female_count / total, 3) if total else 0.0
        s.rural_pct = round(s.rural_count / total, 3) if total else 0.0
        s.privacy_violations = self._privacy_violations

        # Compute diversity score
        targets = self._trial.diversity_targets
        female_score = min(s.female_pct / targets.female_min_pct, 1.0) if targets.female_min_pct else 1.0
        rural_score = min(s.rural_pct / targets.rural_min_pct, 1.0) if targets.rural_min_pct else 1.0
        filled = sum(1 for v in s.age_distribution.values() if v > 0)
        age_score = filled / 4.0
        s.diversity_score = round((female_score + rural_score + age_score) / 3.0, 3)

    def _get_current_patient(self) -> Optional[Patient]:
        if self._current_idx < len(self._patients):
            return self._patients[self._current_idx]
        return None

    def _advance_patient(self) -> None:
        self._current_idx += 1

    def _make_observation(self) -> Observation:
        current_patient = self._get_current_patient()
        current_patient_dict = None
        if current_patient:
            # Anonymize: remove PII, keep clinical data
            current_patient_dict = {
                "id": current_patient.id,
                "age": current_patient.age,
                "gender": current_patient.gender,
                "location": current_patient.location,
                "ethnicity": current_patient.ethnicity,
                "language": current_patient.language,
                "comorbidities": current_patient.comorbidities,
                "labs": current_patient.labs.model_dump(),
                "opted_out": current_patient.opted_out,
                "contacted": current_patient.contacted,
                "enrolled": current_patient.enrolled,
            }

        trial_dict = self._trial.model_dump() if self._trial else {}

        return Observation(
            trial=trial_dict,
            current_patient=current_patient_dict,
            patient_index=self._current_idx,
            total_patients=len(self._patients),
            cohort_stats=self._cohort_stats,
            target_cohort_size=self._trial.target_cohort_size if self._trial else 20,
            action_history=list(self._action_history[-8:]),
            invites_sent=len(self._contacted_ids),
            follow_ups_sent=sum(
                1 for a in self._action_history if a.get("action", "").startswith("follow_up")
            ),
            opt_outs_recorded=len(self._opted_out_ids),
            steps_remaining=max(0, self._max_steps - self._step_count),
            done=self._done,
            task_name=self._task.name if self._task else "",
        )
