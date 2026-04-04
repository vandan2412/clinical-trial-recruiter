"""
tests/test_env.py - Integration tests for ClinicalTrialRecruiterEnv.
Uses stdlib unittest only.
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import ClinicalTrialRecruiterEnv
from src.models import Observation, StepResult, EnvState

class TestEnvReset(unittest.TestCase):
    def setUp(self):
        self.env = ClinicalTrialRecruiterEnv(seed=42)

    def test_reset_returns_observation(self):
        obs = self.env.reset(task="easy_single_criterion")
        self.assertIsInstance(obs, Observation)
        self.assertEqual(obs.task_name, "easy_single_criterion")
        self.assertGreater(obs.total_patients, 0)
        self.assertIsNotNone(obs.current_patient)

    def test_reset_all_tasks(self):
        for task in ["easy_single_criterion", "medium_comorbidities", "hard_diversity"]:
            obs = self.env.reset(task=task)
            self.assertEqual(obs.task_name, task)
            self.assertFalse(obs.done)

    def test_reset_default_task(self):
        obs = self.env.reset()
        self.assertEqual(obs.task_name, "easy_single_criterion")

    def test_reset_clears_state(self):
        self.env.reset(task="easy_single_criterion")
        self.env.step("screen_eligible")
        self.env.reset(task="easy_single_criterion")
        state = self.env.state()
        self.assertEqual(state.step_count, 0)
        self.assertEqual(len(state.enrolled_patient_ids), 0)

    def test_reset_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            self.env.reset(task="nonexistent_task")

class TestEnvStep(unittest.TestCase):
    def setUp(self):
        self.env = ClinicalTrialRecruiterEnv(seed=42)
        self.env.reset(task="easy_single_criterion")

    def test_step_returns_step_result(self):
        result = self.env.step("screen_eligible")
        self.assertIsInstance(result, StepResult)
        self.assertIsInstance(result.reward, float)
        self.assertIsInstance(result.done, bool)

    def test_screen_eligible_advances_patient(self):
        idx_before = self.env._current_idx
        self.env.step("screen_eligible")
        self.assertEqual(self.env._current_idx, idx_before + 1)

    def test_draft_invite_opted_out_penalizes(self):
        # Find opted-out patient
        for i, p in enumerate(self.env._patients):
            if p.opted_out:
                self.env._current_idx = i
                break
        result = self.env.step("draft_invite[Test]")
        self.assertLess(result.reward, 0)
        self.assertGreater(self.env._privacy_violations, 0)

    def test_episode_ends_when_cohort_full(self):
        self.env.reset(task="easy_single_criterion")
        self.env._trial.target_cohort_size = 2
        for p in self.env._patients:
            p.comorbidities = ["tb_active"]
            p.opted_out = False
        done = False
        for _ in range(50):
            result = self.env.step("draft_invite[Hi]")
            if result.done:
                done = True; break
        self.assertTrue(done)

    def test_max_steps_terminates_episode(self):
        self.env.reset(task="easy_single_criterion")
        self.env._max_steps = 3
        result = None
        for _ in range(5):
            result = self.env.step("screen_eligible")
        self.assertTrue(result.done)

    def test_step_after_done_returns_done(self):
        self.env._done = True
        result = self.env.step("screen_eligible")
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)

    def test_all_actions_accepted(self):
        actions = ["screen_eligible","draft_invite[Hello!]","follow_up","mark_optout","prioritize_next"]
        for action in actions:
            self.env.reset(task="easy_single_criterion")
            result = self.env.step(action)
            self.assertIsInstance(result, StepResult)

    def test_reward_varies_across_trajectory(self):
        self.env.reset(task="easy_single_criterion")
        rewards = []
        for _ in range(15):
            result = self.env.step("screen_eligible")
            rewards.append(result.reward)
            if result.done: break
        # At least some different values
        self.assertGreater(len(set(round(r,3) for r in rewards)), 0)

class TestEnvState(unittest.TestCase):
    def test_state_returns_env_state(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        state = env.state()
        self.assertIsInstance(state, EnvState)
        self.assertEqual(state.task_name, "easy_single_criterion")
        self.assertGreater(len(state.patients), 0)

    def test_state_tracks_steps(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        env.step("screen_eligible")
        env.step("screen_eligible")
        self.assertEqual(env.state().step_count, 2)

class TestGraderScore(unittest.TestCase):
    def test_grader_score_range(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        for task in ["easy_single_criterion","medium_comorbidities","hard_diversity"]:
            env.reset(task=task)
            for _ in range(10):
                result = env.step("screen_eligible")
                if result.done: break
            score = env.grader_score(task)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_grader_score_deterministic(self):
        scores = []
        for _ in range(3):
            env = ClinicalTrialRecruiterEnv(seed=42)
            env.reset(task="easy_single_criterion", seed=42)
            for _ in range(5):
                env.step("screen_eligible")
            scores.append(env.grader_score("easy_single_criterion"))
        self.assertEqual(len(set(scores)), 1, f"Non-deterministic: {scores}")

if __name__ == "__main__": unittest.main()
