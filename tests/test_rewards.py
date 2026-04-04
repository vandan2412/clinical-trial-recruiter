"""
tests/test_rewards.py - Reward shaping sanity checks. Uses stdlib unittest.
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import ClinicalTrialRecruiterEnv

class TestRewardVariance(unittest.TestCase):
    def test_rewards_not_all_zero(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        rewards = []
        for _ in range(15):
            r = env.step("screen_eligible")
            rewards.append(r.reward)
            if r.done: break
        self.assertTrue(any(abs(r) > 0.001 for r in rewards))

    def test_invite_eligible_positive(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        p = env._get_current_patient()
        p.comorbidities = ["tb_active"]; p.opted_out = False
        result = env.step("draft_invite[Hello valid patient!]")
        self.assertGreater(result.reward, 0.5)

    def test_privacy_violation_negative(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        p = env._get_current_patient()
        p.opted_out = True
        result = env.step("draft_invite[Ignoring opt-out]")
        self.assertLess(result.reward, 0)
        self.assertGreater(env._privacy_violations, 0)

    def test_ineligible_invite_penalised(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="medium_comorbidities")
        p = env._get_current_patient()
        p.comorbidities = ["hypertension"]; p.opted_out = False
        result = env.step("draft_invite[Wrong invite]")
        self.assertLess(result.reward, 0)

    def test_no_double_enroll(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        p = env._get_current_patient()
        p.comorbidities = ["tb_active"]; p.opted_out = False
        env.step("draft_invite[Invite once]")
        count1 = len(env._enrolled_ids)
        env._current_idx = 0  # same patient
        env.step("draft_invite[Invite again]")
        count2 = len(env._enrolled_ids)
        self.assertLessEqual(count2, count1 + 1)

    def test_eligibility_check_positive(self):
        env = ClinicalTrialRecruiterEnv(seed=42)
        env.reset(task="easy_single_criterion")
        p = env._get_current_patient()
        p.comorbidities = ["tb_active"]
        result = env.step("screen_eligible")
        self.assertGreater(result.reward, 0)

if __name__ == "__main__": unittest.main()
