"""
tests/test_graders.py - Unit tests for all three task graders.
Uses stdlib unittest only.
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graders import (grade_easy_single_criterion, grade_medium_comorbidities,
    grade_hard_diversity, get_ground_truth_eligible, Grader)
from src.models import Patient, Trial, TrialCriteria, DiversityTargets, CohortStats, LabValues

def make_tb_trial():
    return Trial(id="ICMR-TB-TEST", name="TB Trial", disease="tuberculosis", sponsor="ICMR",
        phase="III", target_cohort_size=10,
        criteria=TrialCriteria(age_min=18, age_max=65, required_comorbidities=["tb_active"],
            excluded_comorbidities=["heart_disease"], required_labs={}),
        diversity_targets=DiversityTargets(female_min_pct=0.30, rural_min_pct=0.20),
        description="Test", status="recruiting", window_days=30)

def make_diab_trial():
    return Trial(id="ICMR-DIAB-TEST", name="Diab Trial", disease="diabetes", sponsor="ICMR",
        phase="II", target_cohort_size=10,
        criteria=TrialCriteria(age_min=30, age_max=65, required_comorbidities=["diabetes"],
            excluded_comorbidities=["heart_disease"], required_labs={"hba1c_min":7.0,"hba1c_max":9.5}),
        diversity_targets=DiversityTargets(female_min_pct=0.35, rural_min_pct=0.25),
        description="Test", status="recruiting", window_days=40)

def make_dengue_trial():
    return Trial(id="ICMR-DENGUE-TEST", name="Dengue Trial", disease="dengue", sponsor="ICMR",
        phase="II", target_cohort_size=10,
        criteria=TrialCriteria(age_min=20, age_max=60, required_comorbidities=["dengue_history"],
            excluded_comorbidities=["tb_active","heart_disease"], required_labs={"wbc_max":6.0}),
        diversity_targets=DiversityTargets(female_min_pct=0.30, rural_min_pct=0.20),
        description="Test", status="recruiting", window_days=25)

def mkp(pid, age=35, gender="male", location="urban", comorbidities=None, opted_out=False, hba1c=7.5, wbc=5.0):
    return Patient(id=pid, age=age, gender=gender, location=location, ethnicity="Tamil",
        language="Tamil-English", comorbidities=comorbidities or [],
        labs=LabValues(hba1c=hba1c, wbc=wbc, hemoglobin=12.0), opted_out=opted_out)

class TestEasyGrader(unittest.TestCase):
    def test_perfect_score(self):
        t = make_tb_trial()
        patients = [mkp("P1", comorbidities=["tb_active"]), mkp("P2", comorbidities=["tb_active"]), mkp("P3")]
        self.assertEqual(grade_easy_single_criterion(["P1","P2"], patients, t, 0), 1.0)

    def test_zero_score(self):
        t = make_tb_trial()
        patients = [mkp("P1", comorbidities=["tb_active"]), mkp("P2")]
        self.assertEqual(grade_easy_single_criterion(["P2"], patients, t, 0), 0.0)

    def test_partial_score(self):
        t = make_tb_trial()
        patients = [mkp(f"P{i}", comorbidities=["tb_active"]) for i in range(3)]
        score = grade_easy_single_criterion(["P0"], patients, t, 0)
        self.assertGreater(score, 0.3); self.assertLess(score, 0.4)

    def test_privacy_penalty(self):
        t = make_tb_trial()
        patients = [mkp("P1", comorbidities=["tb_active"])]
        self.assertLess(grade_easy_single_criterion(["P1"], patients, t, 3),
                        grade_easy_single_criterion(["P1"], patients, t, 0))

    def test_score_range(self):
        t = make_tb_trial()
        patients = [mkp(f"P{i}", comorbidities=["tb_active"]) for i in range(5)]
        score = grade_easy_single_criterion(["P0","P1","P2"], patients, t, 0)
        self.assertGreaterEqual(score, 0.0); self.assertLessEqual(score, 1.0)

class TestMediumGrader(unittest.TestCase):
    def test_perfect_enrollment(self):
        t = make_diab_trial()
        patients = [mkp("P1", age=40, comorbidities=["diabetes"], hba1c=7.5),
                    mkp("P2", age=45, comorbidities=["diabetes"], hba1c=8.0),
                    mkp("P3", age=50, comorbidities=["hypertension"])]
        score = grade_medium_comorbidities(["P1","P2"], patients, t, 0)
        self.assertGreater(score, 0.8)

    def test_ineligible_penalty(self):
        t = make_diab_trial()
        patients = [mkp("P1", age=40, comorbidities=["diabetes"], hba1c=7.5),
                    mkp("P2", age=45, comorbidities=["hypertension"])]
        self.assertLess(grade_medium_comorbidities(["P1","P2"], patients, t, 0),
                        grade_medium_comorbidities(["P1"], patients, t, 0))

    def test_excluded_comorbidity(self):
        t = make_diab_trial()
        patients = [mkp("P1", age=40, comorbidities=["diabetes","heart_disease"], hba1c=7.5)]
        self.assertNotIn("P1", get_ground_truth_eligible(patients, t))

    def test_hba1c_out_of_range(self):
        t = make_diab_trial()
        patients = [mkp("P1", age=40, comorbidities=["diabetes"], hba1c=10.5)]
        self.assertNotIn("P1", get_ground_truth_eligible(patients, t))

    def test_score_in_range(self):
        t = make_diab_trial()
        patients = [mkp(f"P{i}", age=40, comorbidities=["diabetes"], hba1c=7.5) for i in range(5)]
        score = grade_medium_comorbidities(["P0","P1"], patients, t, 0)
        self.assertGreaterEqual(score, 0.0); self.assertLessEqual(score, 1.0)

class TestHardGrader(unittest.TestCase):
    def mk_stats(self, enrolled=5, female=2, rural=1):
        return CohortStats(total_enrolled=enrolled, female_count=female, rural_count=rural,
            age_distribution={"18-30":1,"31-45":2,"46-60":2,"61+":0},
            female_pct=round(female/enrolled,3), rural_pct=round(rural/enrolled,3),
            privacy_violations=0, diversity_score=0.7)

    def test_privacy_violation_penalty(self):
        t = make_dengue_trial()
        patients = [mkp("P1", age=30, comorbidities=["dengue_history"], wbc=5.0)]
        stats = self.mk_stats()
        self.assertLess(grade_hard_diversity(["P1"], patients, t, stats, 3),
                        grade_hard_diversity(["P1"], patients, t, stats, 0))

    def test_empty_cohort(self):
        t = make_dengue_trial()
        self.assertEqual(grade_hard_diversity([], [], t, CohortStats(), 0), 0.0)

    def test_score_in_range(self):
        t = make_dengue_trial()
        patients = [mkp("P1", age=30, comorbidities=["dengue_history"], wbc=5.0),
                    mkp("P2", age=35, gender="female", location="rural", comorbidities=["dengue_history"], wbc=4.5)]
        score = grade_hard_diversity(["P1","P2"], patients, t, self.mk_stats(), 0)
        self.assertGreaterEqual(score, 0.0); self.assertLessEqual(score, 1.0)

    def test_diversity_improves_score(self):
        t = make_dengue_trial()
        patients = [mkp("P1", age=30, comorbidities=["dengue_history"], wbc=5.0)]
        lo = CohortStats(total_enrolled=5, female_count=0, rural_count=0,
            age_distribution={"18-30":5,"31-45":0,"46-60":0,"61+":0}, female_pct=0.0, rural_pct=0.0)
        hi = CohortStats(total_enrolled=5, female_count=2, rural_count=1,
            age_distribution={"18-30":1,"31-45":2,"46-60":1,"61+":1}, female_pct=0.4, rural_pct=0.2)
        self.assertGreater(grade_hard_diversity(["P1"], patients, t, hi, 0),
                           grade_hard_diversity(["P1"], patients, t, lo, 0))

class TestGraderInterface(unittest.TestCase):
    def test_unknown_task_raises(self):
        grader = Grader(patients=[mkp("P1")], trial=make_tb_trial())
        with self.assertRaises(ValueError):
            grader.score("unknown_task", [], CohortStats(), 0)

    def test_all_tasks_return_float_in_range(self):
        grader = Grader(patients=[mkp("P1", comorbidities=["tb_active"])], trial=make_tb_trial())
        for task in ["easy_single_criterion","medium_comorbidities","hard_diversity"]:
            score = grader.score(task, ["P1"], CohortStats(total_enrolled=1), 0)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0); self.assertLessEqual(score, 1.0)

if __name__ == "__main__": unittest.main()
