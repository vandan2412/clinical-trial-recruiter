# 🏥 ClinicalTrialRecruiter — OpenEnv Environment

> AI agent environment for clinical trial patient recruitment from anonymised EHR data.
> ICMR-compliant. Tamil-English bilingual. Diversity-first.

---

## Motivation

Clinical trial recruitment is one of the most expensive bottlenecks in drug development — responsible for **$1B+ in annual losses** due to delays, poor cohort diversity, and manual screening inefficiency. Up to 86% of trials fail to meet enrollment timelines.

This environment simulates the full recruitment pipeline for **ICMR (Indian Council of Medical Research) trials**, with a focus on:

- Tuberculosis and dengue antiviral trials (high-burden diseases in India)
- Ethical handling of patient opt-outs and privacy
- Diversity-optimised cohort building (gender, geography, age)
- Bilingual outreach (Tamil-English) for underserved populations

Training AI agents on this environment directly produces capability that can be deployed in real-world pharmaceutical settings.

---

## Environment Overview

| Property | Value |
|---|---|
| **Name** | `ClinicalTrialRecruiter` |
| **Version** | 1.0.0 |
| **Tasks** | 3 (easy → medium → hard) |
| **Action space** | Discrete text (5 actions) |
| **Observation space** | Structured JSON (patient EHR + cohort state) |
| **Reward** | Dense, shaped (partial progress signals) |
| **Episode end** | Cohort full OR max steps OR all patients visited |
| **Reproducible** | Yes (seed=42 by default) |

---

## Action Space

| Action | Description |
|---|---|
| `screen_eligible` | Evaluate current patient against trial inclusion/exclusion criteria |
| `draft_invite[<message>]` | Send personalised recruitment invite (bilingual Tamil-English encouraged) |
| `follow_up` | Send follow-up to previously contacted patient |
| `mark_optout` | Record patient has opted out (privacy protection) |
| `prioritize_next` | Skip current patient, advance to next |

**Example action strings:**
```
screen_eligible
draft_invite[Vanakkam! You have been selected for the ICMR TB Trial. Age 39, your profile matches our criteria. Please contact us.]
follow_up
mark_optout
prioritize_next
```

---

## Observation Space

Each step, the agent receives a JSON observation containing:

```json
{
  "trial": {
    "id": "ICMR-TB-2024-001",
    "name": "ICMR Phase III TB Drug Efficacy Trial",
    "criteria": { "age_min": 18, "age_max": 65, "required_comorbidities": ["tb_active"], ... },
    "diversity_targets": { "female_min_pct": 0.30, "rural_min_pct": 0.20 }
  },
  "current_patient": {
    "id": "P034",
    "age": 36,
    "gender": "male",
    "location": "rural",
    "ethnicity": "Hindi",
    "language": "Hindi-English",
    "comorbidities": ["tb_active"],
    "labs": { "hba1c": 5.3, "wbc": 12.5, "hemoglobin": 9.1 },
    "opted_out": false,
    "contacted": false,
    "enrolled": false
  },
  "cohort_stats": {
    "total_enrolled": 4,
    "female_count": 1,
    "rural_count": 2,
    "female_pct": 0.25,
    "rural_pct": 0.50,
    "diversity_score": 0.72,
    "privacy_violations": 0
  },
  "target_cohort_size": 20,
  "steps_remaining": 18,
  "action_history": [...]
}
```

---

## Tasks

### 🟢 Task 1: `easy_single_criterion` (Easy)

**Goal:** Screen patients for an ICMR TB trial using a single criterion: active TB diagnosis (`tb_active` in comorbidities).

**Grader:** Jaccard similarity between enrolled set and ground-truth eligible set.

```
score = |enrolled ∩ eligible| / |enrolled ∪ eligible|
```

**Baseline score:** `0.85`
**Max steps:** 25

---

### 🟡 Task 2: `medium_comorbidities` (Medium)

**Goal:** Build a diabetes cohort. Patients must have `diabetes` AND NOT have `heart_disease`. HbA1c must be in range `[7.0, 9.5]`.

**Grader:** F1 score on enrolled vs eligible, with penalties:
- −0.08 per ineligible patient enrolled
- −0.05 per privacy violation

**Baseline score:** `0.65`
**Max steps:** 35

---

### 🔴 Task 3: `hard_diversity` (Hard)

**Goal:** Build a dengue antiviral cohort meeting strict ICMR diversity requirements:
- ≥30% female
- ≥20% rural
- Age spread across all brackets (18–30, 31–45, 46–60, 61+)
- Respect all patient opt-outs (privacy critical)
- Use personalised bilingual invites

**Grader:** Weighted composite:
```
score = eligibility(0.45) × diversity(0.35) × privacy(0.20)
```

**Penalties:**
- −0.20 per privacy violation (contacting opted-out patient)
- −0.15 if female % < 15% (critical imbalance)
- −0.10 if rural % < 10% (critical imbalance)

**Baseline score:** `0.45`
**Max steps:** 40

---

## Reward Shaping

| Event | Reward |
|---|---|
| Enrol eligible patient | +0.90 |
| Diversity progress (fills gap) | +0.40 |
| Correctly identify eligibility | +0.20 |
| Personalised bilingual invite bonus | +0.05 |
| Privacy violation (contact opted-out) | −0.50 |
| Enrol ineligible patient | −0.15 |
| Cohort imbalance | −0.30 |
| Repeat action | −0.05 |
| Invalid action | −0.05 |

---

## Setup & Usage

### Local

```bash
git clone <your-repo>
cd clinical-trial-recruiter
pip install -r requirements.txt

# Run the server
python app.py

# Or run inference directly (requires HF_TOKEN)
HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

### Docker

```bash
docker build -t clinical-trial-recruiter .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx clinical-trial-recruiter
```

### API Usage

```bash
# Reset for a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy_single_criterion", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "screen_eligible"}'

# Get state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

### Python SDK

```python
from src.env import ClinicalTrialRecruiterEnv

env = ClinicalTrialRecruiterEnv(seed=42)

# Easy task
obs = env.reset(task="easy_single_criterion")
print(obs.current_patient)

result = env.step("screen_eligible")
print(result.reward, result.done)

result = env.step("draft_invite[Hello patient, you are eligible!]")
print(result.reward)

# Get final score
score = env.grader_score()
print(f"Score: {score}")
```

---

## Project Structure

```
clinical-trial-recruiter/
├── src/
│   ├── __init__.py       # Package exports
│   ├── env.py            # ClinicalTrialRecruiterEnv (main environment)
│   ├── models.py         # Pydantic models (Observation, Action, StepResult, etc.)
│   ├── tasks.py          # Task definitions and configs
│   └── graders.py        # Deterministic graders for all 3 tasks
├── data/
│   ├── patients.json     # 120 anonymised patient EHRs (ICMR-style)
│   └── trials.json       # 3 ICMR trial specifications
├── tests/
│   ├── test_graders.py   # Grader unit tests
│   ├── test_env.py       # Environment API tests
│   └── test_rewards.py   # Reward shaping tests
├── app.py                # HF Spaces HTTP server
├── inference.py          # Baseline inference script (OpenAI API)
├── Dockerfile            # Container spec
├── openenv.yaml          # OpenEnv metadata
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Baseline Scores

Achieved with `Qwen/Qwen2.5-72B-Instruct` at temperature 0.2, seed=42:

| Task | Score | Difficulty |
|---|---|---|
| `easy_single_criterion` | **0.85** | Easy |
| `medium_comorbidities` | **0.65** | Medium |
| `hard_diversity` | **0.45** | Hard |

---

## Design Notes

**Why clinical trials?** The domain is clinically significant, under-explored in RL research, and has immediate real-world deployment value. ICMR trials targeting TB and dengue affect millions in India.

**Why ICMR / India-specific?** Indian clinical trials face unique challenges: linguistic diversity (Tamil, Hindi, Bengali, etc.), urban-rural disparities, and under-representation of female and rural patients. Our environment bakes these challenges in.

**Why dense rewards?** Sparse rewards cause slow learning. Every action produces signal — correct screening, diversity progress, privacy respect — allowing frontier models to learn efficiently.

**Privacy mechanics:** The `opted_out` flag is a first-class citizen. Contacting opted-out patients triggers −0.5 reward and increments a violation counter that directly degrades the hard task score.

---

## License

MIT License. Data is fully synthetic and anonymised.
