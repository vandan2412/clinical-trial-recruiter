"""
inference.py - Baseline inference script for ClinicalTrialRecruiter.
Strictly follows the OpenEnv sample inference.py template.

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import sys
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — strictly per checklist:
# Defaults set ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # No default — must be set in env
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")    # Optional: for from_docker_image()

# OpenAI client uses HF_TOKEN as the API key
API_KEY = HF_TOKEN or os.getenv("API_KEY")

BENCHMARK               = "clinical_trial_recruiter"
MAX_STEPS               = 20
TEMPERATURE             = 0.2
SUCCESS_SCORE_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# OpenAI client — configured via environment variables per checklist
# ---------------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Required stdout logging — exact format per spec
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ")[:120]
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert AI agent for ICMR clinical trial patient recruitment.

Available actions (respond with EXACTLY ONE per turn, no explanation):
  screen_eligible               - Check if current patient meets trial criteria
  draft_invite[<message>]       - Invite eligible patient (use Tamil-English message)
  follow_up                     - Follow up with previously contacted patient
  mark_optout                   - Record patient opted out (privacy protection)
  prioritize_next               - Skip current patient, move to next

Rules:
- Always check opted_out=true before any contact action
- Only invite patients who are eligible per trial criteria
- For hard_diversity: balance female/rural patients, use bilingual Tamil-English invites
- Respond with the action string ONLY. No preamble. No explanation.
"""


def get_agent_action(obs_json: str, task_name: str,
                     step: int, last_reward: float) -> str:
    """Query LLM for next action using OpenAI client."""
    user_prompt = (
        f"Task: {task_name}\nStep: {step}\nLast reward: {last_reward:.2f}\n"
        f"Observation:\n{obs_json}\n\nYour action:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "screen_eligible"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "screen_eligible"


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------
def run_task(env, task_name: str, max_steps: int = MAX_STEPS,
             seed: int = 42) -> float:
    """Run one episode. Returns final grader score in [0.0, 1.0]."""
    import numpy as np
    np.random.seed(seed)

    obs         = env.reset(task=task_name, seed=seed)
    rewards: List[float] = []
    last_reward = 0.0
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action_str = get_agent_action(
                obs.json(indent=None), task_name, step, last_reward
            )

            result      = env.step(action_str)
            reward      = result.reward
            done        = result.done
            error       = result.info.get("error")

            rewards.append(reward)
            steps_taken  = step
            last_reward  = reward
            obs          = result.observation

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error)

            if done:
                break

        score   = env.grader_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main — three explicit task runs
# ---------------------------------------------------------------------------
def main() -> None:
    from src.env import ClinicalTrialRecruiterEnv

    env = ClinicalTrialRecruiterEnv(seed=42)

    score_easy   = run_task(env, "easy_single_criterion",  seed=42)
    score_medium = run_task(env, "medium_comorbidities",   seed=42)
    score_hard   = run_task(env, "hard_diversity",         seed=42)

    scores = {
        "easy_single_criterion": score_easy,
        "medium_comorbidities":  score_medium,
        "hard_diversity":        score_hard,
    }
    print("TASK SCORES:", scores, flush=True)
    env.close()


if __name__ == "__main__":
    main()
