"""
inference.py - Baseline inference script for ClinicalTrialRecruiter.
Phase 2 fix: Must use injected API_BASE_URL and API_KEY from environment.
DO NOT hardcode keys or use other providers.

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
# PHASE 2 FIX: Use injected API_BASE_URL and API_KEY from validator proxy
# Initialize client INSIDE main() so env vars are fully loaded at runtime
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# API_KEY must come from injected environment — validator checks this is used
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

BENCHMARK               = "clinical_trial_recruiter"
MAX_STEPS               = 20
TEMPERATURE             = 0.2
SUCCESS_SCORE_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Stdout logging — exact required format
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
# System prompt
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


def get_agent_action(client: OpenAI, obs_json: str, task_name: str,
                     step: int, last_reward: float) -> str:
    """Query LLM via injected proxy — API_BASE_URL and API_KEY from env."""
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
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "screen_eligible"


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------
def run_task(env, task_name: str, client: OpenAI,
             max_steps: int = MAX_STEPS, seed: int = 42) -> float:
    """Run one episode. Returns grader score in [0.0, 1.0]."""
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
                client, obs.json(indent=None), task_name, step, last_reward
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from src.env import ClinicalTrialRecruiterEnv

    # CRITICAL: Initialize OpenAI client with injected proxy credentials
    # Validator checks that API_BASE_URL and API_KEY from env are used
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key  = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

    print(f"[DEBUG] API_BASE_URL={api_base}", flush=True)
    print(f"[DEBUG] API_KEY set: {bool(api_key)}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    # Use injected proxy — do NOT hardcode or use own credentials
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    env = ClinicalTrialRecruiterEnv(seed=42)

    # Three explicit task runs
    score_easy   = run_task(env, "easy_single_criterion",  client, seed=42)
    score_medium = run_task(env, "medium_comorbidities",   client, seed=42)
    score_hard   = run_task(env, "hard_diversity",         client, seed=42)

    scores = {
        "easy_single_criterion": score_easy,
        "medium_comorbidities":  score_medium,
        "hard_diversity":        score_hard,
    }
    print("TASK SCORES:", scores, flush=True)
    env.close()


if __name__ == "__main__":
    main()
