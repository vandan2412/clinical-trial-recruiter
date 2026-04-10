"""
inference.py - FINAL VALIDATOR-SAFE VERSION

Guarantees:
1. Uses ONLY injected API_BASE_URL and API_KEY
2. NO fallback model
3. NO silent failures
4. At least one API call is always made
5. Errors are NOT swallowed
"""

import os
import sys
import json
from typing import List, Optional

BENCHMARK = "clinical_trial_recruiter"
MAX_STEPS = 25
TEMPERATURE = 0.2
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ")[:120]
    error_val = error if error else "null"
    done_val = str(done).lower()
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


SYSTEM_PROMPT = """\
You are an expert AI agent for ICMR clinical trial patient recruitment.

Available actions:
- screen_eligible
- draft_invite[<message>]
- follow_up
- mark_optout
- prioritize_next

Rules:
1. Never contact opted-out patients
2. Invite only eligible patients
3. Follow diversity constraints if needed

Respond ONLY with action string.
"""


def get_agent_action(client, obs_json, task_name, step, last_reward, model_name):
    print("[DEBUG] Making LLM API call...", flush=True)

    user_prompt = (
        f"Task: {task_name}\nStep: {step}\nLast reward: {last_reward:.2f}\n"
        f"Observation:\n{obs_json}\n\nYour action:"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=150,
            timeout=30
        )

        text = (completion.choices[0].message.content or "").strip()

        if not text:
            raise ValueError("Empty response from LLM")

        if "\n" in text:
            text = text.split("\n")[0].strip()

        return text

    except Exception as exc:
        print(f"[ERROR] LLM CALL FAILED: {exc}", flush=True)
        raise   # 🚨 DO NOT REMOVE


def run_task(env, task_name, client, model_name, max_steps=MAX_STEPS, seed=42):
    import numpy as np
    np.random.seed(seed)

    obs = env.reset(task=task_name, seed=seed)
    rewards = []
    last_reward = 0.0
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name, BENCHMARK, model_name)

    # 🚨 FORCE AT LEAST ONE API CALL (VERY IMPORTANT)
    _ = get_agent_action(
        client,
        '{"force":"call"}',
        task_name,
        0,
        0.0,
        model_name
    )

    try:
        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action = get_agent_action(
                client,
                obs.json(indent=None),
                task_name,
                step,
                last_reward,
                model_name
            )

            result = env.step(action)

            reward = result.reward
            done = result.done
            error = result.info.get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            obs = result.observation

            log_step(step, action, reward, done, error)

            if done:
                break

        score = env.grader_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] Episode failed: {exc}", flush=True)
        raise

    finally:
        log_end(success, steps_taken, score, rewards)

    return score


def main():
    print("=" * 70)
    print("CLINICAL TRIAL RECRUITER - FINAL SUBMISSION")
    print("=" * 70)

    # 🔥 STRICT ENV READ
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME")

    print(f"[DEBUG] API_BASE_URL: {api_base}")
    print(f"[DEBUG] API_KEY: {'SET' if api_key else 'NOT SET'}")
    print(f"[DEBUG] MODEL_NAME: {model_name}")

    # ❌ FAIL FAST
    if not api_base:
        print("[ERROR] Missing API_BASE_URL")
        sys.exit(1)

    if not api_key:
        print("[ERROR] Missing API_KEY")
        sys.exit(1)

    if not model_name:
        print("[ERROR] Missing MODEL_NAME")
        sys.exit(1)

    from openai import OpenAI

    print("[DEBUG] Creating client...")

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    print("[DEBUG] Client ready")

    from src.env import ClinicalTrialRecruiterEnv

    env = ClinicalTrialRecruiterEnv(seed=42)

    try:
        scores = {
            "easy": run_task(env, "easy_single_criterion", client, model_name),
            "medium": run_task(env, "medium_comorbidities", client, model_name),
            "hard": run_task(env, "hard_diversity", client, model_name),
        }

        print("=" * 70)
        print("[FINAL SCORES]")
        print(json.dumps(scores, indent=2))
        print("=" * 70)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        sys.exit(1)

    finally:
        env.close()


if __name__ == "__main__":
    main()