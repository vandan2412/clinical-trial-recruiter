"""
inference.py - BULLETPROOF VERSION for OpenEnv Phase 2

THIS VERSION GUARANTEES:
1. NO defaults for API credentials
2. NO fallbacks to other providers
3. Explicit validation that env vars are set
4. All API calls go through injected proxy
5. Clear logging of what's being used

CRITICAL: This will fail LOUD if env vars are missing
(Better to fail with error message than silently use wrong endpoint)
"""

import os
import sys
import json
from typing import List, Optional

# =============================================================================
# PHASE 2: STRICT ENVIRONMENT VALIDATION
# =============================================================================
# DO NOT read API credentials at module level
# DO NOT provide defaults
# DO NOT use other providers as fallback
# =============================================================================

# These can have defaults because they're not critical credentials:
BENCHMARK = "clinical_trial_recruiter"
MAX_STEPS = 25
TEMPERATURE = 0.2
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task: str, env_name: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    """Log each step."""
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
    """Log episode end."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """\
You are an expert AI agent for ICMR clinical trial patient recruitment.

Available actions:
- screen_eligible: Check if patient meets trial criteria
- draft_invite[<message>]: Invite eligible patient
- follow_up: Follow up with contacted patient
- mark_optout: Record patient opted out
- prioritize_next: Skip to next patient

CRITICAL RULES:
1. Check opted_out flag BEFORE any contact action
2. Only invite patients who passed screen_eligible
3. For hard_diversity: prioritize women/rural, use Tamil-English invites
4. Never contact opted-out patients

Respond with ONLY the action string. No preamble. No explanation.
Examples: screen_eligible, draft_invite[message], follow_up, mark_optout, prioritize_next
"""


def get_agent_action(
    client,  # OpenAI client initialized with injected credentials
    obs_json: str,
    task_name: str,
    step: int,
    last_reward: float,
    model_name: str,
) -> str:
    """
    Get action from LLM via the INJECTED PROXY.
    
    CRITICAL: The client MUST be initialized with:
    - base_url = os.environ["API_BASE_URL"]
    - api_key = os.environ["API_KEY"]
    
    This function makes an actual API call through the proxy.
    """
    user_prompt = (
        f"Task: {task_name}\nStep: {step}\nLast reward: {last_reward:.2f}\n"
        f"Observation:\n{obs_json}\n\nYour action:"
    )

    try:
        # THIS CALL GOES THROUGH THE INJECTED PROXY
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=150,
        )

        text = (completion.choices[0].message.content or "").strip()

        if not text:
            return "screen_eligible"

        # Extract first line if multiline
        if "\n" in text:
            text = text.split("\n")[0].strip()

        return text

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "screen_eligible"


def run_task(env, task_name: str, client, model_name: str,
             max_steps: int = MAX_STEPS, seed: int = 42) -> float:
    """Run single task episode."""
    import numpy as np
    np.random.seed(seed)

    obs = env.reset(task=task_name, seed=seed)
    rewards: List[float] = []
    last_reward = 0.0
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env_name=BENCHMARK, model=model_name)

    try:
        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action_str = get_agent_action(
                client, obs.json(indent=None), task_name, step, last_reward, model_name
            )

            result = env.step(action_str)
            reward = result.reward
            done = result.done
            error = result.info.get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            obs = result.observation

            log_step(step=step, action=action_str, reward=reward,
                    done=done, error=error)

            if done:
                break

        score = env.grader_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
               score=score, rewards=rewards)

    return score


# =============================================================================
# MAIN ENTRY POINT - WHERE ENV VARS ARE READ
# =============================================================================

def main() -> None:
    """
    Main execution.
    
    CRITICAL CHECKS:
    1. Read API_BASE_URL from os.environ (no default)
    2. Read API_KEY from os.environ (no default, no fallback)
    3. Validate both are present
    4. Create OpenAI client with validated values
    5. Make API calls through the client
    """

    print("=" * 70, flush=True)
    print("CLINICAL TRIAL RECRUITER - PHASE 2 SUBMISSION", flush=True)
    print("=" * 70, flush=True)

    # =========================================================================
    # STEP 1: READ FROM ENVIRONMENT (NO DEFAULTS, NO FALLBACKS)
    # =========================================================================

    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    print(f"[DEBUG] API_BASE_URL from env: {api_base}", flush=True)
    print(f"[DEBUG] API_KEY from env: {'SET' if api_key else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME from env: {model_name}", flush=True)

    # =========================================================================
    # STEP 2: STRICT VALIDATION - FAIL FAST IF MISSING
    # =========================================================================

    if not api_base:
        print("[ERROR] API_BASE_URL not set in environment", flush=True)
        print("[ERROR] Validator must inject this variable", flush=True)
        sys.exit(1)

    if not api_key:
        print("[ERROR] API_KEY not set in environment", flush=True)
        print("[ERROR] Validator must inject this variable", flush=True)
        sys.exit(1)

    print("[DEBUG] ✓ All required environment variables are set", flush=True)

    # =========================================================================
    # STEP 3: IMPORT OpenAI AFTER VALIDATION
    # =========================================================================

    from openai import OpenAI

    # =========================================================================
    # STEP 4: CREATE CLIENT WITH INJECTED CREDENTIALS
    # =========================================================================

    print(f"[DEBUG] Creating OpenAI client with base_url={api_base}", flush=True)

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    print("[DEBUG] ✓ OpenAI client initialized", flush=True)

    # =========================================================================
    # STEP 5: IMPORT ENVIRONMENT
    # =========================================================================

    from src.env import ClinicalTrialRecruiterEnv

    env = ClinicalTrialRecruiterEnv(seed=42)

    print("[DEBUG] ✓ Environment initialized", flush=True)
    print("[DEBUG] Starting task execution...", flush=True)
    print("=" * 70, flush=True)

    # =========================================================================
    # STEP 6: RUN TASKS - EACH WILL MAKE API CALLS THROUGH THE PROXY
    # =========================================================================

    try:
        score_easy = run_task(
            env, "easy_single_criterion", client, model_name, seed=42
        )
        score_medium = run_task(
            env, "medium_comorbidities", client, model_name, seed=42
        )
        score_hard = run_task(
            env, "hard_diversity", client, model_name, seed=42
        )

        scores = {
            "easy_single_criterion": score_easy,
            "medium_comorbidities": score_medium,
            "hard_diversity": score_hard,
        }

        print("=" * 70, flush=True)
        print("[DEBUG] EXECUTION COMPLETE", flush=True)
        print(f"[DEBUG] FINAL SCORES: {json.dumps(scores, indent=2)}", flush=True)
        print("=" * 70, flush=True)

    except Exception as exc:
        print(f"[ERROR] Execution failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        env.close()


if __name__ == "__main__":
    main()
