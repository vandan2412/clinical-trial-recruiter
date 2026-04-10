"""
inference.py - FINAL HARDENED VERSION for OpenEnv Phase 2

THIS VERSION GUARANTEES:
1. NO defaults for API credentials OR model name
2. NO silent fallbacks - all errors propagate loudly
3. Explicit validation that ALL env vars are set
4. Smoke test verifies proxy BEFORE tasks run
5. All API calls go through injected proxy
6. Full tracebacks on every failure path

CRITICAL: This will fail LOUD if anything is wrong.
(Better to fail with a clear error than silently make zero API calls)
"""

import os
import sys
import json
import traceback
from typing import List, Optional

# =============================================================================
# PHASE 2: STRICT ENVIRONMENT VALIDATION
# =============================================================================
# DO NOT read API credentials at module level
# DO NOT provide defaults for credentials or model name
# DO NOT use other providers as fallback
# DO NOT swallow exceptions silently
# =============================================================================

# Non-credential constants only:
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
    client,
    obs_json: str,
    task_name: str,
    step: int,
    last_reward: float,
    model_name: str,
) -> str:
    """
    Get action from LLM via the INJECTED PROXY.

    NO silent fallback. If the API call fails, the exception propagates
    so the caller sees the real error. The validator MUST see API traffic.
    """
    user_prompt = (
        f"Task: {task_name}\nStep: {step}\nLast reward: {last_reward:.2f}\n"
        f"Observation:\n{obs_json}\n\nYour action:"
    )

    # THIS CALL GOES THROUGH THE INJECTED PROXY - NO TRY/EXCEPT, NO FALLBACK
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
        print(f"[WARN] Empty LLM response at step {step}, defaulting to screen_eligible", flush=True)
        return "screen_eligible"

    # Extract first line if multiline
    if "\n" in text:
        text = text.split("\n")[0].strip()

    print(f"[DEBUG] LLM action at step {step}: {text}", flush=True)
    return text


def obs_to_str(obs) -> str:
    """
    Safely convert observation to string.
    Handles objects with .json(), dicts, or anything else.
    """
    try:
        return obs.json(indent=None)
    except AttributeError:
        pass
    try:
        return json.dumps(obs)
    except (TypeError, ValueError):
        pass
    return str(obs)


def run_task(env, task_name: str, client, model_name: str,
             max_steps: int = MAX_STEPS, seed: int = 42) -> float:
    """
    Run single task episode.

    API errors from get_agent_action are NOT caught here - they propagate
    to main() so the run fails loud instead of completing with zero API calls.
    """
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

            obs_str = obs_to_str(obs)

            # NO try/except here - API failures must propagate loudly
            action_str = get_agent_action(
                client, obs_str, task_name, step, last_reward, model_name
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
        print(f"[ERROR] Episode '{task_name}' failed at step {steps_taken}: {exc}", flush=True)
        traceback.print_exc()
        raise  # propagate - do not hide failures

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main execution with strict validation at every step.
    Nothing is silent. Everything is logged. Errors always propagate.
    """

    print("=" * 70, flush=True)
    print("CLINICAL TRIAL RECRUITER - PHASE 2 SUBMISSION", flush=True)
    print("=" * 70, flush=True)

    # =========================================================================
    # STEP 1: READ ALL ENV VARS - NO DEFAULTS FOR CREDENTIALS OR MODEL NAME
    # =========================================================================

    api_base   = os.environ.get("API_BASE_URL")
    api_key    = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME")   # NO DEFAULT - must be injected

    print(f"[DEBUG] API_BASE_URL  : {api_base}", flush=True)
    print(f"[DEBUG] API_KEY       : {'SET' if api_key else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME    : {model_name}", flush=True)

    # =========================================================================
    # STEP 2: STRICT VALIDATION - FAIL FAST IF ANYTHING IS MISSING
    # =========================================================================

    missing = []
    if not api_base:
        missing.append("API_BASE_URL")
    if not api_key:
        missing.append("API_KEY")
    if not model_name:
        missing.append("MODEL_NAME")

    if missing:
        for var in missing:
            print(f"[ERROR] {var} is not set in environment", flush=True)
        print("[ERROR] Validator must inject all required variables", flush=True)
        sys.exit(1)

    print("[DEBUG] ✓ All required environment variables are set", flush=True)

    # =========================================================================
    # STEP 3: IMPORT openai - FAIL LOUD IF NOT INSTALLED OR WRONG VERSION
    # =========================================================================

    try:
        from openai import OpenAI
        import openai as _oai
        print(f"[DEBUG] ✓ openai package imported (version: {_oai.__version__})", flush=True)
    except ImportError as exc:
        print(f"[ERROR] Failed to import openai: {exc}", flush=True)
        print("[ERROR] Install with: pip install openai>=1.0.0", flush=True)
        sys.exit(1)

    # =========================================================================
    # STEP 4: CREATE CLIENT WITH INJECTED CREDENTIALS
    # =========================================================================

    print(f"[DEBUG] Initializing OpenAI client with base_url={api_base}", flush=True)

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    print("[DEBUG] ✓ OpenAI client initialized", flush=True)

    # =========================================================================
    # STEP 4.5: SMOKE TEST - VERIFY PROXY IS REACHABLE BEFORE ANYTHING ELSE
    # =========================================================================

    print("[DEBUG] Running proxy smoke test...", flush=True)

    try:
        test_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=10,
            temperature=0.0,
        )
        test_text = (test_response.choices[0].message.content or "").strip()
        print(f"[DEBUG] ✓ Proxy smoke test PASSED. Response: '{test_text}'", flush=True)
    except Exception as exc:
        print(f"[ERROR] Proxy smoke test FAILED: {exc}", flush=True)
        print(f"[ERROR] base_url   = {api_base}", flush=True)
        print(f"[ERROR] model_name = {model_name}", flush=True)
        print("[ERROR] Check that API_BASE_URL is correct and model is registered on the proxy", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 5: IMPORT ENVIRONMENT - SEPARATE IMPORT vs INIT FOR CLEAR ERRORS
    # =========================================================================

    print("[DEBUG] Importing ClinicalTrialRecruiterEnv...", flush=True)

    try:
        from src.env import ClinicalTrialRecruiterEnv
        print("[DEBUG] ✓ src.env module imported", flush=True)
    except ImportError as exc:
        print(f"[ERROR] Cannot import ClinicalTrialRecruiterEnv: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    try:
        env = ClinicalTrialRecruiterEnv(seed=42)
        print("[DEBUG] ✓ Environment instantiated", flush=True)
    except Exception as exc:
        print(f"[ERROR] ClinicalTrialRecruiterEnv() raised: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("[DEBUG] Starting task execution...", flush=True)
    print("=" * 70, flush=True)

    # =========================================================================
    # STEP 6: RUN ALL TASKS - EVERY STEP MAKES A REAL API CALL THROUGH PROXY
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
        print("[DEBUG] ALL TASKS COMPLETE", flush=True)
        print(f"[DEBUG] FINAL SCORES: {json.dumps(scores, indent=2)}", flush=True)
        print("=" * 70, flush=True)

    except Exception as exc:
        print(f"[ERROR] Task execution failed: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        env.close()


if __name__ == "__main__":
    main()