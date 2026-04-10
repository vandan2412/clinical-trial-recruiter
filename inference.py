"""
inference.py - Baseline inference script for ClinicalTrialRecruiter.

Phase 2 FIXED: Properly uses injected API_BASE_URL and API_KEY from environment.
Critical changes:
1. Initialize OpenAI client with os.environ variables (NOT hardcoded)
2. Make actual LLM API calls through the injected proxy
3. Validator tracks all API calls - they MUST go through the proxy

STDOUT FORMAT:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# CRITICAL PHASE 2 FIX
# ---------------------------------------------------------------------------
# DO NOT set default values - let environment injection work
# API_BASE_URL and API_KEY MUST come from os.environ (injected by validator)
# DO NOT fall back to HuggingFace or other endpoints

BENCHMARK = "clinical_trial_recruiter"
MAX_STEPS = 25
TEMPERATURE = 0.2
SUCCESS_SCORE_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Stdout logging — exact required format
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    """Log episode start with task and model info."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    """Log each step: action, reward, done status, any error."""
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
    """Log episode end: success, final score, all rewards."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt for LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI agent for ICMR clinical trial patient recruitment.

Available actions (respond with EXACTLY ONE per turn, no explanation):

screen_eligible - Check if current patient meets trial criteria
draft_invite[<message>] - Invite eligible patient (use Tamil-English message for hard task)
follow_up - Follow up with previously contacted patient  
mark_optout - Record patient opted out (privacy protection)
prioritize_next - Skip current patient, move to next

Critical rules:
- Always check opted_out=true flag BEFORE any contact action (draft_invite, follow_up)
- Only invite patients who passed screen_eligible check
- For hard_diversity: prioritize women and rural patients, use bilingual Tamil-English invites
- Never contact opted-out patients (massive penalty for privacy violation)
- Respond with action string ONLY. No preamble. No explanation. No markdown.

Examples of valid responses:
screen_eligible
draft_invite[Vanakkam! You are eligible for our trial. Please contact us.]
follow_up
mark_optout
prioritize_next
"""


def get_agent_action(client: OpenAI, obs_json: str, task_name: str,
                     step: int, last_reward: float, model_name: str) -> str:
    """
    Query LLM via the INJECTED proxy.
    
    CRITICAL: This function MUST make an actual API call through the proxy
    that the validator provides. The validator checks that:
    1. We use os.environ["API_BASE_URL"] as base_url
    2. We use os.environ["API_KEY"] as api_key
    3. We make ACTUAL API calls (tracked via proxy)
    
    If no calls are made, Phase 2 fails with "No API calls were made".
    """
    user_prompt = (
        f"Task: {task_name}\nStep: {step}\nLast reward: {last_reward:.2f}\n"
        f"Observation:\n{obs_json}\n\nYour action:"
    )

    try:
        # THIS CALL MUST GO THROUGH THE PROXY
        # The client was initialized with os.environ["API_BASE_URL"] and os.environ["API_KEY"]
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
        
        # Validate response is a valid action
        if not text:
            return "screen_eligible"
        
        # If response is too long, extract first line
        if "\n" in text:
            text = text.split("\n")[0].strip()
        
        return text

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "screen_eligible"


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(env, task_name: str, client: OpenAI, model_name: str,
             max_steps: int = MAX_STEPS, seed: int = 42) -> float:
    """
    Run one complete episode for a task.
    
    Returns: grader_score (float in [0.0, 1.0])
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
            # Check episode termination
            if obs.done:
                break

            # CRITICAL: Get action from LLM via PROXY
            action_str = get_agent_action(
                client, obs.json(indent=None), task_name, step, last_reward, model_name
            )

            # Execute action in environment
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

        # Get final score from grader
        score = env.grader_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
               score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point. Runs all 3 tasks.
    
    CRITICAL PHASE 2 CHECKS:
    1. OpenAI client MUST be initialized with:
       - base_url = os.environ["API_BASE_URL"]
       - api_key = os.environ["API_KEY"]
    2. Model name from os.environ["MODEL_NAME"]
    3. Each LLM call MUST go through the proxy (validator tracks this)
    4. Output must follow [START], [STEP], [END] format
    """
    from src.env import ClinicalTrialRecruiterEnv
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    # Log what we're using (for debugging)
    print(f"[DEBUG] Using API_BASE_URL: {api_base}", flush=True)
    print(f"[DEBUG] Using API_KEY: {'SET' if api_key else 'NOT SET'}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME: {model_name}", flush=True)

    if not api_base or not api_key:
        print("[ERROR] API_BASE_URL and API_KEY must be set in environment", flush=True)
        sys.exit(1)

    # Initialize OpenAI client with INJECTED credentials
    # DO NOT use defaults, DO NOT use other providers
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    # Initialize environment
    env = ClinicalTrialRecruiterEnv(seed=42)

    # Run all 3 tasks
    print("[DEBUG] Starting Phase 2 submission with API proxy", flush=True)
    
    score_easy = run_task(env, "easy_single_criterion", client, model_name, seed=42)
    score_medium = run_task(env, "medium_comorbidities", client, model_name, seed=42)
    score_hard = run_task(env, "hard_diversity", client, model_name, seed=42)

    scores = {
        "easy_single_criterion": score_easy,
        "medium_comorbidities": score_medium,
        "hard_diversity": score_hard,
    }

    print(f"[DEBUG] FINAL SCORES: {json.dumps(scores)}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
