#This is for huggingface
"""
inference.py — Baseline LLM agent for Humanitarian Aid Allocation OpenEnv

Reads:
  API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
  HF_TOKEN       API key
  MODEL_NAME     Model identifier

Runs the agent against all 3 tasks (easy/medium/hard) and prints scores.
Must complete in < 20 minutes on vcpu=2, 8GB RAM.
"""

import json
import os
import sys
import textwrap
import time
from typing import Dict, List, Optional

from openai import OpenAI

from humanitarian_env import Action, HumanitarianAidEnv

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS_PER_TASK = 25   # hard cap (task horizon ≤ 20)
TEMPERATURE = 0.1
MAX_TOKENS = 256
TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent operating a humanitarian aid allocation system.
Your goal is to distribute limited relief supplies across disaster-affected zones
to maximize survival coverage and minimize waste.

At each step you MUST output ONLY a valid JSON action with these keys:
  "zone_id"  : integer index of the zone to supply (0-based)
  "quantity" : integer units to send (must not exceed remaining_supply)
  "priority" : one of "low", "med", "high"
                (road-blocked zones REQUIRE "high" priority for airdrop delivery)

STRATEGY HINTS:
- Prioritize zones with high severity (σ ≥ 0.8) first.
- Road-blocked zones need priority="high"; otherwise supply is wasted.
- Do not oversend beyond a zone's deficit — it wastes supply.
- Spread allocations across high-severity zones before covering low ones.
- Output ONLY the JSON object, nothing else.
""").strip()


def build_user_prompt(obs_dict: dict, step: int) -> str:
    zones = obs_dict["zones"]
    g = obs_dict["global_state"]

    zone_lines = []
    for z in zones:
        blocked = "ROAD BLOCKED (use priority=high)" if z["road_blocked"] else "accessible"
        covered = "✓ COVERED" if z["covered"] else f"deficit={z['deficit']}"
        zone_lines.append(
            f"  Zone {z['zone_id']}: pop={z['population']} sev={z['severity']:.1f} "
            f"{covered} | {blocked}"
        )

    return textwrap.dedent(f"""
Step {step} | Remaining supply: {g['remaining_supply']} | Steps left: {g['total_steps'] - g['current_step']}
{'⚠ SUPPLY SHOCK APPLIED' if g.get('supply_shock_applied') else ''}

Zones:
{chr(10).join(zone_lines)}

Hints: {obs_dict.get('feasible_actions_hint', '')}

Output your action as JSON only.
""").strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"  TASK: {task.upper()}")
    print(f"{'='*60}")

    env = HumanitarianAidEnv(task=task, seed=42)
    obs = env.reset()
    obs_dict = obs.model_dump()

    total_reward = 0.0
    history: List[str] = []

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        user_prompt = build_user_prompt(obs_dict, step)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # include last 3 history turns for context
        if history:
            history_text = "\n".join(history[-3:])
            messages.insert(1, {
                "role": "user",
                "content": f"Recent actions:\n{history_text}",
            })
            messages.insert(2, {"role": "assistant", "content": "Understood."})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  [Step {step}] LLM call failed: {exc}. Using noop.")
            response_text = '{"zone_id": 0, "quantity": 0, "priority": "med"}'

        # parse JSON action
        try:
            # strip markdown fences if any
            clean = response_text.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(clean)
            action = Action(
                zone_id=int(action_dict.get("zone_id", 0)),
                quantity=int(action_dict.get("quantity", 0)),
                priority=str(action_dict.get("priority", "med")),
            )
        except Exception:
            print(f"  [Step {step}] Bad JSON from model, using noop. Raw: {response_text[:80]}")
            action = Action(zone_id=0, quantity=0, priority="med")

        result = env.step(action)
        obs_dict = result.observation.model_dump()
        total_reward += result.reward.value

        print(
            f"  Step {step:2d}: zone={action.zone_id} qty={action.quantity} pri={action.priority}"
            f" → reward={result.reward.value:+.3f} | supply_left={result.observation.global_state.remaining_supply}"
        )

        history.append(
            f"Step {step}: zone={action.zone_id} qty={action.quantity} pri={action.priority} "
            f"reward={result.reward.value:+.3f}"
        )

        if result.done:
            print(f"  Episode done at step {step}.")
            break

    grader = env.grade()
    print(f"\n  ── GRADER SCORE: {grader.score:.4f} ──")
    print(f"     survival_rate={grader.survival_rate:.3f}  efficiency={grader.efficiency:.3f}  time_bonus={grader.time_bonus}")
    print(f"     {grader.details}")
    print(f"  Total cumulative reward: {total_reward:.4f}")

    return {
        "task": task,
        "score": grader.score,
        "survival_rate": grader.survival_rate,
        "efficiency": grader.efficiency,
        "time_bonus": grader.time_bonus,
        "total_reward": round(total_reward, 4),
        "details": grader.details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Humanitarian Aid Allocation — OpenEnv Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  API_KEY      : {'set ✓' if API_KEY and API_KEY != 'hf_placeholder' else 'NOT SET ✗'}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    t0 = time.time()

    for task in TASKS:
        result = run_episode(client, task)
        results.append(result)

    elapsed = time.time() - t0

    print("\n" + "="*60)
    print("  BASELINE SCORES SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['task']:8s}  score={r['score']:.4f}  "
              f"survival={r['survival_rate']:.3f}  efficiency={r['efficiency']:.3f}")

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print("="*60)

    # write scores JSON for CI/CD validation
    with open("baseline_scores.json", "w") as f:
        json.dump({"results": results, "average": avg, "elapsed_sec": elapsed}, f, indent=2)
    print("\n  Scores written to baseline_scores.json")


if __name__ == "__main__":
    main()
