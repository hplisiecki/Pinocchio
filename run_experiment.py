# run_experiment.py
#
# Sends every item from every questionnaire in data/questionnaires.json to a list
# of LLMs via OpenRouter, under three different prompt conditions, and saves all
# responses to data/results.json.
#
# Runs calls concurrently (up to MAX_CONCURRENT at a time) to keep runtime manageable.
# Supports checkpoint/resume: re-running the script skips already-completed calls.
#
# Prerequisites:
#   pip install openai
#   set OPENROUTER_API_KEY=<your key>   (or export on Mac/Linux)
#
# Usage:
#   python run_experiment.py

import asyncio
import json
import os

from openai import AsyncOpenAI

from prompts import (
    get_human_simulation_prompt,
    get_neutral_prompt,
    get_system_prompt,
    load_questionnaires,
)

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

MODELS_PATH = "data/models.json"

# Each condition maps a label to the prompt-building function.
# All three functions share the signature: (scale_prompt, item, pre_prompt=None)
CONDITIONS = {
    "llm_analog": get_system_prompt,
    "neutral": get_neutral_prompt,
    "human_simulation": get_human_simulation_prompt,
}

OUTPUT_PATH = "data/results.json"

# How many API calls to run at the same time. Raise for more speed, lower if you
# hit rate limit errors (429s). 20 is a safe starting point for OpenRouter.
MAX_CONCURRENT = 20


def load_models():
    with open(MODELS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Async OpenRouter client
# ---------------------------------------------------------------------------

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


async def query_model(model: str, prompt: str) -> str:
    """Send a single prompt to a model and return the raw text response."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-OpenRouter-Title": "llm-latent-experiment",
        },
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_results() -> list:
    """Load existing results from disk, or return an empty list if none exist yet."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def build_completed_set(results: list) -> set:
    """
    Build a set of already-completed (questionnaire, condition, item_index, model) tuples.
    Used to skip calls that were already made in a previous run.
    """
    return {
        (r["questionnaire"], r["condition"], r["item_index"], r["model"])
        for r in results
    }


def save_results(results: list) -> None:
    """Write the full results list to disk, overwriting the previous checkpoint."""
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


async def run_experiment():
    questionnaires = load_questionnaires()
    models = load_models()

    # Resume from whatever is already on disk
    results = load_results()
    completed = build_completed_set(results)

    if completed:
        print(
            f"Resuming — {len(completed)} calls already completed, skipping those.\n"
        )

    # Build the full list of pending calls up front so we can show accurate progress
    pending = []
    for questionnaire in questionnaires:
        quest_name = questionnaire["quest"]
        scale_prompt = questionnaire["scale_prompt"]
        pre_prompt = questionnaire["pre_prompt"]
        items = questionnaire["items"]

        for condition_name, prompt_fn in CONDITIONS.items():
            for item_index, item in enumerate(items):
                full_prompt = prompt_fn(scale_prompt, item, pre_prompt)
                for model in models:
                    key = (quest_name, condition_name, item_index, model)
                    if key not in completed:
                        pending.append((key, full_prompt, item))

    total = len(completed) + len(pending)
    done = len(completed)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Protects results list and save from concurrent writes
    lock = asyncio.Lock()

    async def run_one(key, prompt, item):
        nonlocal done
        quest_name, condition_name, item_index, model = key

        async with semaphore:
            try:
                raw_response = await query_model(model, prompt)
            except Exception as e:
                raw_response = f"ERROR: {e}"

        async with lock:
            done += 1
            print(
                f"[{done}/{total}] {quest_name} | {condition_name} | item {item_index + 1} | {model}"
            )
            results.append(
                {
                    "questionnaire": quest_name,
                    "condition": condition_name,
                    "item_index": item_index,
                    "item": item,
                    "model": model,
                    "response": raw_response.strip(),
                }
            )
            # Save after every response so a crash loses at most one call
            save_results(results)

    await asyncio.gather(
        *[run_one(key, prompt, item) for key, prompt, item in pending]
    )

    print(f"\nDone. {len(results)} responses saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
