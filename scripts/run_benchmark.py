"""Run benchmark: send each question + dataset to each model via provider SDKs."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "prepared"
Q_DIR = BASE_DIR / "questions"
RESULTS_DIR = BASE_DIR / "results"

# Each run config is fully explicit: name -> provider + native kwargs.
# Names use "model@effort" convention for easy grouping in scoring.
CONFIGS = {
    # Claude Opus 4.6: adaptive thinking + effort via output_config
    "claude-opus-4.6@low": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
        "max_tokens": 16000,
    },
    "claude-opus-4.6@medium": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "max_tokens": 16000,
    },
    "claude-opus-4.6@high": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "max_tokens": 16000,
    },
    "claude-opus-4.6@max": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
        "max_tokens": 32000,
    },

    # GPT-5.1: supports none, low, medium, high
    "gpt-5.1@low": {
        "provider": "openai",
        "model": "gpt-5.1",
        "reasoning": {"effort": "low", "summary": "auto"},
    },
    "gpt-5.1@medium": {
        "provider": "openai",
        "model": "gpt-5.1",
        "reasoning": {"effort": "medium", "summary": "auto"},
    },
    "gpt-5.1@high": {
        "provider": "openai",
        "model": "gpt-5.1",
        "reasoning": {"effort": "high", "summary": "auto"},
    },

    # GPT-5.2: supports none, low, medium, high, xhigh
    "gpt-5.2@low": {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning": {"effort": "low", "summary": "auto"},
    },
    "gpt-5.2@medium": {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning": {"effort": "medium", "summary": "auto"},
    },
    "gpt-5.2@high": {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning": {"effort": "high", "summary": "auto"},
    },
    "gpt-5.2@xhigh": {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning": {"effort": "xhigh", "summary": "auto"},
    },

    # Gemini 3.1 Pro: thinking_level controls reasoning depth
    "gemini-3.1-pro@low": {
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
        "thinking_level": "low",
    },
    "gemini-3.1-pro@high": {
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
        "thinking_level": "high",
    },
}

SYSTEM_PROMPT = (
    "Answer the user's question about the provided data. Do not use tools -- "
    "just examine the data directly and provide your answer.\n"
    "End your response with exactly:\n"
    "ANSWER: <your answer>\n"
    "For numbers, no commas or units. For text, exact match."
)

ALL_DATASETS = ["census_southeast", "census_national", "imdb_top", "pokemon"]

# Lazy-initialized SDK clients
_anthropic_client = None
_openai_client = None
_gemini_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI()
    return _openai_client


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client


async def call_anthropic(model_cfg, user_content):
    """Call Anthropic API via native SDK. Returns (content, reasoning, usage)."""
    client = get_anthropic_client()
    kwargs = {}
    if "thinking" in model_cfg:
        kwargs["thinking"] = model_cfg["thinking"]
    if "output_config" in model_cfg:
        kwargs["output_config"] = model_cfg["output_config"]
    if "max_tokens" in model_cfg:
        kwargs["max_tokens"] = model_cfg["max_tokens"]

    resp = await client.messages.create(
        model=model_cfg["model"],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        timeout=2000,
        **kwargs,
    )

    # Extract content and reasoning from response blocks
    content = ""
    reasoning = None
    for block in resp.content:
        if block.type == "thinking":
            reasoning = block.thinking
        elif block.type == "text":
            content = block.text

    usage = {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }
    return content, reasoning, usage


async def call_openai(model_cfg, user_content):
    """Call OpenAI Responses API via native SDK. Returns (content, reasoning, usage)."""
    client = get_openai_client()
    kwargs = {}
    if "reasoning" in model_cfg:
        kwargs["reasoning"] = model_cfg["reasoning"]

    resp = await client.responses.create(
        model=model_cfg["model"],
        instructions=SYSTEM_PROMPT,
        input=user_content,
        timeout=2000,
        **kwargs,
    )

    # Extract content and reasoning summary from response output
    content = ""
    reasoning = None
    for item in resp.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    content += part.text
        elif item.type == "reasoning":
            summaries = []
            for s in item.summary:
                if s.type == "summary_text":
                    summaries.append(s.text)
            if summaries:
                reasoning = "\n".join(summaries)

    usage = {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }
    return content, reasoning, usage


async def call_gemini(model_cfg, user_content):
    """Call Google Gemini API via native SDK. Returns (content, reasoning, usage)."""
    client = get_gemini_client()

    thinking_kwargs = {"include_thoughts": True}
    if "thinking_level" in model_cfg:
        thinking_kwargs["thinking_level"] = model_cfg["thinking_level"]

    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        thinking_config=genai_types.ThinkingConfig(**thinking_kwargs),
    )

    resp = await client.aio.models.generate_content(
        model=model_cfg["model"],
        contents=user_content,
        config=config,
    )

    # Extract content and thinking from response
    content = ""
    reasoning = None
    if resp.candidates:
        for part in resp.candidates[0].content.parts:
            if part.thought:
                reasoning = part.text
            else:
                content += part.text

    usage = {
        "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
    }
    return content, reasoning, usage


def load_completed(results_file):
    """Load already-completed (question_id, model_name) pairs from JSONL."""
    completed = set()
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            completed.add((rec["question_id"], rec["model_name"]))
    return completed


async def call_model(run_name, model_cfg, question_text, json_data, semaphore, max_retries=3, answer_type=None):
    """Dispatch to provider-specific function. Returns None on total failure."""
    user_content = f"Question: {question_text}\n\nHere is the dataset:\n{json_data}"
    if answer_type == "table":
        user_content += (
            "\n\nIMPORTANT: Your answer must be a valid JSON object (dictionary). "
            "Use ANSWER: followed by the JSON object. Example: ANSWER: {\"key1\": value1, \"key2\": value2}"
        )
    provider = model_cfg["provider"]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                if provider == "anthropic":
                    return await call_anthropic(model_cfg, user_content)
                elif provider == "openai":
                    return await call_openai(model_cfg, user_content)
                elif provider == "gemini":
                    return await call_gemini(model_cfg, user_content)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            wait = 2 ** (attempt + 1)
            if attempt < max_retries - 1:
                print(f"\n    Error on {run_name} ({type(e).__name__}). Retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"\n    FAILED {run_name} after {max_retries} retries ({type(e).__name__}: {e}). Skipping.")
                return None


def extract_answer(response_text):
    """Extract the answer after 'ANSWER:' marker."""
    if "ANSWER:" in response_text:
        return response_text.split("ANSWER:")[-1].strip()
    return response_text.strip().split("\n")[-1].strip()


async def process_one(run_name, model_cfg, q, ds_name, json_data,
                      semaphore, results_file, file_lock, completed):
    """Handle one (question, model_config) pair: call model, extract, write result."""
    print(f"  Running {q['id']} / {run_name}...", flush=True)
    t0 = time.time()

    result_tuple = await call_model(
        run_name, model_cfg, q["question"], json_data, semaphore,
        answer_type=q.get("answer_type"),
    )
    if result_tuple is None:
        return

    response, reasoning, usage = result_tuple
    extracted = extract_answer(response)
    elapsed = round(time.time() - t0, 2)
    print(f"  Done {q['id']} / {run_name} ({elapsed}s) -> {extracted[:80]}")

    effort = run_name.split("@")[1] if "@" in run_name else "default"
    result = {
        "question_id": q["id"],
        "dataset": ds_name,
        "model_name": run_name,
        "model_id": model_cfg["model"],
        "reasoning_effort": effort,
        "question": q["question"],
        "expected_answer": q["answer"],
        "answer_type": q["answer_type"],
        "tolerance": q["tolerance"],
        "raw_response": response,
        "reasoning_trace": reasoning,
        "extracted_answer": extracted,
        "usage": usage,
        "elapsed_seconds": elapsed,
    }
    if "value_types" in q:
        result["value_types"] = q["value_types"]
    if "tolerances" in q:
        result["tolerances"] = q["tolerances"]

    async with file_lock:
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        completed.add((q["id"], run_name))


async def run(run_names, datasets, question_ids=None, concurrency=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "results.jsonl"
    completed = load_completed(results_file)

    # Per-provider semaphores for rate limiting
    # Tier 3 Anthropic (2,000 RPM), Tier 4 OpenAI (10,000 RPM), Tier 1 Gemini (~150 RPM)
    default_concurrency = {
        "anthropic": 20,
        "openai": 20,
        "gemini": 2,
    }
    provider_semaphores: dict[str, asyncio.Semaphore] = {}
    for run_name in run_names:
        provider = CONFIGS[run_name]["provider"]
        if provider not in provider_semaphores:
            if concurrency is None:
                c = default_concurrency.get(provider, 5)
            elif isinstance(concurrency, dict):
                c = concurrency.get(provider, default_concurrency.get(provider, 5))
            else:
                c = concurrency
            provider_semaphores[provider] = asyncio.Semaphore(c)

    file_lock = asyncio.Lock()

    # Collect all tasks
    tasks = []
    for ds_name in datasets:
        data_path = DATA_DIR / f"{ds_name}.json"
        json_data = data_path.read_text()
        questions = json.loads((Q_DIR / f"{ds_name}.json").read_text())

        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]
            if not questions:
                continue

        print(f"\n=== Dataset: {ds_name} ({len(questions)} questions) ===")

        for q in questions:
            for run_name in run_names:
                key = (q["id"], run_name)
                if key in completed:
                    print(f"  SKIP {q['id']} / {run_name} (already done)")
                    continue

                model_cfg = CONFIGS[run_name]
                provider = model_cfg["provider"]
                semaphore = provider_semaphores[provider]

                tasks.append(
                    process_one(
                        run_name, model_cfg, q, ds_name, json_data,
                        semaphore, results_file, file_lock, completed,
                    )
                )

    if not tasks:
        print("\nAll tasks already completed.")
        return

    sem_info = {p: s._value for p, s in provider_semaphores.items()}
    print(f"\nDispatching {len(tasks)} API calls "
          f"(concurrency per provider: {sem_info})")

    await asyncio.gather(*tasks)
    print("\nAll tasks finished.")


def main():
    parser = argparse.ArgumentParser(description="Run Calcubench benchmark")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Run configs to test (model@effort)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Datasets to test",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Only run specific question IDs (e.g. imdb_002 census_nat_001)",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help="Concurrent API calls per provider. Single int (e.g. 10) applies to all, "
             "or comma-separated provider=N pairs (e.g. anthropic=20,openai=20,gemini=5). "
             "Defaults: anthropic=20, openai=20, gemini=5",
    )
    args = parser.parse_args()

    # Parse concurrency arg
    concurrency = None
    if args.concurrency:
        if "=" in args.concurrency:
            concurrency = {}
            for pair in args.concurrency.split(","):
                provider, n = pair.split("=")
                concurrency[provider.strip()] = int(n.strip())
        else:
            concurrency = int(args.concurrency)

    print(f"Run configs ({len(args.configs)}):")
    for name in args.configs:
        print(f"  {name} -> {CONFIGS[name]['model']}")

    asyncio.run(run(args.configs, args.datasets,
                     question_ids=args.questions,
                     concurrency=concurrency))


if __name__ == "__main__":
    main()
