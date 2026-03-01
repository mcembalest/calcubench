"""Concurrency stress test: spray simple questions across all providers and effort levels.

Tests:
1. No errors under full concurrency (20/20/2 per provider)
2. Latency and token counts are captured correctly
3. Higher thinking effort -> more completion tokens (sanity check)
"""

import asyncio
import json
import time
from pathlib import Path

from run_benchmark import call_model, CONFIGS

# Simple questions that any model can answer — enough to saturate concurrency
QUESTIONS = [
    "What is 137 * 29?",
    "What is the sum of all integers from 1 to 50?",
    "If a train travels 240 miles in 4 hours, what is its speed in mph?",
    "What is 15% of 860?",
    "A rectangle has length 17 and width 13. What is its area?",
    "What is 2^10?",
    "What is the average of 23, 47, 89, 12, and 54?",
    "If you buy 7 items at $13.50 each, what is the total cost?",
    "What is 999 - 573?",
    "A circle has radius 7. What is its area? Round to 2 decimal places.",
    "What is 144 / 12?",
    "What is 31 * 43 - 17 * 29?",
    "If 3x + 7 = 28, what is x?",
    "What is the factorial of 7?",
    "A bag has 3 red, 5 blue, and 2 green marbles. What fraction are blue? Simplify.",
    "What is 17^2 + 13^2?",
    "Convert 68 degrees Fahrenheit to Celsius. Round to 1 decimal place.",
    "What is the GCD of 84 and 36?",
    "If a car gets 32 mpg and gas costs $3.50/gallon, how much does a 480-mile trip cost?",
    "What is the 10th term of the arithmetic sequence 3, 7, 11, 15, ...?",
    "What is 256 / 16 + 73 * 2?",
    "How many seconds are in 3.5 hours?",
    "What is 19 * 21?",
    "A triangle has sides 5, 12, and 13. What is its area?",
    "What is the cube root of 729?",
]

# Test configs: pick low and high effort for each provider to test thinking level scaling
TEST_CONFIGS = {
    "anthropic@low": CONFIGS["claude-opus-4.6@low"],
    "anthropic@high": CONFIGS["claude-opus-4.6@high"],
    "openai@low": CONFIGS["gpt-5.2@low"],
    "openai@high": CONFIGS["gpt-5.2@high"],
    "gemini@low": CONFIGS["gemini-3.1-pro@low"],
    "gemini@high": CONFIGS["gemini-3.1-pro@high"],
}

# Default concurrency matching run_benchmark.py defaults
CONCURRENCY = {
    "anthropic": 20,
    "openai": 20,
    "gemini": 2,
}


async def run_one(name, model_cfg, question, semaphore):
    t0 = time.time()
    result_tuple = await call_model(name, model_cfg, question, "", semaphore)
    elapsed = round(time.time() - t0, 2)
    if result_tuple is None:
        return {
            "name": name,
            "provider": model_cfg["provider"],
            "question": question,
            "content": None,
            "reasoning_len": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "elapsed_seconds": elapsed,
            "error": "call_model returned None after retries",
        }
    content, reasoning, usage = result_tuple
    return {
        "name": name,
        "provider": model_cfg["provider"],
        "question": question,
        "content": content,
        "reasoning_len": len(reasoning) if reasoning else 0,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "error": None,
    }


async def main():
    print("=== Concurrency Stress Test ===")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Configs: {list(TEST_CONFIGS.keys())}")
    total = len(QUESTIONS) * len(TEST_CONFIGS)
    print(f"Total calls: {total}")
    print(f"Concurrency: {CONCURRENCY}\n")

    # Build per-provider semaphores
    semaphores = {p: asyncio.Semaphore(c) for p, c in CONCURRENCY.items()}

    # Build all tasks: every question x every config
    tasks = []
    for q in QUESTIONS:
        for name, cfg in TEST_CONFIGS.items():
            provider = cfg["provider"]
            sem = semaphores[provider]
            tasks.append(run_one(name, cfg, q, sem))

    print(f"Dispatching {len(tasks)} calls...\n")
    t_start = time.time()
    results = await asyncio.gather(*tasks)
    wall_time = round(time.time() - t_start, 2)

    # Separate successes from errors
    successes = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    # === Report 1: Error summary ===
    print(f"{'='*60}")
    print(f"RESULTS: {len(successes)} succeeded, {len(errors)} failed, {wall_time}s wall time")
    print(f"{'='*60}")
    if errors:
        print(f"\nERRORS:")
        by_provider = {}
        for e in errors:
            by_provider.setdefault(e["provider"], []).append(e)
        for provider, errs in sorted(by_provider.items()):
            print(f"  {provider}: {len(errs)} failures")
    else:
        print("No errors!")

    # === Report 2: Latency and token stats per config ===
    print(f"\n{'='*60}")
    print(f"LATENCY & TOKENS PER CONFIG")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'N':<5} {'Avg Lat':<10} {'Min Lat':<10} {'Max Lat':<10} "
          f"{'Avg Prompt':<12} {'Avg Compl':<12} {'Avg Reason':<12}")
    print("-" * 101)

    config_stats = {}
    for name in TEST_CONFIGS:
        runs = [r for r in successes if r["name"] == name]
        if not runs:
            print(f"{name:<20} {'0':<5} {'--':<10} {'--':<10} {'--':<10} {'--':<12} {'--':<12} {'--':<12}")
            continue
        lats = [r["elapsed_seconds"] for r in runs]
        prompt_toks = [r["usage"]["prompt_tokens"] for r in runs]
        compl_toks = [r["usage"]["completion_tokens"] for r in runs]
        reason_lens = [r["reasoning_len"] for r in runs]

        stats = {
            "count": len(runs),
            "avg_lat": round(sum(lats) / len(lats), 2),
            "min_lat": round(min(lats), 2),
            "max_lat": round(max(lats), 2),
            "avg_prompt": round(sum(prompt_toks) / len(prompt_toks)),
            "avg_compl": round(sum(compl_toks) / len(compl_toks)),
            "avg_reason": round(sum(reason_lens) / len(reason_lens)),
        }
        config_stats[name] = stats

        print(f"{name:<20} {stats['count']:<5} {stats['avg_lat']:<10} {stats['min_lat']:<10} "
              f"{stats['max_lat']:<10} {stats['avg_prompt']:<12} {stats['avg_compl']:<12} "
              f"{stats['avg_reason']:<12}")

    # === Report 3: Thinking level sanity check ===
    print(f"\n{'='*60}")
    print(f"THINKING LEVEL SANITY CHECK (high vs low)")
    print(f"{'='*60}")
    providers = sorted(set(cfg["provider"] for cfg in TEST_CONFIGS.values()))
    all_pass = True
    for provider in providers:
        low_key = f"{provider}@low"
        high_key = f"{provider}@high"
        if low_key not in config_stats or high_key not in config_stats:
            print(f"\n  {provider}: SKIP (missing data)")
            continue
        low = config_stats[low_key]
        high = config_stats[high_key]

        compl_ratio = high["avg_compl"] / low["avg_compl"] if low["avg_compl"] > 0 else float("inf")
        lat_ratio = high["avg_lat"] / low["avg_lat"] if low["avg_lat"] > 0 else float("inf")

        # High effort should use more completion tokens than low
        token_check = "PASS" if high["avg_compl"] > low["avg_compl"] else "FAIL"
        if token_check == "FAIL":
            all_pass = False

        print(f"\n  {provider}:")
        print(f"    Completion tokens:  low={low['avg_compl']}, high={high['avg_compl']} "
              f"(ratio={compl_ratio:.1f}x) [{token_check}]")
        print(f"    Latency:            low={low['avg_lat']}s, high={high['avg_lat']}s "
              f"(ratio={lat_ratio:.1f}x)")
        print(f"    Reasoning length:   low={low['avg_reason']} chars, high={high['avg_reason']} chars")

    print(f"\n{'='*60}")
    if all_pass and not errors:
        print("ALL CHECKS PASSED")
    else:
        if errors:
            print(f"FAILED: {len(errors)} API call(s) returned None after retries")
        if not all_pass:
            print("FAILED: thinking level scaling check failed for some providers")
    print(f"{'='*60}")

    # Write raw results for inspection
    out = Path(__file__).parent.parent / "results" / "test_concurrency.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in successes:
            f.write(json.dumps(r) + "\n")
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    asyncio.run(main())
