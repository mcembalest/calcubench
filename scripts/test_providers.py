"""Smoke test: call all 3 providers at multiple effort levels, verify reasoning capture and token scaling."""

import asyncio
import json
import time
from pathlib import Path

from run_benchmark import (
    call_anthropic, call_openai, call_gemini, CONFIGS,
)

OUTPUT_FILE = Path(__file__).parent.parent / "results" / "test_providers.jsonl"

# A question hard enough to trigger reasoning at higher effort levels
PROMPT = "What is 17 * 23 + 89 - 14 * 3?"

EXPECTED = str(17 * 23 + 89 - 14 * 3)  # 438

# Test configs: multiple effort levels per provider
TEST_CONFIGS = {
    "anthropic@low": CONFIGS["claude-opus-4.6@low"],
    "anthropic@high": CONFIGS["claude-opus-4.6@high"],
    "openai@low": CONFIGS["gpt-5.2@low"],
    "openai@high": CONFIGS["gpt-5.2@high"],
    "gemini@low": CONFIGS["gemini-3.1-pro@low"],
    "gemini@high": CONFIGS["gemini-3.1-pro@high"],
}

CALLERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini,
}


async def run_one(name, model_cfg):
    provider = model_cfg["provider"]
    user_content = f"Question: {PROMPT}\n\nCompute the answer step by step."
    t0 = time.time()
    content, reasoning, usage = await CALLERS[provider](model_cfg, user_content)
    elapsed = round(time.time() - t0, 2)
    return {
        "name": name,
        "provider": provider,
        "model": model_cfg["model"],
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "elapsed_seconds": elapsed,
    }


async def main():
    print(f"=== Provider Reasoning Test ===")
    print(f"Prompt: {PROMPT}")
    print(f"Expected: {EXPECTED}\n")

    # Run all in parallel
    names = list(TEST_CONFIGS.keys())
    coros = [run_one(name, TEST_CONFIGS[name]) for name in names]

    print(f"Dispatching {len(coros)} calls in parallel...\n")
    results = await asyncio.gather(*coros)

    # Write results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Display results grouped by provider
    print(f"{'Config':<20} {'Answer OK':<12} {'Reasoning':<45} {'Prompt Tok':<12} {'Completion Tok':<16} {'Time':<8}")
    print("-" * 155)

    for r in sorted(results, key=lambda x: (x["provider"], x["name"])):
        answer_ok = EXPECTED in r["content"]
        reasoning = r["reasoning"]
        if reasoning is None:
            reasoning_summary = "(none)"
        else:
            reasoning_summary = f"{len(reasoning)} chars: {reasoning[:80]!r}..."

        print(f"{r['name']:<20} {'PASS' if answer_ok else 'FAIL':<12} {reasoning_summary:<45} "
              f"{r['usage']['prompt_tokens']:<12} {r['usage']['completion_tokens']:<16} {r['elapsed_seconds']:<8}")

    # Compare token usage across effort levels per provider
    print(f"\n=== Token Scaling by Provider ===")
    providers = sorted(set(r["provider"] for r in results))
    for provider in providers:
        provider_results = sorted(
            [r for r in results if r["provider"] == provider],
            key=lambda x: x["name"],
        )
        print(f"\n  {provider}:")
        for r in provider_results:
            effort = r["name"].split("@")[1]
            has_reasoning = r["reasoning"] is not None
            reasoning_len = len(r["reasoning"]) if has_reasoning else 0
            print(f"    {effort:<8} -> completion_tokens={r['usage']['completion_tokens']:<8} "
                  f"reasoning={'YES (' + str(reasoning_len) + ' chars)' if has_reasoning else 'NO':<30}")

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
