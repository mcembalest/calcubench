#!/usr/bin/env python3
"""Disposable script to estimate token counts per dataset and calculate safe concurrency."""

import json
from pathlib import Path

import tiktoken

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "prepared"
Q_DIR = REPO / "questions"

SYSTEM_PROMPT = (
    "Answer the user's question about the provided data. Do not use tools -- "
    "just examine the data directly and provide your answer.\n"
    "End your response with exactly:\n"
    "ANSWER: <your answer>\n"
    "For numbers, no commas or units. For text, exact match."
)

# From rate_limit_info.md
RATE_LIMITS = {
    "anthropic": {"tpm": 2_000_000, "rpm": 4_000},
    "openai":    {"tpm": 4_000_000, "rpm": 10_000},
    "gemini":    {"tpm": 1_000_000, "rpm": 25, "rpd": 250},
}

# Conservative output estimates by effort level
OUTPUT_ESTIMATES = {
    "low": 2_000,
    "medium": 12_000,
    "high": 20_000,
    "max": 35_000,
    "xhigh": 60_000,
}


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    print("=" * 70)
    print("Dataset Token Estimates (tiktoken cl100k_base)")
    print("=" * 70)

    dataset_tokens = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        ds_name = path.stem
        json_data = path.read_text()
        ds_toks = len(enc.encode(json_data))

        # Average question tokens for this dataset
        q_path = Q_DIR / f"{ds_name}.json"
        avg_q_toks = 0
        if q_path.exists():
            questions = json.loads(q_path.read_text())
            q_toks = [len(enc.encode(q["question"])) for q in questions]
            avg_q_toks = sum(q_toks) // len(q_toks)

        sys_toks = len(enc.encode(SYSTEM_PROMPT))
        total_input = ds_toks + avg_q_toks + sys_toks + 20  # overhead
        dataset_tokens[ds_name] = total_input

        print(f"  {ds_name:25s}  {ds_toks:>8,} dataset  + {avg_q_toks:>4} question  + {sys_toks} system  = {total_input:>9,} input tokens")
        print(f"  {'':25s}  ({len(json_data):>8,} bytes,  ratio: {ds_toks/len(json_data):.2f} tok/byte)")

    avg_input = sum(dataset_tokens.values()) // len(dataset_tokens)
    print(f"\n  Average input tokens per call: {avg_input:,}")

    # Heuristic check: compare len//4 to tiktoken
    print(f"\n  len//4 heuristic vs tiktoken:")
    for path in sorted(DATA_DIR.glob("*.json")):
        ds_name = path.stem
        heuristic = len(path.read_text()) // 4
        actual = dataset_tokens[ds_name]
        error_pct = abs(heuristic - actual) / actual * 100
        print(f"    {ds_name:25s}  heuristic={heuristic:>8,}  tiktoken={actual:>8,}  error={error_pct:.1f}%")

    print("\n" + "=" * 70)
    print("Safe Concurrency by Provider")
    print("=" * 70)

    for provider, limits in RATE_LIMITS.items():
        tpm = limits["tpm"]
        rpm = limits.get("rpm", 99999)

        print(f"\n  {provider.upper()}: {tpm:,} TPM, {rpm:,} RPM")

        for effort, out_est in OUTPUT_ESTIMATES.items():
            total_per_call = avg_input + out_est
            calls_per_min_tpm = tpm / total_per_call
            calls_per_min = min(calls_per_min_tpm, rpm)
            binding = "TPM" if calls_per_min_tpm < rpm else "RPM"

            for avg_duration in [30, 60, 90]:
                safe_conc = int(calls_per_min * avg_duration / 60)
                marker = " <-- " if avg_duration == 60 else ""
                print(f"    @{effort:7s}  ~{total_per_call:>7,} tok/call  →  {calls_per_min:.1f} calls/min ({binding})  →  concurrency {safe_conc:>3} @ {avg_duration}s/call{marker}")

    rpd = RATE_LIMITS.get("gemini", {}).get("rpd")
    if rpd:
        print(f"\n  Note: Gemini also has {rpd} RPD limit (100 calls per full run is fine)")


if __name__ == "__main__":
    main()
