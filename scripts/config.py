"""Benchmark configuration: model configs, datasets, rate limits, prompts."""

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

ALL_DATASETS = ["census_southeast", "census_national", "imdb_top", "pokemon", "state_finance"]

RATE_LIMITS = {
    "anthropic": {"tpm": 2_000_000, "rpm": 4_000},
    "openai":    {"tpm": 4_000_000, "rpm": 10_000},
    "gemini":    {"tpm": 1_000_000, "rpm": 25},
}

# Conservative output token estimates by effort level (used for pre-reservation)
OUTPUT_TOKEN_ESTIMATES = {
    "low": 2_000,
    "medium": 12_000,
    "high": 20_000,
    "max": 35_000,
    "xhigh": 60_000,
    "default": 4_000,
}


def get_effort(run_name: str) -> str:
    """Extract effort level from a 'model@effort' run name."""
    return run_name.split("@")[1] if "@" in run_name else "default"
