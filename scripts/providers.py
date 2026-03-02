"""Provider SDK clients and API call functions."""

import asyncio

import anthropic
import openai
from google import genai
from google.genai import types as genai_types

from config import CONFIGS, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Lazy-initialized SDK clients
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Provider-specific call functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_PROVIDER_FNS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini,
}


async def call_provider(model_cfg, user_content):
    """Dispatch to the correct provider. Returns (content, reasoning, usage)."""
    provider = model_cfg["provider"]
    fn = _PROVIDER_FNS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}")
    return await fn(model_cfg, user_content)


async def call_model(run_name, model_cfg, question_text, json_data, semaphore, max_retries=3, answer_type=None):
    """Build prompt, retry with backoff, acquire semaphore. Returns None on total failure."""
    user_content = f"Question: {question_text}\n\nHere is the dataset:\n{json_data}"
    if answer_type == "table":
        user_content += (
            "\n\nIMPORTANT: Your answer must be a valid JSON object (dictionary). "
            "Use ANSWER: followed by the JSON object. Example: ANSWER: {\"key1\": value1, \"key2\": value2}"
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await call_provider(model_cfg, user_content)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            wait = 2 ** (attempt + 1)
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
            else:
                return None, last_error
