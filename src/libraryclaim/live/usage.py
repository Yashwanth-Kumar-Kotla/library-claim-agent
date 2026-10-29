"""What a live session actually costs, from the usage the Live API reports with its replies."""

from __future__ import annotations

from ..config import LIVE_PRICES


def by_modality(details) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in details or []:
        name = getattr(getattr(d, "modality", None), "name", None) or str(getattr(d, "modality", "TEXT"))
        out[name] = out.get(name, 0) + int(getattr(d, "token_count", 0) or 0)
    return out


def usage_entry(u) -> dict:
    """One usage report, per modality, with its cost. Thinking tokens are billed as text output."""
    prompt = by_modality(u.prompt_tokens_details)
    if not prompt and u.prompt_token_count:
        prompt = {"TEXT": int(u.prompt_token_count)}
    response = by_modality(u.response_tokens_details)
    if not response and u.response_token_count:
        response = {"AUDIO": int(u.response_token_count)}
    thoughts = int(u.thoughts_token_count or 0)
    usd = sum(n * LIVE_PRICES["in"].get(m, LIVE_PRICES["in"]["TEXT"]) for m, n in prompt.items()) / 1e6
    usd += sum(n * LIVE_PRICES["out"].get(m, LIVE_PRICES["out"]["TEXT"]) for m, n in response.items()) / 1e6
    usd += thoughts * LIVE_PRICES["out"]["TEXT"] / 1e6
    return {"prompt": prompt, "response": response, "thoughts": thoughts,
            "cached": int(u.cached_content_token_count or 0), "usd": round(usd, 6)}
