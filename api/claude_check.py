"""Use the Claude API to review ASR (speech-to-text) output for likely errors.

There is no reference transcript here — Claude reads the Hebrew ASR text and
flags spans that look like recognition mistakes (misheard words, typos,
nonsense tokens, punctuation/spacing problems), proposes a corrected version,
and gives an overall quality score.
"""
from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from api.config import CLAUDE_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger("api.claude_check")

_SYSTEM = (
    "You are a proofreader for the raw output of a Hebrew speech-to-text (ASR) "
    "system. You do NOT have the reference transcript or the audio — judge only "
    "from the text itself. Find spans that are likely ASR recognition errors: "
    "misheard or invented words, nonsense tokens, wrong/missing punctuation, and "
    "spacing problems. Do not rewrite for style; only fix what is plausibly a "
    "transcription mistake. Keep the original meaning and language (Hebrew). "
    "Always answer by calling the `report` tool."
)

# Forced tool — guarantees structured, validated output instead of free text.
_REPORT_TOOL = {
    "name": "report",
    "description": "Report likely ASR transcription errors and a corrected version.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall transcription quality, 0 (garbage) to 100 (clean).",
            },
            "summary": {
                "type": "string",
                "description": "One- or two-sentence assessment of the transcription.",
            },
            "issues": {
                "type": "array",
                "description": "Specific suspected errors. Empty if the text looks clean.",
                "items": {
                    "type": "object",
                    "properties": {
                        "span": {"type": "string", "description": "The problematic text as it appears."},
                        "type": {
                            "type": "string",
                            "enum": ["recognition_error", "typo", "nonsense", "punctuation", "spacing", "other"],
                        },
                        "suggestion": {"type": "string", "description": "Proposed correction for the span."},
                        "explanation": {"type": "string", "description": "Short reason this looks wrong."},
                    },
                    "required": ["span", "type", "suggestion"],
                },
            },
            "corrected_text": {
                "type": "string",
                "description": "Full transcription with the suspected errors corrected.",
            },
        },
        "required": ["score", "summary", "issues", "corrected_text"],
    },
}

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        if not CLAUDE_API_KEY:
            raise RuntimeError("CLAUDE_API_KEY is not set (check .env)")
        _client = AsyncAnthropic(api_key=CLAUDE_API_KEY)
    return _client


async def review_transcription(text: str) -> dict:
    """Send ASR text to Claude and return the structured review dict."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")

    client = _get_client()
    message = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=_SYSTEM,
        tools=[_REPORT_TOOL],
        tool_choice={"type": "tool", "name": "report"},
        messages=[{
            "role": "user",
            "content": f"Review this Hebrew ASR transcription:\n\n{text}",
        }],
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "report":
            report = dict(block.input)
            report["model"] = CLAUDE_MODEL
            report["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            return report

    raise RuntimeError("Claude did not return a report tool call")
