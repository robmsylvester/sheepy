"""Zero-shot and few-shot text classification via LLM API calls.

Uses ``litellm`` for provider-agnostic LLM access (OpenAI, Anthropic, local, etc.).
Install with ``pip install sheepy[llm]``.
"""
from __future__ import annotations

import json
import time
from typing import Any

from sheepy.common.logger import get_logger

try:
    import litellm
except ImportError:  # pragma: no cover — optional dependency; tests patch this name
    litellm = None  # type: ignore[misc, assignment]

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a text classification assistant. "
    "Classify the given text into exactly one of the provided categories. "
    "Respond with ONLY the category name, nothing else."
)

DEFAULT_TEMPLATE = (
    "Categories: {categories}\n\n"
    "{few_shot_section}"
    "Text to classify:\n{text}\n\n"
    "Category:"
)


class PromptClassifier:
    """Classify text using an LLM via litellm.

    Args:
        model: Model identifier (e.g. ``"gpt-4o-mini"``, ``"claude-3-haiku-20240307"``).
        labels: List of valid label strings.
        system_prompt: System prompt for the LLM.
        template: User message template with ``{categories}``, ``{text}``, ``{few_shot_section}`` placeholders.
        few_shot_examples: Optional list of ``{"text": ..., "label": ...}`` dicts for few-shot.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        api_key: API key (if not set via environment variable).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        labels: list[str] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        template: str = DEFAULT_TEMPLATE,
        few_shot_examples: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        api_key: str | None = None,
    ) -> None:
        """Store model id, prompts, few-shot examples, and zeroed usage counters."""
        self.model = model
        self.labels = labels or []
        self.system_prompt = system_prompt
        self.template = template
        self.few_shot_examples = few_shot_examples or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    def _build_few_shot_section(self) -> str:
        """Format few-shot examples block for the user template, or empty string if none."""
        if not self.few_shot_examples:
            return ""
        lines = ["Examples:\n"]
        for ex in self.few_shot_examples:
            lines.append(f"  Text: {ex['text']}\n  Category: {ex['label']}\n")
        lines.append("\n")
        return "".join(lines)

    def _build_prompt(self, text: str) -> str:
        """Fill the user template with categories, *text*, and few-shot section."""
        return self.template.format(
            categories=", ".join(self.labels),
            text=text,
            few_shot_section=self._build_few_shot_section(),
        )

    def classify(self, text: str) -> dict[str, Any]:
        """Classify a single text.

        Returns:
            Dict with ``label``, ``raw_response``, ``prompt_tokens``,
            ``completion_tokens``, ``cost``, ``latency_ms``.
        """
        if litellm is None:
            raise ImportError("litellm is required for PromptClassifier. Install: pip install sheepy[llm]")

        prompt = self._build_prompt(text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        start = time.time()
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )
        latency_ms = (time.time() - start) * 1000

        raw = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        cost = self._estimate_cost(prompt_tokens, completion_tokens)

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += cost
        self._call_count += 1

        predicted = self._parse_label(raw)

        return {
            "label": predicted,
            "raw_response": raw,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of texts sequentially."""
        return [self.classify(t) for t in texts]

    def _parse_label(self, raw: str) -> str:
        """Match the LLM output to one of the valid labels."""
        raw_lower = raw.lower().strip()
        for label in self.labels:
            if label.lower() == raw_lower:
                return label
        for label in self.labels:
            if label.lower() in raw_lower:
                return label
        logger.warning("label_parse_failed", raw=raw, valid=self.labels)
        return raw

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Rough cost estimate — override for precise per-model pricing."""
        try:
            from litellm import completion_cost
            return completion_cost(model=self.model, prompt="x" * prompt_tokens, completion="x" * completion_tokens) or 0.0
        except Exception:
            return 0.0

    @property
    def usage_summary(self) -> dict[str, Any]:
        """Aggregate token counts, estimated cost, and number of API calls."""
        return {
            "total_calls": self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost_usd": self._total_cost,
        }
