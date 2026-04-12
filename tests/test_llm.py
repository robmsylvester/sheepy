"""Tests for sheepy.llm module with mocked LLM responses."""
from unittest.mock import MagicMock, patch

import pytest


class TestPromptClassifier:
    def test_build_prompt(self):
        from sheepy.llm.prompt_classifier import PromptClassifier

        clf = PromptClassifier(labels=["spam", "ham"])
        prompt = clf._build_prompt("free money now!!!")
        assert "spam" in prompt
        assert "ham" in prompt
        assert "free money now!!!" in prompt

    def test_build_prompt_with_few_shot(self):
        from sheepy.llm.prompt_classifier import PromptClassifier

        examples = [
            {"text": "buy now!", "label": "spam"},
            {"text": "hello friend", "label": "ham"},
        ]
        clf = PromptClassifier(labels=["spam", "ham"], few_shot_examples=examples)
        prompt = clf._build_prompt("test text")
        assert "buy now!" in prompt
        assert "hello friend" in prompt
        assert "Examples:" in prompt

    def test_parse_label_exact(self):
        from sheepy.llm.prompt_classifier import PromptClassifier

        clf = PromptClassifier(labels=["positive", "negative", "neutral"])
        assert clf._parse_label("positive") == "positive"
        assert clf._parse_label("NEGATIVE") == "negative"

    def test_parse_label_substring(self):
        from sheepy.llm.prompt_classifier import PromptClassifier

        clf = PromptClassifier(labels=["spam", "ham"])
        assert clf._parse_label("I think this is spam.") == "spam"

    @patch("sheepy.llm.prompt_classifier.litellm")
    def test_classify_mocked(self, mock_litellm):
        from sheepy.llm.prompt_classifier import PromptClassifier

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "spam"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 1
        mock_litellm.completion.return_value = mock_response

        clf = PromptClassifier(model="test-model", labels=["spam", "ham"])
        result = clf.classify("buy cheap meds now!")

        assert result["label"] == "spam"
        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 1
        assert "latency_ms" in result

    @patch("sheepy.llm.prompt_classifier.litellm")
    def test_usage_summary(self, mock_litellm):
        from sheepy.llm.prompt_classifier import PromptClassifier

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ham"
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 1
        mock_litellm.completion.return_value = mock_response

        clf = PromptClassifier(model="test-model", labels=["spam", "ham"])
        clf.classify("hello")
        clf.classify("world")

        summary = clf.usage_summary
        assert summary["total_calls"] == 2
        assert summary["total_prompt_tokens"] == 60
