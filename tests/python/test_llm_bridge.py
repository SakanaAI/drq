import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import sys
sys.path.insert(0, 'src/python')

from llm_bridge import generate_with_hints, build_prompt


def test_build_prompt_includes_constraints():
    hints = {"constraints": [{"type": "min_length", "value": 10}]}
    prompt = build_prompt(hints)
    assert "10" in prompt or "min" in prompt.lower()


def test_build_prompt_includes_parent_for_mutation():
    hints = {"constraints": [{"type": "parent", "id": "w123"}]}
    prompt = build_prompt(hints)
    assert "mutate" in prompt.lower() or "parent" in prompt.lower()


def test_generate_with_hints_returns_source():
    with patch('llm_bridge._get_completion') as mock_get:
        mock_get.return_value = "ORG start\nstart: MOV 0, 1\nEND"

        result = generate_with_hints({"constraints": []})

        assert result["status"] == "ok"
        assert "MOV" in result["source"]
