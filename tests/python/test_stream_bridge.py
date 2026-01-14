import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, 'src/python')

from stream_bridge import start_evaluation, next_result, cancel, _stream_registry


def test_start_evaluation_returns_stream_id():
    warriors = [{"id": "w1", "source": "MOV 0, 1"}]
    opponents = [{"source": "DAT 0, 0"}]

    with patch('stream_bridge.AsyncStream') as mock_stream:
        stream_id = start_evaluation(warriors, opponents, {"rounds": 1})
        assert isinstance(stream_id, int)


def test_next_result_returns_done_when_complete():
    warriors = []
    opponents = []

    with patch('stream_bridge.AsyncStream') as mock_stream:
        mock_instance = MagicMock()
        mock_instance.next.return_value = {"status": "done"}
        mock_stream.return_value = mock_instance

        stream_id = start_evaluation(warriors, opponents, {})
        result = next_result(stream_id)
        assert result["status"] == "done"


def test_cancel_removes_stream():
    with patch('stream_bridge.AsyncStream') as mock_stream:
        mock_instance = MagicMock()
        mock_stream.return_value = mock_instance

        stream_id = start_evaluation([], [], {})
        assert stream_id in _stream_registry

        cancel(stream_id)
        assert stream_id not in _stream_registry
