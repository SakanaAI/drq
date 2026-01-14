import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import sys
sys.path.insert(0, 'src/python')

from sim_bridge import parse_redcode, run_battle, run_multiple_rounds


def test_parse_redcode_success():
    source = "MOV 0, 1"
    result = parse_redcode(source)
    assert result["status"] == "ok"
    assert result["warrior"] is not None


def test_parse_redcode_failure():
    source = "INVALID GARBAGE @@#$%"
    result = parse_redcode(source)
    assert result["status"] == "error"
    assert "error" in result


@patch('sim_bridge.run_multiple_rounds')
def test_run_battle_returns_metrics(mock_run):
    mock_run.return_value = {
        "score": np.array([[0.5, 0.6, 0.7]]),
        "total_spawned_procs": np.array([[100, 100, 100]]),
        "memory_coverage": np.array([[500, 500, 500]]),
    }

    warrior = {"source": "MOV 0, 1"}
    opponents = [{"source": "DAT 0, 0"}]
    simargs = {"rounds": 3, "size": 8000}

    result = run_battle(warrior, opponents, simargs)

    assert "scores" in result
    assert "total_spawned_procs" in result
    assert "memory_coverage" in result
