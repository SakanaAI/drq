"""Streaming evaluation bridge for Janus interop."""

import threading
from multiprocessing import Pool
from typing import Iterator
import traceback

import sys
sys.path.insert(0, '.')

from src.python.sim_bridge import run_battle, parse_redcode


_stream_registry: dict[int, "AsyncStream"] = {}
_registry_lock = threading.Lock()


class AsyncStream:
    """Iterator over battle results from parallel evaluation."""

    def __init__(self, warriors: list[dict], opponents: list[dict], simargs: dict):
        self.warriors = warriors
        self.opponents = opponents
        self.simargs = simargs
        self.results: list[dict] = []
        self.index = 0
        self.pool = None
        self._start_evaluation()

    def _start_evaluation(self):
        """Start parallel evaluation of all warriors."""
        if not self.warriors:
            return

        def evaluate_one(warrior):
            try:
                metrics = run_battle(warrior, self.opponents, self.simargs)
                return {
                    "status": "ok",
                    "warrior_id": warrior["id"],
                    "fitness": sum(metrics["scores"]) / len(metrics["scores"]),
                    "bc": (metrics["total_spawned_procs"], metrics["memory_coverage"]),
                    "metrics": metrics,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "warrior_id": warrior.get("id", "unknown"),
                    "msg": str(e),
                }

        # Run evaluations (single-threaded for simplicity)
        self.results = [evaluate_one(w) for w in self.warriors]

    def next(self) -> dict:
        """Return next result or done status."""
        if self.index >= len(self.results):
            return {"status": "done"}

        result = self.results[self.index]
        self.index += 1
        return result

    def cancel(self):
        """Cancel pending evaluations."""
        if self.pool:
            self.pool.terminate()


def register_stream(stream: AsyncStream) -> int:
    """Register stream and return ID."""
    with _registry_lock:
        stream_id = id(stream)
        _stream_registry[stream_id] = stream
        return stream_id


def get_stream(stream_id: int) -> AsyncStream | None:
    """Get stream by ID."""
    with _registry_lock:
        return _stream_registry.get(stream_id)


def start_evaluation(warriors: list[dict], opponents: list[dict], simargs: dict) -> int:
    """Start parallel evaluation, return stream ID."""
    stream = AsyncStream(warriors, opponents, simargs)
    return register_stream(stream)


def next_result(stream_id: int) -> dict:
    """Get next result from stream."""
    stream = get_stream(stream_id)
    if stream is None:
        return {"status": "error", "msg": f"Unknown stream: {stream_id}"}
    return stream.next()


def cancel(stream_id: int) -> bool:
    """Cancel stream and remove from registry."""
    with _registry_lock:
        stream = _stream_registry.pop(stream_id, None)
        if stream:
            stream.cancel()
            return True
        return False
