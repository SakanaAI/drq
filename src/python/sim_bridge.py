"""Simulation bridge for Janus interop."""

import sys
sys.path.insert(0, '.')

from corewar.corewar import redcode, MARS, Core
from corewar.corewar.redcode import parse

import random
import numpy as np
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool


@dataclass
class SimulationArgs:
    """Core War simulation configuration."""
    rounds: int = 24
    size: int = 8000
    cycles: int = 80000
    processes: int = 8000
    warrior_length: int = 100
    distance: int = 100


class MyMARS(MARS):
    """Extended MARS with memory coverage tracking."""

    def __init__(self, core=None, warriors=None, minimum_separation=100,
                 randomize=True, max_processes=None):
        self.core = core if core else Core()
        self.minimum_separation = minimum_separation
        self.max_processes = max_processes if max_processes else len(self.core)
        self.warriors = warriors if warriors else []
        self.warrior_cov = {w: np.zeros(len(self.core), dtype=bool)
                           for w in self.warriors}
        if self.warriors:
            self.load_warriors(randomize)

    def core_event(self, warrior, address, event_type):
        i = self.core[address]
        i.a_number = min(max(i.a_number, -999999999), 999999999)
        i.b_number = min(max(i.b_number, -999999999), 999999999)

    def enqueue(self, warrior, address):
        """Enqueue process with coverage tracking."""
        if len(warrior.task_queue) < self.max_processes:
            warrior.task_queue.append(self.core.trim(address))
            self.warrior_cov[warrior][self.core.trim(address)] = True


def run_single_round(simargs: SimulationArgs, warriors: list, seed: int) -> dict:
    """Run a single battle round."""
    random.seed(seed)
    simulation = MyMARS(
        warriors=warriors,
        minimum_separation=simargs.distance,
        max_processes=simargs.processes,
        randomize=True
    )

    n = len(warriors)
    score = np.zeros(n, dtype=float)
    alive_score = np.zeros(n, dtype=float)
    prev_nprocs = np.array([len(w.task_queue) for w in simulation.warriors])
    total_spawned_procs = np.zeros(n, dtype=int)

    for _ in range(simargs.cycles):
        simulation.step()
        nprocs = np.array([len(w.task_queue) for w in simulation.warriors])
        alive_flags = (nprocs > 0).astype(int)
        n_alive = alive_flags.sum()

        if n_alive == 0:
            break

        score += (alive_flags * (1.0 / n_alive)) / simargs.cycles
        alive_score += alive_flags / simargs.cycles
        total_spawned_procs += np.maximum(0, nprocs - prev_nprocs)
        prev_nprocs = nprocs

    memory_coverage = np.array(
        [cov.sum() for cov in simulation.warrior_cov.values()], dtype=int
    )
    score = score * n

    return {
        "score": score,
        "alive_score": alive_score,
        "total_spawned_procs": total_spawned_procs,
        "memory_coverage": memory_coverage,
    }


def run_multiple_rounds(simargs: SimulationArgs, warriors: list,
                        n_processes: int = 1) -> dict:
    """Run multiple battle rounds and aggregate results."""
    run_fn = partial(run_single_round, simargs, warriors)
    seeds = list(range(simargs.rounds))

    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            outputs = pool.map(run_fn, seeds)
    else:
        outputs = [run_fn(s) for s in seeds]

    return {
        k: np.stack([o[k] for o in outputs], axis=-1)
        for k in outputs[0].keys()
    }


def parse_redcode(source: str) -> dict:
    """Parse Redcode source into warrior object.

    Returns:
        {"status": "ok", "warrior": Warrior} on success
        {"status": "error", "error": str} on failure
    """
    try:
        lines = source.split("\n") if isinstance(source, str) else source
        warrior = parse(lines)
        return {"status": "ok", "warrior": warrior}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_battle(warrior: dict, opponents: list[dict], simargs: dict) -> dict:
    """Run battle simulation and return metrics.

    Args:
        warrior: Dict with 'source' key containing Redcode
        opponents: List of dicts with 'source' keys
        simargs: Dict with 'rounds', 'size', 'cycles', etc.

    Returns:
        Dict with 'scores', 'total_spawned_procs', 'memory_coverage'
    """
    def parse_source(src):
        lines = src.split("\n") if isinstance(src, str) else src
        return parse(lines)

    parsed_warrior = parse_source(warrior["source"])
    parsed_opponents = [parse_source(o["source"]) for o in opponents]
    all_warriors = [parsed_warrior] + parsed_opponents

    sim_args = SimulationArgs(
        rounds=simargs.get("rounds", 24),
        size=simargs.get("size", 8000),
        cycles=simargs.get("cycles", 80000),
        processes=simargs.get("processes", 8000),
        warrior_length=simargs.get("warrior_length", 100),
        distance=simargs.get("distance", 100),
    )

    outputs = run_multiple_rounds(sim_args, all_warriors)

    return {
        "scores": outputs["score"][0].tolist(),
        "total_spawned_procs": int(outputs["total_spawned_procs"][0].mean()),
        "memory_coverage": int(outputs["memory_coverage"][0].mean()),
    }
