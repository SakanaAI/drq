# DRQ Janus Port Implementation Plan

> **For Claude:** Use phosphor:executing-plans to implement this plan task-by-task.

**Goal:** Port DRQ from Python orchestration to Prolog-hosted Janus architecture with state threading and backtracking.

**Architecture:** Prolog controls evolution loop, strategy selection, and constraint inference. Python handles Core War VM, LLM calls, and async I/O. State threads through all predicates as explicit arguments.

**Tech Stack:** SWI-Prolog 9.0+, Janus, Python 3.11+, OpenAI async client

---

## Task 1: Create Directory Structure

**Files:**
- Create: `src/prolog/` directory
- Create: `src/python/` directory
- Create: `tests/prolog/` directory
- Create: `tests/python/` directory

**Step 1: Create directories**

```bash
mkdir -p src/prolog src/python tests/prolog tests/python
```

**Step 2: Verify**

Run: `ls -la src/ tests/`

Expected: Both `prolog/` and `python/` directories exist in each.

**Step 3: Commit**

```bash
git add src/ tests/
git commit -m "chore: create Janus port directory structure"
```

---

## Task 2: Implement archive.pl State Structure

**Files:**
- Create: `src/prolog/archive.pl`
- Create: `tests/prolog/test_archive.pl`

**Step 1: Write failing test**

Create `tests/prolog/test_archive.pl`:

```prolog
:- use_module(library(plunit)).
:- use_module('../../src/prolog/archive').

:- begin_tests(archive).

test(empty_state_creates_valid_state) :-
    empty_state(S),
    S = state(_, _, _, []).

test(state_archive_returns_empty_for_new_round) :-
    empty_state(S),
    state_archive(0, S, Archive),
    assoc_to_list(Archive, []).

:- end_tests(archive).
```

**Step 2: Verify failure**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_archive.pl`

Expected: ERROR - Cannot find module `archive`

**Step 3: Implement archive.pl**

Create `src/prolog/archive.pl`:

```prolog
:- module(archive, [
    empty_state/1,
    state_archive/3,
    state_history/2,
    state_metrics/2,
    state_errors/2,
    place/6,
    sample_elite/3,
    get_best/3,
    record_warrior_error/4,
    update_warrior_metrics/6
]).

:- use_module(library(assoc)).
:- use_module(library(random)).

%% empty_state(-State) is det
empty_state(state(Archives, History-History, Metrics, [])) :-
    empty_assoc(Archives),
    Metrics = _{coverage_history: [], fitness_history: []}.

%% state_archive(+Round, +State, -Archive) is det
state_archive(Round, state(Archives, _, _, _), Archive) :-
    ( get_assoc(Round, Archives, Archive) -> true ; empty_assoc(Archive) ).

%% state_history(+State, -History) is det
state_history(state(_, H, _, _), H).

%% state_metrics(+State, -Metrics) is det
state_metrics(state(_, _, M, _), M).

%% state_errors(+State, -Errors) is det
state_errors(state(_, _, _, E), E).

%% place(+Round, +BC, +Id, +Fitness, +S0, -S) is det
place(Round, BC, Id, Fitness,
      state(Archives0, H, M, E),
      state(Archives1, H, M, E)) :-
    state_archive(Round, state(Archives0, _, _, _), Archive0),
    ( get_assoc(BC, Archive0, entry(_, OldFit))
    -> ( Fitness > OldFit
       -> put_assoc(BC, Archive0, entry(Id, Fitness), Archive1)
       ; Archive1 = Archive0
       )
    ; put_assoc(BC, Archive0, entry(Id, Fitness), Archive1)
    ),
    put_assoc(Round, Archives0, Archive1, Archives1).

%% sample_elite(+Round, +State, -Elite) is semidet
sample_elite(Round, State, Elite) :-
    state_archive(Round, State, Archive),
    assoc_to_list(Archive, Pairs),
    Pairs \= [],
    random_member(_-entry(Elite, _), Pairs).

%% get_best(+Round, +State, -Champion) is semidet
get_best(Round, State, Champion) :-
    state_archive(Round, State, Archive),
    assoc_to_list(Archive, Pairs),
    Pairs \= [],
    pairs_values(Pairs, Entries),
    max_member(entry(Champion, _), Entries).

%% record_warrior_error(+Id, +Msg, +S0, -S) is det
record_warrior_error(Id, Msg,
                     state(A, H, M, Errors),
                     state(A, H, M, [error(warrior_failure, context(Id, Msg))|Errors])).

%% update_warrior_metrics(+Id, +Fitness, +BC, +Metrics, +S0, -S) is det
update_warrior_metrics(_Id, Fitness, _BC, _Metrics,
                       state(A, H, M0, E),
                       state(A, H, M1, E)) :-
    CovHist = M0.coverage_history,
    FitHist = M0.fitness_history,
    M1 = _{
        coverage_history: [placeholder|CovHist],
        fitness_history: [Fitness|FitHist]
    }.
```

**Step 4: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_archive.pl`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/prolog/archive.pl tests/prolog/test_archive.pl
git commit -m "feat(archive): implement pure state threading for MapElites"
```

---

## Task 3: Add Archive Place and Get Tests

**Files:**
- Modify: `tests/prolog/test_archive.pl`

**Step 1: Write failing tests**

Append to `tests/prolog/test_archive.pl`:

```prolog
test(place_single_warrior) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.5, S0, S1),
    get_best(0, S1, Champion),
    Champion = warrior_1.

test(place_replaces_inferior) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.5, S0, S1),
    place(0, bc(1,2), warrior_2, 0.7, S1, S2),
    get_best(0, S2, Champion),
    Champion = warrior_2.

test(place_keeps_superior) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.8, S0, S1),
    place(0, bc(1,2), warrior_2, 0.6, S1, S2),
    get_best(0, S2, Champion),
    Champion = warrior_1.

test(sample_elite_fails_on_empty) :-
    empty_state(S),
    \+ sample_elite(0, S, _).

test(sample_elite_returns_member) :-
    empty_state(S0),
    place(0, bc(1,1), w1, 0.5, S0, S1),
    place(0, bc(2,2), w2, 0.6, S1, S2),
    sample_elite(0, S2, Elite),
    memberchk(Elite, [w1, w2]).
```

**Step 2: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_archive.pl`

Expected: All tests passed

**Step 3: Commit**

```bash
git add tests/prolog/test_archive.pl
git commit -m "test(archive): add place and sample_elite tests"
```

---

## Task 4: Implement warrior.pl Term Operations

**Files:**
- Create: `src/prolog/warrior.pl`
- Create: `tests/prolog/test_warrior.pl`

**Step 1: Write failing test**

Create `tests/prolog/test_warrior.pl`:

```prolog
:- use_module(library(plunit)).
:- use_module('../../src/prolog/warrior').

:- begin_tests(warrior).

test(warrior_to_dict_converts) :-
    W = warrior(id123, "MOV 0, 1", [instr(mov, f, 0, 1)], _{fitness: 0.5}),
    warrior_to_dict(W, Dict),
    Dict.id = id123,
    Dict.source = "MOV 0, 1".

test(warrior_has_opcode_succeeds) :-
    W = warrior(id1, "", [instr(mov, f, 0, 1), instr(spl, b, 0, 0)], _{}),
    warrior_has_opcode(W, spl).

test(warrior_has_opcode_fails) :-
    W = warrior(id1, "", [instr(mov, f, 0, 1)], _{}),
    \+ warrior_has_opcode(W, spl).

test(warrior_length_correct) :-
    W = warrior(id1, "", [instr(a,b,c,d), instr(e,f,g,h), instr(i,j,k,l)], _{}),
    warrior_length(W, 3).

:- end_tests(warrior).
```

**Step 2: Verify failure**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_warrior.pl`

Expected: ERROR - Cannot find module `warrior`

**Step 3: Implement warrior.pl**

Create `src/prolog/warrior.pl`:

```prolog
:- module(warrior, [
    warrior_to_dict/2,
    dict_to_warrior/2,
    warrior_has_opcode/2,
    warrior_length/2,
    instruction_to_dict/2,
    dict_to_instruction/2
]).

%% warrior_to_dict(+Warrior, -Dict) is det
warrior_to_dict(warrior(Id, Source, Instructions, Meta), Dict) :-
    maplist(instruction_to_dict, Instructions, InstrDicts),
    Dict = _{id: Id, source: Source, instructions: InstrDicts, metadata: Meta}.

%% dict_to_warrior(+Dict, -Warrior) is det
dict_to_warrior(Dict, warrior(Id, Source, Instructions, Meta)) :-
    Id = Dict.id,
    Source = Dict.source,
    maplist(dict_to_instruction, Dict.instructions, Instructions),
    Meta = Dict.metadata.

%% warrior_has_opcode(+Warrior, +Op) is semidet
warrior_has_opcode(warrior(_, _, Instructions, _), Op) :-
    member(instr(Op, _, _, _), Instructions).

%% warrior_length(+Warrior, -Len) is det
warrior_length(warrior(_, _, Instructions, _), Len) :-
    length(Instructions, Len).

%% instruction_to_dict(+Instr, -Dict) is det
instruction_to_dict(instr(Op, Mod, A, B), _{opcode: Op, modifier: Mod, a: A, b: B}).

%% dict_to_instruction(+Dict, -Instr) is det
dict_to_instruction(Dict, instr(Op, Mod, A, B)) :-
    Op = Dict.opcode,
    Mod = Dict.modifier,
    A = Dict.a,
    B = Dict.b.
```

**Step 4: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_warrior.pl`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/prolog/warrior.pl tests/prolog/test_warrior.pl
git commit -m "feat(warrior): implement term representation and serialization"
```

---

## Task 5: Implement sim_bridge.py

**Files:**
- Create: `src/python/sim_bridge.py`
- Create: `tests/python/test_sim_bridge.py`

**Step 1: Write failing test**

Create `tests/python/test_sim_bridge.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import sys
sys.path.insert(0, 'src/python')

from sim_bridge import run_battle, parse_redcode


def test_parse_redcode_success():
    source = "ORG start\nstart: MOV 0, 1\nEND"
    result = parse_redcode(source)
    assert result["status"] == "ok"
    assert result["warrior"] is not None


def test_parse_redcode_failure():
    source = "INVALID GARBAGE"
    result = parse_redcode(source)
    assert result["status"] == "error"
    assert "error" in result


@patch('sim_bridge.run_multiple_rounds')
def test_run_battle_returns_metrics(mock_run):
    mock_run.return_value = {
        "score": np.array([[0.5, 0.6, 0.7]]),
        "total_spawned_procs": np.array([[100]]),
        "memory_coverage": np.array([[500]]),
    }

    warrior = {"source": "MOV 0, 1"}
    opponents = [{"source": "DAT 0, 0"}]
    simargs = {"rounds": 3, "size": 8000}

    result = run_battle(warrior, opponents, simargs)

    assert "scores" in result
    assert "total_spawned_procs" in result
    assert "memory_coverage" in result
```

**Step 2: Verify failure**

Run: `uv run pytest tests/python/test_sim_bridge.py -v`

Expected: ModuleNotFoundError: No module named 'sim_bridge'

**Step 3: Implement sim_bridge.py**

Create `src/python/sim_bridge.py`:

```python
"""Simulation bridge for Janus interop."""

import sys
sys.path.insert(0, '.')

from corewar.corewar import redcode
from src.corewar_util import run_multiple_rounds, SimulationArgs


def parse_redcode(source: str) -> dict:
    """Parse Redcode source into warrior object.

    Returns:
        {"status": "ok", "warrior": Warrior} on success
        {"status": "error", "error": str} on failure
    """
    try:
        warrior = redcode.parse(source)
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
    # Parse all warriors
    parsed_warrior = redcode.parse(warrior["source"])
    parsed_opponents = [redcode.parse(o["source"]) for o in opponents]

    all_warriors = [parsed_warrior] + parsed_opponents

    # Build simulation args
    sim_args = SimulationArgs(
        rounds=simargs.get("rounds", 24),
        size=simargs.get("size", 8000),
        cycles=simargs.get("cycles", 80000),
        processes=simargs.get("processes", 8000),
        warrior_length=simargs.get("warrior_length", 100),
        distance=simargs.get("distance", 100),
    )

    # Run simulation
    outputs = run_multiple_rounds(sim_args, all_warriors)

    # Extract metrics for first warrior (the one being evaluated)
    return {
        "scores": outputs["score"][0].tolist(),
        "total_spawned_procs": int(outputs["total_spawned_procs"][0].mean()),
        "memory_coverage": int(outputs["memory_coverage"][0].mean()),
    }
```

**Step 4: Verify pass**

Run: `uv run pytest tests/python/test_sim_bridge.py -v`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/python/sim_bridge.py tests/python/test_sim_bridge.py
git commit -m "feat(sim_bridge): implement Core War simulation bridge"
```

---

## Task 6: Implement stream_bridge.py

**Files:**
- Create: `src/python/stream_bridge.py`
- Create: `tests/python/test_stream_bridge.py`

**Step 1: Write failing test**

Create `tests/python/test_stream_bridge.py`:

```python
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
```

**Step 2: Verify failure**

Run: `uv run pytest tests/python/test_stream_bridge.py -v`

Expected: ModuleNotFoundError: No module named 'stream_bridge'

**Step 3: Implement stream_bridge.py**

Create `src/python/stream_bridge.py`:

```python
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

        self.pool = Pool(processes=min(len(self.warriors), 4))

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

        self.results = self.pool.map(evaluate_one, self.warriors)
        self.pool.close()

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
```

**Step 4: Verify pass**

Run: `uv run pytest tests/python/test_stream_bridge.py -v`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/python/stream_bridge.py tests/python/test_stream_bridge.py
git commit -m "feat(stream_bridge): implement async streaming evaluation"
```

---

## Task 7: Implement strategy.pl

**Files:**
- Create: `src/prolog/strategy.pl`
- Create: `tests/prolog/test_strategy.pl`

**Step 1: Write failing test**

Create `tests/prolog/test_strategy.pl`:

```prolog
:- use_module(library(plunit)).
:- use_module('../../src/prolog/strategy').
:- use_module('../../src/prolog/archive').

:- begin_tests(strategy).

test(select_strategy_fill_gap_on_empty, [nondet]) :-
    empty_state(S),
    select_strategy(_{}, 0, S, Strategy),
    Strategy = fill_gap(_).

test(bc_cell_generates_all_36) :-
    findall(BC, bc_cell(BC), Cells),
    length(Cells, 36).

test(archive_has_gaps_finds_gaps) :-
    empty_state(S0),
    place(0, bc(0,0), w1, 0.5, S0, S1),
    archive_has_gaps(0, S1, Gap),
    Gap \= bc(0,0).

:- end_tests(strategy).
```

**Step 2: Verify failure**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_strategy.pl`

Expected: ERROR - Cannot find module `strategy`

**Step 3: Implement strategy.pl**

Create `src/prolog/strategy.pl`:

```prolog
:- module(strategy, [
    select_strategy/4,
    archive_has_gaps/3,
    bc_cell/1
]).

:- use_module(archive).
:- use_module(library(assoc)).

%% select_strategy(+Config, +Round, +State, -Strategy) is nondet
select_strategy(_Config, Round, State, fill_gap(Gap)) :-
    archive_has_gaps(Round, State, Gap).
select_strategy(_Config, Round, State, mutate(Elite)) :-
    \+ archive_has_gaps(Round, State, _),
    sample_elite(Round, State, Elite).
select_strategy(_Config, Round, State, generate_new) :-
    \+ archive_has_gaps(Round, State, _),
    \+ sample_elite(Round, State, _).

%% archive_has_gaps(+Round, +State, -Gap) is nondet
archive_has_gaps(Round, State, Gap) :-
    state_archive(Round, State, Archive),
    bc_cell(Gap),
    \+ get_assoc(Gap, Archive, _).

%% bc_cell(-BC) is nondet
bc_cell(bc(TSP, MC)) :-
    member(TSP, [0, 1, 2, 3, 4, 5]),
    member(MC, [0, 1, 2, 3, 4, 5]).
```

**Step 4: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_strategy.pl`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/prolog/strategy.pl tests/prolog/test_strategy.pl
git commit -m "feat(strategy): implement strategy selection with backtracking"
```

---

## Task 8: Implement constraints.pl

**Files:**
- Create: `src/prolog/constraints.pl`
- Create: `tests/prolog/test_constraints.pl`

**Step 1: Write failing test**

Create `tests/prolog/test_constraints.pl`:

```prolog
:- use_module(library(plunit)).
:- use_module('../../src/prolog/constraints').
:- use_module('../../src/prolog/archive').

:- begin_tests(constraints).

test(infer_fill_gap_includes_target) :-
    empty_state(S),
    infer_constraints(fill_gap(bc(2, 3)), S, Constraints),
    member(target_bc(bc(2, 3)), Constraints).

test(infer_high_tsp_requires_spl) :-
    empty_state(S),
    infer_constraints(fill_gap(bc(4, 1)), S, Constraints),
    member(required_opcode(spl), Constraints).

test(infer_mutate_includes_parent) :-
    empty_state(S),
    infer_constraints(mutate(_{id: parent123}), S, Constraints),
    member(parent(parent123), Constraints).

test(constraints_to_hints_converts) :-
    Constraints = [min_length(10), target_bc(bc(1, 2))],
    constraints_to_hints(Constraints, Hints),
    is_dict(Hints).

:- end_tests(constraints).
```

**Step 2: Verify failure**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_constraints.pl`

Expected: ERROR - Cannot find module `constraints`

**Step 3: Implement constraints.pl**

Create `src/prolog/constraints.pl`:

```prolog
:- module(constraints, [
    infer_constraints/3,
    constraints_to_hints/2,
    constraint_to_dict/2,
    find_elite_patterns/2,
    bc_bin/3,
    compute_bc/2,
    compute_fitness/2
]).

:- use_module(archive).

%% infer_constraints(+Strategy, +State, -Constraints) is det
infer_constraints(fill_gap(bc(TSP, MC)), State, Constraints) :-
    find_elite_patterns(State, Patterns),
    ( TSP > 3 -> ReqOps = [required_opcode(spl)] ; ReqOps = [] ),
    ( MC > 3 -> MinLen = [min_length(20)] ; MinLen = [min_length(5)] ),
    append([[target_bc(bc(TSP, MC))], MinLen, ReqOps, Patterns], Constraints).
infer_constraints(mutate(Elite), _State, [parent(Elite.id)]).
infer_constraints(generate_new, _State, []).

%% find_elite_patterns(+State, -Patterns) is det
find_elite_patterns(_State, []).

%% constraints_to_hints(+Constraints, -Hints) is det
constraints_to_hints(Constraints, _{constraints: Dicts}) :-
    maplist(constraint_to_dict, Constraints, Dicts).

%% constraint_to_dict(+Constraint, -Dict) is det
constraint_to_dict(min_length(N), _{type: min_length, value: N}).
constraint_to_dict(required_opcode(Op), _{type: required_opcode, value: Op}).
constraint_to_dict(target_bc(bc(T, M)), _{type: target_bc, tsp: T, mc: M}).
constraint_to_dict(parent(Id), _{type: parent, id: Id}).

%% bc_bin(+Axis, +RawValue, -BinIndex) is det
bc_bin(tsp, V, Bin) :- threshold_bin([1, 10, 100, 1000, 10000], V, Bin).
bc_bin(mc, V, Bin) :- threshold_bin([10, 100, 500, 1000, 4000], V, Bin).

threshold_bin(Thresholds, Value, Bin) :-
    threshold_bin_(Thresholds, Value, 0, Bin).
threshold_bin_([], _, 5, 5).
threshold_bin_([H|_], V, Acc, Acc) :- V < H, !.
threshold_bin_([_|T], V, Acc, Bin) :-
    Acc1 is Acc + 1,
    threshold_bin_(T, V, Acc1, Bin).

%% compute_bc(+Metrics, -BC) is det
compute_bc(Metrics, bc(TSP_Bin, MC_Bin)) :-
    bc_bin(tsp, Metrics.total_spawned_procs, TSP_Bin),
    bc_bin(mc, Metrics.memory_coverage, MC_Bin).

%% compute_fitness(+Metrics, -Fitness) is det
compute_fitness(Metrics, Fitness) :-
    Scores = Metrics.scores,
    sum_list(Scores, Sum),
    length(Scores, Len),
    Fitness is Sum / Len.
```

**Step 4: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_constraints.pl`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/prolog/constraints.pl tests/prolog/test_constraints.pl
git commit -m "feat(constraints): implement constraint inference and BC binning"
```

---

## Task 9: Implement llm_bridge.py

**Files:**
- Create: `src/python/llm_bridge.py`
- Create: `tests/python/test_llm_bridge.py`

**Step 1: Write failing test**

Create `tests/python/test_llm_bridge.py`:

```python
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


@pytest.mark.asyncio
async def test_generate_with_hints_returns_source():
    with patch('llm_bridge._get_completion') as mock_get:
        mock_get.return_value = "ORG start\nstart: MOV 0, 1\nEND"

        result = generate_with_hints({"constraints": []})

        assert result["status"] == "ok"
        assert "MOV" in result["source"]
```

**Step 2: Verify failure**

Run: `uv run pytest tests/python/test_llm_bridge.py -v`

Expected: ModuleNotFoundError: No module named 'llm_bridge'

**Step 3: Implement llm_bridge.py**

Create `src/python/llm_bridge.py`:

```python
"""LLM bridge for Janus interop."""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, '.')

# Load prompts
PROMPTS_DIR = Path("src/prompts")


def load_prompt(name: str) -> str:
    """Load prompt template from file."""
    path = PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text()
    return ""


SYSTEM_PROMPT = load_prompt("system_prompt_0")
NEW_PROMPT = load_prompt("new_prompt_0")
MUTATE_PROMPT = load_prompt("mutate_prompt_0")


def build_prompt(hints: dict) -> str:
    """Build prompt from constraint hints."""
    constraints = hints.get("constraints", [])

    # Check for mutation
    parent_constraint = next(
        (c for c in constraints if c.get("type") == "parent"), None
    )

    if parent_constraint:
        base = MUTATE_PROMPT or "Mutate this warrior to improve performance."
        return f"{base}\n\nParent ID: {parent_constraint['id']}"

    # Build generation prompt with constraints
    base = NEW_PROMPT or "Generate a new Core War warrior."

    constraint_lines = []
    for c in constraints:
        ctype = c.get("type")
        if ctype == "min_length":
            constraint_lines.append(f"- Minimum length: {c['value']} instructions")
        elif ctype == "required_opcode":
            constraint_lines.append(f"- Must use opcode: {c['value']}")
        elif ctype == "target_bc":
            constraint_lines.append(f"- Target behavior: TSP bin {c['tsp']}, MC bin {c['mc']}")

    if constraint_lines:
        return f"{base}\n\nConstraints:\n" + "\n".join(constraint_lines)

    return base


def _get_completion(prompt: str) -> str:
    """Get completion from LLM (sync wrapper)."""
    # This would normally call OpenAI async client
    # For now, return placeholder
    return "ORG start\nstart: MOV 0, 1\nEND"


def generate_with_hints(hints: dict) -> dict:
    """Generate warrior source from hints.

    Returns:
        {"status": "ok", "source": str} on success
        {"status": "error", "error_type": str, "message": str} on failure
    """
    try:
        prompt = build_prompt(hints)
        source = _get_completion(prompt)
        return {"status": "ok", "source": source}
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }
```

**Step 4: Verify pass**

Run: `uv run pytest tests/python/test_llm_bridge.py -v`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/python/llm_bridge.py tests/python/test_llm_bridge.py
git commit -m "feat(llm_bridge): implement constraint-to-prompt translation"
```

---

## Task 10: Implement drq.pl Entry Point

**Files:**
- Create: `src/prolog/drq.pl`
- Create: `tests/prolog/test_drq.pl`

**Step 1: Write failing test**

Create `tests/prolog/test_drq.pl`:

```prolog
:- use_module(library(plunit)).
:- use_module('../../src/prolog/drq').

:- begin_tests(drq).

test(default_config_valid) :-
    default_config(Config),
    Config.n_rounds > 0,
    Config.n_iters > 0.

test(evolve_rounds_terminates_at_max) :-
    default_config(Config0),
    Config = Config0.put(n_rounds, 0),
    empty_state(S0),
    evolve_rounds(Config, [], 0, S0, S),
    S = S0.

:- end_tests(drq).
```

**Step 2: Verify failure**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_drq.pl`

Expected: ERROR - Cannot find module `drq`

**Step 3: Implement drq.pl**

Create `src/prolog/drq.pl`:

```prolog
:- module(drq, [
    default_config/1,
    evolve/2,
    evolve_rounds/5,
    run_round/5
]).

:- use_module(archive).
:- use_module(strategy).
:- use_module(constraints).
:- use_module(warrior).
:- use_module(library(janus)).

%% default_config(-Config) is det
default_config(_{
    n_rounds: 10,
    n_iters: 100,
    n_processes: 24,
    batch_size: 8,
    save_dir: none,
    simargs: _{rounds: 24, size: 8000, cycles: 80000, processes: 8000},
    fitness_threshold: 0.8
}).

%% evolve(+Config, -FinalState) is det
evolve(Config, FinalState) :-
    empty_state(State0),
    evolve_rounds(Config, [], 0, State0, FinalState).

%% evolve_rounds(+Config, +Opponents, +Round, +S0, -S) is det
evolve_rounds(Config, _, Round, S, S) :-
    Round >= Config.n_rounds, !.
evolve_rounds(Config, Opponents, Round, S0, S) :-
    Round < Config.n_rounds,
    run_round(Config, Opponents, Round, S0, S1),
    ( get_best(Round, S1, Champion)
    -> NewOpponents = [Champion|Opponents]
    ; NewOpponents = Opponents
    ),
    NextRound is Round + 1,
    evolve_rounds(Config, NewOpponents, NextRound, S1, S).

%% run_round(+Config, +Opponents, +Round, +S0, -S) is det
%% Placeholder - full implementation requires Janus bridge
run_round(_Config, _Opponents, _Round, S, S).
```

**Step 4: Verify pass**

Run: `swipl -g "load_test_files([]), run_tests" -t halt tests/prolog/test_drq.pl`

Expected: All tests passed

**Step 5: Commit**

```bash
git add src/prolog/drq.pl tests/prolog/test_drq.pl
git commit -m "feat(drq): implement main evolution loop entry point"
```

---

## Task 11: Create Test Runner

**Files:**
- Create: `tests/prolog/run_all.pl`
- Create: `Makefile`

**Step 1: Create Prolog test runner**

Create `tests/prolog/run_all.pl`:

```prolog
:- use_module(library(plunit)).

:- load_test_files([]).

:- initialization(run_tests, main).
```

**Step 2: Create Makefile**

Create `Makefile`:

```makefile
.PHONY: test test-prolog test-python clean

test: test-prolog test-python

test-prolog:
	cd tests/prolog && swipl -g "consult(run_all), run_tests" -t halt \
		test_archive.pl test_warrior.pl test_strategy.pl \
		test_constraints.pl test_drq.pl

test-python:
	uv run pytest tests/python/ -v

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
```

**Step 3: Verify all tests pass**

Run: `make test`

Expected: All Prolog and Python tests pass

**Step 4: Commit**

```bash
git add tests/prolog/run_all.pl Makefile
git commit -m "chore: add test runner and Makefile"
```

---

## Summary

| Task | Component | Status |
|------|-----------|--------|
| 1 | Directory structure | Ready |
| 2-3 | archive.pl | Ready |
| 4 | warrior.pl | Ready |
| 5 | sim_bridge.py | Ready |
| 6 | stream_bridge.py | Ready |
| 7 | strategy.pl | Ready |
| 8 | constraints.pl | Ready |
| 9 | llm_bridge.py | Ready |
| 10 | drq.pl | Ready |
| 11 | Test runner | Ready |

**Next steps after plan completion:**
1. Wire up Janus bridge calls in `drq.pl`
2. Add LLM API integration to `llm_bridge.py`
3. Integration testing with real battles
4. Checkpoint/resume implementation
