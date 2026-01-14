# DRQ Janus Port Design

Port Digital Red Queen to Janus, using state threading and backtracking for bidirectional reasoning.

## Overview

Prolog replaces Python orchestration. Prolog controls the evolution loop, strategy selection, and constraint inference. Python retains the Core War VM, LLM client, simulation, and async I/O.

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Archive | Threaded state | Pure logic, safe backtracking |
| Warriors | Full Prolog terms | Maximum reasoning power |
| Parallelism | Async streaming | Flexible early stopping |
| LLM integration | Constraint-driven | Backward reasoning from goals |

## System Architecture

```dot
digraph DRQ_ARCHITECTURE {
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10];

    // Entry point (doublecircle)
    drq [label="drq.pl\nEvolution loop\nRound control\nCheckpoint" shape=doublecircle];

    // Pure state layer (box)
    archive [label="archive.pl\nMapElites assoc\nplace/5, sample/3\nHistory diff list" shape=box];

    // Logic modules (box)
    strategy [label="strategy.pl\nselect_strategy/4\nbacktrack on fail" shape=box];
    constraints [label="constraints.pl\ninfer_constraints/3\ngap_analysis\nelite_patterns" shape=box];
    warrior [label="warrior.pl\nTerm repr\nparse/serialize\nvalidation" shape=box];

    // Boundary (plaintext)
    boundary [label="═══ Janus Boundary ═══" shape=plaintext];

    // Python layer (box, filled)
    llm_bridge [label="llm_bridge.py\ngenerate_with_hints\nasync OpenAI" shape=box style=filled fillcolor=lightgray];
    sim_bridge [label="sim_bridge.py\nrun_battle\nparse_redcode" shape=box style=filled fillcolor=lightgray];
    stream_bridge [label="stream_bridge.py\nstart_evaluation\nnext_result" shape=box style=filled fillcolor=lightgray];

    // Edges
    drq -> archive [label="threads state"];
    archive -> strategy;
    archive -> constraints;
    archive -> warrior;
    strategy -> boundary;
    constraints -> boundary;
    warrior -> boundary;
    boundary -> llm_bridge [label="py_call/2"];
    boundary -> sim_bridge [label="py_call/2"];
    boundary -> stream_bridge [label="py_call/2"];
}
```

**Key invariant**: Prolog predicates thread all state as explicit arguments. The Python layer holds no state.

## Data Flow

### Predicate Specifications

```prolog
%% Mode declarations and determinism
%% evolve(+Config, -FinalState) is det
%% evolve_rounds(+Config, +Opponents, +Round, +S0, -S) is det
%% run_round(+Config, +Opponents, +Round, +S0, -S) is det
%% process_results(+StreamId, +Round, +S0, -S) is det
%% generate_warrior(+Config, +Round, +S0, -Warrior, -S) is nondet
%% select_strategy(+Config, +Round, +State, -Strategy) is nondet
%% place(+Round, +BC, +Id, +Fitness, +S0, -S) is det
%% sample_elite(+Round, +State, -Elite) is semidet
%% validate_warrior(+Warrior, +Constraints) is semidet
```

### State Structure

```prolog
%% State = state(Archives, History, Metrics, Errors)
%%   Archives: dict Round -> assoc(BC -> entry(WarriorId, Fitness))
%%   History: difference list of all evaluated warriors
%%   Metrics: py{coverage_history: [...], fitness_history: [...]}
%%   Errors: list of error(Type, Context) terms

%% empty_state(-State) is det
empty_state(state(Archives, History-History, Metrics, [])) :-
    empty_assoc(Archives),
    Metrics = py{coverage_history: [], fitness_history: []}.

%% State accessors
state_archive(Round, state(Archives, _, _, _), Archive) :-
    ( get_assoc(Round, Archives, Archive) -> true ; empty_assoc(Archive) ).
state_history(state(_, H, _, _), H).
state_metrics(state(_, _, M, _), M).
state_errors(state(_, _, _, E), E).
```

### Main Evolution Loop

```prolog
%% evolve(+Config, -FinalState) is det
evolve(Config, FinalState) :-
    empty_state(State0),
    load_initial_opponents(Config, Opponents),
    evolve_rounds(Config, Opponents, 0, State0, FinalState).

%% evolve_rounds(+Config, +Opponents, +Round, +S0, -S) is det
evolve_rounds(Config, _, Round, S, S) :-
    Round >= Config.n_rounds.
evolve_rounds(Config, Opponents, Round, S0, S) :-
    Round < Config.n_rounds,
    run_round(Config, Opponents, Round, S0, S1),
    checkpoint(Config, Round, S1),
    get_best(Round, S1, Champion),
    NextRound is Round + 1,
    evolve_rounds(Config, [Champion|Opponents], NextRound, S1, S).
```

### Single Round with Streaming

```prolog
run_round(Config, Opponents, Round, S0, S) :-
    init_batch(Config, Round, S0, Batch, S1),
    start_stream(Batch, Opponents, StreamId),
    process_results(StreamId, Round, S1, S).

start_stream(Warriors, Opponents, StreamId) :-
    maplist(warrior_to_dict, Warriors, WDicts),
    opponents_to_dicts(Opponents, ODicts),
    py_call(stream_bridge:start_evaluation(WDicts, ODicts), StreamId).

process_results(StreamId, Round, S0, S) :-
    py_call(stream_bridge:next_result(StreamId), Result),
    ( Result = py{status: done}
    -> S = S0
    ; Result = py{status: error, warrior_id: Id, msg: Msg}
    -> record_warrior_error(Id, Msg, S0, S1),
       process_results(StreamId, Round, S1, S)
    ; Result = py{warrior_id: Id, fitness: F, bc: BC, metrics: M}
    -> update_warrior_metrics(Id, F, BC, M, S0, S1),
       place(Round, BC, Id, F, S1, S2),
       process_results(StreamId, Round, S2, S)
    ).
```

### Constraint-Driven Generation with Backtracking

```prolog
%% generate_warrior(+Config, +Round, +S0, -Warrior, -S) is nondet
generate_warrior(Config, Round, S0, Warrior, S) :-
    select_strategy(Config, Round, S0, Strategy),
    infer_constraints(Strategy, S0, Constraints),
    attempt_generation(Config, Constraints, S0, Warrior, S1),
    validate_warrior(Warrior, Constraints),
    record_generation(Warrior, S1, S).

%% select_strategy(+Config, +Round, +State, -Strategy) is nondet
%% Choice points: fill_gap(Gap), mutate(Elite), generate_new
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

%% attempt_generation(+Config, +Constraints, +S0, -Warrior, -S) is det
attempt_generation(_Config, Constraints, S, Warrior, S1) :-
    constraints_to_hints(Constraints, Hints),
    py_call(llm_bridge:generate_with_hints(Hints), Response),
    parse_warrior_response(Response, Warrior, S, S1).
```

### Constraint Types and Inference

```prolog
%% Constraint terms:
%%   min_length(N)        - Minimum instruction count
%%   required_opcode(Op)  - Must include opcode
%%   target_bc(BC)        - Target behavior characterization
%%   parent(WarriorId)    - Mutate from this parent

%% infer_constraints(+Strategy, +State, -Constraints) is det
infer_constraints(fill_gap(bc(TSP, MC)), State, Constraints) :-
    find_elite_patterns(State, Patterns),
    ( TSP > 3 -> ReqOps = [required_opcode(spl)] ; ReqOps = [] ),
    ( MC > 3 -> MinLen = [min_length(20)] ; MinLen = [min_length(5)] ),
    append([[target_bc(bc(TSP, MC))], MinLen, ReqOps, Patterns], Constraints).
infer_constraints(mutate(Elite), _State, [parent(Elite.id)]).
infer_constraints(generate_new, _State, []).

%% constraints_to_hints(+Constraints, -Hints) is det
constraints_to_hints(Constraints, py{constraints: Dicts}) :-
    maplist(constraint_to_dict, Constraints, Dicts).

constraint_to_dict(min_length(N), py{type: min_length, value: N}).
constraint_to_dict(required_opcode(Op), py{type: required_opcode, value: Op}).
constraint_to_dict(target_bc(bc(T, M)), py{type: target_bc, tsp: T, mc: M}).
constraint_to_dict(parent(Id), py{type: parent, id: Id}).
```

### State Threading

```dot
digraph STATE_THREADING {
    rankdir=LR;
    node [fontname="Helvetica", fontsize=10];

    // States (ellipse)
    State0 [label="State0" shape=ellipse];
    State1 [label="State1" shape=ellipse];
    State2 [label="State2" shape=ellipse];
    backtrack [label="Backtrack" shape=ellipse];

    // Actions (box)
    generate [label="Generate warrior" shape=box];
    attempt [label="Attempt generation" shape=box];
    parse [label="Parse response" shape=box];
    record [label="Record generation" shape=box];
    place [label="Place in archive" shape=box];

    // Decisions (diamond)
    validate [label="Valid?" shape=diamond];

    // Flow
    State0 -> generate;
    generate -> attempt [label="strategy"];
    attempt -> parse [label="LLM call"];
    parse -> validate;
    validate -> record [label="yes"];
    validate -> backtrack [label="no"];
    backtrack -> generate [style=dashed label="next strategy"];
    record -> State1;
    State1 -> place;
    place -> State2 [label="if elite"];
}
```

## Fitness and BC Calculation

### Fitness Calculation

Fitness is the mean score across battle rounds:

```prolog
%% compute_fitness(+Metrics, -Fitness) is det
compute_fitness(Metrics, Fitness) :-
    Scores = Metrics.scores,
    sum_list(Scores, Sum),
    length(Scores, Len),
    Fitness is Sum / Len.
```

Python bridge returns aggregated metrics:

```python
# sim_bridge.py
def run_battle(warrior: dict, opponents: list[dict], simargs: dict) -> dict:
    outputs = run_multiple_rounds(simargs, [warrior] + opponents)
    return {
        "scores": outputs["score"][0].tolist(),  # Warrior's scores per round
        "total_spawned_procs": int(outputs["total_spawned_procs"][0].mean()),
        "memory_coverage": int(outputs["memory_coverage"][0].mean()),
    }
```

### BC Binning Logic

Map raw metrics to 6x6 behavior grid:

```prolog
%% bc_bin(+Axis, +RawValue, -BinIndex) is det
bc_bin(tsp, V, Bin) :- threshold_bin([1, 10, 100, 1000, 10000], V, Bin).
bc_bin(mc, V, Bin) :- threshold_bin([10, 100, 500, 1000, 4000], V, Bin).

%% threshold_bin(+Thresholds, +Value, -Bin) is det
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
```

## Error Handling

### Error Categories

| Type | Source | Recovery |
|------|--------|----------|
| Validation failed | `validate_warrior/2` | Backtrack to next strategy |
| Parse failed | `parse_warrior_response/2` | Backtrack to next strategy |
| Rate limit | Python LLM client | Retry with exponential backoff |
| Timeout | Python simulation | Record failure, continue batch |
| Stream error | Python ProcessPool | Cancel stream, keep partial results |
| Fatal | Unhandled exception | Log, checkpoint, exit |

### Recoverable Errors (Backtracking)

```prolog
generate_warrior(Config, Round, S0, Warrior, S) :-
    select_strategy(Config, Round, S0, Strategy),
    infer_constraints(Strategy, S0, Constraints),
    attempt_generation(Config, Constraints, S0, Warrior, S1),
    validate_warrior(Warrior, Constraints),
    record_generation(Warrior, S1, S).
%% Validation failure triggers backtracking to next strategy via select_strategy/4
```

### Retryable Errors (LLM/Network)

```prolog
attempt_with_retry(_, 0, _, _, _) :-
    throw(error(retry_exhausted, context(llm_generation, "Max retries exceeded"))).
attempt_with_retry(Constraints, N, Response, S0, S) :-
    N > 0,
    constraints_to_hints(Constraints, Hints),
    catch(
        py_call(llm_bridge:generate_with_hints(Hints), Response),
        PythonError,
        handle_python_error(PythonError, Constraints, N, Response, S0, S)
    ).

handle_python_error(PythonError, Constraints, N, Response, S0, S) :-
    classify_error(PythonError, retryable),
    N1 is N - 1,
    record_retry(PythonError, S0, S1),
    attempt_with_retry(Constraints, N1, Response, S1, S).
handle_python_error(PythonError, _, _, _, _, _) :-
    classify_error(PythonError, fatal),
    throw(PythonError).

classify_error(py_error('RateLimitError', _), retryable).
classify_error(py_error('APIConnectionError', _), retryable).
classify_error(py_error('Timeout', _), retryable).
classify_error(_, fatal).
```

### Stream Error Recovery

```prolog
process_results(StreamId, Round, S0, S) :-
    catch(
        py_call(stream_bridge:next_result(StreamId), Result),
        StreamError,
        handle_stream_error(StreamError, StreamId, Round, S0, S)
    ),
    process_result(Result, StreamId, Round, S0, S).

handle_stream_error(Error, StreamId, Round, S0, S) :-
    py_call(stream_bridge:cancel(StreamId), _),
    record_stream_failure(Error, Round, S0, S1),
    S = S1.  % Degrade gracefully: retain partial results
```

## Testing

### Test Structure

```
tests/
├── prolog/
│   ├── test_archive.pl      # Pure state operations
│   ├── test_strategy.pl     # Strategy selection + backtracking
│   ├── test_constraints.pl  # Constraint inference
│   ├── test_warrior.pl      # Term manipulation
│   └── test_integration.pl  # Full round with mocked Python
├── python/
│   ├── test_llm_bridge.py   # LLM client isolation
│   ├── test_sim_bridge.py   # Simulation wrapper
│   └── test_stream_bridge.py # Async streaming
└── e2e/
    └── test_evolution.pl    # Full evolution run
```

### Archive Unit Tests

```prolog
:- begin_tests(archive).

test(empty_archive) :-
    empty_state(S),
    \+ get_best(0, S, _).

test(place_single) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.5, S0, S1),
    get_best(0, S1, Best),
    Best = entry(warrior_1, 0.5).

test(place_replaces_inferior) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.5, S0, S1),
    place(0, bc(1,2), warrior_2, 0.7, S1, S2),
    get_best(0, S2, Best),
    Best = entry(warrior_2, 0.7).

test(place_keeps_superior) :-
    empty_state(S0),
    place(0, bc(1,2), warrior_1, 0.8, S0, S1),
    place(0, bc(1,2), warrior_2, 0.6, S1, S2),
    get_best(0, S2, Best),
    Best = entry(warrior_1, 0.8).

:- end_tests(archive).
```

### Strategy Backtracking Tests

```prolog
:- begin_tests(strategy_backtracking).

test(backtrack_on_failure, [nondet]) :-
    empty_state(S0),
    place(0, bc(0,1), w1, 0.5, S0, S1),
    findall(Strat, select_strategy(config{}, 0, S1, Strat), Strategies),
    member(fill_gap(bc(0,0)), Strategies),
    member(fill_gap(bc(1,1)), Strategies).

test(fallback_to_mutation) :-
    full_archive(S),
    select_strategy(config{}, 0, S, Strategy),
    Strategy = mutate(_).

:- end_tests(strategy_backtracking).
```

### Python Bridge Tests

```python
# test_llm_bridge.py
def test_generate_with_hints_success(mock_openai):
    mock_openai.chat.completions.create = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(
            content="ORG start\nstart: MOV 0,1\nEND"
        ))])
    )
    hints = {"required_opcodes": ["mov"], "min_length": 1}
    result = generate_with_hints(hints)
    assert "MOV" in result

def test_generate_retries_on_rate_limit(mock_openai):
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[RateLimitError("rate limited"),
                     Mock(choices=[Mock(message=Mock(content="valid"))])]
    )
    result = generate_with_hints({})
    assert result == "valid"
    assert mock_openai.chat.completions.create.call_count == 2
```

### Test Execution

```bash
# Prolog tests
swipl -g "run_tests" -t halt tests/run_all.pl

# Python tests
uv run pytest tests/python/ -v

# Full suite
make test
```

### Coverage Goals

| Component | Target | Strategy |
|-----------|--------|----------|
| archive.pl | 100% | All state transitions |
| strategy.pl | 90% | All choice points exercised |
| constraints.pl | 95% | All constraint combinations |
| Python bridges | 85% | Mock external services |
| Integration | 80% | Happy path + error recovery |

## File Structure

```
drq/
├── src/
│   ├── prolog/
│   │   ├── drq.pl           # Entry point, evolution loop
│   │   ├── archive.pl       # Pure MapElites state
│   │   ├── strategy.pl      # Strategy selection
│   │   ├── constraints.pl   # Constraint inference
│   │   └── warrior.pl       # Warrior term operations
│   └── python/
│       ├── llm_bridge.py    # LLM client wrapper
│       ├── sim_bridge.py    # Core War simulation
│       └── stream_bridge.py # Async result streaming
├── corewar/                  # Unchanged: VM + parser
├── tests/
│   ├── prolog/
│   └── python/
└── docs/
    └── plans/
```

## Migration Path

1. **Phase 1**: Implement `archive.pl` with pure state threading
2. **Phase 2**: Implement `warrior.pl` term representation and serialization
3. **Phase 3**: Implement Python bridges (`sim_bridge.py`, `stream_bridge.py`)
4. **Phase 4**: Implement `strategy.pl` with backtracking
5. **Phase 5**: Implement `constraints.pl` with elite pattern extraction
6. **Phase 6**: Implement `llm_bridge.py` with constraint-to-hints translation
7. **Phase 7**: Wire up `drq.pl` entry point
8. **Phase 8**: Integration testing and checkpoint compatibility

## Janus Interop Conventions

### Term Conversion

Prolog terms crossing the Janus boundary follow these patterns:

| Prolog Term | Python Type | Converter |
|-------------|-------------|-----------|
| `warrior(Id, Source, Meta)` | `dict` | `warrior_to_dict/2` |
| `bc(TSP, MC)` | `tuple` | Direct unification |
| `[Constraint|Cs]` | `list` | `constraints_to_hints/2` |
| `py{key: Value}` | `dict` | Janus dict literal |

### py_call/2 Return Value Handling

Python functions must return JSON-serializable types:

| Python Return | Prolog Term |
|---------------|-------------|
| `None` | `@none` |
| `True`/`False` | `@true`/`@false` |
| `int`, `float` | number |
| `str` | atom |
| `list` | list |
| `dict` | `py{key: value, ...}` |

```prolog
%% Example: handling LLM response with error checking
call_llm(Hints, Response) :-
    py_call(llm_bridge:generate_with_hints(Hints), PyResult),
    ( PyResult = @none
    -> throw(error(llm_returned_none, context(call_llm, Hints)))
    ; PyResult = py{status: error, message: Msg}
    -> throw(error(llm_error(Msg), context(call_llm, Hints)))
    ; Response = PyResult
    ).
```

Python bridge functions return error dicts for recoverable errors:

```python
def generate_with_hints(hints: dict) -> dict:
    try:
        result = _do_generation(hints)
        return {"status": "ok", "source": result}
    except RateLimitError as e:
        return {"status": "error", "error_type": "rate_limit", "message": str(e)}
```

### Async Interop

Python async functions wrap in synchronous bridge functions using `asyncio.run()`. The streaming pattern uses an iterator interface:

```python
# stream_bridge.py
def start_evaluation(warriors, opponents):
    """Return stream ID; results via next_result()."""
    stream = AsyncStream(warriors, opponents)
    return register_stream(stream)

def next_result(stream_id):
    """Block until next result ready."""
    return get_stream(stream_id).next()
```

### Python Exception Mapping

Python exceptions appear in Prolog as `py_error(Type, Message)`:

| Python Exception | Prolog Pattern |
|-----------------|----------------|
| `openai.RateLimitError` | `py_error('RateLimitError', _)` |
| `openai.APIConnectionError` | `py_error('APIConnectionError', _)` |
| `TimeoutError` | `py_error('Timeout', _)` |

### Thread Safety

The streaming design uses a single-threaded iterator pattern. Python ProcessPool workers run as isolated processes:

```python
# stream_bridge.py
import threading

_stream_registry: dict[int, AsyncStream] = {}
_registry_lock = threading.Lock()

def register_stream(stream: AsyncStream) -> int:
    with _registry_lock:
        stream_id = id(stream)
        _stream_registry[stream_id] = stream
        return stream_id
```

Key constraints:
- Stream registry uses thread-safe access
- Each `run_single_round` receives copies of warriors
- Prolog remains single-threaded; `py_call/2` invocations are synchronous

## Warrior Term Operations

```prolog
%% warrior/4 term structure
%% warrior(Id, Source, Instructions, Metadata)

%% warrior_to_dict(+Warrior, -Dict) is det
warrior_to_dict(warrior(Id, Source, Instructions, Meta), Dict) :-
    maplist(instruction_to_dict, Instructions, InstrDicts),
    Dict = py{id: Id, source: Source, instructions: InstrDicts, metadata: Meta}.

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
```

## Archive Operations

```prolog
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

%% update_warrior_metrics(+Id, +F, +BC, +Metrics, +S0, -S) is det
update_warrior_metrics(_Id, Fitness, _BC, _Metrics,
                       state(A, H, M0, E),
                       state(A, H, M1, E)) :-
    M0.coverage_history = CovHist,
    M0.fitness_history = FitHist,
    M1 = py{
        coverage_history: [coverage_placeholder|CovHist],
        fitness_history: [Fitness|FitHist]
    }.
```

## Dependencies

### Prolog
- SWI-Prolog 9.0+ (Janus support)
- library(janus)
- library(assoc)
- library(plunit)

### Python
- janus-swi
- openai (async client)
- multiprocessing

## Open Questions

1. **Checkpoint format**: Serialize Prolog state to JSON or keep pickle for Python objects?
2. **BC binning**: Port binning logic to Prolog or call Python utility?
3. **History limits**: Cap difference list length or allow unbounded growth?
