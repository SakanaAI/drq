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
