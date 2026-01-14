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
% :- use_module(library(janus)).  % Uncomment when Janus is available

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
