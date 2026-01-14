:- use_module(library(plunit)).
:- use_module('../../src/prolog/drq').
:- use_module('../../src/prolog/archive').

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
