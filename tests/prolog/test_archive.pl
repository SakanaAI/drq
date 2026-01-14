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
