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

:- end_tests(archive).
