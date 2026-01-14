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
