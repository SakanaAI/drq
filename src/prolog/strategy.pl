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
