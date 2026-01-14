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
