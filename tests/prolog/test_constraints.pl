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
