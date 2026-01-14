:- use_module(library(plunit)).
:- use_module('../../src/prolog/warrior').

:- begin_tests(warrior).

test(warrior_to_dict_converts) :-
    W = warrior(id123, "MOV 0, 1", [instr(mov, f, 0, 1)], _{fitness: 0.5}),
    warrior_to_dict(W, Dict),
    Dict.id = id123,
    Dict.source = "MOV 0, 1".

test(warrior_has_opcode_succeeds) :-
    W = warrior(id1, "", [instr(mov, f, 0, 1), instr(spl, b, 0, 0)], _{}),
    warrior_has_opcode(W, spl).

test(warrior_has_opcode_fails) :-
    W = warrior(id1, "", [instr(mov, f, 0, 1)], _{}),
    \+ warrior_has_opcode(W, spl).

test(warrior_length_correct) :-
    W = warrior(id1, "", [instr(a,b,c,d), instr(e,f,g,h), instr(i,j,k,l)], _{}),
    warrior_length(W, 3).

:- end_tests(warrior).
