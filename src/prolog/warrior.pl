:- module(warrior, [
    warrior_to_dict/2,
    dict_to_warrior/2,
    warrior_has_opcode/2,
    warrior_length/2,
    instruction_to_dict/2,
    dict_to_instruction/2
]).

%% warrior_to_dict(+Warrior, -Dict) is det
warrior_to_dict(warrior(Id, Source, Instructions, Meta), Dict) :-
    maplist(instruction_to_dict, Instructions, InstrDicts),
    Dict = _{id: Id, source: Source, instructions: InstrDicts, metadata: Meta}.

%% dict_to_warrior(+Dict, -Warrior) is det
dict_to_warrior(Dict, warrior(Id, Source, Instructions, Meta)) :-
    Id = Dict.id,
    Source = Dict.source,
    maplist(dict_to_instruction, Dict.instructions, Instructions),
    Meta = Dict.metadata.

%% warrior_has_opcode(+Warrior, +Op) is semidet
warrior_has_opcode(warrior(_, _, Instructions, _), Op) :-
    member(instr(Op, _, _, _), Instructions).

%% warrior_length(+Warrior, -Len) is det
warrior_length(warrior(_, _, Instructions, _), Len) :-
    length(Instructions, Len).

%% instruction_to_dict(+Instr, -Dict) is det
instruction_to_dict(instr(Op, Mod, A, B), _{opcode: Op, modifier: Mod, a: A, b: B}).

%% dict_to_instruction(+Dict, -Instr) is det
dict_to_instruction(Dict, instr(Op, Mod, A, B)) :-
    Op = Dict.opcode,
    Mod = Dict.modifier,
    A = Dict.a,
    B = Dict.b.
