#! /usr/bin/env python
#! coding: utf-8

import unittest

from corewar.redcode import *

DEFAULT_ENV = {'CORESIZE': 8000}

class TestRedcodeAssembler(unittest.TestCase):

    def test_1(self):

        input = """
                ;name dwarf
                ;author A. K. Dewdney
                ;assert CORESIZE % 4 == 0

                org start
                step equ 2004

                loop    add.ab  #step,  start
                start   mov     2, 2
                        jmp.f   $loop ;go back and start over
                """
        warrior = parse(input.split('\n'), DEFAULT_ENV)

        self.assertEqual(1, warrior.start)
        self.assertEqual('dwarf', warrior.name)
        self.assertEqual('A. K. Dewdney', warrior.author)
        self.assertEqual(3, len(warrior))

        self.assertEqual(Instruction(ADD, M_AB, IMMEDIATE, 2004, DIRECT, 1),
                          warrior.instructions[0])
        self.assertEqual(Instruction(MOV, M_I, DIRECT, 2, DIRECT, 2),
                          warrior.instructions[1])
        self.assertEqual(Instruction(JMP, M_F, DIRECT, -2, DIRECT, 0),
                          warrior.instructions[2])

if __name__ == '__main__':
    unittest.main()

