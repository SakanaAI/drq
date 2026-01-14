#! /usr/bin/env python
#! coding: utf-8

import os
import re
import unittest

from corewar import redcode, mars

DEFAULT_ENV = {'CORESIZE': 8000, 'MAXLENGTH': 100}

class TestMars(unittest.TestCase):

    def test_dwarf_versus_sitting_duck(self):

        dwarf_code = """
            ;name dwarf
            ;author A. K. Dewdney

            org start

            loop    add.ab  #2004, start
            start   mov     2,     2
                    jmp     loop
        """
        sitting_duck_code = """
            nop
            nop
            nop
            nop
            nop
        """

        dwarf        = redcode.parse(dwarf_code.split('\n'), DEFAULT_ENV)
        sitting_duck = redcode.parse(sitting_duck_code.split('\n'), DEFAULT_ENV)

        simulation = mars.MARS(warriors=[dwarf, sitting_duck])

        # run simulation for at most
        for x in range(8000):
            simulation.step()
            if not dwarf.task_queue or not sitting_duck.task_queue:
                break
        else:
            self.fail("Running for too long and both warriors still alive")

        self.assertEqual(1, len(dwarf.task_queue))
        self.assertEqual(0, len(sitting_duck.task_queue))

    def test_validate(self):

        current_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(current_path, "..", "warriors", "validate.red")) as f:
            validate = redcode.parse(f, DEFAULT_ENV)

        simulation = mars.MARS(warriors=[validate], randomize=False)

        for i in range(8000):
            simulation.step()
            if not validate.task_queue:
                self.fail("Interpreter is not ICWS88-compliant. died in %d steps" % i)

    def test_crazy_warrrior(self):
        self.warrior_step_by_step("crazy.red", "crazy-steps.red", -22, 22)

    def test_validate_warrior(self):
        self.warrior_step_by_step("validate.red", "validate-steps.red", 0, 90)

    def warrior_step_by_step(self, warrior_filename, log_filename, core_start, core_end):

        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_path, "..", "warriors", warrior_filename)) as f:
            test_w = redcode.parse(f, DEFAULT_ENV)

        simulation = mars.MARS(warriors=[test_w], randomize=False)

        nth = 0

        with open(os.path.join(current_path, log_filename)) as f:
            accum_lines = []
            for n, line in enumerate(f):
                m = re.match(r';ACTIVE: ([0-9]{5})', line)
                if line.startswith(';ACTIVE:') and not m:
                    self.fail("Fatal error in regular expression line %d" % n)

                if m:
                    next_queued = int(m.group(1))
                    # has a full program, parse it
                    expected = redcode.parse(accum_lines)

                    # compare with next in queue
                    if not test_w.task_queue:
                        self.fail("No tasks in queue. step %d, line %d" % (nth, n))
                    if test_w.task_queue[0] != next_queued:
                        self.fail("Task address does not match (%d != %d). step %d, line %d" %
                                  (next_queued, test_w.task_queue[0], nth, n))

                    # compare it with the current state
                    for e, i in zip(expected, simulation.core[core_start:core_end]):
                        if e != i:
                            print()
                            x = core_start
                            for e, i in zip(expected, simulation.core[core_start:core_end]):
                                if e != i:
                                    print("%05d %s != %s" % (x, str(e), str(i)))
                                else:
                                    print("%05d %s == %s" % (x, str(e), str(i)))
                                x += 1
                            self.fail("Core don't match, step %d, line %d" % (nth, n))

                    # next state
                    simulation.step()

                    # throw away and start over
                    accum_lines = []
                    nth += 1
                else:
                    accum_lines.append(line)


if __name__ == '__main__':
    unittest.main()

