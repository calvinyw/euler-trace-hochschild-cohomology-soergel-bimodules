import unittest

import sympy as sp

from computations.euler_trace_extfree import khovanov_rozansky_euler_trace_extfree
from computations.khovanov_rozansky import DynkinDiagram
from computations.parallel_euler_trace import (
    khovanov_rozansky_parallel_euler_trace_extfree,
)
from computations.parralell_euler_trace import (
    parallel_euler_trace_extfree as misspelled_parallel_euler_trace_extfree,
)


class ParallelEulerTraceTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_matches_serial_extfree(self):
        diagram = DynkinDiagram.from_data([0], [])
        serial = khovanov_rozansky_euler_trace_extfree(diagram, []).polynomial
        parallel = khovanov_rozansky_parallel_euler_trace_extfree(
            diagram,
            [],
            max_workers=1,
        ).polynomial

        self.assert_polynomials_equal(parallel, serial)

    def test_rank_one_crossing_threaded_matches_serial_extfree(self):
        diagram = DynkinDiagram.from_data([0], [])
        braid = [(0, "+")]
        serial = khovanov_rozansky_euler_trace_extfree(diagram, braid).polynomial
        parallel = khovanov_rozansky_parallel_euler_trace_extfree(
            diagram,
            braid,
            max_workers=2,
            executor="thread",
        ).polynomial

        self.assert_polynomials_equal(parallel, serial)

    def test_misspelled_module_alias_matches_serial_extfree(self):
        diagram = DynkinDiagram.from_data([0], [])
        braid = [(0, "-")]
        serial = khovanov_rozansky_euler_trace_extfree(diagram, braid).polynomial
        parallel = misspelled_parallel_euler_trace_extfree(
            diagram,
            braid,
            max_workers=1,
        ).polynomial

        self.assert_polynomials_equal(parallel, serial)


if __name__ == "__main__":
    unittest.main()
