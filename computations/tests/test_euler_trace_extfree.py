import unittest

import sympy as sp

from computations.slower_old_euler_trace.euler_trace import khovanov_rozansky_euler_trace
from computations.euler_trace_extfree import khovanov_rozansky_euler_trace_extfree
from computations.khovanov_rozansky import DynkinDiagram


class ExtFreeEulerTraceTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_matches_groebner_euler_trace(self):
        diagram = DynkinDiagram.from_data([0], [])
        extfree = khovanov_rozansky_euler_trace_extfree(diagram, []).polynomial
        groebner = khovanov_rozansky_euler_trace(diagram, []).polynomial

        self.assert_polynomials_equal(extfree, groebner)

    def test_rank_one_crossings_match_groebner_euler_trace(self):
        diagram = DynkinDiagram.from_data([0], [])
        for braid in ([(0, "+")], [(0, "-")]):
            extfree = khovanov_rozansky_euler_trace_extfree(diagram, braid).polynomial
            groebner = khovanov_rozansky_euler_trace(diagram, braid).polynomial
            self.assert_polynomials_equal(extfree, groebner)

    def test_type_a2_length_four_matches_groebner_euler_trace(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = [(1, "+"), (2, "+"), (1, "+"), (2, "+")]
        extfree = khovanov_rozansky_euler_trace_extfree(diagram, braid).polynomial
        groebner = khovanov_rozansky_euler_trace(diagram, braid).polynomial

        self.assert_polynomials_equal(extfree, groebner)


if __name__ == "__main__":
    unittest.main()
