import unittest

import sympy as sp

from computations.euler_trace_2_0 import khovanov_rozansky_euler_trace_2_0
from computations.khovanov_rozansky import DynkinDiagram
from computations.slower_old_euler_trace.euler_trace import khovanov_rozansky_euler_trace


class EulerTrace20Tests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_rank_one_cases_match_groebner_euler_trace(self):
        diagram = DynkinDiagram.from_data([0], [])
        for braid in ([], [(0, "+")], [(0, "-")]):
            new = khovanov_rozansky_euler_trace_2_0(diagram, braid).polynomial
            groebner = khovanov_rozansky_euler_trace(diagram, braid).polynomial
            self.assert_polynomials_equal(new, groebner)

    def test_type_a2_length_four_matches_groebner_euler_trace(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = [(1, "+"), (2, "+"), (1, "+"), (2, "+")]

        new = khovanov_rozansky_euler_trace_2_0(diagram, braid).polynomial
        groebner = khovanov_rozansky_euler_trace(diagram, braid).polynomial

        self.assert_polynomials_equal(new, groebner)


if __name__ == "__main__":
    unittest.main()
