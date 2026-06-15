import unittest

import sympy as sp

from computations.slower_old_euler_trace.euler_trace import khovanov_rozansky_euler_trace
from computations.khovanov_rozansky import DynkinDiagram, T, khovanov_rozansky_cohomology


class EulerTraceTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_matches_unreduced_t_minus_one(self):
        diagram = DynkinDiagram.from_data([0], [])
        euler = khovanov_rozansky_euler_trace(diagram, []).polynomial
        unreduced = khovanov_rozansky_cohomology(diagram, []).polynomial.xreplace({T: -1})

        self.assert_polynomials_equal(euler, unreduced)

    def test_rank_one_crossings_match_unreduced_t_minus_one(self):
        diagram = DynkinDiagram.from_data([0], [])
        for braid in ([(0, "+")], [(0, "-")]):
            euler = khovanov_rozansky_euler_trace(diagram, braid).polynomial
            unreduced = khovanov_rozansky_cohomology(diagram, braid).polynomial.xreplace({T: -1})
            self.assert_polynomials_equal(euler, unreduced)

    def test_type_a2_length_four_trace(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        euler = khovanov_rozansky_euler_trace(
            diagram,
            [(1, "+"), (2, "+"), (1, "+"), (2, "+")],
        ).polynomial

        self.assert_polynomials_equal(euler, sp.Symbol("A") + sp.Symbol("Q") ** 4 + 1)


if __name__ == "__main__":
    unittest.main()
