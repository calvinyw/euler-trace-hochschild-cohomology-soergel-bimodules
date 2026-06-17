import unittest

import sympy as sp

from computations.euler_trace_extfree import khovanov_rozansky_euler_trace_extfree
from computations.khovanov_rozansky import DynkinDiagram
from computations.slower_old_KR.khovanov_rozansky_extfree_bounded import (
    khovanov_rozansky_extfree_bounded_homology,
)
from computations.khovanov_rozansky_extfree import khovanov_rozansky_extfree_homology


class KhovanovRozanskyExtFreeTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_is_exterior_algebra_over_r(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_extfree_homology(diagram, [])

        self.assertFalse(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(1, 0)].module.is_zero())

    def test_positive_generator_has_expected_zero_pattern(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_extfree_homology(diagram, [(0, "+")])

        self.assertTrue(result.horizontal_homology[(0, -1)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(1, -1)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(1, 0)].module.is_zero())

    def test_negative_generator_has_expected_zero_pattern(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_extfree_homology(diagram, [(0, "-")])

        self.assertTrue(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(0, 1)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(1, 0)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(1, 1)].module.is_zero())

    def test_horizontal_euler_characteristic_matches_extfree_euler_trace(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = [(1, "+"), (2, "+"), (1, "+"), (2, "+")]

        hhh = khovanov_rozansky_extfree_homology(diagram, braid)
        euler = khovanov_rozansky_euler_trace_extfree(diagram, braid)

        self.assert_polynomials_equal(hhh.euler_trace_polynomial, euler.polynomial)

    def test_bounded_extfree_matches_main_extfree_on_finite_a2_case(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = [(1, "+"), (2, "+"), (1, "+"), (2, "+")]

        hhh = khovanov_rozansky_extfree_homology(diagram, braid)
        bounded = khovanov_rozansky_extfree_bounded_homology(
            diagram,
            braid,
            max_q_degree=8,
        )

        self.assert_polynomials_equal(hhh.polynomial, bounded.polynomial)


if __name__ == "__main__":
    unittest.main()
