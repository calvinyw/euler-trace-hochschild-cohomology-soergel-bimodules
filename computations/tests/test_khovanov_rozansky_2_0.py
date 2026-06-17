import unittest

import sympy as sp

from computations.euler_trace_2_0 import khovanov_rozansky_euler_trace_2_0
from computations.khovanov_rozansky import DynkinDiagram
from computations.khovanov_rozansky_2_0 import khovanov_rozansky_2_0_homology
from computations.khovanov_rozansky_extfree import khovanov_rozansky_extfree_homology


class KhovanovRozansky20Tests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_is_exterior_algebra_over_r(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_2_0_homology(diagram, [])

        self.assertFalse(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(1, 0)].module.is_zero())

    def test_rank_one_generators_have_expected_zero_patterns(self):
        diagram = DynkinDiagram.from_data([0], [])

        positive = khovanov_rozansky_2_0_homology(diagram, [(0, "+")])
        self.assertTrue(positive.horizontal_homology[(0, -1)].module.is_zero())
        self.assertFalse(positive.horizontal_homology[(0, 0)].module.is_zero())
        self.assertTrue(positive.horizontal_homology[(1, -1)].module.is_zero())
        self.assertTrue(positive.horizontal_homology[(1, 0)].module.is_zero())

        negative = khovanov_rozansky_2_0_homology(diagram, [(0, "-")])
        self.assertTrue(negative.horizontal_homology[(0, 0)].module.is_zero())
        self.assertTrue(negative.horizontal_homology[(0, 1)].module.is_zero())
        self.assertTrue(negative.horizontal_homology[(1, 0)].module.is_zero())
        self.assertFalse(negative.horizontal_homology[(1, 1)].module.is_zero())

    def test_rank_one_euler_characteristic_matches_2_0_euler_trace(self):
        diagram = DynkinDiagram.from_data([0], [])
        for braid in ([], [(0, "+")], [(0, "-")]):
            hhh = khovanov_rozansky_2_0_homology(diagram, braid)
            euler = khovanov_rozansky_euler_trace_2_0(diagram, braid)
            self.assert_polynomials_equal(hhh.euler_trace_polynomial, euler.polynomial)

    def test_type_a2_length_four_matches_extfree_homology(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = [(1, "+"), (2, "+"), (1, "+"), (2, "+")]

        new = khovanov_rozansky_2_0_homology(diagram, braid)
        extfree = khovanov_rozansky_extfree_homology(diagram, braid)

        self.assert_polynomials_equal(new.polynomial, extfree.polynomial)
        self.assert_polynomials_equal(
            new.euler_trace_polynomial,
            khovanov_rozansky_euler_trace_2_0(diagram, braid).polynomial,
        )


if __name__ == "__main__":
    unittest.main()
