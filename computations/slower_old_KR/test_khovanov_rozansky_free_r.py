import unittest

import sympy as sp

from computations.khovanov_rozansky import DynkinDiagram
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    khovanov_rozansky_free_r_homology,
)


class KhovanovRozanskyFreeRTests(unittest.TestCase):
    def test_rank_one_bott_samelson_koszul_matrix_is_over_left_r(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_free_r_homology(diagram, [(0, "+")])
        x = result.rouquier_free.r_variables[0]
        b_term = result.term_data[(1,)]

        self.assertEqual(
            b_term.koszul.differentials[0],
            sp.Matrix([[x, -x**2], [-1, x]]),
        )

    def test_positive_generator_has_r_torsion_homology(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_free_r_homology(diagram, [(0, "+")])

        self.assertTrue(result.horizontal_homology[(0, -1)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertEqual(str(result.horizontal_homology[(0, 0)].module), "<[1] + <[2*x_0]>>")
        self.assertTrue(result.horizontal_homology[(1, -1)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(1, 0)].module.is_zero())

    def test_negative_generator_has_top_ext_torsion_homology(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_free_r_homology(diagram, [(0, "-")])

        self.assertTrue(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(0, 1)].module.is_zero())
        self.assertTrue(result.horizontal_homology[(1, 0)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(1, 1)].module.is_zero())

    def test_empty_braid_is_exterior_algebra_over_r(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_free_r_homology(diagram, [])

        self.assertFalse(result.horizontal_homology[(0, 0)].module.is_zero())
        self.assertFalse(result.horizontal_homology[(1, 0)].module.is_zero())
        self.assertEqual(str(result.horizontal_homology[(0, 0)].module), "<[1] + <>>")
        self.assertEqual(str(result.horizontal_homology[(1, 0)].module), "<[1] + <>>")

    def test_type_a2_smoke_test(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        result = khovanov_rozansky_free_r_homology(
            diagram,
            [(0, "+"), (1, "-")],
        )

        self.assertEqual(str(result.ring), "QQ[x_0,x_1]")
        self.assertEqual(len(result.horizontal_homology), 9)


if __name__ == "__main__":
    unittest.main()
