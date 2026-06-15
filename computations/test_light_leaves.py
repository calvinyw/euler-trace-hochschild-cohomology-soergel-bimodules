import unittest

import sympy as sp

from computations.khovanov_rozansky import DynkinDiagram
from computations.light_leaves import (
    bott_samelson_light_leaves,
    rouquier_complex_as_free_left_r_modules,
    v,
)


class LightLeavesTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.expand(left - right), 0)

    def test_rank_one_subexpressions(self):
        diagram = DynkinDiagram.from_data([0], [])
        basis = bott_samelson_light_leaves(diagram, [0])

        self.assertEqual(basis.rank, 2)
        self.assertEqual([leaf.bit_string for leaf in basis.leaves], ["0", "1"])
        self.assertEqual([leaf.step_types for leaf in basis.leaves], [("U0",), ("U1",)])
        self.assertEqual([leaf.endpoint_word for leaf in basis.leaves], [(), (0,)])
        self.assertEqual([leaf.tensor_factors for leaf in basis.leaves], [("1",), ("alpha_0",)])
        self.assertEqual([leaf.module_degree for leaf in basis.leaves], [-1, 1])
        self.assertEqual([leaf.defect for leaf in basis.leaves], [1, 0])
        self.assert_polynomials_equal(basis.graded_rank(), v**-1 + v)

    def test_type_a2_word_has_one_leaf_per_subexpression(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        basis = bott_samelson_light_leaves(diagram, [0, 1, 0])

        self.assertEqual(basis.rank, 8)
        self.assertEqual(
            [leaf.bit_string for leaf in basis.leaves],
            ["000", "001", "010", "011", "100", "101", "110", "111"],
        )
        self.assert_polynomials_equal(
            basis.graded_rank(degree="module"),
            v**-3 + 3 * v**-1 + 3 * v + v**3,
        )

    def test_type_a2_endpoint_counts(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        basis = bott_samelson_light_leaves(diagram, [0, 1, 0])

        self.assertEqual(
            basis.counts_by_endpoint(),
            {
                (): 2,
                (0,): 2,
                (1,): 1,
                (0, 1): 1,
                (1, 0): 1,
                (0, 1, 0): 1,
            },
        )

    def test_type_a2_step_types_and_defects(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        basis = bott_samelson_light_leaves(diagram, [0, 1, 0])
        by_bits = {leaf.bit_string: leaf for leaf in basis.leaves}

        self.assertEqual(by_bits["100"].step_types, ("U1", "U0", "D0"))
        self.assertEqual(by_bits["100"].endpoint_word, (0,))
        self.assertEqual(by_bits["100"].defect, 0)
        self.assertEqual(by_bits["101"].step_types, ("U1", "U0", "D1"))
        self.assertEqual(by_bits["101"].endpoint_word, ())
        self.assertEqual(by_bits["101"].defect, 1)
        self.assertEqual(by_bits["111"].endpoint_word, (0, 1, 0))
        self.assertEqual(by_bits["111"].defect, 0)

    def test_light_leaf_graded_ranks_by_endpoint(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        basis = bott_samelson_light_leaves(diagram, [0, 1, 0])
        ranks = basis.graded_rank_by_endpoint(degree="light_leaf")

        self.assert_polynomials_equal(ranks[()], v**3 + v)
        self.assert_polynomials_equal(ranks[(0,)], v**2 + 1)
        self.assert_polynomials_equal(ranks[(1,)], v**2)
        self.assert_polynomials_equal(ranks[(0, 1)], v)
        self.assert_polynomials_equal(ranks[(1, 0)], v)
        self.assert_polynomials_equal(ranks[(0, 1, 0)], 1)

    def test_unshifted_module_grading(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        basis = bott_samelson_light_leaves(diagram, [0, 1], normalization="unshifted")

        self.assertEqual([leaf.module_degree for leaf in basis.leaves], [0, 2, 2, 4])
        self.assert_polynomials_equal(basis.graded_rank(), 1 + 2 * v**2 + v**4)

    def test_positive_generator_as_free_left_r_complex(self):
        diagram = DynkinDiagram.from_data([0], [])
        complex_ = rouquier_complex_as_free_left_r_modules(diagram, [(0, "+")])
        x = complex_.r_variables[0]

        self.assertEqual(complex_.degrees, [-1, 0])
        self.assertEqual(complex_.module_rank(-1), 2)
        self.assertEqual(complex_.module_rank(0), 1)
        self.assertEqual([basis.q_degree for basis in complex_.basis(-1)], [0, 2])
        self.assertEqual(complex_.differential(-1), sp.Matrix([[1, x]]))
        self.assertTrue(complex_.check_d_squared())

    def test_negative_generator_as_free_left_r_complex(self):
        diagram = DynkinDiagram.from_data([0], [])
        complex_ = rouquier_complex_as_free_left_r_modules(diagram, [(0, "-")])
        x = complex_.r_variables[0]

        self.assertEqual(complex_.degrees, [0, 1])
        self.assertEqual(complex_.module_rank(0), 1)
        self.assertEqual(complex_.module_rank(1), 2)
        self.assertEqual([basis.q_degree for basis in complex_.basis(1)], [-2, 0])
        self.assertEqual(complex_.differential(0), sp.Matrix([[x], [1]]))
        self.assertTrue(complex_.check_d_squared())

    def test_two_crossing_rouquier_differential_squares_to_zero(self):
        diagram = DynkinDiagram.from_data([0], [])
        complex_ = rouquier_complex_as_free_left_r_modules(
            diagram,
            [(0, "+"), (0, "+")],
        )

        self.assertEqual(complex_.degrees, [-2, -1, 0])
        self.assertTrue(complex_.check_d_squared())

    def test_type_a2_rouquier_matrices_have_left_r_entries(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        complex_ = rouquier_complex_as_free_left_r_modules(
            diagram,
            [(0, "+"), (1, "-"), (0, "+")],
        )
        r_symbols = set(complex_.r_variables)

        for matrix in complex_.differentials.values():
            for entry in matrix:
                self.assertLessEqual(entry.free_symbols, r_symbols)
        self.assertTrue(complex_.check_d_squared())


if __name__ == "__main__":
    unittest.main()
