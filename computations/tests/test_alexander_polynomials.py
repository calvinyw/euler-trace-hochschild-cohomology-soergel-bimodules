import unittest

import sympy as sp

from computations.alexander_polynomials import (
    alexander_polynomial_determinant,
    alexander_polynomial_forest,
    matching_numbers_forest,
)


t = sp.Symbol("t")


class AlexanderPolynomialTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.expand(left - right), 0)

    def test_single_arrow_normalizations(self):
        arrows = [(0, 1)]

        determinant = alexander_polynomial_determinant(2, arrows, t=t)
        forest_determinant = alexander_polynomial_forest(
            2,
            arrows,
            t=t,
            normalization="determinant",
        )
        forest_corollary = alexander_polynomial_forest(2, arrows, t=t)

        self.assert_polynomials_equal(determinant, t**2 - t + 1)
        self.assert_polynomials_equal(forest_determinant, determinant)
        self.assert_polynomials_equal(forest_corollary, t**-1 * determinant)

    def test_line_quivers_match_known_type_a_formula(self):
        for n in range(1, 8):
            arrows = [(i, i + 1) for i in range(n - 1)]
            determinant = alexander_polynomial_determinant(n, arrows, t=t)
            forest = alexander_polynomial_forest(
                n,
                arrows,
                t=t,
                normalization="determinant",
            )
            known_type_a = sum((-1) ** (n - k) * t**k for k in range(n + 1))

            self.assert_polynomials_equal(determinant, forest)
            self.assert_polynomials_equal(determinant, known_type_a)

    def test_e6_matching_numbers_and_polynomial(self):
        arrows = [(0, 1), (1, 3), (2, 3), (3, 4), (4, 5)]
        expected = t**6 - t**5 + t**3 - t + 1

        self.assertEqual(matching_numbers_forest(6, arrows), [1, 5, 5, 1])
        self.assert_polynomials_equal(
            alexander_polynomial_forest(6, arrows, t=t, normalization="determinant"),
            expected,
        )
        self.assert_polynomials_equal(
            alexander_polynomial_determinant(6, arrows, t=t),
            expected,
        )

    def test_forest_formula_ignores_orientation(self):
        vertices = range(5)
        first_orientation = [(0, 1), (2, 1), (1, 3), (4, 3)]
        second_orientation = [(1, 0), (1, 2), (3, 1), (3, 4)]

        first_forest = alexander_polynomial_forest(
            vertices,
            first_orientation,
            t=t,
            normalization="determinant",
        )
        second_forest = alexander_polynomial_forest(
            vertices,
            second_orientation,
            t=t,
            normalization="determinant",
        )
        first_determinant = alexander_polynomial_determinant(vertices, first_orientation, t=t)
        second_determinant = alexander_polynomial_determinant(vertices, second_orientation, t=t)

        self.assert_polynomials_equal(first_forest, second_forest)
        self.assert_polynomials_equal(first_forest, first_determinant)
        self.assert_polynomials_equal(second_forest, second_determinant)

    def test_non_forest_is_rejected_by_forest_formula(self):
        with self.assertRaisesRegex(ValueError, "not a forest"):
            alexander_polynomial_forest(3, [(0, 1), (1, 2), (0, 2)], t=t)


if __name__ == "__main__":
    unittest.main()
