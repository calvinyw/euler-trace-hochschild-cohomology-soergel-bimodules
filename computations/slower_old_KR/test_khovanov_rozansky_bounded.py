import unittest

import sympy as sp

from computations.khovanov_rozansky import A, Q, DynkinDiagram
from computations.slower_old_KR.khovanov_rozansky_bounded import (
    compute_bounded_hhh,
    khovanov_rozansky_bounded_cohomology,
)


class BoundedKhovanovRozanskyTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_keeps_leftmost_polynomial_variable(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_bounded_cohomology(
            diagram,
            [],
            max_q_degree=4,
        )

        self.assert_polynomials_equal(
            result.polynomial,
            1 + Q**2 + Q**4 + A / Q**2 + A + A * Q**2 + A * Q**4,
        )

    def test_positive_crossing_rank_one(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = compute_bounded_hhh(
            diagram,
            [(0, "+")],
            max_q_degree=4,
        )

        self.assert_polynomials_equal(result.polynomial, 1)

    def test_inverse_pair_is_identity_through_cutoff(self):
        diagram = DynkinDiagram.from_data([0], [])
        identity = compute_bounded_hhh(diagram, [], max_q_degree=4).polynomial
        pair = compute_bounded_hhh(
            diagram,
            [(0, "+"), (0, "-")],
            max_q_degree=4,
        ).polynomial

        self.assert_polynomials_equal(pair, identity)


if __name__ == "__main__":
    unittest.main()
