import unittest

import sympy as sp

from computations.slower_old_euler_trace.euler_trace_bounded import (
    khovanov_rozansky_bounded_euler_trace,
)
from computations.khovanov_rozansky import A, Q, DynkinDiagram


class BoundedEulerTraceTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_empty_braid_rank_one_keeps_bounded_r_tail(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_bounded_euler_trace(
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
        result = khovanov_rozansky_bounded_euler_trace(
            diagram,
            [(0, "+")],
            max_q_degree=4,
        )

        self.assert_polynomials_equal(result.polynomial, 1)

    def test_type_a2_length_four_trace(self):
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        result = khovanov_rozansky_bounded_euler_trace(
            diagram,
            [(1, "+"), (2, "+"), (1, "+"), (2, "+")],
            max_q_degree=4,
        )

        self.assert_polynomials_equal(result.polynomial, A + Q**4 + 1)


if __name__ == "__main__":
    unittest.main()
