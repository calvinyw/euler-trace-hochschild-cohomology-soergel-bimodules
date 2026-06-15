import unittest

import sympy as sp

from computations.khovanov_rozansky import (
    A,
    Q,
    T,
    DynkinDiagram,
    Realization,
    khovanov_rozansky_cohomology,
    polynomial_ring_hilbert_series,
    rouquier_complex,
)


class KhovanovRozanskyTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_rouquier_complex_for_positive_generator(self):
        diagram = DynkinDiagram.from_data([0], [])
        complex_ = rouquier_complex(diagram, [(0, "+")])

        self.assertEqual(sorted(term.degree for term in complex_.terms.values()), [-1, 0])
        self.assertEqual(len(complex_.arrows), 1)
        self.assertEqual(complex_.arrows[0].kind, "multiplication")

    def test_unreduced_is_the_default_hilbert_series(self):
        diagram = DynkinDiagram.from_data([0], [])
        unreduced = khovanov_rozansky_cohomology(diagram, [])
        reduced = khovanov_rozansky_cohomology(diagram, [], reduced=True)
        hilbert_factor = polynomial_ring_hilbert_series(diagram)

        self.assertFalse(unreduced.reduced)
        self.assertEqual(unreduced.hilbert_factor, hilbert_factor)
        # For a single Bott-Samelson term (no horizontal differential) the
        # unreduced HHH is free over R, so the honest computation agrees with
        # the freeness shortcut here.
        self.assert_polynomials_equal(unreduced.polynomial, reduced.polynomial * hilbert_factor)

    def test_unreduced_empty_braid_is_free_over_R(self):
        diagram = DynkinDiagram.from_data([0], [])
        result = khovanov_rozansky_cohomology(diagram, [])
        self.assert_polynomials_equal(result.polynomial, (1 + A / Q**2) / (1 - Q**2))

    def test_unreduced_inverse_pair_is_identity(self):
        diagram = DynkinDiagram.from_data([0], [])
        identity = khovanov_rozansky_cohomology(diagram, []).polynomial
        positive_negative = khovanov_rozansky_cohomology(
            diagram, [(0, "+"), (0, "-")]
        ).polynomial
        self.assert_polynomials_equal(positive_negative, identity)

    def test_unreduced_single_crossings(self):
        diagram = DynkinDiagram.from_data([0], [])
        positive = khovanov_rozansky_cohomology(diagram, [(0, "+")]).polynomial
        negative = khovanov_rozansky_cohomology(diagram, [(0, "-")]).polynomial
        self.assert_polynomials_equal(positive, sp.Integer(1))
        self.assert_polynomials_equal(negative, A * T / Q**4)

    def test_unreduced_is_cutoff_independent(self):
        diagram = DynkinDiagram.from_data([0], [])
        braid = [(0, "+"), (0, "+")]
        auto = khovanov_rozansky_cohomology(diagram, braid).polynomial
        fixed = khovanov_rozansky_cohomology(
            diagram, braid, max_total_degree=9
        ).polynomial
        self.assert_polynomials_equal(auto, fixed)

    def test_unreduced_euler_characteristic_matches_shortcut(self):
        # The graded Euler characteristic does not depend on the freeness
        # assumption, so the honest series must agree with the shortcut after
        # specializing T -> -1 and A -> -1, even when the series themselves
        # differ.
        diagram = DynkinDiagram.from_data([0], [])
        for braid in ([(0, "+")], [(0, "+"), (0, "+")], [(0, "-")]):
            honest = khovanov_rozansky_cohomology(diagram, braid).polynomial
            reduced = khovanov_rozansky_cohomology(diagram, braid, reduced=True).polynomial
            shortcut = reduced * polynomial_ring_hilbert_series(diagram)
            substitution = {T: -1, A: -1}
            self.assertEqual(
                sp.simplify(honest.xreplace(substitution) - shortcut.xreplace(substitution)),
                0,
            )

    def test_empty_braid_is_reduced_exterior_algebra(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        result = khovanov_rozansky_cohomology(diagram, [], reduced=True)

        self.assert_polynomials_equal(result.polynomial, (1 + A / Q**2) ** 2)

    def test_rank_one_inverse_pairs_are_identity(self):
        diagram = DynkinDiagram.from_data([0], [])
        identity = khovanov_rozansky_cohomology(diagram, [], reduced=True).polynomial
        positive_negative = khovanov_rozansky_cohomology(
            diagram,
            [(0, "+"), (0, "-")],
            reduced=True,
        ).polynomial
        negative_positive = khovanov_rozansky_cohomology(
            diagram,
            [(0, "-"), (0, "+")],
            reduced=True,
        ).polynomial

        self.assert_polynomials_equal(positive_negative, identity)
        self.assert_polynomials_equal(negative_positive, identity)

    def test_type_a2_braid_relation(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        left = khovanov_rozansky_cohomology(
            diagram,
            [(0, "+"), (1, "+"), (0, "+")],
            reduced=True,
        ).polynomial
        right = khovanov_rozansky_cohomology(
            diagram,
            [(1, "+"), (0, "+"), (1, "+")],
            reduced=True,
        ).polynomial

        self.assert_polynomials_equal(left, right)

    def test_single_crossing_outputs_are_laurent_polynomials(self):
        diagram = DynkinDiagram.from_data([0], [])
        positive = khovanov_rozansky_cohomology(
            diagram,
            [(0, "+")],
            reduced=True,
        ).polynomial
        negative = khovanov_rozansky_cohomology(
            diagram,
            [(0, "-")],
            reduced=True,
        ).polynomial

        self.assert_polynomials_equal(positive, 1 + Q**2 / T)
        self.assert_polynomials_equal(negative, A / Q**2 + A * T / Q**4)


class RealizationTests(unittest.TestCase):
    def assert_polynomials_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_standard_realization_reproduces_default(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        braid = [(0, "+"), (1, "+"), (0, "+")]
        default = khovanov_rozansky_cohomology(diagram, braid, reduced=True).polynomial
        explicit = khovanov_rozansky_cohomology(
            diagram,
            braid,
            reduced=True,
            realization=Realization.standard(diagram),
        ).polynomial
        self.assert_polynomials_equal(default, explicit)

    def test_user_supplied_matrices_match_standard(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        standard = Realization.standard(diagram)
        custom = Realization.from_matrices(
            diagram, {vertex: standard.action[vertex] for vertex in diagram.vertices}, dual=True
        )
        braid = [(0, "-"), (1, "+")]
        default = khovanov_rozansky_cohomology(diagram, braid, reduced=True).polynomial
        viacustom = khovanov_rozansky_cohomology(
            diagram, braid, reduced=True, realization=custom
        ).polynomial
        self.assert_polynomials_equal(default, viacustom)

    def test_higher_dimensional_representation_changes_rank(self):
        # The 3-dimensional permutation representation of S_3 = W(A_2).
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        permutation = Realization.from_matrices(
            diagram,
            {
                0: sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                1: sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            },
        )
        self.assertEqual(permutation.dim, 3)
        empty = khovanov_rozansky_cohomology(
            diagram, [], reduced=True, realization=permutation
        ).polynomial
        self.assert_polynomials_equal(empty, (1 + A / Q**2) ** 3)
        # The braid relation is still an invariance of the construction.
        left = khovanov_rozansky_cohomology(
            diagram, [(0, "+"), (1, "+"), (0, "+")], reduced=True, realization=permutation
        ).polynomial
        right = khovanov_rozansky_cohomology(
            diagram, [(1, "+"), (0, "+"), (1, "+")], reduced=True, realization=permutation
        ).polynomial
        self.assert_polynomials_equal(left, right)

    def test_rejects_non_involution(self):
        diagram = DynkinDiagram.from_data([0], [])
        with self.assertRaises(ValueError):
            Realization.from_matrices(diagram, {0: sp.Matrix([[1, 1], [0, 1]])}, dual=True)

    def test_rejects_non_reflection(self):
        diagram = DynkinDiagram.from_data([0], [])
        # An involution whose (-1)-eigenspace is 2-dimensional is not a reflection.
        with self.assertRaises(ValueError):
            Realization.from_matrices(diagram, {0: -sp.eye(2)}, dual=True)

    def test_rejects_braid_relation_failure(self):
        diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
        # Two reflections that commute violate the order-3 braid relation.
        with self.assertRaises(ValueError):
            Realization.from_matrices(
                diagram,
                {0: sp.Matrix([[0, 1], [1, 0]]), 1: sp.Matrix([[1, 0], [0, -1]])},
                dual=True,
            )


if __name__ == "__main__":
    unittest.main()
