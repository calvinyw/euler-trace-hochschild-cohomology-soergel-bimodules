"""Bounded unreduced Khovanov--Rozansky computations by linear algebra.

This module computes the unreduced Soergel-bimodule model for HHH without
setting the leftmost variables ``z_{0,i}`` equal to zero.  Unlike
``khovanov_rozansky.py``, it does not use Groebner bases or normal-form
calculations.  Instead, it fixes a maximum internal ``Q``-degree and replaces
every homogeneous polynomial quotient by the finite vector-space quotient

    S_d / I_d

in each total polynomial degree ``d`` needed below that cutoff.  Here ``I_d``
is spanned by all degree-``d`` multiples of the Bott-Samelson relations.  The
Koszul and Rouquier differentials preserve the total ``Q``-degree, so the HHH
groups in degrees ``q <= max_q_degree`` are computed by finite-dimensional
linear algebra over ``QQ``.

The main entry point is ``khovanov_rozansky_bounded_cohomology``.  It returns a
finite Laurent polynomial

    sum_{q <= max_q_degree} dim HHH^{a,r,q} A^a Q^q T^r,

so it should be read as a bounded truncation of the unreduced invariant rather
than as the full Hilbert series.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import TypeAlias

import sympy as sp

try:  # pragma: no cover - exercised when run as a package.
    from .khovanov_rozansky import (
        A,
        Q,
        T,
        DEFAULT_SHIFTS,
        BraidLetter,
        DynkinDiagram,
        RouquierComplex,
        RouquierTerm,
        ShiftConvention,
        Vertex,
        parse_braid,
        parse_edges,
        parse_vertices,
        rouquier_complex,
    )
except ImportError:  # pragma: no cover - useful for direct script execution.
    from khovanov_rozansky import (  # type: ignore
        A,
        Q,
        T,
        DEFAULT_SHIFTS,
        BraidLetter,
        DynkinDiagram,
        RouquierComplex,
        RouquierTerm,
        ShiftConvention,
        Vertex,
        parse_braid,
        parse_edges,
        parse_vertices,
        rouquier_complex,
    )


Exponent: TypeAlias = tuple[int, ...]
PolyDict: TypeAlias = dict[Exponent, sp.Rational]


@dataclass
class HomogeneousQuotient:
    """Linear model for ``S_d / I_d`` in one homogeneous degree."""

    degree: int
    monomials: tuple[Exponent, ...]
    rref_relations: sp.Matrix
    pivots: tuple[int, ...]
    free_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        self.monomial_index = {exponent: index for index, exponent in enumerate(self.monomials)}
        self.free_index = {column: index for index, column in enumerate(self.free_columns)}

    @property
    def dimension(self) -> int:
        return len(self.free_columns)

    def project_coefficients(self, coefficients: Sequence[sp.Rational]) -> sp.Matrix:
        """Project a vector in ``S_d`` to quotient coordinates."""

        coords = sp.zeros(self.dimension, 1)
        for column in self.free_columns:
            value = coefficients[column]
            if value:
                coords[self.free_index[column], 0] += value

        for row, pivot in enumerate(self.pivots):
            pivot_value = coefficients[pivot]
            if not pivot_value:
                continue
            for column in self.free_columns:
                relation_value = self.rref_relations[row, column]
                if relation_value:
                    coords[self.free_index[column], 0] -= pivot_value * relation_value

        return coords


class BoundedQuotientModel:
    """Degree-bounded vector-space model for one Bott-Samelson quotient."""

    def __init__(
        self,
        diagram: DynkinDiagram,
        term: RouquierTerm,
        *,
        max_q_degree: int,
        shifts: ShiftConvention = DEFAULT_SHIFTS,
        max_monomials: int = 20000,
    ) -> None:
        self.diagram = diagram
        self.term = term
        self.max_q_degree = max_q_degree
        self.shifts = shifts
        self.max_monomials = max_monomials
        self.layer_count = len(term.word) + 1
        self.variable_count = self.layer_count * diagram.rank
        self.vertex_positions = {
            vertex: position for position, vertex in enumerate(diagram.vertices)
        }
        self.relations = self._relations()
        self.max_variable_degree = self._max_variable_degree()

        self.degree_spaces: dict[int, HomogeneousQuotient] = {}
        self.basis_exponents: list[Exponent] = []
        self.basis_degrees: list[int] = []
        self.basis_q_degrees: list[int] = []
        self._degree_local_to_global: dict[tuple[int, int], int] = {}
        self._build_degree_spaces()

    @property
    def dimension(self) -> int:
        return len(self.basis_exponents)

    def variable_index(self, layer: int, vertex: Vertex) -> int:
        return layer * self.diagram.rank + self.vertex_positions[vertex]

    def _max_variable_degree(self) -> int:
        if self.variable_count == 0:
            return 0 if self.term.q_shift <= self.max_q_degree else -1

        candidates = []
        for hochschild_degree in range(self.diagram.rank + 1):
            numerator = (
                self.max_q_degree
                - self.term.q_shift
                - hochschild_degree * self.shifts.koszul_dual_q_shift
            )
            if numerator >= 0:
                candidates.append(numerator // self.shifts.variable_q_degree)
        return max(candidates, default=-1)

    def _relations(self) -> list[tuple[int, PolyDict]]:
        relations: list[tuple[int, PolyDict]] = []
        for position, generator in enumerate(self.term.word, start=1):
            left = position - 1
            right = position
            for vertex in self.diagram.vertices:
                if vertex == generator:
                    continue
                coefficients = self.diagram.invariant_linear_coefficients(generator, vertex)
                relation: PolyDict = {}
                for coefficient_vertex, coefficient in coefficients.items():
                    _add_poly_term(
                        relation,
                        _unit_exponent(
                            self.variable_count,
                            self.variable_index(left, coefficient_vertex),
                        ),
                        coefficient,
                    )
                    _add_poly_term(
                        relation,
                        _unit_exponent(
                            self.variable_count,
                            self.variable_index(right, coefficient_vertex),
                        ),
                        -coefficient,
                    )
                relations.append((1, relation))

            alpha_left = self.variable_index(left, generator)
            alpha_right = self.variable_index(right, generator)
            relation = {}
            _add_poly_term(
                relation,
                _unit_exponent(self.variable_count, alpha_left, power=2),
                sp.Rational(1),
            )
            _add_poly_term(
                relation,
                _unit_exponent(self.variable_count, alpha_right, power=2),
                sp.Rational(-1),
            )
            relations.append((2, relation))
        return relations

    def _build_degree_spaces(self) -> None:
        if self.max_variable_degree < 0:
            return

        for degree in range(self.max_variable_degree + 1):
            monomials = _monomials_of_degree(self.variable_count, degree)
            if len(monomials) > self.max_monomials:
                raise ValueError(
                    f"degree {degree} has {len(monomials)} monomials; increase "
                    "max_monomials if this bounded computation is intentional"
                )
            monomial_index = {exponent: index for index, exponent in enumerate(monomials)}
            relation_rows = self._relation_rows_for_degree(
                degree,
                monomials,
                monomial_index,
            )
            relation_matrix = (
                sp.Matrix(relation_rows)
                if relation_rows
                else sp.zeros(0, len(monomials))
            )
            rref, pivots = relation_matrix.rref()
            pivot_set = set(pivots)
            free_columns = tuple(
                column for column in range(len(monomials)) if column not in pivot_set
            )
            space = HomogeneousQuotient(
                degree=degree,
                monomials=monomials,
                rref_relations=rref,
                pivots=tuple(pivots),
                free_columns=free_columns,
            )
            self.degree_spaces[degree] = space

            for local_index, column in enumerate(free_columns):
                self._degree_local_to_global[(degree, local_index)] = len(self.basis_exponents)
                exponent = monomials[column]
                self.basis_exponents.append(exponent)
                self.basis_degrees.append(degree)
                self.basis_q_degrees.append(
                    self.term.q_shift + self.shifts.variable_q_degree * degree
                )

    def _relation_rows_for_degree(
        self,
        degree: int,
        monomials: Sequence[Exponent],
        monomial_index: dict[Exponent, int],
    ) -> list[list[sp.Rational]]:
        rows: list[list[sp.Rational]] = []
        for relation_degree, relation in self.relations:
            if relation_degree > degree:
                continue
            for multiplier in _monomials_of_degree(
                self.variable_count,
                degree - relation_degree,
            ):
                row = [sp.Rational(0) for _ in monomials]
                for exponent, coefficient in relation.items():
                    shifted = _add_exponents(exponent, multiplier)
                    row[monomial_index[shifted]] += coefficient
                if any(row):
                    rows.append(row)
        return rows

    def diagonal_difference(self, vertex: Vertex) -> PolyDict:
        left = self.variable_index(0, vertex)
        right = self.variable_index(self.layer_count - 1, vertex)
        result: PolyDict = {}
        _add_poly_term(result, _unit_exponent(self.variable_count, left), sp.Rational(1))
        _add_poly_term(result, _unit_exponent(self.variable_count, right), sp.Rational(-1))
        return result

    def project_polynomial(self, polynomial: PolyDict) -> sp.Matrix:
        result = sp.zeros(self.dimension, 1)
        if not polynomial:
            return result

        by_degree: dict[int, list[tuple[Exponent, sp.Rational]]] = defaultdict(list)
        for exponent, coefficient in polynomial.items():
            if coefficient:
                by_degree[sum(exponent)].append((exponent, coefficient))

        for degree, terms in by_degree.items():
            space = self.degree_spaces.get(degree)
            if space is None:
                continue
            coefficients = [sp.Rational(0) for _ in space.monomials]
            for exponent, coefficient in terms:
                coefficients[space.monomial_index[exponent]] += coefficient
            local = space.project_coefficients(coefficients)
            for local_index, value in enumerate(local):
                if value:
                    global_index = self._degree_local_to_global[(degree, local_index)]
                    result[global_index, 0] += value
        return result

    def map_to(self, target: "BoundedQuotientModel", arrow) -> sp.Matrix:
        """Matrix of a Rouquier differential summand on quotient spaces."""

        if arrow.kind == "multiplication":
            variable_map = self._multiplication_variable_map(target, arrow.b_position)
            multiplier = _constant_polynomial(target.variable_count)
        elif arrow.kind == "coevaluation":
            variable_map = self._coevaluation_variable_map(target, arrow.b_position)
            left = target.variable_index(arrow.b_position, arrow.generator)
            right = target.variable_index(arrow.b_position + 1, arrow.generator)
            multiplier = {}
            _add_poly_term(multiplier, _unit_exponent(target.variable_count, left), sp.Rational(1))
            _add_poly_term(multiplier, _unit_exponent(target.variable_count, right), sp.Rational(1))
        else:
            raise ValueError(f"unknown Rouquier arrow kind {arrow.kind!r}")

        matrix = sp.zeros(target.dimension, self.dimension)
        for column, exponent in enumerate(self.basis_exponents):
            image = _substitute_monomial(exponent, variable_map, target.variable_count)
            image = _multiply_polynomials(image, multiplier)
            vector = target.project_polynomial(image)
            for row, coefficient in enumerate(vector):
                if coefficient:
                    matrix[row, column] += arrow.sign * coefficient
        return matrix

    def _multiplication_variable_map(
        self,
        target: "BoundedQuotientModel",
        b_position: int,
    ) -> list[int]:
        variable_map = []
        for layer in range(self.layer_count):
            target_layer = layer if layer <= b_position else layer - 1
            for vertex in self.diagram.vertices:
                variable_map.append(target.variable_index(target_layer, vertex))
        return variable_map

    def _coevaluation_variable_map(
        self,
        target: "BoundedQuotientModel",
        b_position: int,
    ) -> list[int]:
        variable_map = []
        for layer in range(self.layer_count):
            target_layer = layer if layer <= b_position else layer + 1
            for vertex in self.diagram.vertices:
                variable_map.append(target.variable_index(target_layer, vertex))
        return variable_map


@dataclass
class BoundedKoszulComplex:
    model: BoundedQuotientModel
    wedge_bases: dict[int, list[tuple[int, ...]]]
    basis: dict[int, list[tuple[int, tuple[int, ...]]]]
    basis_index: dict[int, dict[tuple[int, tuple[int, ...]], int]]
    q_degrees: dict[int, list[int]]
    differentials: dict[int, sp.Matrix]


@dataclass
class HomologyBlock:
    ambient_indices: list[int]
    image_dimension: int
    cohomology_indices: list[int]
    combined_basis: sp.Matrix


@dataclass
class CohomologyData:
    ambient_dimension: int
    q_degrees: list[int]
    representatives: sp.Matrix
    blocks: dict[int, HomologyBlock]

    @property
    def dimension(self) -> int:
        return len(self.q_degrees)

    def coordinates(self, vector: sp.Matrix) -> sp.Matrix:
        coordinates = sp.zeros(self.dimension, 1)
        if self.dimension == 0:
            return coordinates
        for _q_degree, block in self.blocks.items():
            restricted = sp.Matrix([vector[index, 0] for index in block.ambient_indices])
            if all(entry == 0 for entry in restricted):
                continue
            solution, parameters = block.combined_basis.gauss_jordan_solve(restricted)
            substitutions = {parameter: 0 for parameter in list(parameters)}
            solution = solution.xreplace(substitutions)
            offset = block.image_dimension
            for local_index, global_index in enumerate(block.cohomology_indices):
                coordinates[global_index, 0] = solution[offset + local_index, 0]
        return coordinates


@dataclass
class BoundedExtTermData:
    term: RouquierTerm
    model: BoundedQuotientModel
    koszul: BoundedKoszulComplex
    cohomology: dict[int, CohomologyData]


@dataclass
class BoundedExtComputation:
    rouquier: RouquierComplex
    term_data: dict[tuple[int, ...], BoundedExtTermData]
    induced_maps: dict[tuple[int, int], sp.Matrix]


@dataclass
class BoundedKRResult:
    polynomial: sp.Expr
    ext: BoundedExtComputation
    homology_dimensions: dict[tuple[int, int, int], int]
    max_q_degree: int


def khovanov_rozansky_bounded_cohomology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedKRResult:
    """Compute unreduced HHH through a bounded internal ``Q``-degree.

    The leftmost polynomial variables are kept.  The answer is the finite
    Laurent polynomial containing exactly the computed summands with
    ``q <= max_q_degree``.
    """

    rouquier = rouquier_complex(diagram, braid, shifts=shifts)
    ext = bounded_koszul_ext_complex(
        rouquier,
        max_q_degree=max_q_degree,
        max_monomials=max_monomials,
        validate=validate,
    )
    homology_dimensions = _horizontal_homology_dimensions(rouquier, ext, validate=validate)

    polynomial = sp.Integer(0)
    for (hochschild_degree, rouquier_degree, q_degree), dimension in sorted(
        homology_dimensions.items()
    ):
        if q_degree <= max_q_degree:
            polynomial += dimension * A**hochschild_degree * Q**q_degree * T**rouquier_degree

    return BoundedKRResult(
        polynomial=sp.expand(polynomial),
        ext=ext,
        homology_dimensions=homology_dimensions,
        max_q_degree=max_q_degree,
    )


def compute_bounded_hhh(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedKRResult:
    """Short alias for ``khovanov_rozansky_bounded_cohomology``."""

    return khovanov_rozansky_bounded_cohomology(
        diagram,
        braid,
        max_q_degree=max_q_degree,
        shifts=shifts,
        max_monomials=max_monomials,
        validate=validate,
    )


def bounded_koszul_complex(
    model: BoundedQuotientModel,
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
) -> BoundedKoszulComplex:
    rank = model.diagram.rank
    wedge_bases = {
        degree: [tuple(combo) for combo in _combinations(range(rank), degree)]
        for degree in range(rank + 1)
    }

    basis: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    basis_index: dict[int, dict[tuple[int, tuple[int, ...]], int]] = {}
    q_degrees: dict[int, list[int]] = {}
    for degree in range(rank + 1):
        entries = []
        degrees = []
        for wedge in wedge_bases[degree]:
            for monomial_index, monomial_q_degree in enumerate(model.basis_q_degrees):
                q_degree = monomial_q_degree + degree * shifts.koszul_dual_q_shift
                if q_degree <= max_q_degree:
                    entries.append((monomial_index, wedge))
                    degrees.append(q_degree)
        basis[degree] = entries
        basis_index[degree] = {entry: index for index, entry in enumerate(entries)}
        q_degrees[degree] = degrees

    multiplication_by_diagonal = {
        vertex_position: _multiplication_matrix(model, model.diagonal_difference(vertex))
        for vertex_position, vertex in enumerate(model.diagram.vertices)
    }

    differentials = {}
    for degree in range(rank):
        source = basis[degree]
        target_index = basis_index[degree + 1]
        matrix = sp.zeros(len(basis[degree + 1]), len(source))
        for column, (monomial_index, wedge) in enumerate(source):
            wedge_set = set(wedge)
            for vertex_position in range(rank):
                if vertex_position in wedge_set:
                    continue
                insertion_position = sum(1 for item in wedge if item < vertex_position)
                sign = -1 if insertion_position % 2 else 1
                target_wedge = tuple(sorted((*wedge, vertex_position)))
                multiplication_column = multiplication_by_diagonal[vertex_position][
                    :,
                    monomial_index,
                ]
                for target_monomial, coefficient in enumerate(multiplication_column):
                    if coefficient:
                        row = target_index.get((target_monomial, target_wedge))
                        if row is not None:
                            matrix[row, column] += sign * coefficient
        differentials[degree] = matrix
    differentials[rank] = sp.zeros(0, len(basis[rank]))

    return BoundedKoszulComplex(
        model=model,
        wedge_bases=wedge_bases,
        basis=basis,
        basis_index=basis_index,
        q_degrees=q_degrees,
        differentials=differentials,
    )


def bounded_koszul_ext_complex(
    rouquier: RouquierComplex,
    *,
    max_q_degree: int,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedExtComputation:
    term_data: dict[tuple[int, ...], BoundedExtTermData] = {}
    for choices, term in sorted(rouquier.terms.items(), key=lambda item: item[1].term_id):
        model = BoundedQuotientModel(
            rouquier.diagram,
            term,
            max_q_degree=max_q_degree,
            shifts=rouquier.shifts,
            max_monomials=max_monomials,
        )
        koszul = bounded_koszul_complex(
            model,
            max_q_degree=max_q_degree,
            shifts=rouquier.shifts,
        )
        cohomology = {}
        rank = rouquier.diagram.rank
        for degree in range(rank + 1):
            previous = (
                koszul.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(koszul.basis[degree]), 0)
            )
            next_map = koszul.differentials[degree]
            previous_degrees = koszul.q_degrees[degree - 1] if degree > 0 else []
            cohomology[degree] = _graded_homology_basis(
                previous,
                next_map,
                previous_degrees,
                koszul.q_degrees[degree],
                koszul.q_degrees[degree + 1] if degree < rank else [],
                validate=validate,
                context=f"bounded Koszul term {term.choices}, A-degree {degree}",
            )
        term_data[choices] = BoundedExtTermData(term, model, koszul, cohomology)

    induced_maps = {}
    for arrow in rouquier.arrows:
        source_data = term_data[arrow.source]
        target_data = term_data[arrow.target]
        term_map = source_data.model.map_to(target_data.model, arrow)
        for degree in range(rouquier.diagram.rank + 1):
            chain_map = _koszul_chain_map(
                source_data.koszul,
                target_data.koszul,
                degree,
                term_map,
            )
            if validate:
                _assert_degree_zero(
                    chain_map,
                    source_data.koszul.q_degrees[degree],
                    target_data.koszul.q_degrees[degree],
                    f"bounded Rouquier arrow {arrow.arrow_id} on A-degree {degree}",
                )
            source_cohomology = source_data.cohomology[degree]
            target_cohomology = target_data.cohomology[degree]
            induced = sp.zeros(target_cohomology.dimension, source_cohomology.dimension)
            for column in range(source_cohomology.dimension):
                representative = source_cohomology.representatives[:, column]
                image = chain_map * representative
                coordinates = target_cohomology.coordinates(image)
                for row, coefficient in enumerate(coordinates):
                    if coefficient:
                        induced[row, column] += coefficient
            induced_maps[(arrow.arrow_id, degree)] = induced

    return BoundedExtComputation(
        rouquier=rouquier,
        term_data=term_data,
        induced_maps=induced_maps,
    )


def _multiplication_matrix(model: BoundedQuotientModel, polynomial: PolyDict) -> sp.Matrix:
    matrix = sp.zeros(model.dimension, model.dimension)
    for column, exponent in enumerate(model.basis_exponents):
        monomial = {exponent: sp.Rational(1)}
        image = _multiply_polynomials(monomial, polynomial)
        vector = model.project_polynomial(image)
        for row, coefficient in enumerate(vector):
            if coefficient:
                matrix[row, column] = coefficient
    return matrix


def _koszul_chain_map(
    source: BoundedKoszulComplex,
    target: BoundedKoszulComplex,
    degree: int,
    term_map: sp.Matrix,
) -> sp.Matrix:
    matrix = sp.zeros(len(target.basis[degree]), len(source.basis[degree]))
    target_index = target.basis_index[degree]
    for column, (source_monomial, wedge) in enumerate(source.basis[degree]):
        image_column = term_map[:, source_monomial]
        for target_monomial, coefficient in enumerate(image_column):
            if coefficient:
                row = target_index.get((target_monomial, wedge))
                if row is not None:
                    matrix[row, column] += coefficient
    return matrix


def _horizontal_homology_dimensions(
    rouquier: RouquierComplex,
    ext: BoundedExtComputation,
    *,
    validate: bool,
) -> dict[tuple[int, int, int], int]:
    homology_dimensions: dict[tuple[int, int, int], int] = {}

    for hochschild_degree in range(rouquier.diagram.rank + 1):
        degrees = rouquier.degrees
        group_data = {}
        for degree in degrees:
            term_keys = [
                key
                for key, term in sorted(rouquier.terms.items(), key=lambda item: item[1].term_id)
                if term.degree == degree
            ]
            q_degrees = []
            offsets = {}
            offset = 0
            for key in term_keys:
                cohomology = ext.term_data[key].cohomology[hochschild_degree]
                offsets[key] = offset
                q_degrees.extend(cohomology.q_degrees)
                offset += cohomology.dimension
            group_data[degree] = {
                "term_keys": term_keys,
                "offsets": offsets,
                "dimension": offset,
                "q_degrees": q_degrees,
            }

        differentials = {}
        for degree in degrees:
            target_degree = degree + 1
            if target_degree not in group_data:
                differentials[degree] = sp.zeros(0, group_data[degree]["dimension"])
                continue
            source_dimension = group_data[degree]["dimension"]
            target_dimension = group_data[target_degree]["dimension"]
            matrix = sp.zeros(target_dimension, source_dimension)
            for arrow in rouquier.arrows:
                source_term = rouquier.terms[arrow.source]
                target_term = rouquier.terms[arrow.target]
                if source_term.degree != degree or target_term.degree != target_degree:
                    continue
                induced = ext.induced_maps[(arrow.arrow_id, hochschild_degree)]
                source_offset = group_data[degree]["offsets"][arrow.source]
                target_offset = group_data[target_degree]["offsets"][arrow.target]
                for row in range(induced.shape[0]):
                    for column in range(induced.shape[1]):
                        if induced[row, column]:
                            matrix[target_offset + row, source_offset + column] += induced[
                                row,
                                column,
                            ]
            differentials[degree] = matrix

        for degree in degrees:
            previous_degree = degree - 1
            previous = (
                differentials[previous_degree]
                if previous_degree in differentials
                else sp.zeros(group_data[degree]["dimension"], 0)
            )
            next_map = differentials[degree]
            previous_q_degrees = (
                group_data[previous_degree]["q_degrees"] if previous_degree in group_data else []
            )
            homology = _graded_homology_basis(
                previous,
                next_map,
                previous_q_degrees,
                group_data[degree]["q_degrees"],
                group_data[degree + 1]["q_degrees"] if degree + 1 in group_data else [],
                validate=validate,
                context=f"bounded horizontal A-degree {hochschild_degree}, T-degree {degree}",
            )
            for q_degree in homology.q_degrees:
                key = (hochschild_degree, degree, q_degree)
                homology_dimensions[key] = homology_dimensions.get(key, 0) + 1

    return homology_dimensions


def _graded_homology_basis(
    previous: sp.Matrix,
    next_map: sp.Matrix,
    previous_degrees: Sequence[int],
    current_degrees: Sequence[int],
    next_degrees: Sequence[int],
    *,
    validate: bool,
    context: str,
) -> CohomologyData:
    if validate:
        if next_map.shape[1] != previous.shape[0]:
            raise ValueError(f"{context}: incompatible differential dimensions")
        if previous.shape[1] and next_map.shape[0]:
            composition = next_map * previous
            if any(entry != 0 for entry in composition):
                raise ValueError(f"{context}: differentials do not compose to zero")
        _assert_degree_zero(previous, previous_degrees, current_degrees, f"{context} previous")
        _assert_degree_zero(next_map, current_degrees, next_degrees, f"{context} next")

    representatives = []
    representative_degrees = []
    blocks: dict[int, HomologyBlock] = {}
    current_by_q = _indices_by_degree(current_degrees)
    previous_by_q = _indices_by_degree(previous_degrees)
    next_by_q = _indices_by_degree(next_degrees)

    for q_degree in sorted(current_by_q):
        current_indices = current_by_q[q_degree]
        previous_indices = previous_by_q.get(q_degree, [])
        next_indices = next_by_q.get(q_degree, [])

        previous_block = _extract_matrix(previous, current_indices, previous_indices)
        next_block = _extract_matrix(next_map, next_indices, current_indices)
        image_columns = previous_block.columnspace()
        kernel_columns = next_block.nullspace()

        image_span = _matrix_from_columns(image_columns, len(current_indices))
        quotient_columns = _independent_extension(image_span, kernel_columns)

        if not quotient_columns:
            continue

        image_basis = _matrix_from_columns(image_columns, len(current_indices))
        quotient_basis = _matrix_from_columns(quotient_columns, len(current_indices))
        combined = image_basis.row_join(quotient_basis)
        cohomology_indices = []
        for quotient_vector in quotient_columns:
            full = sp.zeros(len(current_degrees), 1)
            for local_row, ambient_row in enumerate(current_indices):
                full[ambient_row, 0] = quotient_vector[local_row, 0]
            cohomology_indices.append(len(representatives))
            representatives.append(full)
            representative_degrees.append(q_degree)
        blocks[q_degree] = HomologyBlock(
            ambient_indices=current_indices,
            image_dimension=image_basis.shape[1],
            cohomology_indices=cohomology_indices,
            combined_basis=combined,
        )

    representative_matrix = _matrix_from_columns(representatives, len(current_degrees))
    return CohomologyData(
        ambient_dimension=len(current_degrees),
        q_degrees=representative_degrees,
        representatives=representative_matrix,
        blocks=blocks,
    )


def _independent_extension(
    image_span: sp.Matrix,
    candidate_columns: Sequence[sp.Matrix],
) -> list[sp.Matrix]:
    if not candidate_columns:
        return []
    image_count = image_span.shape[1]
    combined = (
        image_span.row_join(_matrix_from_columns(candidate_columns, image_span.shape[0]))
        if image_count
        else _matrix_from_columns(candidate_columns, candidate_columns[0].shape[0])
    )
    _, pivots = combined.rref()
    return [candidate_columns[index - image_count] for index in pivots if index >= image_count]


def _extract_matrix(matrix: sp.Matrix, rows: Sequence[int], columns: Sequence[int]) -> sp.Matrix:
    if not rows or not columns:
        return sp.zeros(len(rows), len(columns))
    return matrix.extract(rows, columns)


def _matrix_from_columns(columns: Sequence[sp.Matrix], row_count: int) -> sp.Matrix:
    if not columns:
        return sp.zeros(row_count, 0)
    return sp.Matrix.hstack(*columns)


def _indices_by_degree(degrees: Sequence[int]) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = defaultdict(list)
    for index, degree in enumerate(degrees):
        indices[degree].append(index)
    return dict(indices)


def _assert_degree_zero(
    matrix: sp.Matrix,
    source_degrees: Sequence[int],
    target_degrees: Sequence[int],
    context: str,
) -> None:
    if matrix.shape != (len(target_degrees), len(source_degrees)):
        raise ValueError(f"{context}: matrix shape does not match grading data")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if matrix[row, column] and target_degrees[row] != source_degrees[column]:
                raise ValueError(
                    f"{context}: nonzero entry shifts Q-degree "
                    f"{source_degrees[column]} -> {target_degrees[row]}"
                )


def _multiply_polynomials(left: PolyDict, right: PolyDict) -> PolyDict:
    result: PolyDict = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            _add_poly_term(
                result,
                _add_exponents(left_exponent, right_exponent),
                left_coefficient * right_coefficient,
            )
    return result


def _substitute_monomial(
    exponent: Exponent,
    variable_map: Sequence[int],
    target_variable_count: int,
) -> PolyDict:
    target = [0 for _ in range(target_variable_count)]
    for source_variable, power in enumerate(exponent):
        if power:
            target[variable_map[source_variable]] += power
    return {tuple(target): sp.Rational(1)}


def _constant_polynomial(variable_count: int) -> PolyDict:
    return {tuple(0 for _ in range(variable_count)): sp.Rational(1)}


def _add_poly_term(poly: PolyDict, exponent: Exponent, coefficient: sp.Rational) -> None:
    if not coefficient:
        return
    value = poly.get(exponent, sp.Rational(0)) + coefficient
    if value:
        poly[exponent] = sp.Rational(value)
    elif exponent in poly:
        del poly[exponent]


def _add_exponents(left: Exponent, right: Exponent) -> Exponent:
    return tuple(left_value + right_value for left_value, right_value in zip(left, right))


def _unit_exponent(variable_count: int, variable: int, *, power: int = 1) -> Exponent:
    exponent = [0 for _ in range(variable_count)]
    exponent[variable] = power
    return tuple(exponent)


@lru_cache(maxsize=None)
def _monomials_of_degree(variable_count: int, degree: int) -> tuple[Exponent, ...]:
    if degree < 0:
        return tuple()
    if variable_count == 0:
        return (tuple(),) if degree == 0 else tuple()
    if variable_count == 1:
        return ((degree,),)
    monomial_count = comb(variable_count + degree - 1, degree)
    if monomial_count == 0:
        return tuple()

    monomials = []
    for first_power in range(degree + 1):
        for rest in _monomials_of_degree(variable_count - 1, degree - first_power):
            monomials.append((first_power, *rest))
    return tuple(monomials)


def _combinations(items: range, degree: int):
    if degree == 0:
        yield tuple()
        return
    items_tuple = tuple(items)
    if degree > len(items_tuple):
        return
    if degree == 1:
        for item in items_tuple:
            yield (item,)
        return
    from itertools import combinations

    yield from combinations(items_tuple, degree)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", required=True, help="comma-separated vertices, e.g. 0,1,2")
    parser.add_argument("--edges", default="", help="comma-separated edges, e.g. 0-1,1-2")
    parser.add_argument("--braid", default="", help="comma-separated braid letters, e.g. 0:+,1:-")
    parser.add_argument(
        "--max-q-degree",
        type=int,
        required=True,
        help="compute all HHH summands with internal Q-degree at most this value",
    )
    parser.add_argument("--max-monomials", type=int, default=20000)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip differential and degree-zero consistency checks",
    )
    args = parser.parse_args(argv)

    diagram = DynkinDiagram.from_data(parse_vertices(args.vertices), parse_edges(args.edges))
    result = khovanov_rozansky_bounded_cohomology(
        diagram,
        parse_braid(args.braid),
        max_q_degree=args.max_q_degree,
        max_monomials=args.max_monomials,
        validate=not args.no_validate,
    )
    print(result.polynomial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
