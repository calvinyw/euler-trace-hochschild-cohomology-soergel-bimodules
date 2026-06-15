"""Reduced Khovanov--Rozansky homology from Soergel bimodules.

This module computes the reduced Soergel-module version of Khovanov--Rozansky
homology,

    H(C tensor_R HH),

where ``C`` is the Rouquier complex of an Artin braid and ``HH`` is the
Hochschild cohomology of each Bott--Samelson term, computed via the Koszul
resolution for the diagonal ideal ``(x_i - y_i)``.

The implementation is intentionally explicit.  A Bott--Samelson term

    B_{s_1} tensor_R ... tensor_R B_{s_k}

is represented as the coordinate ring with variables for ``k + 1`` tensor
positions.  The relation for the factor ``B_s`` between adjacent positions is
that the standard generators of ``R^s`` agree: all invariant linear forms agree,
and ``alpha_s^2`` agrees.

After forming each term, the leftmost polynomial variables are set to zero.
This reduced specialization turns the Koszul complexes into finite-dimensional
vector spaces and produces a Laurent polynomial in ``A,Q,T``.

The default internal grading convention has ``deg(alpha_i) = 2`` and uses the
cohomological convention in the manuscript:

* a selected positive ``B_s(-1)`` term has ``Q``-shift ``0``;
* a selected negative ``B_s(1)`` term has ``Q``-shift ``-2``;
* a Koszul dual generator has ``Q``-shift ``-2`` and contributes one ``A``.

With these shifts the multiplication map and the coevaluation/dot map are
degree-zero maps for the finite-dimensional complexes computed here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import TypeAlias

import sympy as sp


Vertex: TypeAlias = int
Edge: TypeAlias = tuple[Vertex, Vertex]
BraidLetter: TypeAlias = tuple[Vertex, int | str]


A, Q, T = sp.symbols("A Q T")


@dataclass(frozen=True)
class ShiftConvention:
    """Internal grading shifts used by the finite-dimensional model."""

    variable_q_degree: int = 2
    positive_b_q_shift: int = 0
    negative_b_q_shift: int = -2
    koszul_dual_q_shift: int = -2


DEFAULT_SHIFTS = ShiftConvention()


@dataclass(frozen=True)
class DynkinDiagram:
    """A simply-laced Dynkin/Coxeter graph with integer-labelled vertices."""

    vertices: tuple[Vertex, ...]
    edges: frozenset[Edge]

    def __post_init__(self) -> None:
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("vertices must be distinct")
        vertex_set = set(self.vertices)
        normalized_edges = set()
        for u, v in self.edges:
            if u == v:
                raise ValueError(f"loops are not allowed: {(u, v)!r}")
            if u not in vertex_set or v not in vertex_set:
                raise ValueError(f"edge {(u, v)!r} uses a vertex outside the graph")
            normalized_edges.add(tuple(sorted((u, v))))
        object.__setattr__(self, "edges", frozenset(normalized_edges))

    @classmethod
    def from_data(
        cls,
        vertices: int | Iterable[Vertex],
        edges: Iterable[Edge],
    ) -> "DynkinDiagram":
        if isinstance(vertices, int):
            if vertices < 0:
                raise ValueError("the number of vertices must be nonnegative")
            vertex_list = tuple(range(vertices))
        else:
            vertex_list = tuple(vertices)
        return cls(vertex_list, frozenset(edges))

    @property
    def rank(self) -> int:
        return len(self.vertices)

    def vertex_index(self, vertex: Vertex) -> int:
        try:
            return self.vertices.index(vertex)
        except ValueError as exc:
            raise ValueError(f"unknown vertex/generator {vertex!r}") from exc

    def cartan_entry(self, row: Vertex, column: Vertex) -> int:
        if row == column:
            return 2
        return -1 if tuple(sorted((row, column))) in self.edges else 0

    def invariant_linear_coefficients(self, reflection: Vertex, vertex: Vertex) -> dict[Vertex, sp.Rational]:
        """Return coefficients for a linear ``s``-invariant form.

        For ``vertex != reflection`` this is

            beta_vertex = alpha_vertex - a_{s,vertex}/2 alpha_s.

        In the simply-laced case this is ``alpha_vertex + alpha_s/2`` when the
        two vertices are adjacent, and just ``alpha_vertex`` otherwise.
        """

        if vertex == reflection:
            raise ValueError("alpha_s is not a linear invariant for s")
        return {
            vertex: sp.Rational(1),
            reflection: -sp.Rational(self.cartan_entry(reflection, vertex), 2),
        }


@dataclass(frozen=True)
class RouquierTerm:
    """One direct summand/term in the tensor product Rouquier complex."""

    term_id: int
    choices: tuple[int, ...]
    word: tuple[Vertex, ...]
    degree: int
    q_shift: int


@dataclass(frozen=True)
class RouquierArrow:
    """One summand of the Rouquier differential."""

    arrow_id: int
    source: tuple[int, ...]
    target: tuple[int, ...]
    crossing_index: int
    generator: Vertex
    kind: str
    b_position: int
    sign: int


@dataclass
class RouquierComplex:
    """The Rouquier complex before applying Hochschild cohomology."""

    diagram: DynkinDiagram
    braid: tuple[tuple[Vertex, int], ...]
    terms: dict[tuple[int, ...], RouquierTerm]
    arrows: list[RouquierArrow]
    shifts: ShiftConvention

    def terms_in_degree(self, degree: int) -> list[RouquierTerm]:
        return [term for term in self.terms.values() if term.degree == degree]

    @property
    def degrees(self) -> list[int]:
        return sorted({term.degree for term in self.terms.values()})


def normalize_braid(braid: Iterable[BraidLetter]) -> tuple[tuple[Vertex, int], ...]:
    normalized = []
    for letter in braid:
        if len(letter) != 2:
            raise ValueError(f"braid letter {letter!r} must have length 2")
        generator, sign = letter
        if sign in (1, "+", "plus", "+1", True):
            epsilon = 1
        elif sign in (-1, "-", "minus", "-1", False):
            epsilon = -1
        else:
            raise ValueError(f"braid sign must be plus or minus: {letter!r}")
        normalized.append((generator, epsilon))
    return tuple(normalized)


def rouquier_complex(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
) -> RouquierComplex:
    """Build the Rouquier complex of an Artin braid.

    The local two-term complexes are

    * positive: ``B_s(-1) -> R`` with ``R`` in degree ``0``;
    * negative: ``R -> B_s(1)`` with ``R`` in degree ``0``.

    A term is encoded by a 0/1 tuple saying whether each crossing contributes
    the ``R`` summand or the ``B_s`` summand.  The Bott-Samelson word of a term
    is obtained by deleting the crossings whose choice is ``R``.
    """

    normalized_braid = normalize_braid(braid)
    for generator, _epsilon in normalized_braid:
        diagram.vertex_index(generator)

    terms: dict[tuple[int, ...], RouquierTerm] = {}
    raw_terms = []
    for choices in product((0, 1), repeat=len(normalized_braid)):
        degree = 0
        q_shift = 0
        word = []
        for choice, (generator, epsilon) in zip(choices, normalized_braid):
            if choice == 0:
                continue
            word.append(generator)
            if epsilon == 1:
                degree -= 1
                q_shift += shifts.positive_b_q_shift
            else:
                degree += 1
                q_shift += shifts.negative_b_q_shift
        raw_terms.append((degree, choices, tuple(word), q_shift))

    raw_terms.sort(key=lambda item: (item[0], item[1]))
    for term_id, (degree, choices, word, q_shift) in enumerate(raw_terms):
        terms[choices] = RouquierTerm(term_id, choices, word, degree, q_shift)

    arrows = []
    arrow_id = 0
    for choices, source_term in sorted(terms.items(), key=lambda item: item[1].term_id):
        prefix_degree = 0
        b_before = 0
        for crossing_index, (choice, (generator, epsilon)) in enumerate(zip(choices, normalized_braid)):
            sign = -1 if prefix_degree % 2 else 1
            if epsilon == 1 and choice == 1:
                target = list(choices)
                target[crossing_index] = 0
                arrows.append(
                    RouquierArrow(
                        arrow_id=arrow_id,
                        source=choices,
                        target=tuple(target),
                        crossing_index=crossing_index,
                        generator=generator,
                        kind="multiplication",
                        b_position=b_before,
                        sign=sign,
                    )
                )
                arrow_id += 1
            elif epsilon == -1 and choice == 0:
                target = list(choices)
                target[crossing_index] = 1
                arrows.append(
                    RouquierArrow(
                        arrow_id=arrow_id,
                        source=choices,
                        target=tuple(target),
                        crossing_index=crossing_index,
                        generator=generator,
                        kind="coevaluation",
                        b_position=b_before,
                        sign=sign,
                    )
                )
                arrow_id += 1

            if choice == 1:
                b_before += 1
            prefix_degree += _local_degree(choice, epsilon)

    return RouquierComplex(diagram, normalized_braid, terms, arrows, shifts)


def _local_degree(choice: int, epsilon: int) -> int:
    if choice == 0:
        return 0
    return -1 if epsilon == 1 else 1


class QuotientModel:
    """Finite quotient model for one Bott-Samelson term."""

    def __init__(
        self,
        diagram: DynkinDiagram,
        term: RouquierTerm,
        *,
        shifts: ShiftConvention = DEFAULT_SHIFTS,
        max_basis_size: int = 5000,
    ) -> None:
        self.diagram = diagram
        self.term = term
        self.shifts = shifts
        self.max_basis_size = max_basis_size
        self.layer_count = len(term.word) + 1

        names = [
            f"z{term.term_id}_{layer}_{position}"
            for layer in range(self.layer_count)
            for position in range(diagram.rank)
        ]
        symbols = sp.symbols(" ".join(names)) if names else tuple()
        # sympy.symbols("x") returns a Symbol rather than a tuple.
        self.variables = (symbols,) if isinstance(symbols, sp.Symbol) else tuple(symbols)
        self.layers = [
            {
                vertex: self.variables[layer * diagram.rank + position]
                for position, vertex in enumerate(diagram.vertices)
            }
            for layer in range(self.layer_count)
        ]

        relations = self._relations()
        if self.variables:
            self.groebner = sp.groebner(relations, *self.variables, order="lex", domain=sp.QQ)
            if not self.groebner.is_zero_dimensional:
                raise ValueError(
                    f"term {term.choices} did not produce a zero-dimensional quotient; "
                    "the reduced specialization may be missing relations"
                )
            leading = [tuple(poly.LM(order=self.groebner.order)) for poly in self.groebner.polys]
            self.basis_exponents = _standard_monomials(leading, len(self.variables), max_basis_size)
        else:
            self.groebner = None
            self.basis_exponents = [tuple()]
        self.basis_index = {exp: index for index, exp in enumerate(self.basis_exponents)}
        self.basis_exprs = tuple(self._monomial_expr(exp) for exp in self.basis_exponents)
        self.basis_q_degrees = tuple(
            self.term.q_shift + self.shifts.variable_q_degree * sum(exp)
            for exp in self.basis_exponents
        )

    def _relations(self) -> list[sp.Expr]:
        relations: list[sp.Expr] = []
        relations.extend(self.layers[0][vertex] for vertex in self.diagram.vertices)
        for position, generator in enumerate(self.term.word, start=1):
            left = position - 1
            right = position
            for vertex in self.diagram.vertices:
                if vertex == generator:
                    continue
                coefficients = self.diagram.invariant_linear_coefficients(generator, vertex)
                relations.append(
                    self._linear_form(left, coefficients) - self._linear_form(right, coefficients)
                )
            alpha_left = self.alpha(left, generator)
            alpha_right = self.alpha(right, generator)
            relations.append(alpha_left**2 - alpha_right**2)
        return [sp.expand(relation) for relation in relations]

    def _linear_form(self, layer: int, coefficients: dict[Vertex, sp.Rational]) -> sp.Expr:
        return sp.expand(
            sum(coefficient * self.layers[layer][vertex] for vertex, coefficient in coefficients.items())
        )

    def alpha(self, layer: int, vertex: Vertex) -> sp.Symbol:
        return self.layers[layer][vertex]

    def diagonal_difference(self, vertex: Vertex) -> sp.Expr:
        return self.alpha(0, vertex) - self.alpha(self.layer_count - 1, vertex)

    def _monomial_expr(self, exponents: tuple[int, ...]) -> sp.Expr:
        result = sp.Integer(1)
        for variable, exponent in zip(self.variables, exponents):
            if exponent:
                result *= variable**exponent
        return result

    def normal_form(self, polynomial: sp.Expr) -> sp.Expr:
        if not self.variables:
            return sp.expand(polynomial)
        _quotients, remainder = self.groebner.reduce(sp.expand(polynomial))
        return sp.expand(remainder)

    def vector(self, polynomial: sp.Expr) -> sp.Matrix:
        remainder = self.normal_form(polynomial)
        vector = sp.zeros(len(self.basis_exponents), 1)
        if remainder == 0:
            return vector
        if not self.variables:
            vector[0, 0] = sp.Rational(remainder)
            return vector
        poly = sp.Poly(remainder, *self.variables, domain=sp.QQ)
        for exponents, coefficient in poly.terms():
            try:
                row = self.basis_index[tuple(exponents)]
            except KeyError as exc:
                raise ValueError(f"normal form contains non-standard monomial {exponents}") from exc
            vector[row, 0] += coefficient
        return vector

    def map_to(self, target: "QuotientModel", arrow: RouquierArrow) -> sp.Matrix:
        """Matrix of a Rouquier differential summand on quotient rings."""

        if arrow.kind == "multiplication":
            substitution = self._multiplication_substitution(target, arrow.b_position)
            multiplier = sp.Integer(1)
        elif arrow.kind == "coevaluation":
            substitution = self._coevaluation_substitution(target, arrow.b_position)
            left = target.alpha(arrow.b_position, arrow.generator)
            right = target.alpha(arrow.b_position + 1, arrow.generator)
            multiplier = left + right
        else:
            raise ValueError(f"unknown arrow kind {arrow.kind!r}")

        matrix = sp.zeros(len(target.basis_exponents), len(self.basis_exponents))
        for column, monomial in enumerate(self.basis_exprs):
            image = sp.expand(monomial.xreplace(substitution) * multiplier)
            vector = target.vector(image)
            for row, coefficient in enumerate(vector):
                if coefficient:
                    matrix[row, column] += arrow.sign * coefficient
        return matrix

    def _multiplication_substitution(
        self,
        target: "QuotientModel",
        b_position: int,
    ) -> dict[sp.Symbol, sp.Expr]:
        substitution = {}
        for layer in range(self.layer_count):
            if layer <= b_position:
                target_layer = layer
            else:
                target_layer = layer - 1
            for vertex in self.diagram.vertices:
                substitution[self.alpha(layer, vertex)] = target.alpha(target_layer, vertex)
        return substitution

    def _coevaluation_substitution(
        self,
        target: "QuotientModel",
        b_position: int,
    ) -> dict[sp.Symbol, sp.Expr]:
        substitution = {}
        for layer in range(self.layer_count):
            if layer <= b_position:
                target_layer = layer
            else:
                target_layer = layer + 1
            for vertex in self.diagram.vertices:
                substitution[self.alpha(layer, vertex)] = target.alpha(target_layer, vertex)
        return substitution


def _standard_monomials(
    leading_monomials: Sequence[tuple[int, ...]],
    variable_count: int,
    max_basis_size: int,
) -> list[tuple[int, ...]]:
    if variable_count == 0:
        return [tuple()]

    def divisible(exponent: tuple[int, ...], leading: tuple[int, ...]) -> bool:
        return all(left >= right for left, right in zip(exponent, leading))

    def standard(exponent: tuple[int, ...]) -> bool:
        return not any(divisible(exponent, leading) for leading in leading_monomials)

    basis = []
    seen = set()
    queue: deque[tuple[int, ...]] = deque([tuple(0 for _ in range(variable_count))])

    while queue:
        exponent = queue.popleft()
        if exponent in seen:
            continue
        seen.add(exponent)
        if not standard(exponent):
            continue
        basis.append(exponent)
        if len(basis) > max_basis_size:
            raise ValueError(
                "quotient basis is too large; increase max_basis_size if this is expected"
            )
        for index in range(variable_count):
            next_exponent = list(exponent)
            next_exponent[index] += 1
            queue.append(tuple(next_exponent))

    return sorted(basis, key=lambda exponent: (sum(exponent), exponent))


@dataclass
class KoszulComplex:
    model: QuotientModel
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
    """Basis and coordinate data for one cohomology group."""

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
class ExtTermData:
    term: RouquierTerm
    model: QuotientModel
    koszul: KoszulComplex
    cohomology: dict[int, CohomologyData]


@dataclass
class ExtComputation:
    """Koszul Ext data and the maps induced by the Rouquier differential."""

    rouquier: RouquierComplex
    term_data: dict[tuple[int, ...], ExtTermData]
    induced_maps: dict[tuple[int, int], sp.Matrix]


def koszul_complex(
    model: QuotientModel,
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
) -> KoszulComplex:
    """Build ``Hom_{R^e}(Koszul(x_i-y_i), M)`` for one term ``M``."""

    rank = model.diagram.rank
    wedge_bases = {
        degree: [tuple(combo) for combo in combinations(range(rank), degree)]
        for degree in range(rank + 1)
    }
    basis: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    basis_index: dict[int, dict[tuple[int, tuple[int, ...]], int]] = {}
    q_degrees: dict[int, list[int]] = {}
    for degree in range(rank + 1):
        entries = [
            (monomial_index, wedge)
            for wedge in wedge_bases[degree]
            for monomial_index in range(len(model.basis_exponents))
        ]
        basis[degree] = entries
        basis_index[degree] = {entry: index for index, entry in enumerate(entries)}
        q_degrees[degree] = [
            model.basis_q_degrees[monomial_index] + degree * shifts.koszul_dual_q_shift
            for monomial_index, _wedge in entries
        ]

    differentials = {}
    multiplication_by_diagonal = {
        vertex_position: _multiplication_matrix(model, model.diagonal_difference(vertex))
        for vertex_position, vertex in enumerate(model.diagram.vertices)
    }

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
                multiplication_column = multiplication_by_diagonal[vertex_position][:, monomial_index]
                for target_monomial, coefficient in enumerate(multiplication_column):
                    if coefficient:
                        row = target_index[(target_monomial, target_wedge)]
                        matrix[row, column] += sign * coefficient
        differentials[degree] = matrix
    differentials[rank] = sp.zeros(0, len(basis[rank]))

    return KoszulComplex(model, wedge_bases, basis, basis_index, q_degrees, differentials)


def _multiplication_matrix(model: QuotientModel, polynomial: sp.Expr) -> sp.Matrix:
    matrix = sp.zeros(len(model.basis_exponents), len(model.basis_exponents))
    for column, monomial in enumerate(model.basis_exprs):
        vector = model.vector(sp.expand(polynomial * monomial))
        for row, coefficient in enumerate(vector):
            if coefficient:
                matrix[row, column] = coefficient
    return matrix


def koszul_ext_complex(
    rouquier: RouquierComplex,
    *,
    max_basis_size: int = 5000,
    validate: bool = True,
) -> ExtComputation:
    """Apply the finite reduced Koszul Ext computation termwise.

    The returned ``ExtComputation.induced_maps`` dictionary is keyed by
    ``(arrow_id, hochschild_degree)`` and contains the maps

        Ext^a(R, C^r) -> Ext^a(R, C^{r+1})

    induced by each summand of the Rouquier differential.
    """

    term_data: dict[tuple[int, ...], ExtTermData] = {}
    for choices, term in sorted(rouquier.terms.items(), key=lambda item: item[1].term_id):
        model = QuotientModel(
            rouquier.diagram,
            term,
            shifts=rouquier.shifts,
            max_basis_size=max_basis_size,
        )
        koszul = koszul_complex(model, shifts=rouquier.shifts)
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
                context=f"Koszul term {term.choices}, A-degree {degree}",
            )
        term_data[choices] = ExtTermData(term, model, koszul, cohomology)

    induced_maps = {}
    for arrow in rouquier.arrows:
        source_data = term_data[arrow.source]
        target_data = term_data[arrow.target]
        term_map = source_data.model.map_to(target_data.model, arrow)
        for degree in range(rouquier.diagram.rank + 1):
            chain_map = _koszul_chain_map(source_data.koszul, target_data.koszul, degree, term_map)
            if validate:
                _assert_degree_zero(
                    chain_map,
                    source_data.koszul.q_degrees[degree],
                    target_data.koszul.q_degrees[degree],
                    f"Rouquier arrow {arrow.arrow_id} on Koszul A-degree {degree}",
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

    return ExtComputation(rouquier, term_data, induced_maps)


def _koszul_chain_map(
    source: KoszulComplex,
    target: KoszulComplex,
    degree: int,
    term_map: sp.Matrix,
) -> sp.Matrix:
    matrix = sp.zeros(len(target.basis[degree]), len(source.basis[degree]))
    target_index = target.basis_index[degree]
    for column, (source_monomial, wedge) in enumerate(source.basis[degree]):
        image_column = term_map[:, source_monomial]
        for target_monomial, coefficient in enumerate(image_column):
            if coefficient:
                row = target_index[(target_monomial, wedge)]
                matrix[row, column] += coefficient
    return matrix


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

        span = _matrix_from_columns(image_columns, len(current_indices))
        quotient_columns = []
        rank = span.rank()
        for kernel_vector in kernel_columns:
            candidate = span.row_join(kernel_vector)
            candidate_rank = candidate.rank()
            if candidate_rank > rank:
                quotient_columns.append(kernel_vector)
                span = candidate
                rank = candidate_rank

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


@dataclass
class KRResult:
    polynomial: sp.Expr
    ext: ExtComputation
    homology_dimensions: dict[tuple[int, int, int], int]


def khovanov_rozansky_cohomology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    max_basis_size: int = 5000,
    validate: bool = True,
) -> KRResult:
    """Compute the reduced triply graded KR invariant of a braid.

    The returned expression is the Laurent polynomial

        sum dim H(C tensor_R HH)^{a,r,q} A^a Q^q T^r,

    i.e. the Soergel-module version of Khovanov--Rozansky homology.
    """

    rouquier = rouquier_complex(diagram, braid, shifts=shifts)
    ext = koszul_ext_complex(
        rouquier,
        max_basis_size=max_basis_size,
        validate=validate,
    )
    homology_dimensions: dict[tuple[int, int, int], int] = {}
    polynomial = sp.Integer(0)

    for hochschild_degree in range(diagram.rank + 1):
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
                            matrix[target_offset + row, source_offset + column] += induced[row, column]
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
                context=f"horizontal A-degree {hochschild_degree}, T-degree {degree}",
            )
            for q_degree in homology.q_degrees:
                key = (hochschild_degree, degree, q_degree)
                homology_dimensions[key] = homology_dimensions.get(key, 0) + 1

    for (hochschild_degree, rouquier_degree, q_degree), dimension in sorted(homology_dimensions.items()):
        polynomial += dimension * A**hochschild_degree * Q**q_degree * T**rouquier_degree

    return KRResult(
        sp.expand(polynomial),
        ext,
        homology_dimensions,
    )


def parse_edges(text: str) -> list[Edge]:
    if not text.strip():
        return []
    edges = []
    for item in text.split(","):
        left, right = item.split("-")
        edges.append((int(left), int(right)))
    return edges


def parse_braid(text: str) -> list[BraidLetter]:
    if not text.strip():
        return []
    braid = []
    for item in text.split(","):
        generator, sign = item.split(":")
        braid.append((int(generator), sign))
    return braid


def parse_vertices(text: str) -> list[Vertex]:
    if not text.strip():
        return []
    return [int(item) for item in text.split(",")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", required=True, help="comma-separated vertices, e.g. 0,1,2")
    parser.add_argument("--edges", default="", help="comma-separated edges, e.g. 0-1,1-2")
    parser.add_argument("--braid", default="", help="comma-separated braid letters, e.g. 0:+,1:-")
    parser.add_argument("--max-basis-size", type=int, default=5000)
    args = parser.parse_args(argv)

    diagram = DynkinDiagram.from_data(parse_vertices(args.vertices), parse_edges(args.edges))
    result = khovanov_rozansky_cohomology(
        diagram,
        parse_braid(args.braid),
        max_basis_size=args.max_basis_size,
    )
    print(result.polynomial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
