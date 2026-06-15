"""Light-leaves data for Bott--Samelson bimodules.

For an expression ``word = (s_1, ..., s_n)``, the Bott--Samelson bimodule

    B_word = B_{s_1} tensor_R ... tensor_R B_{s_n}

has a left ``R``-basis indexed by subexpressions ``e in {0, 1}^n``.  This file
computes the corresponding light-leaves combinatorics: the endpoint of each
subexpression, its ``U/D`` decoration, and its defect

    defect(e) = #{U0 steps} - #{D0 steps}.

The implementation is intentionally combinatorial.  It does not construct
planar diagrams; it records the labels and degrees one normally uses to choose
the light-leaves basis elements.  The bottom half of the file applies the same
basis to Rouquier complexes: each Bott--Samelson term is written as a free
left ``R``-module and every Rouquier differential is recorded as a matrix with
entries in the leftmost polynomial ring.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

import sympy as sp

from computations.khovanov_rozansky import (
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Realization,
    RouquierArrow,
    RouquierComplex,
    RouquierTerm,
    ShiftConvention,
    Vertex,
    rouquier_complex,
)


Subexpression: TypeAlias = tuple[int, ...]
MatrixKey: TypeAlias = tuple[tuple[sp.Expr, ...], ...]

v = sp.Symbol("v")


@dataclass(frozen=True)
class LightLeaf:
    """One light-leaves basis label indexed by a subexpression."""

    index: int
    word: tuple[Vertex, ...]
    subexpression: Subexpression
    selected_word: tuple[Vertex, ...]
    step_types: tuple[str, ...]
    endpoint_key: MatrixKey
    endpoint_word: tuple[Vertex, ...]
    endpoint_length: int
    module_degree: int
    defect: int

    @property
    def light_leaf_degree(self) -> int:
        """The usual light-leaf morphism degree, equal to the defect."""

        return self.defect

    @property
    def bit_string(self) -> str:
        return "".join(str(bit) for bit in self.subexpression)

    @property
    def step_string(self) -> str:
        return " ".join(self.step_types)

    @property
    def endpoint_string(self) -> str:
        if not self.endpoint_word:
            return "e"
        return " ".join(str(generator) for generator in self.endpoint_word)

    @property
    def tensor_factors(self) -> tuple[str, ...]:
        """Rank-one left-``R`` basis labels for the tensor factors."""

        return tuple(
            f"alpha_{generator}" if bit else "1"
            for bit, generator in zip(self.subexpression, self.word, strict=True)
        )

    @property
    def tensor_string(self) -> str:
        return " tensor ".join(self.tensor_factors) if self.tensor_factors else "1"


@dataclass(frozen=True)
class LightLeavesBasis:
    """The full subexpression/light-leaves basis data for a Bott--Samelson word."""

    diagram: DynkinDiagram
    word: tuple[Vertex, ...]
    leaves: tuple[LightLeaf, ...]
    normalization: str
    root_degree: int

    @property
    def rank(self) -> int:
        return len(self.leaves)

    def graded_rank(
        self,
        *,
        variable: sp.Symbol = v,
        degree: str = "module",
    ) -> sp.Expr:
        """Return ``sum variable^degree(leaf)`` for the chosen grading.

        ``degree="module"`` uses the left-``R`` module basis degree.  In the
        self-dual normalization this is ``-1`` for a skipped rank-one factor
        and ``+1`` for a selected factor.  ``degree="light_leaf"`` uses the
        light-leaf defect.
        """

        return _rank_from_degrees(
            _leaf_degree(leaf, degree) for leaf in self.leaves
        ).as_expr(variable)

    def graded_rank_by_endpoint(
        self,
        *,
        variable: sp.Symbol = v,
        degree: str = "light_leaf",
    ) -> dict[tuple[Vertex, ...], sp.Expr]:
        """Return graded ranks grouped by endpoint reduced word."""

        degrees: dict[tuple[Vertex, ...], list[int]] = defaultdict(list)
        for leaf in self.leaves:
            degrees[leaf.endpoint_word].append(_leaf_degree(leaf, degree))
        return {
            endpoint: _rank_from_degrees(endpoint_degrees).as_expr(variable)
            for endpoint, endpoint_degrees in sorted(
                degrees.items(), key=lambda item: (len(item[0]), item[0])
            )
        }

    def counts_by_endpoint(self) -> dict[tuple[Vertex, ...], int]:
        counts = Counter(leaf.endpoint_word for leaf in self.leaves)
        return dict(sorted(counts.items(), key=lambda item: (len(item[0]), item[0])))

    def ending_at(self, endpoint: Sequence[Vertex]) -> tuple[LightLeaf, ...]:
        endpoint_tuple = tuple(endpoint)
        return tuple(leaf for leaf in self.leaves if leaf.endpoint_word == endpoint_tuple)


@dataclass(frozen=True)
class _RankPolynomial:
    terms: Counter[int]

    def as_expr(self, variable: sp.Symbol = v) -> sp.Expr:
        return sp.Add(
            *(
                multiplicity * variable**degree
                for degree, multiplicity in sorted(self.terms.items())
            ),
            evaluate=False,
        )


class _CoxeterCombinatorics:
    """Coxeter operations needed by the subexpression walk."""

    def __init__(self, diagram: DynkinDiagram) -> None:
        self.diagram = diagram
        self.realization = Realization.standard(diagram)
        self.identity = sp.eye(diagram.rank)
        self.simple = self.realization.action
        self.simple_roots = {
            vertex: _basis_vector(diagram.rank, diagram.vertex_index(vertex))
            for vertex in diagram.vertices
        }

    def matrix_key(self, matrix: sp.Matrix) -> MatrixKey:
        return tuple(
            tuple(sp.simplify(matrix[row, col]) for col in range(matrix.cols))
            for row in range(matrix.rows)
        )

    def right_step_type(self, matrix: sp.Matrix, generator: Vertex) -> str:
        root = matrix * self.simple_roots[generator]
        if _is_positive_root(root):
            return "U"
        if _is_negative_root(root):
            return "D"
        raise ValueError(
            f"could not determine the sign of {matrix!r} applied to alpha_{generator!r}"
        )

    def multiply_right(self, matrix: sp.Matrix, generator: Vertex) -> sp.Matrix:
        return sp.expand(matrix * self.simple[generator])

    def reduced_word(self, matrix: sp.Matrix, length: int) -> tuple[Vertex, ...]:
        """Recover one reduced word by repeatedly stripping right descents."""

        remaining = sp.Matrix(matrix)
        stripped: list[Vertex] = []
        for _ in range(length):
            descent = self._right_descent(remaining)
            stripped.append(descent)
            remaining = self.multiply_right(remaining, descent)
        if self.matrix_key(remaining) != self.matrix_key(self.identity):
            raise ValueError("right-descent stripping did not reach the identity")
        return tuple(reversed(stripped))

    def _right_descent(self, matrix: sp.Matrix) -> Vertex:
        for generator in self.diagram.vertices:
            if self.right_step_type(matrix, generator) == "D":
                return generator
        raise ValueError("non-identity element has no right descent")


def bott_samelson_light_leaves(
    diagram: DynkinDiagram,
    word: Sequence[Vertex],
    *,
    normalization: str = "self_dual",
    root_degree: int = 2,
) -> LightLeavesBasis:
    """Enumerate the light-leaves basis labels for ``B_word``.

    Parameters
    ----------
    diagram:
        Simply-laced Coxeter graph, using the same ``DynkinDiagram`` class as
        the other computation modules in this folder.
    word:
        The Bott--Samelson expression ``(s_1, ..., s_n)``.
    normalization:
        ``"self_dual"`` gives rank-one module degrees ``-1`` and ``+1``.
        ``"unshifted"`` gives degrees ``0`` and ``root_degree`` for the
        unshifted bimodule ``R tensor_{R^s} R``.
    root_degree:
        Degree of a simple root in the unshifted normalization.
    """

    word_tuple = tuple(word)
    _validate_word(diagram, word_tuple)
    if normalization not in {"self_dual", "unshifted"}:
        raise ValueError("normalization must be 'self_dual' or 'unshifted'")

    coxeter = _CoxeterCombinatorics(diagram)
    leaves = []
    for index, subexpression in enumerate(product((0, 1), repeat=len(word_tuple))):
        matrix = coxeter.identity
        length = 0
        step_types: list[str] = []
        selected_word: list[Vertex] = []
        defect = 0

        for bit, generator in zip(subexpression, word_tuple, strict=True):
            direction = coxeter.right_step_type(matrix, generator)
            step_types.append(f"{direction}{bit}")
            if bit == 0:
                defect += 1 if direction == "U" else -1
            else:
                selected_word.append(generator)
                matrix = coxeter.multiply_right(matrix, generator)
                length += 1 if direction == "U" else -1

        endpoint_word = coxeter.reduced_word(matrix, length)
        leaves.append(
            LightLeaf(
                index=index,
                word=word_tuple,
                subexpression=tuple(subexpression),
                selected_word=tuple(selected_word),
                step_types=tuple(step_types),
                endpoint_key=coxeter.matrix_key(matrix),
                endpoint_word=endpoint_word,
                endpoint_length=length,
                module_degree=_module_degree(subexpression, normalization, root_degree),
                defect=defect,
            )
        )

    return LightLeavesBasis(
        diagram=diagram,
        word=word_tuple,
        leaves=tuple(leaves),
        normalization=normalization,
        root_degree=root_degree,
    )


@dataclass(frozen=True)
class RouquierFreeBasisElement:
    """One basis vector in a free-left-``R`` Rouquier chain group."""

    term_choices: tuple[int, ...]
    term_id: int
    local_index: int
    homological_degree: int
    word: tuple[Vertex, ...]
    subexpression: Subexpression
    q_degree: int
    leaf: LightLeaf

    @property
    def label(self) -> str:
        bits = "".join(str(bit) for bit in self.subexpression)
        return f"{self.term_choices}:{bits}"


class BottSamelsonFreeLeftRModel:
    """A Bott--Samelson term as a free left ``R``-module.

    The basis is indexed by ``{0,1}^k`` for ``k = len(term.word)``.  Internally
    the temporary variable ``theta_i`` denotes the simple root on the ``i``th
    new layer.  The triangular relations

        theta_i^2 = alpha_{s_i}(previous layer)^2

    reduce every element to an ``R``-linear combination of squarefree
    ``theta`` monomials.
    """

    def __init__(
        self,
        realization: Realization,
        term: RouquierTerm,
        *,
        shifts: ShiftConvention = DEFAULT_SHIFTS,
        r_variables: Sequence[sp.Symbol] | None = None,
    ) -> None:
        self.realization = realization
        self.diagram = realization.diagram
        self.term = term
        self.shifts = shifts
        self.dim = realization.dim
        self.r_variables = (
            tuple(r_variables)
            if r_variables is not None
            else tuple(sp.Symbol(f"x_{position}") for position in range(self.dim))
        )
        if len(self.r_variables) != self.dim:
            raise ValueError(
                f"expected {self.dim} left R variables, got {len(self.r_variables)}"
            )

        self.theta_variables = tuple(
            sp.Symbol(f"theta_{term.term_id}_{position}")
            for position in range(1, len(term.word) + 1)
        )
        self.layer_coordinates: list[tuple[sp.Expr, ...]] = [tuple(self.r_variables)]
        self.theta_squares: list[sp.Expr] = []

        for position, (generator, theta) in enumerate(
            zip(term.word, self.theta_variables, strict=True),
            start=1,
        ):
            previous = self.layer_coordinates[position - 1]
            previous_alpha = self.coordinate_form(previous, self.realization.root(generator))
            self.theta_squares.append(self.reduce_expression(previous_alpha**2))
            self.layer_coordinates.append(
                self._next_layer_coordinates(previous, generator, theta)
            )

        self.light_leaves = bott_samelson_light_leaves(
            self.diagram,
            term.word,
            normalization="unshifted",
            root_degree=shifts.variable_q_degree,
        )
        self.basis_subexpressions = tuple(
            leaf.subexpression for leaf in self.light_leaves.leaves
        )
        self.basis_index = {
            subexpression: index
            for index, subexpression in enumerate(self.basis_subexpressions)
        }
        self.basis_exprs = tuple(
            self._basis_expr(subexpression)
            for subexpression in self.basis_subexpressions
        )
        self.basis_q_degrees = tuple(
            term.q_shift + shifts.variable_q_degree * sum(subexpression)
            for subexpression in self.basis_subexpressions
        )

    @property
    def rank(self) -> int:
        return len(self.basis_subexpressions)

    def coordinate_form(
        self,
        coordinates: Sequence[sp.Expr],
        form: sp.Matrix,
    ) -> sp.Expr:
        return sp.expand(
            sum(form[position] * coordinates[position] for position in range(self.dim))
        )

    def alpha_at_layer(self, layer: int, generator: Vertex) -> sp.Expr:
        return self.reduce_expression(
            self.coordinate_form(self.layer_coordinates[layer], self.realization.root(generator))
        )

    def reduce_expression(self, expression: sp.Expr) -> sp.Expr:
        reduced = sp.expand(expression)
        for theta, theta_square in reversed(
            list(zip(self.theta_variables, self.theta_squares))
        ):
            reduced = _reduce_squarefree_in_variable(reduced, theta, theta_square)
        return sp.expand(reduced)

    def vector(self, expression: sp.Expr) -> sp.Matrix:
        coefficients = self.coefficients(expression)
        return sp.Matrix([coefficients[subexpression] for subexpression in self.basis_subexpressions])

    def coefficients(self, expression: sp.Expr) -> dict[Subexpression, sp.Expr]:
        reduced = self.reduce_expression(expression)
        coefficients = {
            subexpression: sp.Integer(0)
            for subexpression in self.basis_subexpressions
        }
        if not self.theta_variables:
            coefficients[tuple()] = sp.expand(reduced)
            return coefficients

        polynomial = sp.Poly(reduced, *self.theta_variables, domain="EX")
        for exponents, coefficient in polynomial.terms():
            subexpression = tuple(int(exponent) for exponent in exponents)
            if any(exponent not in (0, 1) for exponent in subexpression):
                raise ValueError(
                    f"expression was not reduced to the Bott-Samelson basis: {reduced!r}"
                )
            coefficients[subexpression] += sp.expand(coefficient)
        return coefficients

    def map_to(
        self,
        target: "BottSamelsonFreeLeftRModel",
        arrow: RouquierArrow,
    ) -> sp.Matrix:
        """Matrix of one Rouquier differential summand over the left ring."""

        substitution = self._theta_substitution(target, arrow)
        if arrow.kind == "multiplication":
            multiplier = sp.Integer(1)
        elif arrow.kind == "coevaluation":
            multiplier = target.alpha_at_layer(
                arrow.b_position,
                arrow.generator,
            ) + target.alpha_at_layer(arrow.b_position + 1, arrow.generator)
        else:
            raise ValueError(f"unknown Rouquier arrow kind {arrow.kind!r}")

        matrix = sp.zeros(target.rank, self.rank)
        for column, basis_expr in enumerate(self.basis_exprs):
            image = sp.expand(basis_expr.xreplace(substitution) * multiplier)
            vector = target.vector(image)
            for row, coefficient in enumerate(vector):
                if coefficient:
                    matrix[row, column] += arrow.sign * sp.expand(coefficient)
        return matrix

    def _basis_expr(self, subexpression: Subexpression) -> sp.Expr:
        result = sp.Integer(1)
        for bit, theta in zip(subexpression, self.theta_variables, strict=True):
            if bit:
                result *= theta
        return result

    def _next_layer_coordinates(
        self,
        previous: Sequence[sp.Expr],
        generator: Vertex,
        theta: sp.Symbol,
    ) -> tuple[sp.Expr, ...]:
        root = self.realization.root(generator)
        current = []
        for coordinate_index in range(self.dim):
            coordinate = _basis_vector(self.dim, coordinate_index)
            reflected = self.realization.action[generator] * coordinate
            fixed_part = (coordinate + reflected) / 2
            anti_part = (coordinate - reflected) / 2
            coefficient = _root_line_coefficient(anti_part, root)
            current.append(
                self.reduce_expression(
                    self.coordinate_form(previous, fixed_part) + coefficient * theta
                )
            )
        return tuple(current)

    def _theta_substitution(
        self,
        target: "BottSamelsonFreeLeftRModel",
        arrow: RouquierArrow,
    ) -> dict[sp.Symbol, sp.Expr]:
        substitution = {}
        for source_index, (theta, generator) in enumerate(
            zip(self.theta_variables, self.term.word, strict=True),
            start=1,
        ):
            if arrow.kind == "multiplication":
                target_layer = source_index if source_index <= arrow.b_position else source_index - 1
            elif arrow.kind == "coevaluation":
                target_layer = source_index if source_index <= arrow.b_position else source_index + 1
            else:
                raise ValueError(f"unknown Rouquier arrow kind {arrow.kind!r}")
            substitution[theta] = target.alpha_at_layer(target_layer, generator)
        return substitution


@dataclass
class RouquierFreeLeftRComplex:
    """Rouquier complex expanded as free left ``R``-modules."""

    rouquier: RouquierComplex
    r_variables: tuple[sp.Symbol, ...]
    models: dict[tuple[int, ...], BottSamelsonFreeLeftRModel]
    arrow_matrices: dict[int, sp.Matrix]
    basis_by_degree: dict[int, tuple[RouquierFreeBasisElement, ...]]
    basis_index_by_degree: dict[int, dict[tuple[tuple[int, ...], int], int]]
    differentials: dict[int, sp.Matrix]

    @property
    def degrees(self) -> list[int]:
        return sorted(self.basis_by_degree)

    def module_rank(self, degree: int) -> int:
        return len(self.basis_by_degree.get(degree, tuple()))

    def basis(self, degree: int) -> tuple[RouquierFreeBasisElement, ...]:
        return self.basis_by_degree.get(degree, tuple())

    def differential(self, degree: int) -> sp.Matrix:
        if degree in self.differentials:
            return self.differentials[degree]
        return sp.zeros(self.module_rank(degree + 1), self.module_rank(degree))

    def check_d_squared(self) -> bool:
        if not self.degrees:
            return True
        for degree in range(min(self.degrees), max(self.degrees) - 1):
            composite = self.differential(degree + 1) * self.differential(degree)
            if any(sp.simplify(entry) != 0 for entry in composite):
                return False
        return True


def rouquier_complex_as_free_left_r_modules(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    r_variables: Sequence[sp.Symbol] | None = None,
) -> RouquierFreeLeftRComplex:
    """Build a Rouquier complex as a complex of free left ``R``-modules."""

    rouquier = rouquier_complex(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    if r_variables is None:
        variables = tuple(sp.Symbol(f"x_{position}") for position in range(rouquier.realization.dim))
    else:
        variables = tuple(r_variables)

    models = {
        choices: BottSamelsonFreeLeftRModel(
            rouquier.realization,
            term,
            shifts=shifts,
            r_variables=variables,
        )
        for choices, term in rouquier.terms.items()
    }
    arrow_matrices = {
        arrow.arrow_id: models[arrow.source].map_to(models[arrow.target], arrow)
        for arrow in rouquier.arrows
    }

    basis_by_degree, basis_index_by_degree = _rouquier_free_basis_by_degree(rouquier, models)
    differentials = _rouquier_free_differentials(
        rouquier,
        arrow_matrices,
        basis_by_degree,
        basis_index_by_degree,
    )
    return RouquierFreeLeftRComplex(
        rouquier=rouquier,
        r_variables=variables,
        models=models,
        arrow_matrices=arrow_matrices,
        basis_by_degree=basis_by_degree,
        basis_index_by_degree=basis_index_by_degree,
        differentials=differentials,
    )


def _rouquier_free_basis_by_degree(
    rouquier: RouquierComplex,
    models: dict[tuple[int, ...], BottSamelsonFreeLeftRModel],
) -> tuple[
    dict[int, tuple[RouquierFreeBasisElement, ...]],
    dict[int, dict[tuple[tuple[int, ...], int], int]],
]:
    grouped: dict[int, list[RouquierFreeBasisElement]] = defaultdict(list)
    for choices, term in sorted(rouquier.terms.items(), key=lambda item: item[1].term_id):
        model = models[choices]
        for local_index, leaf in enumerate(model.light_leaves.leaves):
            grouped[term.degree].append(
                RouquierFreeBasisElement(
                    term_choices=choices,
                    term_id=term.term_id,
                    local_index=local_index,
                    homological_degree=term.degree,
                    word=term.word,
                    subexpression=leaf.subexpression,
                    q_degree=model.basis_q_degrees[local_index],
                    leaf=leaf,
                )
            )

    basis_by_degree = {
        degree: tuple(entries)
        for degree, entries in sorted(grouped.items())
    }
    basis_index_by_degree = {
        degree: {
            (entry.term_choices, entry.local_index): index
            for index, entry in enumerate(entries)
        }
        for degree, entries in basis_by_degree.items()
    }
    return basis_by_degree, basis_index_by_degree


def _rouquier_free_differentials(
    rouquier: RouquierComplex,
    arrow_matrices: dict[int, sp.Matrix],
    basis_by_degree: dict[int, tuple[RouquierFreeBasisElement, ...]],
    basis_index_by_degree: dict[int, dict[tuple[tuple[int, ...], int], int]],
) -> dict[int, sp.Matrix]:
    if not basis_by_degree:
        return {}

    differentials: dict[int, sp.Matrix] = {}
    for degree in range(min(basis_by_degree), max(basis_by_degree) + 1):
        source_basis = basis_by_degree.get(degree, tuple())
        target_basis = basis_by_degree.get(degree + 1, tuple())
        matrix = sp.zeros(len(target_basis), len(source_basis))
        if not source_basis:
            differentials[degree] = matrix
            continue

        source_index = basis_index_by_degree.get(degree, {})
        target_index = basis_index_by_degree.get(degree + 1, {})
        for arrow in rouquier.arrows:
            source_term = rouquier.terms[arrow.source]
            target_term = rouquier.terms[arrow.target]
            if source_term.degree != degree or target_term.degree != degree + 1:
                continue

            block = arrow_matrices[arrow.arrow_id]
            for local_column in range(block.cols):
                column = source_index[(arrow.source, local_column)]
                for local_row in range(block.rows):
                    row = target_index[(arrow.target, local_row)]
                    coefficient = block[local_row, local_column]
                    if coefficient:
                        matrix[row, column] += coefficient
        differentials[degree] = matrix.applyfunc(sp.expand)
    return differentials


def _reduce_squarefree_in_variable(
    expression: sp.Expr,
    variable: sp.Symbol,
    square: sp.Expr,
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), variable, domain="EX")
    result = sp.Integer(0)
    for (exponent,), coefficient in polynomial.terms():
        result += coefficient * variable ** (exponent % 2) * square ** (exponent // 2)
    return sp.expand(result)


def _root_line_coefficient(vector: sp.Matrix, root: sp.Matrix) -> sp.Expr:
    nonzero_positions = [
        index
        for index, entry in enumerate(root)
        if sp.simplify(entry) != 0
    ]
    if not nonzero_positions:
        raise ValueError("simple root vector must be nonzero")
    first = nonzero_positions[0]
    coefficient = sp.simplify(vector[first] / root[first])
    if any(
        sp.simplify(vector[index] - coefficient * root[index]) != 0
        for index in range(root.rows)
    ):
        raise ValueError(f"{vector!r} is not in the line spanned by {root!r}")
    return coefficient


def _module_degree(
    subexpression: Subexpression,
    normalization: str,
    root_degree: int,
) -> int:
    if normalization == "self_dual":
        return sum(1 if bit else -1 for bit in subexpression)
    return sum(root_degree if bit else 0 for bit in subexpression)


def _leaf_degree(leaf: LightLeaf, degree: str) -> int:
    if degree in {"module", "rmodule", "r-module"}:
        return leaf.module_degree
    if degree in {"light_leaf", "light-leaf", "defect"}:
        return leaf.light_leaf_degree
    raise ValueError("degree must be 'module' or 'light_leaf'")


def _rank_from_degrees(degrees: Iterable[int]) -> _RankPolynomial:
    return _RankPolynomial(Counter(degrees))


def _basis_vector(size: int, index: int) -> sp.Matrix:
    vector = sp.zeros(size, 1)
    vector[index, 0] = 1
    return vector


def _is_positive_root(root: sp.Matrix) -> bool:
    entries = [sp.simplify(entry) for entry in root]
    return any(entry != 0 for entry in entries) and all(entry >= 0 for entry in entries)


def _is_negative_root(root: sp.Matrix) -> bool:
    entries = [sp.simplify(entry) for entry in root]
    return any(entry != 0 for entry in entries) and all(entry <= 0 for entry in entries)


def _validate_word(diagram: DynkinDiagram, word: Sequence[Vertex]) -> None:
    vertices = set(diagram.vertices)
    unknown = [generator for generator in word if generator not in vertices]
    if unknown:
        raise ValueError(f"word uses generator(s) outside the diagram: {unknown!r}")


def _parse_int_list(raw: str) -> tuple[int, ...]:
    if raw.strip() == "":
        return tuple()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_edges(raw: str) -> tuple[tuple[int, int], ...]:
    if raw.strip() == "":
        return tuple()
    edges = []
    for part in raw.split(","):
        if not part.strip():
            continue
        left, right = part.split("-", maxsplit=1)
        edges.append((int(left.strip()), int(right.strip())))
    return tuple(edges)


def _parse_braid(raw: str) -> tuple[tuple[int, str], ...]:
    if raw.strip() == "":
        return tuple()
    braid = []
    for part in raw.split(","):
        if not part.strip():
            continue
        generator, sign = part.split(":", maxsplit=1)
        braid.append((int(generator.strip()), sign.strip()))
    return tuple(braid)


def _format_endpoint(endpoint: Sequence[Vertex]) -> str:
    return "e" if not endpoint else " ".join(str(generator) for generator in endpoint)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate light-leaves labels for a Bott-Samelson word, or expand "
            "a Rouquier complex as free left R-modules."
        )
    )
    parser.add_argument(
        "--vertices",
        required=True,
        help="comma-separated generators, e.g. '0,1,2'",
    )
    parser.add_argument(
        "--edges",
        default="",
        help="comma-separated simply-laced Coxeter graph edges, e.g. '0-1,1-2'",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--word",
        help="comma-separated Bott-Samelson word, e.g. '0,1,0'",
    )
    input_group.add_argument(
        "--braid",
        help="comma-separated Rouquier braid letters, e.g. '0:+,1:-,0:+'",
    )
    parser.add_argument(
        "--normalization",
        choices=("self_dual", "unshifted"),
        default="self_dual",
    )
    parser.add_argument("--root-degree", type=int, default=2)
    parser.add_argument(
        "--degree",
        choices=("module", "light_leaf"),
        default="module",
        help="grading used in the displayed total graded rank",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    diagram = DynkinDiagram.from_data(_parse_int_list(args.vertices), _parse_edges(args.edges))
    if args.braid is not None:
        complex_ = rouquier_complex_as_free_left_r_modules(
            diagram,
            _parse_braid(args.braid),
        )
        variables = ", ".join(str(variable) for variable in complex_.r_variables)
        print(f"R = QQ[{variables}]")
        print(f"braid: {args.braid}")
        print(f"d^2 = 0: {complex_.check_d_squared()}")
        print()
        for degree in complex_.degrees:
            print(f"C^{degree}: rank {complex_.module_rank(degree)}")
            for index, basis_element in enumerate(complex_.basis(degree)):
                print(
                    f"  {index}: term={basis_element.term_choices} "
                    f"bits={''.join(str(bit) for bit in basis_element.subexpression) or 'e'} "
                    f"q={basis_element.q_degree}"
                )
            matrix = complex_.differential(degree)
            if matrix.rows or matrix.cols:
                print(f"d^{degree}:")
                print(matrix)
            print()
        return

    basis = bott_samelson_light_leaves(
        diagram,
        _parse_int_list(args.word),
        normalization=args.normalization,
        root_degree=args.root_degree,
    )

    print(f"word: {_format_endpoint(basis.word)}")
    print(f"rank over R: {basis.rank}")
    print(f"{args.degree} graded rank: {basis.graded_rank(degree=args.degree)}")
    print()
    print("bits  endpoint  steps        module_deg  defect")
    print("----  --------  -----------  ----------  ------")
    for leaf in basis.leaves:
        print(
            f"{leaf.bit_string:<4}  "
            f"{leaf.endpoint_string:<8}  "
            f"{leaf.step_string:<11}  "
            f"{leaf.module_degree:>10}  "
            f"{leaf.defect:>6}"
        )


if __name__ == "__main__":
    main()
