"""Small-rank Khovanov--Rozansky computations from Soergel bimodules.

This module implements the three steps requested in the prompt:

* build the Rouquier complex of an Artin braid;
* compute Hochschild cohomology of each Bott-Samelson term with the Koszul
  resolution for the diagonal ideal ``(x_i - y_i)``;
* take homology of the induced maps between the Hochschild cohomology groups.

The implementation is intentionally explicit.  A Bott-Samelson term

    B_{s_1} tensor_R ... tensor_R B_{s_k}

is represented as the coordinate ring with variables for ``k + 1`` tensor
positions.  The relation for the factor ``B_s`` between adjacent positions is
that the standard generators of ``R^s`` agree: all invariant linear forms agree,
and ``alpha_s^2`` agrees.

The executable linear-algebra routine below computes a reduced version when
``reduced=True``: after forming each term, it sets the leftmost polynomial
variables to zero.  This turns the Koszul complexes into finite-dimensional
vector spaces and produces a Laurent polynomial in ``A,Q,T``.

For ``reduced=False`` the module computes the genuine *unreduced* invariant
directly, i.e. **without** imposing the relations ``z_{0,i} = 0``.  The
Bott-Samelson coordinate rings are then free of finite rank over the leftmost
polynomial ring ``R = QQ[z_{0,0}, ..., z_{0,rank-1}]`` instead of being
finite-dimensional, so the graded pieces are infinite-dimensional and the
invariant is recorded as a *Hilbert series* rather than a Laurent polynomial.

The computation exploits the fact that every differential and induced map in
the construction is degree ``0`` for the internal ``Q``-grading.  A degree-zero
map of graded ``R``-modules splits as a direct sum over the total ``Q``-degree,
so the homology in each fixed ``Q``-degree is the homology of a *finite*
complex of ``QQ``-vector spaces.  Enumerating the standard monomials up to a
cutoff therefore computes the graded dimension ``dim HHH^{a,r}_q`` exactly for
every ``q`` below a reliable bound, and the existing finite linear algebra is
reused verbatim.

Because each ``HHH^{a,r}`` is a finitely generated graded module over ``R``,
its Hilbert series is a rational function whose denominator divides
``(1 - Q^2)^{rank}``.  The reconstructed answer is

    HHH^{a,r}(Q) = K_{a,r}(Q) / (1 - Q^2)^rank,

where ``K_{a,r}`` is the (Laurent-polynomial) numerator recovered from the
graded dimensions; the cutoff is increased until the numerators stabilize.
This is an honest computation and does *not* assume the freeness/base-change
shortcut ``HHH_unreduced = HHH_reduced * Hilb(R)``.

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
from sympy.polys.matrices import DomainMatrix


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


def _coxeter_orders(diagram: DynkinDiagram) -> dict[tuple[Vertex, Vertex], int]:
    """Default Coxeter orders ``m_{ij}`` for a simply-laced diagram.

    Adjacent generators braid (``m = 3``); non-adjacent ones commute
    (``m = 2``).  This matches the Cartan/Coxeter data already encoded by the
    edges of ``diagram``.
    """

    orders: dict[tuple[Vertex, Vertex], int] = {}
    for i in diagram.vertices:
        for j in diagram.vertices:
            if i == j:
                continue
            adjacent = tuple(sorted((i, j))) in diagram.edges
            orders[(i, j)] = 3 if adjacent else 2
    return orders


def _alternating_product(first: sp.Matrix, second: sp.Matrix, length: int) -> sp.Matrix:
    """Return ``first second first ...`` with ``length`` alternating factors."""

    result = sp.eye(first.shape[0])
    factors = (first, second)
    for k in range(length):
        result = result * factors[k % 2]
    return result


class Realization:
    """A representation ``V`` of the Coxeter system, giving ``R = Sym(V^*)``.

    ``R`` is the polynomial ring on a fixed coordinate basis
    ``x_0, ..., x_{dim-1}`` of ``V^*``.  ``action[s]`` is the matrix of the
    simple reflection ``s`` acting on ``V^*`` in that basis, i.e.

        s . x_j = sum_k action[s][k, j] x_k.

    The ``s``-invariant linear forms are the ``+1`` eigenvectors of
    ``action[s]`` and the simple root ``alpha_s`` is its (unique) ``-1``
    eigenvector; together with ``alpha_s^2`` these generate ``R^s`` and define
    the Bott-Samelson relations for ``B_s``.

    Use :meth:`standard` for the reflection representation determined by the
    Cartan matrix (the previous, default behaviour) or :meth:`from_matrices`
    to supply your own ``V`` by giving a matrix for each generator.  Every
    generator must act as an involutive reflection (``s^2 = 1`` and a single
    ``-1`` eigenvalue) and the matrices must satisfy the braid relations
    ``s_i s_j s_i ... = s_j s_i s_j ...`` of the diagram.

    All coordinate variables share the internal ``Q``-degree
    ``ShiftConvention.variable_q_degree`` (the manuscript's ``deg alpha = 2``).
    """

    def __init__(
        self,
        diagram: DynkinDiagram,
        action: dict[Vertex, sp.Matrix],
        *,
        coxeter_orders: dict[tuple[Vertex, Vertex], int] | None = None,
        validate: bool = True,
    ) -> None:
        self.diagram = diagram
        missing = [vertex for vertex in diagram.vertices if vertex not in action]
        if missing:
            raise ValueError(f"missing action matrix for generator(s) {missing!r}")

        matrices: dict[Vertex, sp.Matrix] = {}
        dims = set()
        for vertex in diagram.vertices:
            matrix = sp.Matrix(action[vertex])
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"generator {vertex!r} action must be square")
            matrices[vertex] = matrix
            dims.add(matrix.shape[0])
        if len(dims) != 1:
            raise ValueError(f"all action matrices must have the same size, got {sorted(dims)}")

        self.action = matrices
        self.dim = dims.pop()
        self.coxeter_orders = coxeter_orders if coxeter_orders is not None else _coxeter_orders(diagram)

        self._roots: dict[Vertex, sp.Matrix] = {}
        self._fixed: dict[Vertex, tuple[sp.Matrix, ...]] = {}
        for vertex in diagram.vertices:
            matrix = matrices[vertex]
            roots = (matrix + sp.eye(self.dim)).nullspace()
            fixed = (matrix - sp.eye(self.dim)).nullspace()
            self._roots[vertex] = roots[0] if roots else sp.zeros(self.dim, 1)
            self._fixed[vertex] = tuple(fixed)

        if validate:
            self._validate()

    @classmethod
    def standard(cls, diagram: DynkinDiagram) -> "Realization":
        """The reflection representation built from the Cartan matrix."""

        order = diagram.vertices
        action: dict[Vertex, sp.Matrix] = {}
        for reflection in order:
            matrix = sp.eye(diagram.rank)
            row = diagram.vertex_index(reflection)
            for column_vertex in order:
                column = diagram.vertex_index(column_vertex)
                kronecker = 1 if reflection == column_vertex else 0
                matrix[row, column] = kronecker - diagram.cartan_entry(reflection, column_vertex)
            action[reflection] = matrix
        return cls(diagram, action, validate=False)

    @classmethod
    def from_matrices(
        cls,
        diagram: DynkinDiagram,
        matrices: dict[Vertex, sp.Matrix],
        *,
        dual: bool = False,
        coxeter_orders: dict[tuple[Vertex, Vertex], int] | None = None,
        validate: bool = True,
    ) -> "Realization":
        """Build a realization from user-supplied generator matrices.

        ``matrices`` maps each generator to the matrix of its action.  By
        default (``dual=False``) these are read as the action on ``V`` itself;
        the action on ``V^*`` used to build ``R`` is then the contragredient,
        which for an involution is the transpose.  Pass ``dual=True`` if your
        matrices already describe the action on ``V^*`` (the coordinate
        variables / roots).

        With ``validate=True`` (default) the matrices are checked to be
        involutive reflections satisfying the braid relations of ``diagram``.
        """

        action = {
            vertex: (sp.Matrix(matrix) if dual else sp.Matrix(matrix).T)
            for vertex, matrix in matrices.items()
        }
        return cls(diagram, action, coxeter_orders=coxeter_orders, validate=validate)

    def root(self, generator: Vertex) -> sp.Matrix:
        """Coordinate vector of the simple root ``alpha_s`` (``-1`` eigenform)."""

        return self._roots[generator]

    def fixed_forms(self, generator: Vertex) -> tuple[sp.Matrix, ...]:
        """Basis of the ``s``-invariant linear forms (``+1`` eigenforms)."""

        return self._fixed[generator]

    def _validate(self) -> None:
        identity = sp.eye(self.dim)
        for vertex in self.diagram.vertices:
            matrix = self.action[vertex]
            if sp.expand(matrix * matrix - identity) != sp.zeros(self.dim, self.dim):
                raise ValueError(f"generator {vertex!r} does not satisfy s^2 = 1")
            if len(self._roots[vertex]) == 0 or self._roots[vertex] == sp.zeros(self.dim, 1):
                raise ValueError(f"generator {vertex!r} acts trivially; it is not a reflection")
            if len(self._fixed[vertex]) != self.dim - 1:
                raise ValueError(
                    f"generator {vertex!r} is not a reflection: its fixed space has "
                    f"dimension {len(self._fixed[vertex])}, expected {self.dim - 1}"
                )

        for i, j in combinations(self.diagram.vertices, 2):
            order = self.coxeter_orders.get((i, j)) or self.coxeter_orders.get((j, i))
            if order is None:
                continue
            left = _alternating_product(self.action[i], self.action[j], order)
            right = _alternating_product(self.action[j], self.action[i], order)
            if sp.expand(left - right) != sp.zeros(self.dim, self.dim):
                raise ValueError(
                    f"braid relation of order {order} fails for generators {i!r}, {j!r}"
                )


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
    realization: Realization
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
    realization: Realization | None = None,
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

    if realization is None:
        realization = Realization.standard(diagram)
    elif realization.diagram != diagram:
        raise ValueError("realization.diagram does not match the supplied diagram")

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

    return RouquierComplex(diagram, realization, normalized_braid, terms, arrows, shifts)


def _local_degree(choice: int, epsilon: int) -> int:
    if choice == 0:
        return 0
    return -1 if epsilon == 1 else 1


class QuotientModel:
    """Quotient model for one Bott-Samelson term.

    With ``reduced=True`` the leftmost polynomial variables are set to zero and
    the quotient is a finite-dimensional ``QQ``-vector space.  With
    ``reduced=False`` those relations are dropped: the quotient is then free of
    finite rank over ``R = QQ[z_{0,i}]`` and infinite-dimensional over ``QQ``.
    In the unreduced case ``max_total_degree`` bounds the total monomial degree
    (sum of exponents) of the enumerated standard-monomial basis, so that each
    graded ``Q``-degree below the corresponding cutoff is represented exactly.
    """

    def __init__(
        self,
        realization: Realization,
        term: RouquierTerm,
        *,
        reduced: bool = False,
        shifts: ShiftConvention = DEFAULT_SHIFTS,
        max_basis_size: int = 5000,
        max_total_degree: int | None = None,
    ) -> None:
        self.realization = realization
        self.diagram = realization.diagram
        self.dim = realization.dim
        self.term = term
        self.reduced = reduced
        self.shifts = shifts
        self.max_basis_size = max_basis_size
        self.max_total_degree = max_total_degree
        self.layer_count = len(term.word) + 1

        names = [
            f"z{term.term_id}_{layer}_{position}"
            for layer in range(self.layer_count)
            for position in range(self.dim)
        ]
        symbols = sp.symbols(" ".join(names)) if names else tuple()
        # sympy.symbols("x") returns a Symbol rather than a tuple.
        self.variables = (symbols,) if isinstance(symbols, sp.Symbol) else tuple(symbols)
        self.layers = [
            [self.variables[layer * self.dim + position] for position in range(self.dim)]
            for layer in range(self.layer_count)
        ]

        relations = self._relations()
        if self.variables:
            order = "lex" if self.reduced else "grevlex"
            self.groebner = sp.groebner(relations, *self.variables, order=order, domain=sp.QQ)
            leading = [tuple(poly.LM(order=self.groebner.order)) for poly in self.groebner.polys]
            if self.reduced:
                if not self.groebner.is_zero_dimensional:
                    raise ValueError(
                        f"term {term.choices} did not produce a zero-dimensional quotient; "
                        "the reduced specialization may be missing relations"
                    )
                self.basis_exponents = _standard_monomials(leading, len(self.variables), max_basis_size)
            else:
                if self.max_total_degree is None:
                    raise ValueError(
                        "unreduced QuotientModel requires max_total_degree to bound the "
                        "infinite standard-monomial basis"
                    )
                self.basis_exponents = _standard_monomials(
                    leading,
                    len(self.variables),
                    max_basis_size,
                    max_degree=self.max_total_degree,
                )
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
        if self.reduced:
            relations.extend(self.layers[0][position] for position in range(self.dim))
        for position, generator in enumerate(self.term.word, start=1):
            left = position - 1
            right = position
            for fixed_form in self.realization.fixed_forms(generator):
                relations.append(
                    self._coordinate_form(left, fixed_form) - self._coordinate_form(right, fixed_form)
                )
            alpha_left = self.alpha(left, generator)
            alpha_right = self.alpha(right, generator)
            relations.append(alpha_left**2 - alpha_right**2)
        return [sp.expand(relation) for relation in relations]

    def _coordinate_form(self, layer: int, coordinates: sp.Matrix) -> sp.Expr:
        """Linear form ``sum_j coordinates[j] * x_{layer, j}`` in ``V^*``."""

        return sp.expand(
            sum(coordinates[position] * self.layers[layer][position] for position in range(self.dim))
        )

    def alpha(self, layer: int, generator: Vertex) -> sp.Expr:
        """The simple root ``alpha_generator`` realized at ``layer``."""

        return self._coordinate_form(layer, self.realization.root(generator))

    def diagonal_difference(self, position: int) -> sp.Expr:
        return self.layers[0][position] - self.layers[self.layer_count - 1][position]

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
            key = tuple(exponents)
            row = self.basis_index.get(key)
            if row is None:
                if not self.reduced and self.max_total_degree is not None and sum(key) > self.max_total_degree:
                    # A genuine standard monomial that exceeds the degree cutoff.
                    # Dropping it only affects graded pieces above the reliable
                    # bound; the degree-zero maps keep every piece below the
                    # cutoff complete.
                    continue
                raise ValueError(f"normal form contains non-standard monomial {key}")
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
            for position in range(self.dim):
                substitution[self.layers[layer][position]] = target.layers[target_layer][position]
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
            for position in range(self.dim):
                substitution[self.layers[layer][position]] = target.layers[target_layer][position]
        return substitution


def _standard_monomials(
    leading_monomials: Sequence[tuple[int, ...]],
    variable_count: int,
    max_basis_size: int,
    max_degree: int | None = None,
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
            if max_degree is not None and sum(next_exponent) > max_degree:
                continue
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
            solution = _fast_solve(block.combined_basis, restricted)
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

    rank = model.dim
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
        position: _multiplication_matrix(model, model.diagonal_difference(position))
        for position in range(model.dim)
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
    reduced: bool = False,
    max_basis_size: int = 5000,
    max_total_degree: int | None = None,
    validate: bool = True,
) -> ExtComputation:
    """Apply the finite reduced Koszul Ext computation termwise.

    The returned ``ExtComputation.induced_maps`` dictionary is keyed by
    ``(arrow_id, hochschild_degree)`` and contains the maps

        Ext^a(R, C^r) -> Ext^a(R, C^{r+1})

    induced by each summand of the Rouquier differential.

    With ``reduced=True`` this uses finite-dimensional linear algebra.  With
    ``reduced=False`` the quotient models are free of finite rank over
    ``R = QQ[z_{0,i}]``; ``max_total_degree`` then truncates the standard
    monomial basis so that each graded ``Q``-degree below the reliable bound is
    represented exactly.  The unreduced Hilbert series is assembled by
    ``khovanov_rozansky_cohomology(..., reduced=False)``.
    """

    term_data: dict[tuple[int, ...], ExtTermData] = {}
    for choices, term in sorted(rouquier.terms.items(), key=lambda item: item[1].term_id):
        model = QuotientModel(
            rouquier.realization,
            term,
            reduced=reduced,
            shifts=rouquier.shifts,
            max_basis_size=max_basis_size,
            max_total_degree=max_total_degree,
        )
        koszul = koszul_complex(model, shifts=rouquier.shifts)
        cohomology = {}
        rank = rouquier.realization.dim
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
        for degree in range(rouquier.realization.dim + 1):
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
        image_columns = _fast_columnspace(previous_block)
        kernel_columns = _fast_nullspace(next_block)

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


def _domain_matrix(matrix: sp.Matrix) -> DomainMatrix:
    return DomainMatrix.from_Matrix(matrix).convert_to(sp.QQ)


def _fast_rref_pivots(matrix: sp.Matrix) -> tuple[int, ...]:
    """Pivot column indices of ``matrix`` computed over ``QQ``.

    ``DomainMatrix`` uses a fraction-free algorithm, which is dramatically
    faster than ``sympy.Matrix.rref`` on the rational matrices that arise here
    (where naive Gaussian elimination suffers from coefficient blow-up).
    """

    if matrix.rows == 0 or matrix.cols == 0:
        return tuple()
    _, pivots = _domain_matrix(matrix).rref()
    return pivots


def _fast_columnspace(matrix: sp.Matrix) -> list[sp.Matrix]:
    return [matrix[:, column] for column in _fast_rref_pivots(matrix)]


def _fast_solve(matrix: sp.Matrix, target: sp.Matrix) -> sp.Matrix:
    """Solve ``matrix x = target`` when ``matrix`` has full column rank.

    ``target`` is assumed to lie in the column space (as is the case for the
    cohomology coordinate computations), so the solution is unique.  Non-pivot
    components default to zero, matching the previous ``gauss_jordan_solve``
    behaviour with free parameters set to zero.
    """

    columns = matrix.cols
    solution = sp.zeros(columns, 1)
    if columns == 0:
        return solution
    augmented = matrix.row_join(target)
    reduced, pivots = _domain_matrix(augmented).rref()
    reduced_matrix = reduced.to_Matrix()
    for row, pivot in enumerate(pivots):
        if pivot < columns:
            solution[pivot, 0] = reduced_matrix[row, columns]
    return solution


def _fast_nullspace(matrix: sp.Matrix) -> list[sp.Matrix]:
    columns = matrix.cols
    if columns == 0:
        return []
    if matrix.rows == 0:
        identity = sp.eye(columns)
        return [identity[:, index] for index in range(columns)]
    basis = _domain_matrix(matrix).nullspace().to_Matrix()
    return [basis[row, :].T for row in range(basis.rows)]


def _independent_extension(
    image_span: sp.Matrix,
    candidate_columns: Sequence[sp.Matrix],
) -> list[sp.Matrix]:
    """Pick the ``candidate_columns`` that are independent modulo ``image_span``.

    A single ``rref`` of ``[image_span | candidates]`` identifies the pivot
    columns; the pivots that fall in the candidate block are the cohomology
    representatives.  This avoids the quadratic loop of growing ``rank`` calls.
    """

    if not candidate_columns:
        return []
    image_count = image_span.shape[1]
    combined = (
        image_span.row_join(_matrix_from_columns(candidate_columns, image_span.shape[0]))
        if image_count
        else _matrix_from_columns(candidate_columns, candidate_columns[0].shape[0])
    )
    pivots = _fast_rref_pivots(combined)
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


@dataclass
class KRResult:
    polynomial: sp.Expr
    ext: ExtComputation
    homology_dimensions: dict[tuple[int, int, int], int]
    reduced: bool = True
    hilbert_factor: sp.Expr = sp.Integer(1)


def khovanov_rozansky_cohomology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    reduced: bool = False,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    max_basis_size: int = 5000,
    max_total_degree: int | None = None,
    validate: bool = True,
    realization: Realization | None = None,
) -> KRResult:
    """Compute the triply graded KR invariant of a braid.

    For ``reduced=True`` the returned expression is the reduced Laurent
    polynomial

        sum dim HHH^{a,r,q} A^a Q^q T^r.

    For ``reduced=False`` the unreduced invariant is computed directly, without
    imposing ``z_{0,i} = 0``.  The graded dimensions ``dim HHH^{a,r}_q`` are
    computed exactly for every ``q`` below a reliable cutoff (see the module
    docstring), and the returned expression is the Hilbert series

        sum_{a,r} A^a T^r * K_{a,r}(Q) / (1 - Q^variable_q_degree)^rank,

    where ``K_{a,r}`` is the numerator recovered from those dimensions.  No
    freeness/base-change shortcut is assumed.

    By default ``R = Sym(V^*)`` is built from the reflection representation of
    ``diagram``.  Pass ``realization`` (see :class:`Realization`) to use a
    different representation ``V``; ``rank`` above is then ``dim V``.
    """

    if realization is None:
        realization = Realization.standard(diagram)

    if not reduced:
        return _unreduced_khovanov_rozansky_cohomology(
            diagram,
            braid,
            shifts=shifts,
            max_basis_size=max_basis_size,
            max_total_degree=max_total_degree,
            realization=realization,
        )

    rouquier = rouquier_complex(diagram, braid, shifts=shifts, realization=realization)
    ext = koszul_ext_complex(
        rouquier,
        reduced=True,
        max_basis_size=max_basis_size,
        validate=validate,
    )
    homology_dimensions = _horizontal_homology_dimensions(rouquier, ext, validate=validate)

    polynomial = sp.Integer(0)
    for (hochschild_degree, rouquier_degree, q_degree), dimension in sorted(homology_dimensions.items()):
        polynomial += dimension * A**hochschild_degree * Q**q_degree * T**rouquier_degree

    return KRResult(
        sp.expand(polynomial),
        ext,
        homology_dimensions,
        reduced=True,
        hilbert_factor=sp.Integer(1),
    )


def _horizontal_homology_dimensions(
    rouquier: RouquierComplex,
    ext: ExtComputation,
    *,
    validate: bool,
) -> dict[tuple[int, int, int], int]:
    """Take homology of the induced Rouquier differential on Ext groups.

    Returns ``dim HHH^{a,r}_q`` keyed by ``(a, r, q)``.  In the unreduced case
    each value is the dimension of a single graded ``Q``-degree piece.
    """

    homology_dimensions: dict[tuple[int, int, int], int] = {}

    for hochschild_degree in range(rouquier.realization.dim + 1):
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

    return homology_dimensions


def _unreduced_khovanov_rozansky_cohomology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    max_basis_size: int = 5000,
    max_total_degree: int | None = None,
    degree_iterations: int = 8,
    realization: Realization | None = None,
) -> KRResult:
    """Compute the honest unreduced Hilbert series as a rational function.

    The standard-monomial basis of each term is truncated at total degree
    ``D``.  Because every map is degree ``0``, the graded dimension in each
    ``Q``-degree ``q <= reliable_max_qd(D)`` is exact, so the numerator of each
    rational Hilbert series can be reconstructed.  When ``max_total_degree`` is
    ``None`` the cutoff ``D`` is increased until the numerators stabilize.
    """

    if realization is None:
        realization = Realization.standard(diagram)
    rouquier = rouquier_complex(diagram, braid, shifts=shifts, realization=realization)
    rank = realization.dim
    variable_q_degree = shifts.variable_q_degree
    min_q_shift = min((term.q_shift for term in rouquier.terms.values()), default=0)

    def compute(cutoff: int) -> tuple[ExtComputation, dict, dict]:
        ext = koszul_ext_complex(
            rouquier,
            reduced=False,
            max_basis_size=max_basis_size,
            max_total_degree=cutoff,
            validate=False,
        )
        homology_dimensions = _horizontal_homology_dimensions(rouquier, ext, validate=False)
        # A degree-q graded piece of the Koszul complex of a term with q-shift
        # ``s`` uses monomials of total degree up to (q - s)/vqd + rank; that is
        # within the cutoff exactly when q <= s + vqd * (cutoff - rank).
        reliable_max_qd = min_q_shift + variable_q_degree * (cutoff - rank)
        numerators = _hilbert_numerators(
            homology_dimensions, reliable_max_qd, rank, variable_q_degree
        )
        return ext, homology_dimensions, numerators

    if max_total_degree is not None:
        ext, homology_dimensions, numerators = compute(max_total_degree)
    else:
        cutoff = rank + 2
        previous_numerators = None
        ext = homology_dimensions = numerators = None
        for _ in range(degree_iterations):
            ext, homology_dimensions, numerators = compute(cutoff)
            reliable_max_qd = min_q_shift + variable_q_degree * (cutoff - rank)
            stable = previous_numerators is not None and numerators == previous_numerators
            # Each numerator coefficient ``K[j]`` with ``j <= reliable_max_qd`` is
            # exact, so a numerator whose top degree is strictly below the bound
            # is a complete (terminated) polynomial.
            terminated = _numerators_top_degree(numerators) <= reliable_max_qd - variable_q_degree
            if stable and terminated:
                break
            previous_numerators = numerators
            cutoff += 2
        else:
            raise ValueError(
                "unreduced Hilbert numerators did not stabilize; pass a larger "
                "max_total_degree or increase degree_iterations"
            )

    polynomial = _assemble_hilbert_series(numerators, rank, variable_q_degree)
    return KRResult(
        polynomial,
        ext,
        homology_dimensions,
        reduced=False,
        hilbert_factor=polynomial_ring_hilbert_series(diagram, shifts=shifts, realization=realization),
    )


def _numerators_top_degree(
    numerators: dict[tuple[int, int], dict[int, int]],
) -> int:
    """Largest ``Q``-degree appearing in any numerator (``-inf`` if empty)."""

    top = float("-inf")
    for coefficients in numerators.values():
        if coefficients:
            top = max(top, max(coefficients))
    return top


def _hilbert_numerators(
    homology_dimensions: dict[tuple[int, int, int], int],
    reliable_max_qd: int,
    rank: int,
    variable_q_degree: int,
) -> dict[tuple[int, int], dict[int, int]]:
    """Reconstruct numerators ``K_{a,r}(Q)`` from graded dimensions.

    Each ``HHH^{a,r}`` is a finitely generated graded module over a polynomial
    ring in ``rank`` variables of degree ``variable_q_degree``, so its Hilbert
    series is ``K_{a,r}(Q) / (1 - Q^{vqd})^rank``.  Multiplying the (reliable
    range of the) graded-dimension series by ``(1 - Q^{vqd})^rank`` recovers the
    numerator's coefficients in every degree ``<= reliable_max_qd``.
    """

    blocks: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for (hochschild_degree, rouquier_degree, q_degree), dimension in homology_dimensions.items():
        if q_degree <= reliable_max_qd:
            blocks[(hochschild_degree, rouquier_degree)][q_degree] = dimension

    binomials = [sp.binomial(rank, k) for k in range(rank + 1)]
    numerators: dict[tuple[int, int], dict[int, int]] = {}
    for key, dimension_by_q in blocks.items():
        if not dimension_by_q:
            continue
        lowest = min(dimension_by_q)
        coefficients: dict[int, int] = {}
        for degree in range(lowest, reliable_max_qd + 1):
            value = 0
            for k in range(rank + 1):
                term = dimension_by_q.get(degree - variable_q_degree * k)
                if term:
                    value += int((-1) ** k * binomials[k]) * term
            if value:
                coefficients[degree] = value
        if coefficients:
            numerators[key] = coefficients
    return numerators


def _assemble_hilbert_series(
    numerators: dict[tuple[int, int], dict[int, int]],
    rank: int,
    variable_q_degree: int,
) -> sp.Expr:
    denominator = (1 - Q**variable_q_degree) ** rank
    series = sp.Integer(0)
    for (hochschild_degree, rouquier_degree), coefficients in sorted(numerators.items()):
        numerator = sp.Integer(0)
        for degree, coefficient in coefficients.items():
            numerator += coefficient * Q**degree
        series += A**hochschild_degree * T**rouquier_degree * numerator / denominator
    return series


def polynomial_ring_hilbert_series(
    diagram: DynkinDiagram,
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
) -> sp.Expr:
    """Return ``Hilb(R)`` for ``R = Sym(V^*)`` in the chosen grading.

    The number of variables is ``dim V``; this defaults to ``diagram.rank`` for
    the standard realization but follows ``realization`` when one is supplied.
    """

    dimension = diagram.rank if realization is None else realization.dim
    return sp.Integer(1) / (1 - Q**shifts.variable_q_degree) ** dimension


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
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="set the leftmost polynomial variables to zero and compute the finite reduced invariant",
    )
    parser.add_argument("--max-basis-size", type=int, default=5000)
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help=(
            "fix the total-degree cutoff for the unreduced standard-monomial basis; "
            "by default the cutoff grows until the Hilbert numerators stabilize"
        ),
    )
    args = parser.parse_args(argv)

    diagram = DynkinDiagram.from_data(parse_vertices(args.vertices), parse_edges(args.edges))
    result = khovanov_rozansky_cohomology(
        diagram,
        parse_braid(args.braid),
        reduced=args.reduced,
        max_basis_size=args.max_basis_size,
        max_total_degree=args.max_degree,
    )
    print(result.polynomial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
