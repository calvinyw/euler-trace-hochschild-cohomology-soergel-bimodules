"""Alexander-polynomial computations for acyclic quivers.

The two main functions are:

* ``alexander_polynomial_determinant``: the Fomin--Neville determinant
  ``det(t U_Q - U_Q.T)``.
* ``alexander_polynomial_forest``: the forest matching formula from
  Corollary 4.12 of Schwartz's forest-quiver HOMFLY paper.

Vertices may be supplied as an integer ``n`` (meaning ``range(n)``) or as any
finite iterable of hashable labels.  Arrows are ``(tail, head)`` or
``(tail, head, multiplicity)`` tuples.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Sequence
from typing import TypeAlias

import sympy as sp


Vertex: TypeAlias = Hashable
Arrow: TypeAlias = tuple[Vertex, Vertex] | tuple[Vertex, Vertex, int]


def alexander_polynomial_determinant(
    vertices: int | Iterable[Vertex],
    arrows: Iterable[Arrow],
    *,
    vertex_order: Sequence[Vertex] | None = None,
    t: sp.Symbol | None = None,
) -> sp.Expr:
    """Compute the Fomin--Neville Alexander polynomial.

    This is

        det(t U_Q - U_Q.T),

    where ``U_Q`` is the unipotent upper-triangular companion obtained from the
    exchange matrix ``B_Q`` by putting ``1`` on the diagonal and ``-b_ij`` above
    the diagonal.

    If ``vertex_order`` is omitted, a stable topological order is used, so the
    input quiver must be acyclic.  For a tree/forest quiver this determinant
    normalization should equal ``alexander_polynomial_forest(...,
    normalization="determinant")``.
    """

    symbol = t if t is not None else sp.Symbol("t")
    vertex_list, arrow_list = _normalize_inputs(vertices, arrows)
    order = list(vertex_order) if vertex_order is not None else _topological_order(vertex_list, arrow_list)
    _validate_order(vertex_list, order)

    exchange = _exchange_matrix(order, arrow_list)
    companion = _unipotent_companion(exchange)
    return sp.expand((symbol * companion - companion.T).det())


def alexander_polynomial_forest(
    vertices: int | Iterable[Vertex],
    arrows: Iterable[Arrow],
    *,
    t: sp.Symbol | None = None,
    normalization: str = "corollary",
) -> sp.Expr:
    """Compute the forest Alexander polynomial by matching numbers.

    Let ``b_i(Q)`` be the number of size-``i`` matchings in the underlying
    unoriented forest.  Corollary 4.12 gives

        t^(-n/2) * sum_i b_i(Q) t^i (t - 1)^(n - 2i).

    Use ``normalization="corollary"`` for this exact Laurent/half-Laurent
    expression, and ``normalization="determinant"`` for the polynomial

        sum_i b_i(Q) t^i (t - 1)^(n - 2i),

    which is the normalization that can be compared directly with
    ``alexander_polynomial_determinant``.
    """

    symbol = t if t is not None else sp.Symbol("t")
    vertex_list, arrow_list = _normalize_inputs(vertices, arrows)
    matchings = matching_numbers_forest(vertex_list, arrow_list)
    n = len(vertex_list)

    polynomial = sum(
        count * symbol**i * (symbol - 1) ** (n - 2 * i)
        for i, count in enumerate(matchings)
    )
    polynomial = sp.expand(polynomial)

    if normalization == "determinant":
        return polynomial
    if normalization == "corollary":
        return sp.expand(symbol ** sp.Rational(-n, 2) * polynomial)
    raise ValueError('normalization must be either "corollary" or "determinant"')


def matching_numbers_forest(
    vertices: int | Iterable[Vertex],
    arrows: Iterable[Arrow],
) -> list[int]:
    """Return ``[b_0, b_1, ...]`` for matchings in the underlying forest."""

    vertex_list, arrow_list = _normalize_inputs(vertices, arrows)
    adjacency = _underlying_simple_forest_adjacency(vertex_list, arrow_list)

    seen: set[Vertex] = set()
    total = [1]

    for root in vertex_list:
        if root in seen:
            continue
        component_poly, component_seen = _matching_poly_for_tree(root, adjacency)
        seen.update(component_seen)
        total = _poly_mul(total, component_poly)

    return total


def exchange_matrix(
    vertices: int | Iterable[Vertex],
    arrows: Iterable[Arrow],
    *,
    vertex_order: Sequence[Vertex] | None = None,
) -> sp.Matrix:
    """Return the exchange matrix in the convention ``b_ij = #(i -> j)``."""

    vertex_list, arrow_list = _normalize_inputs(vertices, arrows)
    order = list(vertex_order) if vertex_order is not None else _topological_order(vertex_list, arrow_list)
    _validate_order(vertex_list, order)
    return _exchange_matrix(order, arrow_list)


def _normalize_inputs(
    vertices: int | Iterable[Vertex],
    arrows: Iterable[Arrow],
) -> tuple[list[Vertex], list[tuple[Vertex, Vertex, int]]]:
    if isinstance(vertices, int):
        if vertices < 0:
            raise ValueError("the number of vertices must be nonnegative")
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    if len(set(vertex_list)) != len(vertex_list):
        raise ValueError("vertices must be distinct")

    vertex_set = set(vertex_list)
    arrow_list: list[tuple[Vertex, Vertex, int]] = []
    for arrow in arrows:
        if len(arrow) == 2:
            tail, head = arrow
            multiplicity = 1
        elif len(arrow) == 3:
            tail, head, multiplicity = arrow
        else:
            raise ValueError(f"arrow {arrow!r} must have length 2 or 3")

        if tail not in vertex_set or head not in vertex_set:
            raise ValueError(f"arrow {arrow!r} uses a vertex outside the vertex set")
        if tail == head:
            raise ValueError(f"loops are not allowed: {arrow!r}")
        if not isinstance(multiplicity, int) or multiplicity <= 0:
            raise ValueError(f"arrow multiplicity must be a positive integer: {arrow!r}")
        arrow_list.append((tail, head, multiplicity))

    return vertex_list, arrow_list


def _topological_order(
    vertices: Sequence[Vertex],
    arrows: Sequence[tuple[Vertex, Vertex, int]],
) -> list[Vertex]:
    position = {vertex: i for i, vertex in enumerate(vertices)}
    outgoing: dict[Vertex, list[Vertex]] = {vertex: [] for vertex in vertices}
    indegree = {vertex: 0 for vertex in vertices}

    for tail, head, _multiplicity in arrows:
        outgoing[tail].append(head)
        indegree[head] += 1

    for vertex in vertices:
        outgoing[vertex].sort(key=position.__getitem__)

    queue = deque(vertex for vertex in vertices if indegree[vertex] == 0)
    order: list[Vertex] = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for head in outgoing[vertex]:
            indegree[head] -= 1
            if indegree[head] == 0:
                queue.append(head)

    if len(order) != len(vertices):
        raise ValueError("the quiver is not acyclic; pass vertex_order to force a linear order")
    return order


def _validate_order(vertices: Sequence[Vertex], order: Sequence[Vertex]) -> None:
    if len(order) != len(vertices) or set(order) != set(vertices):
        raise ValueError("vertex_order must contain each vertex exactly once")


def _exchange_matrix(
    order: Sequence[Vertex],
    arrows: Sequence[tuple[Vertex, Vertex, int]],
) -> sp.Matrix:
    index = {vertex: i for i, vertex in enumerate(order)}
    matrix = sp.zeros(len(order))
    for tail, head, multiplicity in arrows:
        i = index[tail]
        j = index[head]
        matrix[i, j] += multiplicity
        matrix[j, i] -= multiplicity
    return matrix


def _unipotent_companion(exchange: sp.Matrix) -> sp.Matrix:
    n = exchange.rows
    companion = sp.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            companion[i, j] = -exchange[i, j]
    return companion


def _underlying_simple_forest_adjacency(
    vertices: Sequence[Vertex],
    arrows: Sequence[tuple[Vertex, Vertex, int]],
) -> dict[Vertex, list[Vertex]]:
    adjacency_sets: dict[Vertex, set[Vertex]] = {vertex: set() for vertex in vertices}
    seen_edges: set[frozenset[Vertex]] = set()

    for tail, head, multiplicity in arrows:
        if multiplicity != 1:
            raise ValueError("the forest formula expects a simple underlying graph")
        edge = frozenset((tail, head))
        if edge in seen_edges:
            raise ValueError("the forest formula expects at most one edge between two vertices")
        seen_edges.add(edge)
        adjacency_sets[tail].add(head)
        adjacency_sets[head].add(tail)

    adjacency = {vertex: sorted(neighbors, key=vertices.index) for vertex, neighbors in adjacency_sets.items()}
    _validate_forest(adjacency)
    return adjacency


def _validate_forest(adjacency: dict[Vertex, list[Vertex]]) -> None:
    seen: set[Vertex] = set()

    for root in adjacency:
        if root in seen:
            continue
        stack = [(root, None)]
        while stack:
            vertex, parent = stack.pop()
            if vertex in seen:
                raise ValueError("the underlying unoriented graph is not a forest")
            seen.add(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor != parent:
                    stack.append((neighbor, vertex))


def _matching_poly_for_tree(
    root: Vertex,
    adjacency: dict[Vertex, list[Vertex]],
) -> tuple[list[int], set[Vertex]]:
    seen: set[Vertex] = set()

    def visit(vertex: Vertex, parent: Vertex | None) -> tuple[list[int], list[int]]:
        seen.add(vertex)
        child_data = [
            (child, visit(child, vertex))
            for child in adjacency[vertex]
            if child != parent
        ]

        # blocked: the edge from parent to vertex is in the matching, so vertex
        # cannot be matched to any child.
        blocked = [1]
        for _child, (child_free, _child_blocked) in child_data:
            blocked = _poly_mul(blocked, child_free)

        free = blocked[:]
        for chosen_child, (_child_free, child_blocked) in child_data:
            term = [0, 1]
            for child, (other_free, other_blocked) in child_data:
                term = _poly_mul(term, other_blocked if child == chosen_child else other_free)
            free = _poly_add(free, term)

        return free, blocked

    root_free, _root_blocked = visit(root, None)
    return root_free, seen


def _poly_add(left: list[int], right: list[int]) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for i, coefficient in enumerate(left):
        result[i] += coefficient
    for i, coefficient in enumerate(right):
        result[i] += coefficient
    return _trim(result)


def _poly_mul(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, left_coefficient in enumerate(left):
        for j, right_coefficient in enumerate(right):
            result[i + j] += left_coefficient * right_coefficient
    return _trim(result)


def _trim(poly: list[int]) -> list[int]:
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly
