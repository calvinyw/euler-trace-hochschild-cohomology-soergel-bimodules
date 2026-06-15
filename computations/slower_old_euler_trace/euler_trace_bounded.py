"""Bounded Euler-trace computation by homogeneous linear algebra.

This is a finite-dimensional companion to ``euler_trace.py``.  It uses the
same free-left-``R`` Bott-Samelson model, but avoids the SymPy AGCA Groebner
basis calculation for the termwise Ext modules.  Instead, for each fixed total
``Q``-degree, it expands the free ``R``-module Koszul complex into a finite
complex of ``QQ``-vector spaces and computes homology dimensions by matrix
ranks.

The result is a bounded truncation:

    sum_a A^a sum_j (-1)^j sum_{q <= max_q_degree}
        dim Ext^a_{R-R}(R, C^j)_q Q^q.

Unlike ``euler_trace.py``, this module does not recover a rational Hilbert
series.  It records exactly the homogeneous summands through the requested
``Q``-degree cutoff.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import TypeAlias

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from computations.khovanov_rozansky import (
    A,
    Q,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Realization,
    ShiftConvention,
    parse_braid,
    parse_edges,
    parse_vertices,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    free_r_koszul_complex,
)
from computations.light_leaves import (
    BottSamelsonFreeLeftRModel,
    RouquierFreeLeftRComplex,
    rouquier_complex_as_free_left_r_modules,
)


Exponent: TypeAlias = tuple[int, ...]
SliceBasis: TypeAlias = tuple[tuple[int, Exponent], ...]


@dataclass
class BoundedEulerTraceTermData:
    """Termwise bounded Ext dimensions."""

    model: BottSamelsonFreeLeftRModel
    koszul: FreeRKoszulComplex
    ext_dimensions: dict[int, dict[int, int]]


@dataclass
class BoundedEulerTraceResult:
    """Bounded Euler trace and the termwise data used to compute it."""

    polynomial: sp.Expr
    term_data: dict[tuple[int, ...], BoundedEulerTraceTermData]
    rouquier_free: RouquierFreeLeftRComplex
    max_q_degree: int


def khovanov_rozansky_bounded_euler_trace(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedEulerTraceResult:
    """Compute the Euler trace through the bounded internal ``Q``-degree.

    The computation keeps the left polynomial ring ``R`` but only enumerates the
    finite homogeneous pieces needed for ``q <= max_q_degree``.
    """

    if shifts.variable_q_degree <= 0:
        raise ValueError("bounded Euler trace requires a positive variable Q-degree")

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )

    term_data: dict[tuple[int, ...], BoundedEulerTraceTermData] = {}
    trace = sp.Integer(0)
    rank = rouquier_free.rouquier.realization.dim
    variables = rouquier_free.r_variables

    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        sign = -1 if model.term.degree % 2 else 1
        ext_dimensions: dict[int, dict[int, int]] = {}

        for degree in range(rank + 1):
            previous = (
                koszul.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(koszul.basis[degree]), 0)
            )
            next_map = koszul.differentials[degree]
            dimensions = _bounded_homology_dimensions(
                previous,
                next_map,
                koszul.q_degrees[degree - 1] if degree > 0 else [],
                koszul.q_degrees[degree],
                koszul.q_degrees[degree + 1] if degree < rank else [],
                variables,
                shifts.variable_q_degree,
                max_q_degree=max_q_degree,
                max_monomials=max_monomials,
                validate=validate,
                context=f"bounded Euler trace term {choices}, A-degree {degree}",
            )
            ext_dimensions[degree] = dimensions
            for q_degree, dimension in sorted(dimensions.items()):
                trace += sign * dimension * A**degree * Q**q_degree

        term_data[choices] = BoundedEulerTraceTermData(
            model=model,
            koszul=koszul,
            ext_dimensions=ext_dimensions,
        )

    return BoundedEulerTraceResult(
        polynomial=sp.expand(trace),
        term_data=term_data,
        rouquier_free=rouquier_free,
        max_q_degree=max_q_degree,
    )


def compute_bounded_euler_trace(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedEulerTraceResult:
    """Short alias for ``khovanov_rozansky_bounded_euler_trace``."""

    return khovanov_rozansky_bounded_euler_trace(
        diagram,
        braid,
        max_q_degree=max_q_degree,
        shifts=shifts,
        realization=realization,
        max_monomials=max_monomials,
        validate=validate,
    )


def _bounded_homology_dimensions(
    previous: sp.Matrix,
    next_map: sp.Matrix,
    previous_q_degrees: Sequence[int],
    current_q_degrees: Sequence[int],
    next_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_q_degree: int,
    max_monomials: int,
    validate: bool,
    context: str,
) -> dict[int, int]:
    if previous.rows != next_map.cols:
        raise ValueError(f"{context}: homology matrices have incompatible dimensions")
    if previous.cols != len(previous_q_degrees):
        raise ValueError(f"{context}: previous Q-degree data does not match")
    if previous.rows != len(current_q_degrees):
        raise ValueError(f"{context}: current Q-degree data does not match")
    if next_map.rows != len(next_q_degrees):
        raise ValueError(f"{context}: next Q-degree data does not match")

    dimensions: dict[int, int] = {}
    for q_degree in _possible_q_degrees(
        current_q_degrees,
        variable_q_degree,
        max_q_degree,
    ):
        previous_slice = _homogeneous_slice_matrix(
            previous,
            previous_q_degrees,
            current_q_degrees,
            q_degree,
            variables,
            variable_q_degree,
            max_monomials=max_monomials,
        )
        next_slice = _homogeneous_slice_matrix(
            next_map,
            current_q_degrees,
            next_q_degrees,
            q_degree,
            variables,
            variable_q_degree,
            max_monomials=max_monomials,
        )
        if validate and previous_slice.cols and next_slice.rows:
            composition = next_slice * previous_slice
            if any(entry != 0 for entry in composition):
                raise ValueError(f"{context}: differentials do not compose to zero at q={q_degree}")

        image_rank = _matrix_rank(previous_slice)
        kernel_dimension = next_slice.cols - _matrix_rank(next_slice)
        homology_dimension = kernel_dimension - image_rank
        if homology_dimension < 0:
            raise ValueError(f"{context}: negative homology dimension at q={q_degree}")
        if homology_dimension:
            dimensions[q_degree] = homology_dimension
    return dimensions


def _possible_q_degrees(
    basis_q_degrees: Sequence[int],
    variable_q_degree: int,
    max_q_degree: int,
) -> tuple[int, ...]:
    q_degrees = set()
    for basis_q_degree in basis_q_degrees:
        q_degree = basis_q_degree
        while q_degree <= max_q_degree:
            q_degrees.add(q_degree)
            q_degree += variable_q_degree
    return tuple(sorted(q_degrees))


def _homogeneous_slice_matrix(
    matrix: sp.Matrix,
    source_q_degrees: Sequence[int],
    target_q_degrees: Sequence[int],
    q_degree: int,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_monomials: int,
) -> sp.Matrix:
    source_basis, _source_index = _slice_basis(
        source_q_degrees,
        q_degree,
        len(variables),
        variable_q_degree,
        max_monomials=max_monomials,
    )
    target_basis, target_index = _slice_basis(
        target_q_degrees,
        q_degree,
        len(variables),
        variable_q_degree,
        max_monomials=max_monomials,
    )
    result = sp.zeros(len(target_basis), len(source_basis))
    if matrix.rows == 0 or matrix.cols == 0:
        return result

    polynomial_terms = {
        (row, column): _polynomial_terms(matrix[row, column], variables)
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    }
    for column, (source_component, source_exponent) in enumerate(source_basis):
        for target_component in range(matrix.rows):
            for term_exponent, coefficient in polynomial_terms.get(
                (target_component, source_component),
                (),
            ):
                target_exponent = _add_exponents(source_exponent, term_exponent)
                row = target_index.get((target_component, target_exponent))
                if row is not None:
                    result[row, column] += coefficient
    return result


def _slice_basis(
    basis_q_degrees: Sequence[int],
    q_degree: int,
    variable_count: int,
    variable_q_degree: int,
    *,
    max_monomials: int,
) -> tuple[SliceBasis, dict[tuple[int, Exponent], int]]:
    entries: list[tuple[int, Exponent]] = []
    for component, basis_q_degree in enumerate(basis_q_degrees):
        remainder = q_degree - basis_q_degree
        if remainder < 0 or remainder % variable_q_degree:
            continue
        polynomial_degree = remainder // variable_q_degree
        monomials = _monomials_of_degree(variable_count, polynomial_degree)
        if len(monomials) > max_monomials:
            raise ValueError(
                f"degree {polynomial_degree} has {len(monomials)} monomials; "
                "increase max_monomials if this bounded computation is intentional"
            )
        entries.extend((component, exponent) for exponent in monomials)
    basis = tuple(entries)
    return basis, {entry: index for index, entry in enumerate(basis)}


def _polynomial_terms(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
) -> tuple[tuple[Exponent, sp.Rational], ...]:
    polynomial = sp.expand(polynomial)
    if polynomial == 0:
        return tuple()
    if not variables:
        return ((tuple(), sp.Rational(polynomial)),)
    poly = sp.Poly(polynomial, *variables, domain=sp.QQ)
    return tuple(
        (tuple(int(exponent) for exponent in exponents), sp.Rational(coefficient))
        for exponents, coefficient in poly.terms()
        if coefficient
    )


def _matrix_rank(matrix: sp.Matrix) -> int:
    if matrix.rows == 0 or matrix.cols == 0:
        return 0
    return DomainMatrix.from_Matrix(matrix, domain=sp.QQ).rank()


def _add_exponents(left: Exponent, right: Exponent) -> Exponent:
    return tuple(
        left_exponent + right_exponent
        for left_exponent, right_exponent in zip(left, right, strict=True)
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", required=True, help="comma-separated vertices, e.g. 0,1,2")
    parser.add_argument("--edges", default="", help="comma-separated edges, e.g. 0-1,1-2")
    parser.add_argument("--braid", default="", help="comma-separated braid letters, e.g. 0:+,1:-")
    parser.add_argument(
        "--max-q-degree",
        type=int,
        required=True,
        help="compute Euler-trace summands with internal Q-degree at most this value",
    )
    parser.add_argument("--max-monomials", type=int, default=20000)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip q-sliced differential consistency checks",
    )
    args = parser.parse_args(argv)

    diagram = DynkinDiagram.from_data(parse_vertices(args.vertices), parse_edges(args.edges))
    result = khovanov_rozansky_bounded_euler_trace(
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
