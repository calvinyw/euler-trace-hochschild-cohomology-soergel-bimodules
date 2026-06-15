"""Euler-trace computation assuming termwise Ext groups are free over ``R``.

WARNING: this shortcut is valid only when every termwise
``Ext^a_{R-R}(R, C^j)`` that appears is a free graded left ``R``-module.  If
any termwise Ext group has torsion or another non-free summand, the
specialization to ``k = R/(x_0, ..., x_n)`` can give the wrong Hilbert series.

This module is a faster, conditional companion to ``euler_trace.py``.  It uses
the same free-left-``R`` Bott--Samelson model, but assumes each termwise

    Ext^a_{R-R}(R, C^j)

is a free graded left ``R``-module.  Under that assumption, tensoring the
free-left-``R`` Koszul complex on the left with ``k = R/(x_0, ..., x_n)``
recovers the graded free generators:

    k tensor_R Ext^a_{R-R}(R, C^j)
        = H^a(k tensor_R K(C^j)).

The resulting finite-dimensional complex is computed by setting the left
``R`` variables to zero in the Koszul matrices and taking graded linear
algebra over ``QQ``.  The Hilbert series is then

    (sum_d number_of_generators_in_degree_d * Q^d) / HilbDen(R).

If the Ext groups have torsion, this specialization can overcount generators;
use ``euler_trace.py`` for the general Groebner-basis computation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Q,
    Realization,
    ShiftConvention,
    _graded_homology_basis,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules


@dataclass
class ExtFreeEulerTraceTermData:
    """Termwise free-generator data and Hilbert series."""

    free_generator_q_degrees: dict[int, list[int]]
    hilbert_series: dict[int, sp.Expr]


@dataclass
class ExtFreeEulerTraceResult:
    """Euler trace computed under the termwise Ext-freeness assumption."""

    polynomial: sp.Expr
    term_data: dict[tuple[int, ...], ExtFreeEulerTraceTermData]
    ring: Any


def khovanov_rozansky_euler_trace_extfree(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
) -> ExtFreeEulerTraceResult:
    """Compute the Euler trace assuming termwise Ext modules are free over ``R``.

    The returned polynomial is

        sum_a A^a sum_j (-1)^j Hilb_Q Ext^a_{R-R}(R, C^j),

    where the termwise Hilbert series are computed from the free generator
    degrees detected by ``k tensor_R -``.
    """

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], ExtFreeEulerTraceTermData] = {}
    trace = sp.Integer(0)
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        generator_degrees = extfree_generator_q_degrees_by_degree(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"Euler trace Ext-free term {choices}",
        )
        hilbert_series = extfree_hilbert_series_from_generators(
            generator_degrees,
            rouquier_free.r_variables,
            shifts.variable_q_degree,
        )
        sign = -1 if model.term.degree % 2 else 1
        for degree, series in hilbert_series.items():
            trace += sign * A**degree * series
        term_data[choices] = ExtFreeEulerTraceTermData(
            free_generator_q_degrees=generator_degrees,
            hilbert_series=hilbert_series,
        )

    return ExtFreeEulerTraceResult(
        polynomial=sp.factor(sp.cancel(trace)),
        term_data=term_data,
        ring=ring,
    )


def compute_euler_trace_extfree(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
) -> ExtFreeEulerTraceResult:
    """Short alias for ``khovanov_rozansky_euler_trace_extfree``."""

    return khovanov_rozansky_euler_trace_extfree(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
        validate=validate,
    )


def extfree_generator_q_degrees_by_degree(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
    *,
    validate: bool = True,
    context: str = "Ext-free Koszul complex",
) -> dict[int, list[int]]:
    """Free generator ``Q``-degrees for all Ext degrees after ``k tensor_R -``."""

    specialized = _left_specialized_differentials(koszul, variables)
    generator_degrees: dict[int, list[int]] = {}
    for degree in sorted(koszul.q_degrees):
        cohomology = _graded_homology_basis(
            specialized[degree - 1]
            if degree > 0
            else sp.zeros(len(koszul.basis[degree]), 0),
            specialized[degree],
            koszul.q_degrees[degree - 1] if degree > 0 else [],
            koszul.q_degrees[degree],
            koszul.q_degrees.get(degree + 1, []),
            validate=validate,
            context=f"{context}, Ext degree {degree}",
        )
        generator_degrees[degree] = cohomology.q_degrees
    return generator_degrees


def extfree_hilbert_series_from_generators(
    generator_q_degrees: dict[int, Sequence[int]],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> dict[int, sp.Expr]:
    """Hilbert series of free ``R``-modules from their generator degrees."""

    denominator = (1 - Q**variable_q_degree) ** len(variables)
    return {
        degree: sp.cancel(extfree_generator_numerator(degrees) / denominator)
        for degree, degrees in sorted(generator_q_degrees.items())
    }


def _left_specialized_differentials(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
) -> dict[int, sp.Matrix]:
    """Set the left ``R`` variables to zero in each Koszul differential."""

    zero_substitution = {variable: sp.Integer(0) for variable in variables}
    return {
        degree: matrix.applyfunc(
            lambda entry: sp.expand(sp.sympify(entry).xreplace(zero_substitution))
        )
        for degree, matrix in koszul.differentials.items()
    }


def extfree_generator_numerator(generator_q_degrees: Sequence[int]) -> sp.Expr:
    """Numerator ``sum Q^degree`` for a list of free generator degrees."""

    return sum((Q**q_degree for q_degree in generator_q_degrees), sp.Integer(0))
