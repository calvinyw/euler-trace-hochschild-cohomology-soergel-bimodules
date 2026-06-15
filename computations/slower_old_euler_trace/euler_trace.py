"""Euler-trace computation for unreduced Khovanov--Rozansky data.

This is the fast decategorified-in-the-Rouquier-direction companion to
``khovanov_rozansky_free_r.py``.  It keeps the left polynomial ring ``R`` and
computes the termwise Hochschild/Ext Hilbert series

    Ext^i_{R-R}(R, C^j)

without materializing the Ext modules themselves.  It then stops before the
expensive horizontal homology calculation and records the Euler trace

    sum_i A^i sum_j (-1)^j Hilb_Q Ext^i_{R-R}(R, C^j).

Equivalently, this is the ``T = -1`` trace of the chain complex direction,
computed from the termwise Ext Hilbert series.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
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
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    RModuleHomology,
    _image_submodule,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules


@dataclass
class EulerTraceTermData:
    """Termwise Ext Hilbert series."""

    hilbert_series: dict[int, sp.Expr]


@dataclass
class EulerTraceResult:
    """The Euler trace and the termwise data used to compute it."""

    polynomial: sp.Expr
    term_data: dict[tuple[int, ...], EulerTraceTermData]
    ring: Any


def khovanov_rozansky_euler_trace(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
) -> EulerTraceResult:
    """Compute ``sum_i A^i sum_j (-1)^j Hilb_Q Ext^i(R, C^j)``.

    This preserves the left ``R`` variables.  It does not impose
    ``z_{0,i} = 0``, does not compute horizontal homology, and does not
    materialize the termwise Ext modules.
    """

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], EulerTraceTermData] = {}
    trace = sp.Integer(0)
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        hilbert_series = koszul_ext_hilbert_series_by_degree(
            koszul,
            ring,
            rouquier_free.r_variables,
            shifts.variable_q_degree,
        )
        sign = -1 if model.term.degree % 2 else 1
        for degree, series in hilbert_series.items():
            trace += sign * A**degree * series
        term_data[choices] = EulerTraceTermData(hilbert_series=hilbert_series)

    return EulerTraceResult(
        polynomial=sp.factor(sp.cancel(trace)),
        term_data=term_data,
        ring=ring,
    )


def koszul_ext_hilbert_series_by_degree(
    koszul: FreeRKoszulComplex,
    ring: Any,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> dict[int, sp.Expr]:
    """Hilbert series of all Ext degrees for a free-``R`` Koszul complex."""

    free_series = {
        degree: free_module_hilbert_series(
            koszul.q_degrees[degree],
            variables,
            variable_q_degree,
        )
        for degree in sorted(koszul.q_degrees)
    }
    image_series = {
        degree: image_hilbert_series(
            koszul.differentials[degree],
            koszul.q_degrees.get(degree + 1, []),
            ring,
            variables,
            variable_q_degree,
        )
        for degree in sorted(koszul.q_degrees)
    }
    return {
        degree: sp.cancel(
            free_series[degree]
            - image_series[degree]
            - image_series.get(degree - 1, sp.Integer(0))
        )
        for degree in sorted(koszul.q_degrees)
    }


def koszul_ext_hilbert_series(
    koszul: FreeRKoszulComplex,
    ring: Any,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    degree: int,
) -> sp.Expr:
    """Hilbert series of ``Ext^degree`` for a free-``R`` Koszul complex.

    For a homogeneous complex ``F^a -> F^{a+1}``, exactness of
    ``0 -> ker(d_a) -> F^a -> im(d_a) -> 0`` gives

        Hilb ker(d_a) = Hilb F^a - Hilb im(d_a).

    Thus ``Hilb H^a = Hilb F^a - Hilb im(d_a) - Hilb im(d_{a-1})``.  This
    avoids constructing the kernel/syzygy module when only the Hilbert series is
    needed.
    """

    if degree < 0 or degree not in koszul.q_degrees:
        raise ValueError(f"degree {degree} is outside the Koszul complex")

    current_series = free_module_hilbert_series(
        koszul.q_degrees[degree],
        variables,
        variable_q_degree,
    )
    next_image_series = image_hilbert_series(
        koszul.differentials[degree],
        koszul.q_degrees.get(degree + 1, []),
        ring,
        variables,
        variable_q_degree,
    )
    previous_image_series = (
        image_hilbert_series(
            koszul.differentials[degree - 1],
            koszul.q_degrees[degree],
            ring,
            variables,
            variable_q_degree,
        )
        if degree > 0
        else sp.Integer(0)
    )
    return sp.cancel(current_series - next_image_series - previous_image_series)


def image_hilbert_series(
    matrix: sp.Matrix,
    target_q_degrees: Sequence[int],
    ring: Any,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> sp.Expr:
    """Hilbert series of the image submodule of a homogeneous matrix."""

    if matrix.rows != len(target_q_degrees):
        raise ValueError("target q-degree data does not match matrix rows")

    image = _image_submodule(matrix, ring)
    quotient_series = free_quotient_hilbert_series(
        image,
        target_q_degrees,
        variables,
        variable_q_degree,
    )
    target_series = free_module_hilbert_series(
        target_q_degrees,
        variables,
        variable_q_degree,
    )
    return sp.cancel(target_series - quotient_series)


def free_module_hilbert_series(
    ambient_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> sp.Expr:
    """Hilbert series of a shifted free ``R``-module."""

    denominator = (1 - Q**variable_q_degree) ** len(variables)
    numerator = sum((Q**q_degree for q_degree in ambient_q_degrees), sp.Integer(0))
    return sp.cancel(numerator / denominator)


def subquotient_hilbert_series(
    homology: RModuleHomology,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> sp.Expr:
    """Hilbert series of ``kernel / image`` using the ambient free grading."""

    ambient_degrees = homology.ambient_q_degrees
    image_series = free_quotient_hilbert_series(
        homology.image,
        ambient_degrees,
        variables,
        variable_q_degree,
    )
    kernel_series = free_quotient_hilbert_series(
        homology.kernel,
        ambient_degrees,
        variables,
        variable_q_degree,
    )
    return sp.cancel(image_series - kernel_series)


def free_quotient_hilbert_series(
    submodule: Any,
    ambient_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> sp.Expr:
    """Hilbert series of ``F / submodule`` for a shifted free module ``F``."""

    if len(ambient_q_degrees) != submodule.container.rank:
        raise ValueError("ambient q-degree data does not match submodule rank")

    leading_by_component: dict[int, list[tuple[int, ...]]] = {
        component: [] for component in range(submodule.container.rank)
    }
    for vector in submodule._groebner():
        if not vector:
            continue
        monomial, _coefficient = vector[0]
        component = int(monomial[0])
        exponents = tuple(int(exponent) for exponent in monomial[1:])
        leading_by_component[component].append(exponents)

    result = sp.Integer(0)
    for component, basis_q_degree in enumerate(ambient_q_degrees):
        result += Q**basis_q_degree * _monomial_quotient_hilbert_series(
            leading_by_component[component],
            len(variables),
            variable_q_degree,
        )
    return sp.cancel(result)


def _monomial_quotient_hilbert_series(
    generators: Sequence[tuple[int, ...]],
    variable_count: int,
    variable_q_degree: int,
) -> sp.Expr:
    """Hilbert series of ``R / I`` for a monomial ideal ``I``."""

    minimal = _minimal_monomial_generators(generators)
    if any(all(exponent == 0 for exponent in generator) for generator in minimal):
        return sp.Integer(0)

    denominator = (1 - Q**variable_q_degree) ** variable_count
    numerator = sp.Integer(0)
    for size in range(len(minimal) + 1):
        for subset in combinations(minimal, size):
            if not subset:
                lcm_degree = 0
            else:
                lcm_degree = sum(max(generator[index] for generator in subset) for index in range(variable_count))
            numerator += (-1) ** size * Q ** (variable_q_degree * lcm_degree)
    return sp.cancel(numerator / denominator)


def _minimal_monomial_generators(
    generators: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    unique = sorted(set(generators), key=lambda exponent: (sum(exponent), exponent))
    minimal: list[tuple[int, ...]] = []
    for candidate in unique:
        if not any(_monomial_divides(generator, candidate) for generator in minimal):
            minimal.append(candidate)
    return tuple(minimal)


def _monomial_divides(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(left_exp <= right_exp for left_exp, right_exp in zip(left, right, strict=True))
