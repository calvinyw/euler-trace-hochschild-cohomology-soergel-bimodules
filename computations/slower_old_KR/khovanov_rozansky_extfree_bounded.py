"""Bounded KR computation using termwise Ext-freeness.

This module uses the same termwise Ext-free shortcut as
``khovanov_rozansky_extfree.py`` and then computes the final horizontal
homology by finite-dimensional homogeneous slices up to ``max_q_degree``.
It therefore avoids Groebner bases in both the termwise Ext step and the final
horizontal step, but returns only a bounded truncation of the triply graded
series.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sympy as sp

from computations.slower_old_euler_trace.euler_trace_bounded import _bounded_homology_dimensions
from computations.euler_trace_extfree import extfree_generator_q_degrees_by_degree
from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Q,
    Realization,
    ShiftConvention,
    T,
)
from computations.khovanov_rozansky_extfree import (
    ExtFreeChainGroup,
    ExtFreeTermData,
    _extfree_chain_groups,
    _extfree_horizontal_maps,
    _extfree_module_data,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import free_r_koszul_complex
from computations.light_leaves import (
    RouquierFreeLeftRComplex,
    rouquier_complex_as_free_left_r_modules,
)


@dataclass
class BoundedExtFreeKRResult:
    """Bounded KR polynomial and the data used to compute it."""

    polynomial: sp.Expr
    homology_dimensions: dict[tuple[int, int, int], int]
    rouquier_free: RouquierFreeLeftRComplex
    term_data: dict[tuple[int, ...], ExtFreeTermData]
    ext_chain_groups: dict[tuple[int, int], ExtFreeChainGroup]
    horizontal_maps: dict[tuple[int, int], sp.Matrix]
    max_q_degree: int


def khovanov_rozansky_extfree_bounded_homology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedExtFreeKRResult:
    """Compute bounded KR homology using termwise Ext freeness."""

    if shifts.variable_q_degree <= 0:
        raise ValueError("the bounded Ext-free computation requires positive variable Q-degree")

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    rank = rouquier_free.rouquier.realization.dim
    term_data: dict[tuple[int, ...], ExtFreeTermData] = {}
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        generator_degrees = extfree_generator_q_degrees_by_degree(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"bounded Ext-free KR term {choices}",
        )
        ext = {
            degree: _extfree_module_data(
                koszul,
                degree,
                generator_degrees[degree],
                rouquier_free.r_variables,
                shifts.variable_q_degree,
                max_monomials=max_monomials,
                validate=validate,
                context=f"bounded Ext-free KR term {choices}, Ext degree {degree}",
            )
            for degree in range(rank + 1)
        }
        term_data[choices] = ExtFreeTermData(
            choices=choices,
            model=model,
            koszul=koszul,
            ext=ext,
        )

    ext_chain_groups = _extfree_chain_groups(rouquier_free, term_data)
    horizontal_maps = _extfree_horizontal_maps(
        rouquier_free,
        term_data,
        ext_chain_groups,
        max_monomials=max_monomials,
        validate=validate,
    )
    homology_dimensions = _bounded_horizontal_homology_dimensions(
        rouquier_free,
        ext_chain_groups,
        horizontal_maps,
        max_q_degree=max_q_degree,
        max_monomials=max_monomials,
        validate=validate,
    )

    polynomial = sp.Integer(0)
    for (ext_degree, rouquier_degree, q_degree), dimension in sorted(
        homology_dimensions.items()
    ):
        polynomial += dimension * A**ext_degree * T**rouquier_degree * Q**q_degree

    return BoundedExtFreeKRResult(
        polynomial=sp.expand(polynomial),
        homology_dimensions=homology_dimensions,
        rouquier_free=rouquier_free,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
        max_q_degree=max_q_degree,
    )


def compute_khovanov_rozansky_extfree_bounded(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    max_q_degree: int,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> BoundedExtFreeKRResult:
    """Short alias for ``khovanov_rozansky_extfree_bounded_homology``."""

    return khovanov_rozansky_extfree_bounded_homology(
        diagram,
        braid,
        max_q_degree=max_q_degree,
        shifts=shifts,
        realization=realization,
        max_monomials=max_monomials,
        validate=validate,
    )


def _bounded_horizontal_homology_dimensions(
    rouquier_free: RouquierFreeLeftRComplex,
    ext_chain_groups: dict[tuple[int, int], ExtFreeChainGroup],
    horizontal_maps: dict[tuple[int, int], sp.Matrix],
    *,
    max_q_degree: int,
    max_monomials: int,
    validate: bool,
) -> dict[tuple[int, int, int], int]:
    dimensions: dict[tuple[int, int, int], int] = {}
    rank = rouquier_free.rouquier.realization.dim
    variables = rouquier_free.r_variables
    variable_q_degree = rouquier_free.rouquier.shifts.variable_q_degree

    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            group = ext_chain_groups[(ext_degree, rouquier_degree)]
            previous_group = ext_chain_groups.get((ext_degree, rouquier_degree - 1))
            next_group = ext_chain_groups.get((ext_degree, rouquier_degree + 1))
            previous = horizontal_maps.get(
                (ext_degree, rouquier_degree - 1),
                sp.zeros(group.rank, 0),
            )
            next_map = horizontal_maps.get(
                (ext_degree, rouquier_degree),
                sp.zeros(0, group.rank),
            )
            q_dimensions = _bounded_homology_dimensions(
                previous,
                next_map,
                previous_group.generator_q_degrees if previous_group is not None else [],
                group.generator_q_degrees,
                next_group.generator_q_degrees if next_group is not None else [],
                variables,
                variable_q_degree,
                max_q_degree=max_q_degree,
                max_monomials=max_monomials,
                validate=validate,
                context=(
                    f"bounded Ext-free horizontal A-degree {ext_degree}, "
                    f"T-degree {rouquier_degree}"
                ),
            )
            for q_degree, dimension in q_dimensions.items():
                dimensions[(ext_degree, rouquier_degree, q_degree)] = dimension
    return dimensions
