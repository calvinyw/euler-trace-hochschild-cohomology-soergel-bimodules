"""Khovanov--Rozansky homology with the Euler-trace 2.0 minimization.

This module is the general Groebner-basis KR computation, but with the
termwise Koszul complexes first shrunk by the same field-splitting cancellation
used in ``euler_trace_2_0.py``.  The horizontal Rouquier differential is then
transported to those minimal termwise complexes and horizontal homology is
computed over the left polynomial ring ``R``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from computations.euler_trace_2_0 import (
    MinimalComplexData,
    minimal_koszul_complex_from_field_splitting,
    _unit_pivot_solve_matrix,
)
from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Realization,
    ShiftConvention,
    T,
)
from computations.light_leaves import (
    BottSamelsonFreeLeftRModel,
    RouquierFreeLeftRComplex,
    rouquier_complex_as_free_left_r_modules,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    RModuleHomology,
    _assert_graded_matrix,
    _embedded_direct_sum_submodule,
    _free_module_homology,
    _horizontal_homology,
    _koszul_chain_map,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.slower_old_euler_trace.euler_trace import (
    subquotient_hilbert_series,
)


@dataclass
class KR20TermData:
    """Termwise Koszul data after the 2.0 minimization step."""

    choices: tuple[int, ...]
    model: BottSamelsonFreeLeftRModel
    koszul: FreeRKoszulComplex
    minimal: MinimalComplexData
    block_sizes: dict[int, tuple[int, int, int]]
    ext: dict[int, RModuleHomology]


@dataclass
class KR20ChainGroup:
    """Direct sum of minimized termwise Ext presentations."""

    ext_degree: int
    rouquier_degree: int
    term_keys: tuple[tuple[int, ...], ...]
    offsets: dict[tuple[int, ...], int]
    ambient_q_degrees: list[int]
    kernel: Any
    image: Any
    module: Any


@dataclass
class KR20Result:
    """Khovanov--Rozansky homology computed through the 2.0 pipeline."""

    polynomial: sp.Expr
    euler_trace_polynomial: sp.Expr
    rouquier_free: RouquierFreeLeftRComplex
    ring: Any
    term_data: dict[tuple[int, ...], KR20TermData]
    ext_chain_groups: dict[tuple[int, int], KR20ChainGroup]
    horizontal_maps: dict[tuple[int, int], sp.Matrix]
    horizontal_homology: dict[tuple[int, int], RModuleHomology]


def khovanov_rozansky_2_0_homology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
) -> KR20Result:
    """Compute KR homology after minimizing each termwise Koszul complex."""

    if shifts.variable_q_degree <= 0:
        raise ValueError("KR 2.0 requires a positive variable Q-degree")

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], KR20TermData] = {}
    rank = rouquier_free.rouquier.realization.dim
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        minimal = minimal_koszul_complex_from_field_splitting(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"KR 2.0 term {choices}",
        )
        block_sizes = _minimal_block_sizes(minimal)
        ext = {
            degree: _free_module_homology(
                minimal.complex.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(minimal.complex.q_degrees[degree]), 0),
                minimal.complex.differentials[degree],
                ring,
                minimal.complex.q_degrees[degree],
                rouquier_free.r_variables,
                shifts.variable_q_degree,
                degree=degree,
            )
            for degree in range(rank + 1)
        }
        term_data[choices] = KR20TermData(
            choices=choices,
            model=model,
            koszul=koszul,
            minimal=minimal,
            block_sizes=block_sizes,
            ext=ext,
        )

    ext_chain_groups = _kr20_chain_groups(rouquier_free, term_data, ring)
    horizontal_maps = _kr20_horizontal_maps(
        rouquier_free,
        term_data,
        ext_chain_groups,
        validate=validate,
    )
    horizontal_homology = _horizontal_homology(
        rouquier_free,
        ext_chain_groups,
        horizontal_maps,
        ring,
        shifts,
    )
    polynomial, euler_trace = _homology_hilbert_polynomials(
        horizontal_homology,
        rouquier_free.r_variables,
        shifts.variable_q_degree,
    )

    return KR20Result(
        polynomial=polynomial,
        euler_trace_polynomial=euler_trace,
        rouquier_free=rouquier_free,
        ring=ring,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
        horizontal_homology=horizontal_homology,
    )


def compute_khovanov_rozansky_2_0(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
) -> KR20Result:
    """Short alias for ``khovanov_rozansky_2_0_homology``."""

    return khovanov_rozansky_2_0_homology(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
        validate=validate,
    )


def _minimal_block_sizes(
    minimal: MinimalComplexData,
) -> dict[int, tuple[int, int, int]]:
    block_sizes: dict[int, tuple[int, int, int]] = {}
    for degree in sorted(minimal.complex.q_degrees):
        previous_sources = (
            minimal.field_splitting[degree - 1].source_lifts
            if degree - 1 in minimal.field_splitting
            else []
        )
        splitting = minimal.field_splitting[degree]
        block_sizes[degree] = (
            len(previous_sources),
            len(splitting.homology_lifts),
            len(splitting.source_lifts),
        )

        change = minimal.basis_changes[degree]
        expected_columns = sum(block_sizes[degree])
        expected_rows = (
            len(minimal.complex.q_degrees[degree])
            + block_sizes[degree][0]
            + block_sizes[degree][2]
        )
        if change.cols != expected_columns or change.rows != expected_rows:
            raise ValueError(
                f"minimal basis data in degree {degree} has shape {change.shape}, "
                f"expected {expected_rows} by {expected_columns}"
            )
    return block_sizes


def _kr20_chain_groups(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], KR20TermData],
    ring: Any,
) -> dict[tuple[int, int], KR20ChainGroup]:
    groups: dict[tuple[int, int], KR20ChainGroup] = {}
    rank = rouquier_free.rouquier.realization.dim
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            term_keys = tuple(
                key
                for key, term in sorted(
                    rouquier_free.rouquier.terms.items(),
                    key=lambda item: item[1].term_id,
                )
                if term.degree == rouquier_degree
            )
            offsets: dict[tuple[int, ...], int] = {}
            ambient_q_degrees: list[int] = []
            offset = 0
            for key in term_keys:
                q_degrees = term_data[key].minimal.complex.q_degrees[ext_degree]
                offsets[key] = offset
                ambient_q_degrees.extend(q_degrees)
                offset += len(q_degrees)

            ambient = ring.free_module(offset)
            kernel = _embedded_direct_sum_submodule(
                ambient,
                [
                    (
                        offsets[key],
                        len(term_data[key].minimal.complex.q_degrees[ext_degree]),
                        term_data[key].ext[ext_degree].kernel.gens,
                    )
                    for key in term_keys
                ],
            )
            image = _embedded_direct_sum_submodule(
                ambient,
                [
                    (
                        offsets[key],
                        len(term_data[key].minimal.complex.q_degrees[ext_degree]),
                        term_data[key].ext[ext_degree].image.gens,
                    )
                    for key in term_keys
                ],
            )
            groups[(ext_degree, rouquier_degree)] = KR20ChainGroup(
                ext_degree=ext_degree,
                rouquier_degree=rouquier_degree,
                term_keys=term_keys,
                offsets=offsets,
                ambient_q_degrees=ambient_q_degrees,
                kernel=kernel,
                image=image,
                module=kernel.quotient_module(image),
            )
    return groups


def _kr20_horizontal_maps(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], KR20TermData],
    ext_chain_groups: dict[tuple[int, int], KR20ChainGroup],
    *,
    validate: bool,
) -> dict[tuple[int, int], sp.Matrix]:
    maps: dict[tuple[int, int], sp.Matrix] = {}
    rank = rouquier_free.rouquier.realization.dim
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            source_group = ext_chain_groups[(ext_degree, rouquier_degree)]
            target_group = ext_chain_groups.get((ext_degree, rouquier_degree + 1))
            if target_group is None:
                continue

            matrix = sp.zeros(
                len(target_group.ambient_q_degrees),
                len(source_group.ambient_q_degrees),
            )
            for arrow in rouquier_free.rouquier.arrows:
                source_term = rouquier_free.rouquier.terms[arrow.source]
                target_term = rouquier_free.rouquier.terms[arrow.target]
                if (
                    source_term.degree != rouquier_degree
                    or target_term.degree != rouquier_degree + 1
                ):
                    continue

                source_data = term_data[arrow.source]
                target_data = term_data[arrow.target]
                block = _induced_minimal_map(
                    source_data,
                    target_data,
                    ext_degree,
                    rouquier_free.arrow_matrices[arrow.arrow_id],
                    rouquier_free.r_variables,
                    context=(
                        f"KR 2.0 horizontal arrow {arrow.arrow_id}, "
                        f"Ext {ext_degree}"
                    ),
                )
                source_offset = source_group.offsets[arrow.source]
                target_offset = target_group.offsets[arrow.target]
                for row in range(block.rows):
                    for column in range(block.cols):
                        coefficient = block[row, column]
                        if coefficient:
                            matrix[target_offset + row, source_offset + column] += (
                                coefficient
                            )

            if validate:
                _assert_graded_matrix(
                    matrix,
                    source_group.ambient_q_degrees,
                    target_group.ambient_q_degrees,
                    rouquier_free.r_variables,
                    rouquier_free.rouquier.shifts.variable_q_degree,
                    context=f"KR 2.0 horizontal Ext {ext_degree}, degree {rouquier_degree}",
                )
            maps[(ext_degree, rouquier_degree)] = matrix.applyfunc(sp.expand)
    return maps


def _induced_minimal_map(
    source: KR20TermData,
    target: KR20TermData,
    degree: int,
    term_map: sp.Matrix,
    variables: Sequence[sp.Symbol],
    *,
    context: str,
) -> sp.Matrix:
    source_sizes = source.block_sizes[degree]
    target_sizes = target.block_sizes[degree]
    source_h_count = source_sizes[1]
    target_h_count = target_sizes[1]
    if source_h_count == 0 or target_h_count == 0:
        return sp.zeros(target_h_count, source_h_count)

    chain_map = _koszul_chain_map(source.koszul, target.koszul, degree, term_map)
    source_inclusion = _corrected_minimal_inclusion(source, degree, variables, context=context)
    image = (chain_map * source_inclusion).applyfunc(sp.expand)
    target_coords = _unit_pivot_solve_matrix(
        target.minimal.basis_changes[degree],
        image,
        variables,
        context=f"{context}, target minimal coordinates",
    )
    target_h_start = target_sizes[0]
    target_h_end = target_h_start + target_h_count
    return target_coords.extract(
        range(target_h_start, target_h_end),
        range(source_h_count),
    ).applyfunc(sp.expand)


def _corrected_minimal_inclusion(
    term: KR20TermData,
    degree: int,
    variables: Sequence[sp.Symbol],
    *,
    context: str,
) -> sp.Matrix:
    """Return the chain-level inclusion of the minimized ``H`` summand."""

    sizes = term.block_sizes[degree]
    boundary_count, homology_count, source_count = sizes
    change = term.minimal.basis_changes[degree]
    homology_start = boundary_count
    homology_end = homology_start + homology_count
    homology_columns = change.extract(
        range(change.rows),
        range(homology_start, homology_end),
    )
    if homology_count == 0 or source_count == 0:
        return homology_columns

    source_start = homology_end
    source_end = source_start + source_count
    source_columns = change.extract(range(change.rows), range(source_start, source_end))
    boundary_leak = _homology_to_boundary_block(term, degree, variables, context=context)
    if boundary_leak.shape != (source_count, homology_count):
        raise ValueError(
            f"{context}: boundary-leak block has shape {boundary_leak.shape}, "
            f"expected {(source_count, homology_count)}"
        )
    return (homology_columns - source_columns * boundary_leak).applyfunc(sp.expand)


def _homology_to_boundary_block(
    term: KR20TermData,
    degree: int,
    variables: Sequence[sp.Symbol],
    *,
    context: str,
) -> sp.Matrix:
    """Coordinates of ``d(H_degree)`` in the next boundary block."""

    sizes = term.block_sizes[degree]
    homology_count = sizes[1]
    source_count = sizes[2]
    if homology_count == 0:
        return sp.zeros(source_count, 0)
    if degree + 1 not in term.minimal.basis_changes:
        if source_count != 0:
            raise ValueError(f"{context}: terminal degree has nonzero source block")
        return sp.zeros(0, homology_count)

    change = term.minimal.basis_changes[degree]
    homology_start = sizes[0]
    homology_end = homology_start + homology_count
    homology_columns = change.extract(
        range(change.rows),
        range(homology_start, homology_end),
    )
    image = (term.koszul.differentials[degree] * homology_columns).applyfunc(sp.expand)
    target_coords = _unit_pivot_solve_matrix(
        term.minimal.basis_changes[degree + 1],
        image,
        variables,
        context=f"{context}, source boundary correction",
    )
    target_boundary_count = term.block_sizes[degree + 1][0]
    return target_coords.extract(
        range(target_boundary_count),
        range(homology_count),
    ).applyfunc(sp.expand)


def _homology_hilbert_polynomials(
    horizontal_homology: dict[tuple[int, int], RModuleHomology],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> tuple[sp.Expr, sp.Expr]:
    polynomial = sp.Integer(0)
    euler_trace = sp.Integer(0)
    for (ext_degree, rouquier_degree), homology in sorted(horizontal_homology.items()):
        series = subquotient_hilbert_series(
            homology,
            variables,
            variable_q_degree,
        )
        polynomial += A**ext_degree * T**rouquier_degree * series
        sign = -1 if rouquier_degree % 2 else 1
        euler_trace += sign * A**ext_degree * series
    return sp.factor(sp.cancel(polynomial)), sp.factor(sp.cancel(euler_trace))
