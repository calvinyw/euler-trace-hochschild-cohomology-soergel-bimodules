"""Khovanov--Rozansky data using termwise Ext-freeness.

WARNING: this shortcut is valid only when every termwise
``Ext^a_{R-R}(R, C^j)`` that appears is a free graded left ``R``-module.  If
any termwise Ext group has torsion or another non-free summand, the
specialization to ``k = R/(x_0, ..., x_n)`` can give the wrong generator data.

This module is a conditional companion to ``khovanov_rozansky_free_r.py``.  It
assumes only the termwise modules

    Ext^a_{R-R}(R, C^j)

are free as graded left ``R``-modules.  Under that assumption, the free
generator degrees are found by tensoring the free-left-``R`` Koszul complex on
the left with ``k = R/(x_0, ..., x_n)`` and taking finite-dimensional graded
homology over ``QQ``.

The horizontal Rouquier differential is not assumed to have free homology.  The
induced maps between the free termwise Ext modules are reconstructed as
homogeneous matrices over ``R`` by comparing homogeneous slices of the Koszul
complexes.  Horizontal homology is then computed over ``R`` with the same AGCA
subquotient machinery used by ``khovanov_rozansky_free_r.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import sympy as sp

from computations.slower_old_euler_trace.euler_trace import subquotient_hilbert_series
from computations.slower_old_euler_trace.euler_trace_bounded import (
    _homogeneous_slice_matrix,
    _monomials_of_degree,
    _polynomial_terms,
    _slice_basis,
)
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
    _fast_columnspace,
    _fast_nullspace,
    _fast_solve,
    _independent_extension,
    _matrix_from_columns,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    RModuleHomology,
    _assert_graded_matrix,
    _free_module_homology,
    _koszul_chain_map,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import (
    BottSamelsonFreeLeftRModel,
    RouquierFreeLeftRComplex,
    rouquier_complex_as_free_left_r_modules,
)


Exponent: TypeAlias = tuple[int, ...]


@dataclass
class ExtFreeModuleData:
    """One free termwise Ext module with chosen homogeneous cycle lifts."""

    degree: int
    generator_q_degrees: list[int]
    generator_cycles: list[sp.Matrix]

    @property
    def rank(self) -> int:
        return len(self.generator_q_degrees)


@dataclass
class ExtFreeTermData:
    """Termwise Koszul and Ext-free data for one Rouquier summand."""

    choices: tuple[int, ...]
    model: BottSamelsonFreeLeftRModel
    koszul: FreeRKoszulComplex
    ext: dict[int, ExtFreeModuleData]


@dataclass
class ExtFreeChainGroup:
    """Direct sum of free termwise Ext modules in one Rouquier degree."""

    ext_degree: int
    rouquier_degree: int
    term_keys: tuple[tuple[int, ...], ...]
    offsets: dict[tuple[int, ...], int]
    generator_q_degrees: list[int]

    @property
    def rank(self) -> int:
        return len(self.generator_q_degrees)


@dataclass
class ExtFreeKRResult:
    """KR computation using the termwise Ext-freeness shortcut."""

    polynomial: sp.Expr
    euler_trace_polynomial: sp.Expr
    rouquier_free: RouquierFreeLeftRComplex
    ring: Any
    term_data: dict[tuple[int, ...], ExtFreeTermData]
    ext_chain_groups: dict[tuple[int, int], ExtFreeChainGroup]
    horizontal_maps: dict[tuple[int, int], sp.Matrix]
    horizontal_homology: dict[tuple[int, int], RModuleHomology]


def khovanov_rozansky_extfree_homology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> ExtFreeKRResult:
    """Compute KR data assuming only termwise Ext modules are free over ``R``."""

    if shifts.variable_q_degree <= 0:
        raise ValueError("the Ext-free computation requires a positive variable Q-degree")

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], ExtFreeTermData] = {}
    rank = rouquier_free.rouquier.realization.dim
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        generator_degrees = extfree_generator_q_degrees_by_degree(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"Ext-free KR term {choices}",
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
                context=f"Ext-free KR term {choices}, Ext degree {degree}",
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
    horizontal_homology = _extfree_horizontal_homology(
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

    return ExtFreeKRResult(
        polynomial=polynomial,
        euler_trace_polynomial=euler_trace,
        rouquier_free=rouquier_free,
        ring=ring,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
        horizontal_homology=horizontal_homology,
    )


def compute_khovanov_rozansky_extfree(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    max_monomials: int = 20000,
    validate: bool = True,
) -> ExtFreeKRResult:
    """Short alias for ``khovanov_rozansky_extfree_homology``."""

    return khovanov_rozansky_extfree_homology(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
        max_monomials=max_monomials,
        validate=validate,
    )


def _extfree_module_data(
    koszul: FreeRKoszulComplex,
    degree: int,
    generator_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_monomials: int,
    validate: bool,
    context: str,
) -> ExtFreeModuleData:
    """Choose homogeneous cycle lifts for the free Ext generators."""

    previous = (
        koszul.differentials[degree - 1]
        if degree > 0
        else sp.zeros(len(koszul.basis[degree]), 0)
    )
    next_map = koszul.differentials[degree]
    previous_q_degrees = koszul.q_degrees[degree - 1] if degree > 0 else []
    current_q_degrees = koszul.q_degrees[degree]
    next_q_degrees = koszul.q_degrees.get(degree + 1, [])

    generator_cycles: list[sp.Matrix] = []
    chosen_degrees: list[int] = []
    for q_degree in sorted(set(generator_q_degrees)):
        needed = sum(1 for item in generator_q_degrees if item == q_degree)
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
                raise ValueError(f"{context}: differentials do not compose at Q={q_degree}")

        current_basis, _current_index = _slice_basis(
            current_q_degrees,
            q_degree,
            len(variables),
            variable_q_degree,
            max_monomials=max_monomials,
        )
        row_count = len(current_basis)
        boundary_columns = _fast_columnspace(previous_slice)
        lower_columns = [
            _polynomial_vector_to_slice_coords(
                _multiply_polynomial_vector_by_monomial(cycle, exponent, variables),
                current_q_degrees,
                q_degree,
                variables,
                variable_q_degree,
                max_monomials=max_monomials,
            )
            for cycle, cycle_q_degree in zip(generator_cycles, chosen_degrees, strict=True)
            for exponent in _monomials_for_q_difference(
                q_degree - cycle_q_degree,
                len(variables),
                variable_q_degree,
            )
        ]
        span_columns = _independent_columns(boundary_columns + lower_columns, row_count)
        span = _matrix_from_columns(span_columns, row_count)
        kernel_columns = _fast_nullspace(next_slice)
        new_columns = _independent_extension(span, kernel_columns)
        if len(new_columns) != needed:
            raise ValueError(
                f"{context}: expected {needed} new free generator(s) at Q={q_degree}, "
                f"found {len(new_columns)}"
            )
        for column in new_columns:
            generator_cycles.append(
                _slice_coords_to_polynomial_vector(
                    column,
                    current_basis,
                    len(current_q_degrees),
                    variables,
                )
            )
            chosen_degrees.append(q_degree)

    if sorted(chosen_degrees) != sorted(generator_q_degrees):
        raise ValueError(f"{context}: chosen generator degrees do not match specialization")
    return ExtFreeModuleData(
        degree=degree,
        generator_q_degrees=chosen_degrees,
        generator_cycles=generator_cycles,
    )


def _extfree_chain_groups(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], ExtFreeTermData],
) -> dict[tuple[int, int], ExtFreeChainGroup]:
    groups: dict[tuple[int, int], ExtFreeChainGroup] = {}
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
            generator_q_degrees: list[int] = []
            offset = 0
            for key in term_keys:
                ext_module = term_data[key].ext[ext_degree]
                offsets[key] = offset
                generator_q_degrees.extend(ext_module.generator_q_degrees)
                offset += ext_module.rank
            groups[(ext_degree, rouquier_degree)] = ExtFreeChainGroup(
                ext_degree=ext_degree,
                rouquier_degree=rouquier_degree,
                term_keys=term_keys,
                offsets=offsets,
                generator_q_degrees=generator_q_degrees,
            )
    return groups


def _extfree_horizontal_maps(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], ExtFreeTermData],
    ext_chain_groups: dict[tuple[int, int], ExtFreeChainGroup],
    *,
    max_monomials: int,
    validate: bool,
) -> dict[tuple[int, int], sp.Matrix]:
    maps: dict[tuple[int, int], sp.Matrix] = {}
    rank = rouquier_free.rouquier.realization.dim
    variables = rouquier_free.r_variables
    variable_q_degree = rouquier_free.rouquier.shifts.variable_q_degree
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            source_group = ext_chain_groups[(ext_degree, rouquier_degree)]
            target_group = ext_chain_groups.get((ext_degree, rouquier_degree + 1))
            if target_group is None:
                continue

            matrix = sp.zeros(target_group.rank, source_group.rank)
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
                block = _induced_extfree_map(
                    source_data.ext[ext_degree],
                    target_data.ext[ext_degree],
                    source_data.koszul,
                    target_data.koszul,
                    rouquier_free.arrow_matrices[arrow.arrow_id],
                    ext_degree,
                    variables,
                    variable_q_degree,
                    max_monomials=max_monomials,
                )
                source_offset = source_group.offsets[arrow.source]
                target_offset = target_group.offsets[arrow.target]
                for row in range(block.rows):
                    for column in range(block.cols):
                        coefficient = block[row, column]
                        if coefficient:
                            matrix[target_offset + row, source_offset + column] += coefficient

            if validate:
                _assert_graded_matrix(
                    matrix,
                    source_group.generator_q_degrees,
                    target_group.generator_q_degrees,
                    variables,
                    variable_q_degree,
                    context=f"Ext-free horizontal Ext {ext_degree}, degree {rouquier_degree}",
                )
            maps[(ext_degree, rouquier_degree)] = matrix.applyfunc(sp.expand)
    return maps


def _induced_extfree_map(
    source_ext: ExtFreeModuleData,
    target_ext: ExtFreeModuleData,
    source_koszul: FreeRKoszulComplex,
    target_koszul: FreeRKoszulComplex,
    term_map: sp.Matrix,
    ext_degree: int,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_monomials: int,
) -> sp.Matrix:
    matrix = sp.zeros(target_ext.rank, source_ext.rank)
    chain_map = _koszul_chain_map(source_koszul, target_koszul, ext_degree, term_map)
    previous = (
        target_koszul.differentials[ext_degree - 1]
        if ext_degree > 0
        else sp.zeros(len(target_koszul.basis[ext_degree]), 0)
    )
    previous_q_degrees = target_koszul.q_degrees[ext_degree - 1] if ext_degree > 0 else []
    current_q_degrees = target_koszul.q_degrees[ext_degree]

    for source_index, (source_cycle, source_q_degree) in enumerate(
        zip(source_ext.generator_cycles, source_ext.generator_q_degrees, strict=True)
    ):
        image = _matrix_times_polynomial_vector(chain_map, source_cycle)
        image_coords = _polynomial_vector_to_slice_coords(
            image,
            current_q_degrees,
            source_q_degree,
            variables,
            variable_q_degree,
            max_monomials=max_monomials,
        )
        previous_slice = _homogeneous_slice_matrix(
            previous,
            previous_q_degrees,
            current_q_degrees,
            source_q_degree,
            variables,
            variable_q_degree,
            max_monomials=max_monomials,
        )
        boundary_columns = _fast_columnspace(previous_slice)
        target_basis_entries = _extfree_basis_entries_at_q(
            target_ext,
            source_q_degree,
            variables,
            variable_q_degree,
            max_monomials=max_monomials,
            component_q_degrees=current_q_degrees,
        )
        basis_columns = [entry[2] for entry in target_basis_entries]
        boundary_basis = _independent_columns(boundary_columns, len(image_coords))
        combined_columns = boundary_basis + basis_columns
        independent = _independent_columns(combined_columns, len(image_coords))
        if len(independent) != len(combined_columns):
            raise ValueError("target Ext basis is not independent modulo boundaries")
        combined = _matrix_from_columns(combined_columns, len(image_coords))
        if combined.cols == 0:
            if any(entry != 0 for entry in image_coords):
                raise ValueError("image class is not in the target Ext span")
            continue
        solution = _fast_solve(combined, image_coords)

        basis_offset = len(boundary_basis)
        for basis_index, (target_index, exponent, _basis_column) in enumerate(target_basis_entries):
            solution_column = basis_offset + basis_index
            coefficient = solution[solution_column, 0]
            if coefficient:
                matrix[target_index, source_index] += coefficient * _monomial_expr(
                    exponent,
                    variables,
                )
    return matrix.applyfunc(sp.expand)


def _extfree_basis_entries_at_q(
    ext_module: ExtFreeModuleData,
    q_degree: int,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_monomials: int,
    component_q_degrees: Sequence[int],
) -> list[tuple[int, Exponent, sp.Matrix]]:
    entries: list[tuple[int, Exponent, sp.Matrix]] = []
    for generator_index, (cycle, generator_q_degree) in enumerate(
        zip(
            ext_module.generator_cycles,
            ext_module.generator_q_degrees,
            strict=True,
        )
    ):
        for exponent in _monomials_for_q_difference(
            q_degree - generator_q_degree,
            len(variables),
            variable_q_degree,
        ):
            coords = _polynomial_vector_to_slice_coords(
                _multiply_polynomial_vector_by_monomial(cycle, exponent, variables),
                component_q_degrees,
                q_degree,
                variables,
                variable_q_degree,
                max_monomials=max_monomials,
            )
            entries.append((generator_index, exponent, coords))
    return entries


def _extfree_horizontal_homology(
    rouquier_free: RouquierFreeLeftRComplex,
    ext_chain_groups: dict[tuple[int, int], ExtFreeChainGroup],
    horizontal_maps: dict[tuple[int, int], sp.Matrix],
    ring: Any,
    shifts: ShiftConvention,
) -> dict[tuple[int, int], RModuleHomology]:
    homology: dict[tuple[int, int], RModuleHomology] = {}
    rank = rouquier_free.rouquier.realization.dim
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            group = ext_chain_groups[(ext_degree, rouquier_degree)]
            previous = horizontal_maps.get(
                (ext_degree, rouquier_degree - 1),
                sp.zeros(group.rank, 0),
            )
            next_map = horizontal_maps.get(
                (ext_degree, rouquier_degree),
                sp.zeros(0, group.rank),
            )
            homology[(ext_degree, rouquier_degree)] = _free_module_homology(
                previous,
                next_map,
                ring,
                group.generator_q_degrees,
                rouquier_free.r_variables,
                shifts.variable_q_degree,
                degree=rouquier_degree,
            )
    return homology


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


def _polynomial_vector_to_slice_coords(
    vector: sp.Matrix,
    component_q_degrees: Sequence[int],
    q_degree: int,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    max_monomials: int,
) -> sp.Matrix:
    basis, index = _slice_basis(
        component_q_degrees,
        q_degree,
        len(variables),
        variable_q_degree,
        max_monomials=max_monomials,
    )
    coords = sp.zeros(len(basis), 1)
    for component, polynomial in enumerate(vector):
        for exponent, coefficient in _polynomial_terms(polynomial, variables):
            row = index.get((component, exponent))
            if row is None:
                if coefficient:
                    raise ValueError(
                        f"polynomial vector has term outside Q-degree {q_degree}: "
                        f"component={component}, exponent={exponent}"
                    )
                continue
            coords[row, 0] += coefficient
    return coords


def _slice_coords_to_polynomial_vector(
    coords: sp.Matrix,
    slice_basis: Sequence[tuple[int, Exponent]],
    component_count: int,
    variables: Sequence[sp.Symbol],
) -> sp.Matrix:
    vector = sp.zeros(component_count, 1)
    for row, (component, exponent) in enumerate(slice_basis):
        coefficient = coords[row, 0]
        if coefficient:
            vector[component, 0] += coefficient * _monomial_expr(exponent, variables)
    return vector.applyfunc(sp.expand)


def _matrix_times_polynomial_vector(matrix: sp.Matrix, vector: sp.Matrix) -> sp.Matrix:
    result = sp.zeros(matrix.rows, 1)
    for row in range(matrix.rows):
        result[row, 0] = sp.expand(
            sum(matrix[row, column] * vector[column, 0] for column in range(matrix.cols))
        )
    return result


def _multiply_polynomial_vector_by_monomial(
    vector: sp.Matrix,
    exponent: Exponent,
    variables: Sequence[sp.Symbol],
) -> sp.Matrix:
    monomial = _monomial_expr(exponent, variables)
    return vector.applyfunc(lambda entry: sp.expand(entry * monomial))


def _monomial_expr(exponent: Exponent, variables: Sequence[sp.Symbol]) -> sp.Expr:
    result = sp.Integer(1)
    for variable, power in zip(variables, exponent, strict=True):
        result *= variable**power
    return result


def _monomials_for_q_difference(
    q_difference: int,
    variable_count: int,
    variable_q_degree: int,
) -> tuple[Exponent, ...]:
    if q_difference < 0 or q_difference % variable_q_degree:
        return tuple()
    return _monomials_of_degree(variable_count, q_difference // variable_q_degree)


def _independent_columns(columns: Sequence[sp.Matrix], row_count: int) -> list[sp.Matrix]:
    if not columns:
        return []
    return _fast_columnspace(_matrix_from_columns(columns, row_count))
