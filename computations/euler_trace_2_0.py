"""Euler-trace computation with minimization before the Groebner step.

This module computes the same termwise Euler trace as
``slower_old_euler_trace.euler_trace``:

    sum_a A^a sum_j (-1)^j Hilb_Q Ext^a_{R-R}(R, C^j).

For each Bott--Samelson term ``B`` the Ext groups are computed as the homology
of the free-left-``R`` Koszul complex ``C^* = B tensor Koszul``.  Before asking
SymPy's AGCA machinery for a Groebner-basis Hilbert series, the complex is
shrunk in two linear-algebra passes:

* over ``k = R/(x_0, ..., x_n)``, split each ``k tensor C^i`` as
  ``B^{i,j} + H^{i,j} + S^{i,j}``, with ``d:S^{i,j}->B^{i+1,j}`` an
  isomorphism, then lift and cancel these acyclic free summands over ``R``;
* on the resulting minimal complex, try to split certified free homology
  summands by solving for an inclusion ``J`` and projection ``P`` with
  ``dJ = 0``, ``Pd = 0``, and ``PJ = 1``.  This implementation only removes
  summands after the equations are verified over ``R``; anything not certified
  remains for the final Groebner calculation.

The final Groebner step is therefore still honest, but it should often see
smaller matrices than the direct ``euler_trace.py`` computation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import sympy as sp

from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Q,
    Realization,
    ShiftConvention,
    _fast_nullspace,
    _fast_rref_pivots,
    _independent_extension,
    _matrix_from_columns,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRKoszulComplex,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.slower_old_euler_trace.euler_trace import (
    free_module_hilbert_series,
    koszul_ext_hilbert_series_by_degree,
)
from computations.slower_old_euler_trace.euler_trace_bounded import _polynomial_terms


Exponent: TypeAlias = tuple[int, ...]


@dataclass
class ReducedFreeRComplex:
    """A free ``R`` cochain complex represented by homogeneous matrices."""

    q_degrees: dict[int, list[int]]
    differentials: dict[int, sp.Matrix]


@dataclass
class FieldSplittingDegree:
    """The ``B + H + S`` splitting of one specialized cochain group."""

    homology_lifts: list[sp.Matrix]
    homology_q_degrees: list[int]
    source_lifts: list[sp.Matrix]
    source_q_degrees: list[int]


@dataclass
class MinimalComplexData:
    """The minimal complex obtained by lifting and cancelling ``S -> B``."""

    complex: ReducedFreeRComplex
    field_splitting: dict[int, FieldSplittingDegree]
    cancelled_pairs_by_degree: dict[int, int]
    basis_changes: dict[int, sp.Matrix]


@dataclass
class FreeSplitData:
    """Certified free homology summands split off of a complex."""

    complex: ReducedFreeRComplex
    free_q_degrees: dict[int, list[int]]


@dataclass
class EulerTrace20TermData:
    """All termwise data used by the 2.0 Euler-trace computation."""

    koszul: FreeRKoszulComplex
    minimal_complex: ReducedFreeRComplex
    remaining_complex: ReducedFreeRComplex
    cancelled_pairs_by_degree: dict[int, int]
    split_free_q_degrees: dict[int, list[int]]
    remaining_hilbert_series: dict[int, sp.Expr]
    hilbert_series: dict[int, sp.Expr]


@dataclass
class EulerTrace20Result:
    """Euler trace after minimization, certified splitting, and Groebner."""

    polynomial: sp.Expr
    term_data: dict[tuple[int, ...], EulerTrace20TermData]
    ring: Any


def khovanov_rozansky_euler_trace_2_0(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    split_free: bool = True,
    validate: bool = True,
) -> EulerTrace20Result:
    """Compute the Euler trace via field minimization plus Groebner fallback."""

    if shifts.variable_q_degree <= 0:
        raise ValueError("Euler trace 2.0 requires a positive variable Q-degree")

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], EulerTrace20TermData] = {}
    trace = sp.Integer(0)
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        minimal = minimal_koszul_complex_from_field_splitting(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"Euler trace 2.0 term {choices}",
        )
        split = (
            split_certified_free_summands(
                minimal.complex,
                rouquier_free.r_variables,
                validate=validate,
                context=f"Euler trace 2.0 term {choices}",
            )
            if split_free
            else FreeSplitData(minimal.complex, {})
        )
        remaining_series = koszul_ext_hilbert_series_by_degree(
            split.complex,
            ring,
            rouquier_free.r_variables,
            shifts.variable_q_degree,
        )
        free_series = _free_split_hilbert_series(
            split.free_q_degrees,
            rouquier_free.r_variables,
            shifts.variable_q_degree,
        )
        hilbert_series = {
            degree: sp.cancel(
                remaining_series.get(degree, sp.Integer(0))
                + free_series.get(degree, sp.Integer(0))
            )
            for degree in sorted(set(remaining_series) | set(free_series))
        }

        sign = -1 if model.term.degree % 2 else 1
        for degree, series in hilbert_series.items():
            trace += sign * A**degree * series
        term_data[choices] = EulerTrace20TermData(
            koszul=koszul,
            minimal_complex=minimal.complex,
            remaining_complex=split.complex,
            cancelled_pairs_by_degree=minimal.cancelled_pairs_by_degree,
            split_free_q_degrees=split.free_q_degrees,
            remaining_hilbert_series=remaining_series,
            hilbert_series=hilbert_series,
        )

    return EulerTrace20Result(
        polynomial=sp.factor(sp.cancel(trace)),
        term_data=term_data,
        ring=ring,
    )


def compute_euler_trace_2_0(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    split_free: bool = True,
    validate: bool = True,
) -> EulerTrace20Result:
    """Short alias for ``khovanov_rozansky_euler_trace_2_0``."""

    return khovanov_rozansky_euler_trace_2_0(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
        split_free=split_free,
        validate=validate,
    )


def minimal_koszul_complex_from_field_splitting(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
    *,
    validate: bool = True,
    context: str = "minimal Koszul complex",
) -> MinimalComplexData:
    """Lift the ``k tensor C = B + H + S`` splitting and cancel ``S -> B``."""

    field_splitting = koszul_field_splitting(
        koszul,
        variables,
        validate=validate,
        context=context,
    )
    return minimal_koszul_complex_from_splitting_data(
        koszul,
        variables,
        field_splitting,
        validate=validate,
        context=context,
    )


def koszul_field_splitting(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
    *,
    validate: bool = True,
    context: str = "Koszul field splitting",
) -> dict[int, FieldSplittingDegree]:
    """Compute the ``B + H + S`` splitting of ``k tensor C``."""

    specialized = _left_specialized_differentials(koszul, variables)
    return _field_splitting(koszul, specialized, validate=validate, context=context)


def minimal_koszul_complex_from_splitting_data(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
    field_splitting: dict[int, FieldSplittingDegree],
    *,
    validate: bool = True,
    context: str = "minimal Koszul complex",
) -> MinimalComplexData:
    """Lift an already-computed field splitting and cancel ``S -> B`` over ``R``."""

    degrees = sorted(koszul.q_degrees)

    basis_changes: dict[int, sp.Matrix] = {}
    inverse_changes: dict[int, sp.Matrix] = {}
    block_sizes: dict[int, tuple[int, int, int]] = {}
    cancelled_pairs_by_degree: dict[int, int] = {}

    for degree in degrees:
        previous_sources = (
            field_splitting[degree - 1].source_lifts
            if degree - 1 in field_splitting
            else []
        )
        boundary_lifts = [
            _matrix_times_vector(koszul.differentials[degree - 1], source)
            for source in previous_sources
        ]
        splitting = field_splitting[degree]
        columns = boundary_lifts + splitting.homology_lifts + splitting.source_lifts
        row_count = len(koszul.q_degrees[degree])
        change = _matrix_from_columns(columns, row_count)
        if change.shape != (row_count, row_count):
            raise ValueError(
                f"{context}, degree {degree}: B+H+S has shape {change.shape}, "
                f"expected {(row_count, row_count)}"
            )
        basis_changes[degree] = change
        inverse_changes[degree] = _polynomial_inverse(change, variables, f"{context}, degree {degree}")
        block_sizes[degree] = (
            len(boundary_lifts),
            len(splitting.homology_lifts),
            len(splitting.source_lifts),
        )
        cancelled_pairs_by_degree[degree] = len(splitting.source_lifts)

    minimal_q_degrees = {
        degree: list(field_splitting[degree].homology_q_degrees)
        for degree in degrees
    }
    minimal_differentials: dict[int, sp.Matrix] = {}
    for degree in degrees:
        source_homology_count = block_sizes[degree][1]
        source_homology_start = block_sizes[degree][0]
        source_homology_end = source_homology_start + source_homology_count
        if degree + 1 not in koszul.q_degrees:
            minimal_differentials[degree] = sp.zeros(0, source_homology_count)
            continue

        target_homology_count = block_sizes[degree + 1][1]
        target_homology_start = block_sizes[degree + 1][0]
        target_homology_end = target_homology_start + target_homology_count
        transformed = inverse_changes[degree + 1] * koszul.differentials[degree] * basis_changes[degree]
        transformed = transformed.applyfunc(
            lambda entry: _as_polynomial(entry, variables, f"{context}, transformed d_{degree}")
        )
        if validate:
            _validate_cancelled_block(
                transformed,
                block_sizes[degree],
                block_sizes[degree + 1],
                context=f"{context}, differential {degree}",
            )
        minimal_differentials[degree] = transformed.extract(
            range(target_homology_start, target_homology_end),
            range(source_homology_start, source_homology_end),
        )

    minimal = ReducedFreeRComplex(minimal_q_degrees, minimal_differentials)
    if validate:
        _validate_complex_shapes(minimal, context=context)
        _validate_compositions(minimal, context=context)
    return MinimalComplexData(
        complex=minimal,
        field_splitting=field_splitting,
        cancelled_pairs_by_degree=cancelled_pairs_by_degree,
        basis_changes=basis_changes,
    )


def split_certified_free_summands(
    complex_: ReducedFreeRComplex,
    variables: Sequence[sp.Symbol],
    *,
    validate: bool = True,
    context: str = "free summand splitting",
) -> FreeSplitData:
    """Split free homology summands certified by exact ``J`` and ``P`` data.

    The search is intentionally conservative.  It looks for homogeneous
    generators with a nonzero constant/native-degree part, solves the exact
    polynomial equations ``dJ = 0`` and ``Pd = 0`` using coefficient linear
    algebra over ``QQ``, verifies ``PJ = 1``, then changes basis and deletes the
    resulting zero row and zero column.  Uncertified classes are left in the
    returned complex for the Groebner-basis fallback.
    """

    reduced = _copy_complex(complex_)
    free_q_degrees: dict[int, list[int]] = defaultdict(list)

    changed = True
    while changed:
        changed = False
        for degree in sorted(reduced.q_degrees):
            q_degrees = reduced.q_degrees[degree]
            for q_degree in sorted(set(q_degrees)):
                indices = [index for index, item in enumerate(q_degrees) if item == q_degree]
                split = _find_native_free_split(
                    reduced,
                    degree,
                    indices,
                    variables,
                )
                if split is None:
                    continue
                _apply_native_split(reduced, degree, indices, split, variables, context=context)
                free_q_degrees[degree].append(q_degree)
                changed = True
                break
            if changed:
                break

    result = FreeSplitData(
        complex=reduced,
        free_q_degrees={degree: sorted(values) for degree, values in free_q_degrees.items()},
    )
    if validate:
        _validate_complex_shapes(result.complex, context=context)
        _validate_compositions(result.complex, context=context)
    return result


def _field_splitting(
    koszul: FreeRKoszulComplex,
    specialized: dict[int, sp.Matrix],
    *,
    validate: bool,
    context: str,
) -> dict[int, FieldSplittingDegree]:
    splitting: dict[int, FieldSplittingDegree] = {}
    degrees = sorted(koszul.q_degrees)
    for degree in degrees:
        previous = (
            specialized[degree - 1]
            if degree > min(degrees)
            else sp.zeros(len(koszul.q_degrees[degree]), 0)
        )
        next_map = specialized[degree]
        previous_degrees = koszul.q_degrees[degree - 1] if degree > min(degrees) else []
        current_degrees = koszul.q_degrees[degree]
        next_degrees = koszul.q_degrees.get(degree + 1, [])
        if validate:
            _validate_specialized_differentials(
                previous,
                next_map,
                previous_degrees,
                current_degrees,
                next_degrees,
                context=f"{context}, field degree {degree}",
            )

        homology_lifts: list[sp.Matrix] = []
        homology_q_degrees: list[int] = []
        source_lifts: list[sp.Matrix] = []
        source_q_degrees: list[int] = []

        current_by_q = _indices_by_q_degree(current_degrees)
        previous_by_q = _indices_by_q_degree(previous_degrees)
        next_by_q = _indices_by_q_degree(next_degrees)

        for q_degree in sorted(current_by_q):
            current_indices = current_by_q[q_degree]
            previous_indices = previous_by_q.get(q_degree, [])
            next_indices = next_by_q.get(q_degree, [])
            previous_block = _extract_matrix(previous, current_indices, previous_indices)
            next_block = _extract_matrix(next_map, next_indices, current_indices)

            boundary_columns = [previous_block[:, column] for column in _fast_rref_pivots(previous_block)]
            kernel_columns = _fast_nullspace(next_block)
            boundary_span = _matrix_from_columns(boundary_columns, len(current_indices))
            homology_columns = _independent_extension(boundary_span, kernel_columns)
            source_pivots = _fast_rref_pivots(next_block)
            source_columns = [_standard_basis_column(len(current_indices), pivot) for pivot in source_pivots]

            if validate:
                combined = _matrix_from_columns(
                    boundary_columns + homology_columns + source_columns,
                    len(current_indices),
                )
                if combined.rank() != len(current_indices):
                    raise ValueError(
                        f"{context}, field degree {degree}, Q={q_degree}: "
                        "B+H+S does not span"
                    )

            homology_lifts.extend(
                _lift_slice_vector(column, current_indices, len(current_degrees))
                for column in homology_columns
            )
            homology_q_degrees.extend(q_degree for _ in homology_columns)
            source_lifts.extend(
                _lift_slice_vector(column, current_indices, len(current_degrees))
                for column in source_columns
            )
            source_q_degrees.extend(q_degree for _ in source_columns)

        splitting[degree] = FieldSplittingDegree(
            homology_lifts=homology_lifts,
            homology_q_degrees=homology_q_degrees,
            source_lifts=source_lifts,
            source_q_degrees=source_q_degrees,
        )
    return splitting


def _find_native_free_split(
    complex_: ReducedFreeRComplex,
    degree: int,
    indices: Sequence[int],
    variables: Sequence[sp.Symbol],
) -> tuple[sp.Matrix, sp.Matrix] | None:
    if not indices:
        return None
    previous = complex_.differentials.get(
        degree - 1,
        sp.zeros(len(complex_.q_degrees[degree]), 0),
    )
    next_map = complex_.differentials.get(
        degree,
        sp.zeros(0, len(complex_.q_degrees[degree])),
    )
    next_equations = _column_combination_equations(next_map, indices, variables)
    previous_equations = _row_combination_equations(previous, indices, variables)
    cycle_candidates = _fast_nullspace(next_equations)
    projection_candidates = _fast_nullspace(previous_equations)
    for cycle in cycle_candidates:
        for projection in projection_candidates:
            pairing = (projection.T * cycle)[0, 0]
            if pairing == 0:
                continue
            return cycle.applyfunc(lambda entry: sp.cancel(entry / pairing)), projection
    return None


def _apply_native_split(
    complex_: ReducedFreeRComplex,
    degree: int,
    indices: Sequence[int],
    split: tuple[sp.Matrix, sp.Matrix],
    variables: Sequence[sp.Symbol],
    *,
    context: str,
) -> None:
    cycle, projection = split
    local_change = _native_change_matrix(cycle, projection)
    local_inverse = _polynomial_inverse(local_change, variables, f"{context}, split degree {degree}")
    size = len(complex_.q_degrees[degree])
    change = _embed_square_block(size, indices, local_change)
    inverse = _embed_square_block(size, indices, local_inverse)

    if degree - 1 in complex_.differentials:
        complex_.differentials[degree - 1] = (
            inverse * complex_.differentials[degree - 1]
        ).applyfunc(lambda entry: _as_polynomial(entry, variables, context))
    if degree in complex_.differentials:
        complex_.differentials[degree] = (
            complex_.differentials[degree] * change
        ).applyfunc(lambda entry: _as_polynomial(entry, variables, context))

    split_index = indices[0]
    previous = complex_.differentials.get(degree - 1)
    next_map = complex_.differentials.get(degree)
    if previous is not None and any(previous[split_index, column] != 0 for column in range(previous.cols)):
        raise ValueError(f"{context}: proposed projection does not annihilate previous image")
    if next_map is not None and any(next_map[row, split_index] != 0 for row in range(next_map.rows)):
        raise ValueError(f"{context}: proposed inclusion is not a cycle")

    complex_.q_degrees[degree].pop(split_index)
    if previous is not None:
        complex_.differentials[degree - 1] = _delete_row(previous, split_index)
    if next_map is not None:
        complex_.differentials[degree] = _delete_column(next_map, split_index)


def _native_change_matrix(cycle: sp.Matrix, projection: sp.Matrix) -> sp.Matrix:
    if cycle.rows != projection.rows:
        raise ValueError("cycle and projection dimensions do not match")
    pairing = (projection.T * cycle)[0, 0]
    if pairing != 1:
        raise ValueError("cycle and projection must pair to 1")
    complement = _fast_nullspace(projection.T)
    change = _matrix_from_columns([cycle, *complement], cycle.rows)
    if change.cols != cycle.rows or change.rank() != cycle.rows:
        raise ValueError("could not extend split generator to a basis")
    return change


def _free_split_hilbert_series(
    free_q_degrees: dict[int, Sequence[int]],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> dict[int, sp.Expr]:
    return {
        degree: free_module_hilbert_series(q_degrees, variables, variable_q_degree)
        for degree, q_degrees in sorted(free_q_degrees.items())
        if q_degrees
    }


def _left_specialized_differentials(
    koszul: FreeRKoszulComplex,
    variables: Sequence[sp.Symbol],
) -> dict[int, sp.Matrix]:
    zero_substitution = {variable: sp.Integer(0) for variable in variables}
    return {
        degree: matrix.applyfunc(
            lambda entry: sp.expand(sp.sympify(entry).xreplace(zero_substitution))
        )
        for degree, matrix in koszul.differentials.items()
    }


def _validate_specialized_differentials(
    previous: sp.Matrix,
    next_map: sp.Matrix,
    previous_degrees: Sequence[int],
    current_degrees: Sequence[int],
    next_degrees: Sequence[int],
    *,
    context: str,
) -> None:
    if previous.rows != len(current_degrees) or next_map.cols != len(current_degrees):
        raise ValueError(f"{context}: specialized matrices do not match current degree")
    if previous.cols != len(previous_degrees) or next_map.rows != len(next_degrees):
        raise ValueError(f"{context}: specialized matrices do not match adjacent degrees")
    if previous.cols and next_map.rows:
        composition = next_map * previous
        if any(entry != 0 for entry in composition):
            raise ValueError(f"{context}: differentials do not compose after specialization")


def _validate_cancelled_block(
    transformed: sp.Matrix,
    source_sizes: tuple[int, int, int],
    target_sizes: tuple[int, int, int],
    *,
    context: str,
) -> None:
    source_boundary, source_homology, source_source = source_sizes
    target_boundary, _target_homology, _target_source = target_sizes
    if source_source != target_boundary:
        raise ValueError(f"{context}: source and boundary cancellation sizes differ")
    if source_source == 0:
        return
    source_start = source_boundary + source_homology
    target_rows = range(target_boundary)
    source_columns = range(source_start, source_start + source_source)
    block = transformed.extract(target_rows, source_columns)
    if block != sp.eye(source_source):
        raise ValueError(f"{context}: lifted S->B block is not the identity")
    lower_rows = range(target_boundary, transformed.rows)
    if lower_rows and any(
        transformed[row, column] != 0
        for row in lower_rows
        for column in source_columns
    ):
        raise ValueError(f"{context}: lifted S columns have uncancelled lower entries")


def _validate_complex_shapes(complex_: ReducedFreeRComplex, *, context: str) -> None:
    for degree, matrix in complex_.differentials.items():
        source_count = len(complex_.q_degrees.get(degree, []))
        target_count = len(complex_.q_degrees.get(degree + 1, []))
        if matrix.shape != (target_count, source_count):
            raise ValueError(
                f"{context}: d_{degree} has shape {matrix.shape}, "
                f"expected {(target_count, source_count)}"
            )


def _validate_compositions(complex_: ReducedFreeRComplex, *, context: str) -> None:
    for degree in sorted(complex_.differentials):
        if degree + 1 not in complex_.differentials:
            continue
        left = complex_.differentials[degree + 1]
        right = complex_.differentials[degree]
        if left.cols != right.rows:
            continue
        composition = (left * right).applyfunc(sp.expand)
        if any(entry != 0 for entry in composition):
            raise ValueError(f"{context}: reduced differentials do not compose at {degree}")


def _column_combination_equations(
    matrix: sp.Matrix,
    columns: Sequence[int],
    variables: Sequence[sp.Symbol],
) -> sp.Matrix:
    equations: dict[tuple[int, Exponent], list[sp.Expr]] = {}
    for local_column, column in enumerate(columns):
        for row in range(matrix.rows):
            for exponent, coefficient in _polynomial_terms(matrix[row, column], variables):
                equations.setdefault(
                    (row, exponent),
                    [sp.Integer(0) for _ in columns],
                )[local_column] += coefficient
    return _equation_matrix(equations.values(), len(columns))


def _row_combination_equations(
    matrix: sp.Matrix,
    rows: Sequence[int],
    variables: Sequence[sp.Symbol],
) -> sp.Matrix:
    equations: dict[tuple[int, Exponent], list[sp.Expr]] = {}
    for local_row, row in enumerate(rows):
        for column in range(matrix.cols):
            for exponent, coefficient in _polynomial_terms(matrix[row, column], variables):
                equations.setdefault(
                    (column, exponent),
                    [sp.Integer(0) for _ in rows],
                )[local_row] += coefficient
    return _equation_matrix(equations.values(), len(rows))


def _equation_matrix(rows: Iterable[Sequence[sp.Expr]], column_count: int) -> sp.Matrix:
    row_list = [list(row) for row in rows if any(entry != 0 for entry in row)]
    if not row_list:
        return sp.zeros(0, column_count)
    return sp.Matrix(row_list)


def _polynomial_inverse(matrix: sp.Matrix, variables: Sequence[sp.Symbol], context: str) -> sp.Matrix:
    if matrix.rows != matrix.cols:
        raise ValueError(f"{context}: only square basis-change matrices can be inverted")
    inverse = matrix.inv()
    return inverse.applyfunc(lambda entry: _as_polynomial(entry, variables, context))


def _as_polynomial(expr: sp.Expr, variables: Sequence[sp.Symbol], context: str) -> sp.Expr:
    simplified = sp.cancel(expr)
    numerator, denominator = sp.fraction(simplified)
    if any(variable in denominator.free_symbols for variable in variables):
        raise ValueError(f"{context}: non-polynomial entry {expr!r}")
    return sp.expand(numerator / denominator)


def _indices_by_q_degree(q_degrees: Sequence[int]) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = defaultdict(list)
    for index, q_degree in enumerate(q_degrees):
        indices[q_degree].append(index)
    return dict(indices)


def _extract_matrix(matrix: sp.Matrix, rows: Sequence[int], columns: Sequence[int]) -> sp.Matrix:
    if not rows or not columns:
        return sp.zeros(len(rows), len(columns))
    return matrix.extract(rows, columns)


def _standard_basis_column(row_count: int, index: int) -> sp.Matrix:
    column = sp.zeros(row_count, 1)
    column[index, 0] = 1
    return column


def _lift_slice_vector(
    vector: sp.Matrix,
    indices: Sequence[int],
    ambient_dimension: int,
) -> sp.Matrix:
    lift = sp.zeros(ambient_dimension, 1)
    for local_row, ambient_row in enumerate(indices):
        lift[ambient_row, 0] = vector[local_row, 0]
    return lift


def _matrix_times_vector(matrix: sp.Matrix, vector: sp.Matrix) -> sp.Matrix:
    return (matrix * vector).applyfunc(sp.expand)


def _embed_square_block(
    size: int,
    indices: Sequence[int],
    block: sp.Matrix,
) -> sp.Matrix:
    if block.shape != (len(indices), len(indices)):
        raise ValueError("embedded block shape does not match index set")
    result = sp.eye(size)
    for local_row, row in enumerate(indices):
        for local_column, column in enumerate(indices):
            result[row, column] = block[local_row, local_column]
    return result


def _delete_row(matrix: sp.Matrix, row: int) -> sp.Matrix:
    rows = [index for index in range(matrix.rows) if index != row]
    if not rows:
        return sp.zeros(0, matrix.cols)
    return matrix.extract(rows, range(matrix.cols))


def _delete_column(matrix: sp.Matrix, column: int) -> sp.Matrix:
    columns = [index for index in range(matrix.cols) if index != column]
    if not columns:
        return sp.zeros(matrix.rows, 0)
    return matrix.extract(range(matrix.rows), columns)


def _copy_complex(complex_: ReducedFreeRComplex) -> ReducedFreeRComplex:
    return ReducedFreeRComplex(
        q_degrees={degree: list(q_degrees) for degree, q_degrees in complex_.q_degrees.items()},
        differentials={degree: sp.Matrix(matrix) for degree, matrix in complex_.differentials.items()},
    )
