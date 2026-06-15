"""Khovanov--Rozansky data over the left polynomial ring ``R``.

The existing ``khovanov_rozansky.py`` computes reduced vector spaces and
unreduced Hilbert series by cutting the left ``R``-module direction into finite
graded pieces.  This module keeps that left ``R``-module structure explicit.

The construction is:

* expand every Rouquier term in the light-leaf/free-left-``R`` basis;
* build the Koszul cochain complex computing ``Ext^a_{R-R}(R, term)`` as a
  complex of free left ``R``-modules;
* represent each termwise ``Ext^a`` as the subquotient
  ``ker(d_K^a) / im(d_K^{a-1})``;
* assemble the Rouquier differential on these subquotients and compute
  horizontal homology over ``R`` as ``ker(d_R^r) / im(d_R^{r-1})``.

The homology objects returned here are SymPy AGCA subquotient modules.  Thus
torsion such as ``R/(x_0)`` is retained instead of disappearing after tensoring
with the fraction field of ``R``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import sympy as sp

from computations.khovanov_rozansky import (
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Realization,
    ShiftConvention,
    Vertex,
)
from computations.light_leaves import (
    BottSamelsonFreeLeftRModel,
    RouquierFreeLeftRComplex,
    rouquier_complex_as_free_left_r_modules,
)


@dataclass(frozen=True)
class FreeRKoszulBasisElement:
    """One basis vector in a Koszul cochain group."""

    local_index: int
    wedge: tuple[int, ...]
    q_degree: int


@dataclass
class FreeRKoszulComplex:
    """Koszul complex of one Bott--Samelson term over the left ring."""

    model: BottSamelsonFreeLeftRModel
    wedge_bases: dict[int, tuple[tuple[int, ...], ...]]
    basis: dict[int, tuple[FreeRKoszulBasisElement, ...]]
    basis_index: dict[int, dict[tuple[int, tuple[int, ...]], int]]
    q_degrees: dict[int, list[int]]
    differentials: dict[int, sp.Matrix]


@dataclass
class RModuleHomology:
    """A homology module represented as ``kernel / image`` over ``R``."""

    degree: int
    ambient_q_degrees: list[int]
    kernel: Any
    image: Any
    module: Any
    kernel_generator_q_degrees: list[int | None]

    @property
    def is_zero(self) -> bool:
        return self.module.is_zero()


@dataclass
class FreeRExtTermData:
    """Termwise Koszul data for one Rouquier summand."""

    choices: tuple[int, ...]
    model: BottSamelsonFreeLeftRModel
    koszul: FreeRKoszulComplex
    ext: dict[int, RModuleHomology]


@dataclass
class ExtChainGroup:
    """Direct sum of termwise Ext modules in one Rouquier degree."""

    ext_degree: int
    rouquier_degree: int
    term_keys: tuple[tuple[int, ...], ...]
    offsets: dict[tuple[int, ...], int]
    ambient_q_degrees: list[int]
    kernel: Any
    image: Any
    module: Any


@dataclass
class FreeRKRResult:
    """KR computation retaining left ``R``-module presentations."""

    rouquier_free: RouquierFreeLeftRComplex
    ring: Any
    term_data: dict[tuple[int, ...], FreeRExtTermData]
    ext_chain_groups: dict[tuple[int, int], ExtChainGroup]
    horizontal_maps: dict[tuple[int, int], sp.Matrix]
    horizontal_homology: dict[tuple[int, int], RModuleHomology]


def khovanov_rozansky_free_r_homology(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
) -> FreeRKRResult:
    """Compute KR homology data as modules over the left polynomial ring.

    The returned ``horizontal_homology[(a, r)].module`` is the ``R``-module
    homology in Hochschild/Ext degree ``a`` and Rouquier degree ``r``.
    """

    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    ring = _polynomial_ring(rouquier_free.r_variables)

    term_data: dict[tuple[int, ...], FreeRExtTermData] = {}
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        ext = {
            degree: _free_module_homology(
                koszul.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(koszul.basis[degree]), 0),
                koszul.differentials[degree],
                ring,
                koszul.q_degrees[degree],
                rouquier_free.r_variables,
                shifts.variable_q_degree,
                degree=degree,
            )
            for degree in range(rouquier_free.rouquier.realization.dim + 1)
        }
        term_data[choices] = FreeRExtTermData(choices, model, koszul, ext)

    ext_chain_groups = _ext_chain_groups(rouquier_free, term_data, ring)
    horizontal_maps = _horizontal_ext_maps(rouquier_free, term_data, ext_chain_groups)
    horizontal_homology = _horizontal_homology(
        rouquier_free,
        ext_chain_groups,
        horizontal_maps,
        ring,
        shifts,
    )

    return FreeRKRResult(
        rouquier_free=rouquier_free,
        ring=ring,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
        horizontal_homology=horizontal_homology,
    )


def free_r_koszul_complex(
    model: BottSamelsonFreeLeftRModel,
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
) -> FreeRKoszulComplex:
    """Build ``Hom_{R^e}(Koszul(x_i-y_i), M)`` over the left ring."""

    rank = model.dim
    wedge_bases = {
        degree: tuple(tuple(combo) for combo in combinations(range(rank), degree))
        for degree in range(rank + 1)
    }
    basis: dict[int, tuple[FreeRKoszulBasisElement, ...]] = {}
    basis_index: dict[int, dict[tuple[int, tuple[int, ...]], int]] = {}
    q_degrees: dict[int, list[int]] = {}

    for degree in range(rank + 1):
        entries = tuple(
            FreeRKoszulBasisElement(
                local_index=local_index,
                wedge=wedge,
                q_degree=model.basis_q_degrees[local_index]
                + degree * shifts.koszul_dual_q_shift,
            )
            for wedge in wedge_bases[degree]
            for local_index in range(model.rank)
        )
        basis[degree] = entries
        basis_index[degree] = {
            (entry.local_index, entry.wedge): index
            for index, entry in enumerate(entries)
        }
        q_degrees[degree] = [entry.q_degree for entry in entries]

    diagonal_multiplication = {
        position: _multiplication_matrix(model, _diagonal_difference(model, position))
        for position in range(rank)
    }

    differentials: dict[int, sp.Matrix] = {}
    for degree in range(rank):
        source = basis[degree]
        target_index = basis_index[degree + 1]
        matrix = sp.zeros(len(basis[degree + 1]), len(source))
        for column, entry in enumerate(source):
            wedge_set = set(entry.wedge)
            for position in range(rank):
                if position in wedge_set:
                    continue
                insertion_position = sum(1 for item in entry.wedge if item < position)
                sign = -1 if insertion_position % 2 else 1
                target_wedge = tuple(sorted((*entry.wedge, position)))
                multiplication_column = diagonal_multiplication[position][:, entry.local_index]
                for target_local, coefficient in enumerate(multiplication_column):
                    if coefficient:
                        row = target_index[(target_local, target_wedge)]
                        matrix[row, column] += sign * coefficient
        differentials[degree] = matrix.applyfunc(sp.expand)
        _assert_graded_matrix(
            differentials[degree],
            q_degrees[degree],
            q_degrees[degree + 1],
            model.r_variables,
            shifts.variable_q_degree,
            context=f"Koszul term {model.term.choices}, degree {degree}",
        )
    differentials[rank] = sp.zeros(0, len(basis[rank]))

    return FreeRKoszulComplex(
        model=model,
        wedge_bases=wedge_bases,
        basis=basis,
        basis_index=basis_index,
        q_degrees=q_degrees,
        differentials=differentials,
    )


def _multiplication_matrix(
    model: BottSamelsonFreeLeftRModel,
    polynomial: sp.Expr,
) -> sp.Matrix:
    matrix = sp.zeros(model.rank, model.rank)
    for column, basis_expr in enumerate(model.basis_exprs):
        vector = model.vector(sp.expand(polynomial * basis_expr))
        for row, coefficient in enumerate(vector):
            if coefficient:
                matrix[row, column] += sp.expand(coefficient)
    return matrix


def _diagonal_difference(
    model: BottSamelsonFreeLeftRModel,
    position: int,
) -> sp.Expr:
    return model.reduce_expression(
        model.r_variables[position] - model.layer_coordinates[-1][position]
    )


def _free_module_homology(
    previous: sp.Matrix,
    next_map: sp.Matrix,
    ring: Any,
    ambient_q_degrees: list[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    degree: int,
) -> RModuleHomology:
    """Return ``ker(next_map) / im(previous)`` as an AGCA subquotient."""

    if previous.rows != next_map.cols:
        raise ValueError("homology matrices have incompatible dimensions")
    if previous.rows != len(ambient_q_degrees):
        raise ValueError("ambient q-degree data does not match matrix dimensions")
    if previous.cols and next_map.rows:
        composition = (next_map * previous).applyfunc(sp.expand)
        if any(entry != 0 for entry in composition):
            raise ValueError("differentials do not compose to zero over R")

    kernel = _kernel_submodule(next_map, ring)
    image = _image_submodule(previous, ring)
    if not kernel.is_submodule(image):
        raise ValueError(f"image is not contained in kernel in degree {degree}")
    module = kernel.quotient_module(image)
    generator_q_degrees = [
        _module_vector_q_degree(generator, ambient_q_degrees, variables, variable_q_degree)
        for generator in kernel.gens
    ]
    return RModuleHomology(
        degree=degree,
        ambient_q_degrees=list(ambient_q_degrees),
        kernel=kernel,
        image=image,
        module=module,
        kernel_generator_q_degrees=generator_q_degrees,
    )


def _ext_chain_groups(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], FreeRExtTermData],
    ring: Any,
) -> dict[tuple[int, int], ExtChainGroup]:
    groups: dict[tuple[int, int], ExtChainGroup] = {}
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
                koszul = term_data[key].koszul
                offsets[key] = offset
                ambient_q_degrees.extend(koszul.q_degrees[ext_degree])
                offset += len(koszul.basis[ext_degree])

            ambient = ring.free_module(offset)
            kernel = _embedded_direct_sum_submodule(
                ambient,
                [
                    (
                        offsets[key],
                        len(term_data[key].koszul.basis[ext_degree]),
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
                        len(term_data[key].koszul.basis[ext_degree]),
                        term_data[key].ext[ext_degree].image.gens,
                    )
                    for key in term_keys
                ],
            )
            groups[(ext_degree, rouquier_degree)] = ExtChainGroup(
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


def _horizontal_ext_maps(
    rouquier_free: RouquierFreeLeftRComplex,
    term_data: dict[tuple[int, ...], FreeRExtTermData],
    ext_chain_groups: dict[tuple[int, int], ExtChainGroup],
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
                source_koszul = term_data[arrow.source].koszul
                target_koszul = term_data[arrow.target].koszul
                term_map = rouquier_free.arrow_matrices[arrow.arrow_id]
                block = _koszul_chain_map(
                    source_koszul,
                    target_koszul,
                    ext_degree,
                    term_map,
                )
                source_offset = source_group.offsets[arrow.source]
                target_offset = target_group.offsets[arrow.target]
                for row in range(block.rows):
                    for column in range(block.cols):
                        coefficient = block[row, column]
                        if coefficient:
                            matrix[target_offset + row, source_offset + column] += coefficient

            _assert_graded_matrix(
                matrix,
                source_group.ambient_q_degrees,
                target_group.ambient_q_degrees,
                rouquier_free.r_variables,
                rouquier_free.rouquier.shifts.variable_q_degree,
                context=f"horizontal Ext {ext_degree}, degree {rouquier_degree}",
            )
            maps[(ext_degree, rouquier_degree)] = matrix.applyfunc(sp.expand)
    return maps


def _horizontal_homology(
    rouquier_free: RouquierFreeLeftRComplex,
    ext_chain_groups: dict[tuple[int, int], ExtChainGroup],
    horizontal_maps: dict[tuple[int, int], sp.Matrix],
    ring: Any,
    shifts: ShiftConvention,
) -> dict[tuple[int, int], RModuleHomology]:
    homology: dict[tuple[int, int], RModuleHomology] = {}
    rank = rouquier_free.rouquier.realization.dim
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            group = ext_chain_groups[(ext_degree, rouquier_degree)]
            next_matrix = horizontal_maps.get((ext_degree, rouquier_degree))
            next_group = ext_chain_groups.get((ext_degree, rouquier_degree + 1))
            if next_matrix is None or next_group is None:
                kernel = group.kernel
            else:
                kernel = _preimage_submodule(
                    next_matrix,
                    group.kernel,
                    next_group.image,
                    ring,
                    rouquier_free.r_variables,
                )

            previous_matrix = horizontal_maps.get((ext_degree, rouquier_degree - 1))
            previous_group = ext_chain_groups.get((ext_degree, rouquier_degree - 1))
            previous_image = (
                _image_of_submodule(
                    previous_matrix,
                    previous_group.kernel,
                    ring,
                    rouquier_free.r_variables,
                )
                if previous_matrix is not None and previous_group is not None
                else kernel.submodule()
            )
            image = _union_submodules(group.image, previous_image)
            if not kernel.is_submodule(image):
                raise ValueError(
                    f"horizontal image is not contained in kernel for "
                    f"Ext {ext_degree}, degree {rouquier_degree}"
                )
            module = kernel.quotient_module(image)
            homology[(ext_degree, rouquier_degree)] = RModuleHomology(
                degree=rouquier_degree,
                ambient_q_degrees=group.ambient_q_degrees,
                kernel=kernel,
                image=image,
                module=module,
                kernel_generator_q_degrees=[
                    _module_vector_q_degree(
                        generator,
                        group.ambient_q_degrees,
                        rouquier_free.r_variables,
                        shifts.variable_q_degree,
                    )
                    for generator in kernel.gens
                ],
            )
    return homology


def _union_submodules(*submodules: Any) -> Any:
    if not submodules:
        raise ValueError("expected at least one submodule")
    result = submodules[0]
    for submodule in submodules[1:]:
        result = result.union(submodule)
    return result


def _preimage_submodule(
    matrix: sp.Matrix,
    source_submodule: Any,
    target_submodule: Any,
    ring: Any,
    variables: Sequence[sp.Symbol],
) -> Any:
    """Return ``{v in source_submodule | matrix*v in target_submodule}``."""

    if not source_submodule.gens:
        return source_submodule
    if matrix.rows == 0:
        return source_submodule

    target_free = ring.free_module(matrix.rows)
    source_generators = list(source_submodule.gens)
    image_vectors = [
        _matrix_times_module_vector(matrix, generator, variables)
        for generator in source_generators
    ]
    target_generators = [
        [-entry for entry in _module_element_to_exprs(generator, variables)]
        for generator in target_submodule.gens
    ]
    relation_module = target_free.submodule(*(image_vectors + target_generators))
    syzygies = relation_module.syzygy_module()

    preimage_generators = []
    for syzygy in syzygies.gens:
        coefficients = _module_element_to_exprs(syzygy, variables)
        source_coefficients = coefficients[: len(source_generators)]
        preimage_generators.append(
            _linear_combination_of_module_generators(
                source_generators,
                source_coefficients,
                variables,
                matrix.cols,
            )
        )
    return _submodule_from_vectors(ring.free_module(matrix.cols), preimage_generators)


def _image_of_submodule(
    matrix: sp.Matrix,
    source_submodule: Any,
    ring: Any,
    variables: Sequence[sp.Symbol],
) -> Any:
    target_free = ring.free_module(matrix.rows)
    image_vectors = [
        _matrix_times_module_vector(matrix, generator, variables)
        for generator in source_submodule.gens
    ]
    return _submodule_from_vectors(target_free, image_vectors)


def _matrix_times_module_vector(
    matrix: sp.Matrix,
    vector: Any,
    variables: Sequence[sp.Symbol],
) -> list[sp.Expr]:
    entries = _module_element_to_exprs(vector, variables)
    return [
        sp.expand(
            sum(matrix[row, column] * entries[column] for column in range(matrix.cols))
        )
        for row in range(matrix.rows)
    ]


def _linear_combination_of_module_generators(
    generators: Sequence[Any],
    coefficients: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    length: int,
) -> list[sp.Expr]:
    result = [sp.Integer(0) for _ in range(length)]
    for coefficient, generator in zip(coefficients, generators, strict=True):
        entries = _module_element_to_exprs(generator, variables)
        for index in range(length):
            result[index] += coefficient * entries[index]
    return [sp.expand(entry) for entry in result]


def _submodule_from_vectors(ambient: Any, vectors: Sequence[Sequence[sp.Expr]]) -> Any:
    generators = [
        list(vector)
        for vector in vectors
        if any(entry != 0 for entry in vector)
    ]
    return ambient.submodule(*generators)


def _koszul_chain_map(
    source: FreeRKoszulComplex,
    target: FreeRKoszulComplex,
    degree: int,
    term_map: sp.Matrix,
) -> sp.Matrix:
    matrix = sp.zeros(len(target.basis[degree]), len(source.basis[degree]))
    target_index = target.basis_index[degree]
    for column, source_entry in enumerate(source.basis[degree]):
        image_column = term_map[:, source_entry.local_index]
        for target_local, coefficient in enumerate(image_column):
            if coefficient:
                row = target_index[(target_local, source_entry.wedge)]
                matrix[row, column] += coefficient
    return matrix.applyfunc(sp.expand)


def _zero_ext_chain_group(
    ext_degree: int,
    rouquier_degree: int,
    rouquier_free: RouquierFreeLeftRComplex,
) -> ExtChainGroup:
    ring = _polynomial_ring(rouquier_free.r_variables)
    zero = ring.free_module(0).submodule()
    return ExtChainGroup(
        ext_degree=ext_degree,
        rouquier_degree=rouquier_degree,
        term_keys=tuple(),
        offsets={},
        ambient_q_degrees=[],
        kernel=zero,
        image=zero,
        module=zero.quotient_module(zero),
    )


def _kernel_submodule(matrix: sp.Matrix, ring: Any) -> Any:
    domain = ring.free_module(matrix.cols)
    if matrix.cols == 0:
        return domain.submodule()
    if matrix.rows == 0:
        return domain.submodule(*domain.basis())
    target = ring.free_module(matrix.rows)
    image_with_all_columns = target.submodule(*_matrix_to_column_images(matrix))
    return image_with_all_columns.syzygy_module()


def _image_submodule(matrix: sp.Matrix, ring: Any) -> Any:
    target = ring.free_module(matrix.rows)
    if matrix.cols == 0:
        return target.submodule()
    generators = [
        column
        for column in _matrix_to_column_images(matrix)
        if any(entry != 0 for entry in column)
    ]
    return target.submodule(*generators)


def _embedded_direct_sum_submodule(
    ambient: Any,
    pieces: Sequence[tuple[int, int, Sequence[Any]]],
) -> Any:
    generators = []
    total_rank = ambient.rank
    for offset, width, piece_generators in pieces:
        for generator in piece_generators:
            vector = [sp.Integer(0) for _ in range(total_rank)]
            entries = _module_element_to_exprs(generator, ambient.ring.gens)
            for local_index in range(width):
                vector[offset + local_index] = entries[local_index]
            if any(entry != 0 for entry in vector):
                generators.append(vector)
    return ambient.submodule(*generators)


def _matrix_to_column_images(matrix: sp.Matrix) -> list[list[sp.Expr]]:
    return [
        [sp.expand(matrix[row, column]) for row in range(matrix.rows)]
        for column in range(matrix.cols)
    ]


def _polynomial_ring(variables: Sequence[sp.Symbol]) -> Any:
    if not variables:
        raise ValueError("the free-R computation expects a positive-rank polynomial ring")
    return sp.QQ.old_poly_ring(*variables)


def _module_element_to_exprs(element: Any, variables: Sequence[sp.Symbol]) -> list[sp.Expr]:
    return [_ring_element_to_expr(entry, variables) for entry in element]


def _ring_element_to_expr(element: Any, variables: Sequence[sp.Symbol]) -> sp.Expr:
    if hasattr(element, "to_sympy_dict"):
        result = sp.Integer(0)
        for exponents, coefficient in element.to_sympy_dict().items():
            monomial = sp.Integer(1)
            for variable, exponent in zip(variables, exponents, strict=True):
                monomial *= variable**exponent
            result += coefficient * monomial
        return sp.expand(result)
    return sp.expand(sp.sympify(element))


def _module_vector_q_degree(
    element: Any,
    ambient_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> int | None:
    degrees = set()
    for entry, basis_degree in zip(
        _module_element_to_exprs(element, variables),
        ambient_q_degrees,
        strict=True,
    ):
        polynomial_degree = _polynomial_q_degree(entry, variables, variable_q_degree)
        if polynomial_degree is not None:
            degrees.add(basis_degree + polynomial_degree)
    if not degrees:
        return None
    if len(degrees) != 1:
        raise ValueError(f"module generator is not homogeneous: {element!r}")
    return degrees.pop()


def _polynomial_q_degree(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
) -> int | None:
    polynomial = sp.expand(polynomial)
    if polynomial == 0:
        return None
    poly = sp.Poly(polynomial, *variables, domain=sp.QQ)
    degrees = {
        sum(exponents) * variable_q_degree
        for exponents, coefficient in poly.terms()
        if coefficient
    }
    if len(degrees) != 1:
        raise ValueError(f"polynomial is not homogeneous: {polynomial!r}")
    return degrees.pop()


def _assert_graded_matrix(
    matrix: sp.Matrix,
    source_q_degrees: Sequence[int],
    target_q_degrees: Sequence[int],
    variables: Sequence[sp.Symbol],
    variable_q_degree: int,
    *,
    context: str,
) -> None:
    if matrix.shape != (len(target_q_degrees), len(source_q_degrees)):
        raise ValueError(f"{context}: matrix shape does not match grading data")
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            polynomial_degree = _polynomial_q_degree(
                matrix[row, column],
                variables,
                variable_q_degree,
            )
            if polynomial_degree is None:
                continue
            if target_q_degrees[row] + polynomial_degree != source_q_degrees[column]:
                raise ValueError(
                    f"{context}: entry ({row}, {column}) has degree "
                    f"{polynomial_degree}, but shifts are "
                    f"{source_q_degrees[column]} -> {target_q_degrees[row]}"
                )
