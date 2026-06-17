"""Diagnose Ext^0 generators for the rank-64 affine A2 Bott-Samelson term."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from computations.khovanov_rozansky import (
    DEFAULT_SHIFTS,
    DynkinDiagram,
    Realization,
    _graded_homology_basis,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.slower_old_KR.khovanov_rozansky_free_r import free_r_koszul_complex


DEFAULT_OUTPUT = Path(
    "computations/euler_trace_outputs/affine_a2_rank64_ext0_2d_diagnostic.txt"
)


@dataclass(frozen=True)
class HomogeneousUnknown:
    source_index: int
    monomial: tuple[int, int]
    symbol: sp.Symbol


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def close(self) -> None:
        self.handle.close()

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def file_only(self, message: str = "") -> None:
        self.handle.write(message + "\n")
        self.handle.flush()

    def timing(self, label: str, start: float) -> None:
        self.log(f"{label}: {time.perf_counter() - start:.6f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="file receiving the full matrix and kernel diagnostic",
    )
    args = parser.parse_args()

    logger = Logger(args.output)
    started = time.perf_counter()
    try:
        _run(logger, started)
    finally:
        logger.close()
    return 0


def _run(logger: Logger, started: float) -> None:
    logger.log("Affine A2 rank-64 Bott-Samelson Ext^0 diagnostic")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log("Realization: original 2D affine A2 representation")
    logger.log("Word: s0 s1 s2 s0 s1 s2")
    logger.log("")

    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    braid = ((0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+"))
    matrices_on_v = {
        0: sp.Matrix([[-1, 0], [-1, 1]]),
        1: sp.Matrix([[0, 1], [1, 0]]),
        2: sp.Matrix([[1, -1], [0, -1]]),
    }

    section_start = time.perf_counter()
    realization = Realization.from_matrices(
        diagram,
        matrices_on_v,
        dual=False,
        validate=True,
    )
    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        realization=realization,
    )
    model = rouquier_free.models[(1, 1, 1, 1, 1, 1)]
    koszul = free_r_koszul_complex(model)
    logger.timing("build model and Koszul complex", section_start)
    logger.log(f"R variables: {rouquier_free.r_variables}")
    logger.log(f"Bott-Samelson rank: {model.rank}")
    logger.log(f"d^0 internal shape: {koszul.differentials[0].shape}")
    logger.log(f"d^0 transpose shape requested by user: {koszul.differentials[0].T.shape}")
    logger.log("")

    _write_basis(logger, model, koszul)
    _write_matrix(logger, "d0_internal_128x64", koszul.differentials[0])
    _write_matrix(logger, "d0_transpose_64x128", koszul.differentials[0].T)

    zero_subs = {variable: sp.Integer(0) for variable in rouquier_free.r_variables}
    specialized = koszul.differentials[0].applyfunc(
        lambda entry: sp.expand(entry.xreplace(zero_subs))
    )
    cohomology = _graded_homology_basis(
        sp.zeros(len(koszul.basis[0]), 0),
        specialized,
        [],
        koszul.q_degrees[0],
        koszul.q_degrees[1],
        validate=True,
        context="rank-64 2D Ext^0 after k tensor",
    )

    logger.log("k tensor_R Ext^0 generator degrees:")
    logger.log(str(cohomology.q_degrees))
    logger.log("")
    _write_kernel_vectors(logger, "k_tensor_Ext0_kernel_vectors", cohomology.representatives, model)

    q8_columns = [
        column for column, q_degree in enumerate(cohomology.q_degrees) if q_degree == 8
    ]
    logger.log(f"Number of k tensor_R Q=8 kernel vectors: {len(q8_columns)}")
    logger.log("")

    section_start = time.perf_counter()
    q8_basis, q8_unknowns = _homogeneous_kernel_basis(
        koszul.differentials[0],
        koszul.q_degrees[0],
        rouquier_free.r_variables,
        total_q_degree=8,
        variable_q_degree=DEFAULT_SHIFTS.variable_q_degree,
    )
    logger.timing("solve honest R-kernel in total Q-degree 8", section_start)
    logger.log(f"Honest homogeneous R-kernel dimension in Q=8: {len(q8_basis)}")
    logger.log("")
    _write_homogeneous_kernel(logger, "honest_R_kernel_Q8_basis", q8_basis, model)

    logger.log("Lift tests for k tensor_R Q=8 vectors:")
    for column in q8_columns:
        representative = cohomology.representatives[:, column]
        lift = _lift_with_constant_part(
            q8_basis,
            representative,
            model,
            total_q_degree=8,
        )
        logger.log(f"  k-kernel column {column}: lifts to honest R-cycle? {lift is not None}")
        if lift is not None:
            logger.file_only(f"honest_R_lift_of_k_kernel_column_{column}:")
            for index, coefficient in _nonzero_entries(lift):
                logger.file_only(
                    f"  {index:02d} {''.join(map(str, model.basis_subexpressions[index]))}: "
                    f"{sp.expand(coefficient)}"
                )
            logger.file_only("")
    logger.log("")

    if q8_columns:
        first_q8 = cohomology.representatives[:, q8_columns[0]]
        logger.log("First Q=8 k-kernel vector, by subexpression:")
        for index, coefficient in _nonzero_entries(first_q8):
            logger.log(
                f"  {index:02d} {''.join(map(str, model.basis_subexpressions[index]))}: "
                f"{coefficient}"
            )
        logger.log("")

    logger.log(f"Overall wall time: {time.perf_counter() - started:.6f}s")


def _write_basis(logger: Logger, model, koszul) -> None:
    logger.file_only("Basis of K^0, indexed by local source index:")
    for index, subexpression in enumerate(model.basis_subexpressions):
        logger.file_only(
            f"  {index:02d}: {''.join(map(str, subexpression))} "
            f"q={koszul.q_degrees[0][index]}"
        )
    logger.file_only("")


def _write_matrix(logger: Logger, label: str, matrix: sp.Matrix) -> None:
    logger.file_only(f"{label}: shape={matrix.shape}")
    logger.file_only(str(matrix))
    logger.file_only("")


def _write_kernel_vectors(logger: Logger, label: str, vectors: sp.Matrix, model) -> None:
    logger.file_only(f"{label}: shape={vectors.shape}")
    for column in range(vectors.cols):
        logger.file_only(f"vector {column}:")
        for index, coefficient in _nonzero_entries(vectors[:, column]):
            logger.file_only(
                f"  {index:02d} {''.join(map(str, model.basis_subexpressions[index]))}: "
                f"{coefficient}"
            )
    logger.file_only("")


def _write_homogeneous_kernel(logger: Logger, label: str, vectors: list[sp.Matrix], model) -> None:
    logger.file_only(label)
    for column, vector in enumerate(vectors):
        logger.file_only(f"vector {column}:")
        for index, coefficient in _nonzero_entries(vector):
            logger.file_only(
                f"  {index:02d} {''.join(map(str, model.basis_subexpressions[index]))}: "
                f"{sp.expand(coefficient)}"
            )
    logger.file_only("")


def _homogeneous_kernel_basis(
    matrix: sp.Matrix,
    source_q_degrees: list[int],
    variables: tuple[sp.Symbol, ...],
    *,
    total_q_degree: int,
    variable_q_degree: int,
) -> tuple[list[sp.Matrix], list[HomogeneousUnknown]]:
    unknowns: list[HomogeneousUnknown] = []
    source_entries: list[sp.Expr] = []

    for source_index, q_degree in enumerate(source_q_degrees):
        remaining = total_q_degree - q_degree
        if remaining < 0 or remaining % variable_q_degree:
            source_entries.append(sp.Integer(0))
            continue
        polynomial_degree = remaining // variable_q_degree
        monomials = _monomials(len(variables), polynomial_degree)
        terms = []
        for monomial in monomials:
            symbol = sp.Symbol(f"c_{source_index}_{'_'.join(map(str, monomial))}")
            unknowns.append(HomogeneousUnknown(source_index, monomial, symbol))
            terms.append(symbol * _monomial_expr(variables, monomial))
        source_entries.append(sum(terms, sp.Integer(0)))

    source_vector = sp.Matrix(source_entries)
    product = (matrix * source_vector).applyfunc(sp.expand)
    equations = []
    for entry in product:
        if entry == 0:
            continue
        poly = sp.Poly(entry, *variables, domain="EX")
        for _monomial, coefficient in poly.terms():
            equations.append(sp.expand(coefficient))

    coefficient_matrix, _rhs = sp.linear_eq_to_matrix(
        equations,
        [unknown.symbol for unknown in unknowns],
    )
    nullspace = coefficient_matrix.nullspace()

    basis = []
    for solution in nullspace:
        entries = [sp.Integer(0) for _ in source_q_degrees]
        for unknown, coefficient in zip(unknowns, solution, strict=True):
            if coefficient:
                entries[unknown.source_index] += coefficient * _monomial_expr(
                    variables,
                    unknown.monomial,
                )
        basis.append(sp.Matrix([sp.expand(entry) for entry in entries]))
    return basis, unknowns


def _lift_with_constant_part(
    kernel_basis: list[sp.Matrix],
    representative: sp.Matrix,
    model,
    *,
    total_q_degree: int,
) -> sp.Matrix | None:
    if not kernel_basis:
        return None
    scalar_symbols = [sp.Symbol(f"t_{index}") for index in range(len(kernel_basis))]
    combo = sp.zeros(representative.rows, 1)
    for scalar, vector in zip(scalar_symbols, kernel_basis, strict=True):
        combo += scalar * vector

    equations = []
    for index in range(representative.rows):
        subexpression_q_degree = model.basis_q_degrees[index]
        if subexpression_q_degree == total_q_degree:
            equations.append(sp.expand(combo[index, 0] - representative[index, 0]))
        elif subexpression_q_degree > total_q_degree:
            continue
        else:
            # Lower-degree basis elements have no constant term in total Q=8.
            equations.append(sp.expand(combo[index, 0].xreplace({variable: 0 for variable in model.r_variables})))

    coefficient_matrix, rhs = sp.linear_eq_to_matrix(equations, scalar_symbols)
    solution_set = sp.linsolve((coefficient_matrix, rhs))
    if solution_set == sp.EmptySet:
        return None
    solution = next(iter(solution_set))
    free_symbols = sorted(
        set().union(*(entry.free_symbols for entry in solution)),
        key=lambda symbol: symbol.name,
    )
    substitution = {symbol: sp.Integer(0) for symbol in free_symbols}
    specialized_solution = [sp.expand(entry.xreplace(substitution)) for entry in solution]

    lift = sp.zeros(representative.rows, 1)
    for scalar, vector in zip(specialized_solution, kernel_basis, strict=True):
        lift += scalar * vector
    return lift.applyfunc(sp.expand)


def _nonzero_entries(vector: sp.Matrix) -> list[tuple[int, sp.Expr]]:
    return [
        (index, sp.expand(vector[index, 0]))
        for index in range(vector.rows)
        if vector[index, 0] != 0
    ]


def _monomials(variable_count: int, degree: int) -> list[tuple[int, ...]]:
    if variable_count == 1:
        return [(degree,)]
    result = []
    for first in range(degree + 1):
        for rest in _monomials(variable_count - 1, degree - first):
            result.append((first, *rest))
    return result


def _monomial_expr(variables: tuple[sp.Symbol, ...], monomial: tuple[int, ...]) -> sp.Expr:
    return sp.prod(variable**exponent for variable, exponent in zip(variables, monomial, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
