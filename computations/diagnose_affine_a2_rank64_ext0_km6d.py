"""Diagnose Ext^0 generators for the rank-64 affine A2 Bott-Samelson term.

This version uses the 6D universal Kac-Moody realization.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from computations.diagnose_affine_a2_rank64_ext0_2d import (
    Logger,
    _homogeneous_kernel_basis,
    _lift_with_constant_part,
    _nonzero_entries,
    _write_homogeneous_kernel,
    _write_kernel_vectors,
    _write_matrix,
)
from computations.khovanov_rozansky import DEFAULT_SHIFTS, DynkinDiagram, _graded_homology_basis
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.run_affine_a2_kac_moody_6d_euler_trace_extfree import (
    kac_moody_affine_a2_universal_realization,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import free_r_koszul_complex


DEFAULT_OUTPUT = Path(
    "computations/euler_trace_outputs/affine_a2_rank64_ext0_km6d_diagnostic.txt"
)


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
    logger.log("Realization: 6D universal Kac-Moody affine A2 realization")
    logger.log("Basis: alpha_0, alpha_1, alpha_2, Lambda_0, Lambda_1, Lambda_2")
    logger.log("Word: s0 s1 s2 s0 s1 s2")
    logger.log("")

    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    braid = ((0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+"))

    section_start = time.perf_counter()
    realization = kac_moody_affine_a2_universal_realization(diagram)
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
    logger.log(f"d^0 transpose shape: {koszul.differentials[0].T.shape}")
    logger.log("")

    _write_basis(logger, model, koszul)
    _write_matrix(logger, "d0_internal_384x64", koszul.differentials[0])
    _write_matrix(logger, "d0_transpose_64x384", koszul.differentials[0].T)

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
        context="rank-64 KM6D Ext^0 after k tensor",
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
    q8_basis, _q8_unknowns = _homogeneous_kernel_basis(
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


if __name__ == "__main__":
    raise SystemExit(main())
