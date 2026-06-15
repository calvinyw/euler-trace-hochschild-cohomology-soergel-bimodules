"""Write the image of the Q^8 special-fiber Ext^0 vector in the KM6D complex."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from computations.diagnose_affine_a2_rank64_ext0_2d import Logger, _nonzero_entries
from computations.khovanov_rozansky import DynkinDiagram
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.run_affine_a2_kac_moody_6d_euler_trace_extfree import (
    kac_moody_affine_a2_universal_realization,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import free_r_koszul_complex


DEFAULT_OUTPUT = Path(
    "computations/euler_trace_outputs/affine_a2_rank64_q8_image_km6d.txt"
)


Q8_VECTOR_ENTRIES = {
    15: sp.Integer(4),
    23: sp.Integer(-6),
    29: sp.Integer(2),
    39: sp.Integer(3),
    43: sp.Integer(-4),
    45: sp.Integer(-3),
    57: sp.Integer(1),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="file receiving the image vector diagnostic",
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
    logger.log("Affine A2 KM6D rank-64 Q^8 vector image diagnostic")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log("Word: s0 s1 s2 s0 s1 s2")
    logger.log("Vector v:")
    for index, coefficient in Q8_VECTOR_ENTRIES.items():
        logger.log(f"  e_{index:02d}: {coefficient}")
    logger.log("")

    section_start = time.perf_counter()
    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    braid = ((0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+"))
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
    logger.log(f"d^0 internal shape: {koszul.differentials[0].shape}")
    logger.log("")

    source = sp.zeros(model.rank, 1)
    for index, coefficient in Q8_VECTOR_ENTRIES.items():
        source[index, 0] = coefficient

    image = (koszul.differentials[0] * source).applyfunc(sp.expand)
    nonzero = list(_nonzero_entries(image))
    zero_subs = {variable: sp.Integer(0) for variable in rouquier_free.r_variables}
    specialized_image = image.applyfunc(lambda entry: sp.expand(entry.xreplace(zero_subs)))

    logger.log(f"Image shape: {image.shape}")
    logger.log(f"Nonzero entries: {len(nonzero)}")
    logger.log(f"Specialized at x=0 is zero? {specialized_image == sp.zeros(image.rows, 1)}")
    logger.log("")

    logger.log("Nonzero entries of d^0(v) in R^384:")
    for row, coefficient in nonzero:
        basis_entry = koszul.basis[1][row]
        subexpression = "".join(map(str, model.basis_subexpressions[basis_entry.local_index]))
        logger.log(
            f"  row {row:03d}, wedge={basis_entry.wedge}, "
            f"local={basis_entry.local_index:02d} {subexpression}, "
            f"q={basis_entry.q_degree}: {sp.factor(coefficient)}"
        )
    logger.log("")

    logger.file_only("Dense image vector in R^384:")
    logger.file_only(str(image))
    logger.file_only("")

    logger.log(f"Overall wall time: {time.perf_counter() - started:.6f}s")


if __name__ == "__main__":
    raise SystemExit(main())
