"""Run affine A2 HHH using the termwise Ext-free shortcut.

The script tests the braid

    s0 s1 s2 s0 s1 s2

in the affine A2 Coxeter graph with the two-dimensional representation used by
the other affine A2 scripts.  It writes a progress log and whatever final or
partial data is available before the wall-clock timeout.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import sympy as sp

from computations.euler_trace_extfree import extfree_generator_q_degrees_by_degree
from computations.khovanov_rozansky import DynkinDiagram, Realization
from computations.khovanov_rozansky_extfree import (
    ExtFreeKRResult,
    ExtFreeTermData,
    _extfree_chain_groups,
    _extfree_horizontal_maps,
    _extfree_module_data,
    _homology_hilbert_polynomials,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    RModuleHomology,
    _free_module_homology,
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules


DEFAULT_OUTPUT = Path(
    "computations/euler_trace_outputs/affine_a2_extfree_hhh_output.txt"
)


class TimeoutExpired(Exception):
    """Raised when the script reaches its wall-clock budget."""


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

    def timing(self, label: str, start: float) -> None:
        self.log(f"{label}: {time.perf_counter() - start:.6f}s")


def _timeout_handler(_signum, _frame) -> None:
    raise TimeoutExpired("wall-clock timeout reached")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="file receiving progress logs and final/partial output",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="wall-clock timeout for the computation",
    )
    parser.add_argument(
        "--max-monomials",
        type=int,
        default=20000,
        help="maximum monomials in any homogeneous slice",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip consistency checks in the Ext-free computation",
    )
    args = parser.parse_args()

    logger = Logger(args.output)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout_seconds)

    try:
        _run_computation(
            logger,
            started,
            timeout_seconds=args.timeout_seconds,
            max_monomials=args.max_monomials,
            validate=not args.no_validate,
        )
        return 0
    except TimeoutExpired as exc:
        logger.log("")
        logger.log(f"TIMEOUT: {exc}")
        logger.log(f"Elapsed before timeout: {time.perf_counter() - started:.6f}s")
        return 124
    except Exception as exc:
        logger.log("")
        logger.log(f"ERROR after {time.perf_counter() - started:.6f}s: {exc!r}")
        raise
    finally:
        signal.alarm(0)
        logger.close()


def _run_computation(
    logger: Logger,
    started: float,
    *,
    timeout_seconds: int,
    max_monomials: int,
    validate: bool,
) -> None:
    logger.log("Affine A2 Ext-free HHH computation")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log(f"Timeout seconds: {timeout_seconds}")
    logger.log(f"Validate: {validate}")
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
    logger.timing("setup diagram and realization", section_start)
    logger.log(f"diagram vertices: {diagram.vertices}")
    logger.log(f"diagram edges: {sorted(diagram.edges)}")
    logger.log(f"braid: {braid}")
    logger.log(f"realization dimension: {realization.dim}")
    for generator in diagram.vertices:
        logger.log(f"s{generator} action on V^* used by R:")
        logger.log(str(realization.action[generator]))
        logger.log(f"alpha_{generator}: {list(realization.root(generator))}")
    logger.log("")

    section_start = time.perf_counter()
    rouquier_free = rouquier_complex_as_free_left_r_modules(
        diagram,
        braid,
        realization=realization,
    )
    logger.timing("build Rouquier complex as free left R-modules", section_start)
    logger.log(f"R variables: {rouquier_free.r_variables}")
    logger.log(f"Rouquier degrees: {rouquier_free.rouquier.degrees}")
    logger.log(f"number of Rouquier terms: {len(rouquier_free.rouquier.terms)}")
    logger.log(f"number of Rouquier arrows: {len(rouquier_free.rouquier.arrows)}")
    logger.log(f"d^2=0 on free-left-R Rouquier complex: {rouquier_free.check_d_squared()}")
    logger.log("")

    ring = _polynomial_ring(rouquier_free.r_variables)
    term_data: dict[tuple[int, ...], ExtFreeTermData] = {}
    rank = rouquier_free.rouquier.realization.dim

    logger.log("Computing termwise Ext-free data...")
    section_start = time.perf_counter()
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        term_start = time.perf_counter()
        logger.log(
            f"term {choices}: degree={model.term.degree}, "
            f"word={model.term.word}, rank={model.rank}"
        )

        koszul_start = time.perf_counter()
        koszul = free_r_koszul_complex(model)
        logger.timing("  Koszul complex", koszul_start)

        gen_start = time.perf_counter()
        generator_degrees = extfree_generator_q_degrees_by_degree(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"affine A2 Ext-free HHH term {choices}",
        )
        logger.timing("  k tensor_R Ext generator degrees", gen_start)

        ext = {}
        for ext_degree in range(rank + 1):
            ext_start = time.perf_counter()
            ext_module = _extfree_module_data(
                koszul,
                ext_degree,
                generator_degrees[ext_degree],
                rouquier_free.r_variables,
                rouquier_free.rouquier.shifts.variable_q_degree,
                max_monomials=max_monomials,
                validate=validate,
                context=f"affine A2 term {choices}, Ext degree {ext_degree}",
            )
            ext[ext_degree] = ext_module
            logger.log(
                f"  Ext^{ext_degree}: rank={ext_module.rank}, "
                f"generator_q_degrees={ext_module.generator_q_degrees}, "
                f"lift_time={time.perf_counter() - ext_start:.6f}s"
            )

        term_data[choices] = ExtFreeTermData(
            choices=choices,
            model=model,
            koszul=koszul,
            ext=ext,
        )
        logger.timing(f"term {choices} total", term_start)
        logger.log("")
    logger.timing("termwise Ext-free data total", section_start)
    logger.log("")

    logger.log("Assembling Ext chain groups...")
    section_start = time.perf_counter()
    ext_chain_groups = _extfree_chain_groups(rouquier_free, term_data)
    logger.timing("Ext chain groups", section_start)
    for key, group in sorted(ext_chain_groups.items()):
        logger.log(
            f"  group {key}: rank={group.rank}, "
            f"generator_q_degrees={group.generator_q_degrees}"
        )
    logger.log("")

    logger.log("Building horizontal Ext maps over R...")
    section_start = time.perf_counter()
    horizontal_maps = _extfree_horizontal_maps(
        rouquier_free,
        term_data,
        ext_chain_groups,
        max_monomials=max_monomials,
        validate=validate,
    )
    logger.timing("horizontal Ext maps", section_start)
    for key, matrix in sorted(horizontal_maps.items()):
        nonzero = sum(1 for entry in matrix if entry)
        logger.log(f"  map {key}: shape={matrix.shape}, nonzero_entries={nonzero}")
    logger.log("")

    logger.log("Computing final horizontal homology over R...")
    section_start = time.perf_counter()
    horizontal_homology: dict[tuple[int, int], RModuleHomology] = {}
    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            hom_start = time.perf_counter()
            group = ext_chain_groups[(ext_degree, rouquier_degree)]
            previous = horizontal_maps.get(
                (ext_degree, rouquier_degree - 1),
                sp.zeros(group.rank, 0),
            )
            next_map = horizontal_maps.get(
                (ext_degree, rouquier_degree),
                sp.zeros(0, group.rank),
            )
            homology = _free_module_homology(
                previous,
                next_map,
                ring,
                group.generator_q_degrees,
                rouquier_free.r_variables,
                rouquier_free.rouquier.shifts.variable_q_degree,
                degree=rouquier_degree,
            )
            horizontal_homology[(ext_degree, rouquier_degree)] = homology
            label = "0" if homology.is_zero else str(homology.module)
            logger.log(
                f"  HHH^({ext_degree},{rouquier_degree}): "
                f"{time.perf_counter() - hom_start:.6f}s -> {label[:240]}"
            )
    logger.timing("horizontal homology total", section_start)
    logger.log("")

    logger.log("Assembling Hilbert series...")
    section_start = time.perf_counter()
    polynomial, euler_trace = _homology_hilbert_polynomials(
        horizontal_homology,
        rouquier_free.r_variables,
        rouquier_free.rouquier.shifts.variable_q_degree,
    )
    logger.timing("assemble Hilbert series", section_start)
    logger.log("")
    logger.log("Final HHH polynomial:")
    logger.log(str(polynomial))
    logger.log("")
    logger.log("Horizontal Euler trace:")
    logger.log(str(euler_trace))
    logger.log("")
    logger.log("Nonzero horizontal homology summary:")
    for (ext_degree, rouquier_degree), homology in sorted(horizontal_homology.items()):
        if not homology.is_zero:
            logger.log(
                f"  (a,r)=({ext_degree},{rouquier_degree}): "
                f"{len(homology.module.gens)} generator(s), module={homology.module}"
            )
    logger.log("")
    logger.log(f"Overall wall time: {time.perf_counter() - started:.6f}s")

    # Keep a structured result available for debugging from runpy/import users.
    _ = ExtFreeKRResult(
        polynomial=polynomial,
        euler_trace_polynomial=euler_trace,
        rouquier_free=rouquier_free,
        ring=ring,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
        horizontal_homology=horizontal_homology,
    )


if __name__ == "__main__":
    raise SystemExit(main())
