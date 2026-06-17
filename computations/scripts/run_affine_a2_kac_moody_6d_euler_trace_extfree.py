"""Run affine A2 Euler trace in the 6D universal Kac-Moody realization."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import sympy as sp

from computations.euler_trace_extfree import (
    ExtFreeEulerTraceTermData,
    extfree_generator_q_degrees_by_degree,
    extfree_hilbert_series_from_generators,
)
from computations.khovanov_rozansky import A, DEFAULT_SHIFTS, DynkinDiagram, Realization
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)


DEFAULT_OUTPUT = Path(
    "computations/euler_trace_outputs/affine_a2_kac_moody_6d_extfree_euler_trace_output.txt"
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


def kac_moody_affine_a2_universal_realization(diagram: DynkinDiagram) -> Realization:
    """The 6D universal Kac-Moody realization for affine A2 on ``h^*``.

    The coordinate basis is

        alpha_0, alpha_1, alpha_2, Lambda_0, Lambda_1, Lambda_2.

    The simple reflections act by

        s_i(v) = v - <v, alpha_i^vee> alpha_i,

    with ``<alpha_j, alpha_i^vee>`` given by the affine A2 Cartan matrix and
    ``<Lambda_j, alpha_i^vee> = delta_{ij}``.
    """

    cartan = {
        (0, 0): sp.Integer(2),
        (0, 1): sp.Integer(-1),
        (0, 2): sp.Integer(-1),
        (1, 0): sp.Integer(-1),
        (1, 1): sp.Integer(2),
        (1, 2): sp.Integer(-1),
        (2, 0): sp.Integer(-1),
        (2, 1): sp.Integer(-1),
        (2, 2): sp.Integer(2),
    }

    action_on_dual = {}
    identity = sp.eye(6)
    for vertex in diagram.vertices:
        alpha = sp.zeros(6, 1)
        alpha[vertex, 0] = sp.Integer(1)

        alpha_coroot = sp.zeros(1, 6)
        for basis_root in diagram.vertices:
            alpha_coroot[0, basis_root] = cartan[(vertex, basis_root)]
        alpha_coroot[0, 3 + vertex] = sp.Integer(1)

        action_on_dual[vertex] = identity - alpha * alpha_coroot

    return Realization.from_matrices(
        diagram,
        action_on_dual,
        dual=True,
        validate=True,
    )


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
        default=14400,
        help="wall-clock timeout for the computation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip representative validation in graded linear algebra",
    )
    parser.add_argument(
        "--skip-d2-check",
        action="store_true",
        help="skip the free-left-R d^2=0 check before Ext computations",
    )
    args = parser.parse_args()

    logger = Logger(args.output)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout_seconds)
    try:
        _run(
            logger,
            started,
            timeout_seconds=args.timeout_seconds,
            validate=not args.no_validate,
            check_d2=not args.skip_d2_check,
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


def _run(
    logger: Logger,
    started: float,
    *,
    timeout_seconds: int,
    validate: bool,
    check_d2: bool,
) -> None:
    logger.log("Affine A2 Kac-Moody 6D Euler-trace computation (Ext-free)")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log(f"Timeout seconds: {timeout_seconds}")
    logger.log(f"Validate specialized homology representatives: {validate}")
    logger.log(f"Check d^2=0 before Ext computations: {check_d2}")
    logger.log("")

    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    braid = ((0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+"))

    section_start = time.perf_counter()
    realization = kac_moody_affine_a2_universal_realization(diagram)
    logger.timing("setup universal Kac-Moody realization", section_start)
    logger.log("basis: alpha_0, alpha_1, alpha_2, Lambda_0, Lambda_1, Lambda_2")
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
        shifts=DEFAULT_SHIFTS,
        realization=realization,
    )
    logger.timing("build Rouquier complex as free left R-modules", section_start)
    logger.log(f"R variables: {rouquier_free.r_variables}")
    logger.log(f"Rouquier degrees: {rouquier_free.rouquier.degrees}")
    logger.log(f"number of Rouquier terms: {len(rouquier_free.rouquier.terms)}")
    logger.log(f"number of Rouquier arrows: {len(rouquier_free.rouquier.arrows)}")
    if check_d2:
        check_start = time.perf_counter()
        logger.log(f"d^2=0 on free-left-R Rouquier complex: {rouquier_free.check_d_squared()}")
        logger.timing("check d^2=0", check_start)
    else:
        logger.log("d^2=0 check skipped")
    logger.log("")

    section_start = time.perf_counter()
    ring = _polynomial_ring(rouquier_free.r_variables)
    logger.timing("construct left polynomial ring", section_start)
    logger.log(f"ring: {ring}")
    logger.log("")

    total_trace = sp.Integer(0)
    term_data: dict[tuple[int, ...], ExtFreeEulerTraceTermData] = {}
    generator_total_time = 0.0

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
        koszul = free_r_koszul_complex(model, shifts=DEFAULT_SHIFTS)
        logger.timing("  Koszul complex", koszul_start)

        generator_start = time.perf_counter()
        generator_degrees = extfree_generator_q_degrees_by_degree(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"6D Kac-Moody affine A2 Ext-free term {choices}",
        )
        generator_elapsed = time.perf_counter() - generator_start
        generator_total_time += generator_elapsed
        logger.log(f"  k tensor Ext generator degrees: {generator_elapsed:.6f}s")

        hilbert_series = extfree_hilbert_series_from_generators(
            generator_degrees,
            rouquier_free.r_variables,
            DEFAULT_SHIFTS.variable_q_degree,
        )
        sign = -1 if model.term.degree % 2 else 1
        for ext_degree, series in hilbert_series.items():
            total_trace += sign * A**ext_degree * series
            logger.log(f"  Ext^{ext_degree} free generator Q-degrees={generator_degrees[ext_degree]}")
            logger.log(f"  Ext^{ext_degree}: Hilb_Q={series}")

        term_data[choices] = ExtFreeEulerTraceTermData(
            free_generator_q_degrees=generator_degrees,
            hilbert_series=hilbert_series,
        )
        logger.timing(f"term {choices} total", term_start)
        logger.log("")

    section_start = time.perf_counter()
    simplified = sp.factor(sp.cancel(total_trace))
    logger.timing("simplify final Euler trace", section_start)
    logger.log("")
    logger.log("Final 6D Kac-Moody Ext-free Euler trace:")
    logger.log(str(simplified))
    logger.log("")
    logger.log("Timing summary:")
    logger.log(f"  k tensor generator-degree total: {generator_total_time:.6f}s")
    logger.log(f"  Overall wall time: {time.perf_counter() - started:.6f}s")
    logger.log(f"  Recorded term data entries: {len(term_data)}")


if __name__ == "__main__":
    raise SystemExit(main())
