"""Run affine A2 ``s0 s1 s2`` Euler-trace 2.0 computations.

This module is the shared implementation for the geometric and 6D Kac--Moody
runner scripts.  It logs per Bott--Samelson timing for:

* field splitting of ``k tensor C`` into ``B + H + S``;
* lifted cancellation to the minimal free ``R`` complex;
* certified free-homology splitting via ``J`` and ``P``;
* final Groebner-basis Hilbert-series calculation on the remaining complex.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import sympy as sp

from computations.euler_trace_2_0 import (
    EulerTrace20TermData,
    _free_split_hilbert_series,
    koszul_field_splitting,
    minimal_koszul_complex_from_splitting_data,
    split_certified_free_summands,
)
from computations.khovanov_rozansky import A, DEFAULT_SHIFTS, DynkinDiagram, Realization
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.run_affine_a2_kac_moody_6d_euler_trace_extfree import (
    kac_moody_affine_a2_universal_realization,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.slower_old_euler_trace.euler_trace import koszul_ext_hilbert_series_by_degree


RealizationName = Literal["geometric", "geometric_3d", "kac_moody_6d"]

DEFAULT_OUTPUTS = {
    "geometric": Path(
        "computations/euler_trace_outputs/affine_a2_geometric_012_euler_trace_2_0_output.txt"
    ),
    "geometric_3d": Path(
        "computations/euler_trace_outputs/affine_a2_geometric_3d_012_euler_trace_2_0_output.txt"
    ),
    "kac_moody_6d": Path(
        "computations/euler_trace_outputs/affine_a2_kac_moody_6d_012_euler_trace_2_0_output.txt"
    ),
}


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

    def log_timing(self, label: str, start: float) -> None:
        self.log(f"{label}: {time.perf_counter() - start:.6f}s")

    def log_elapsed(self, label: str, start: float) -> float:
        elapsed = time.perf_counter() - start
        self.log(f"{label}: {elapsed:.6f}s")
        return elapsed


def main_with_realization(realization_name: RealizationName) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUTS[realization_name],
        help="file receiving progress logs and the final Euler trace",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=43200,
        help="wall-clock timeout for the computation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip validation checks in the minimization pipeline",
    )
    parser.add_argument(
        "--skip-d2-check",
        action="store_true",
        help="skip the free-left-R d^2=0 check before Ext computations",
    )
    parser.add_argument(
        "--caffeinate",
        action="store_true",
        help="on macOS, re-run under caffeinate so the system stays awake",
    )
    args = parser.parse_args()

    env_var = f"AFFINE_A2_012_ET20_{realization_name.upper()}_CAFFEINATED"
    if args.caffeinate and not os.environ.get(env_var):
        env = os.environ.copy()
        env[env_var] = "1"
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            sys.executable,
            "-m",
            _wrapper_module_name(realization_name),
            "--output",
            str(args.output),
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        if args.no_validate:
            command.append("--no-validate")
        if args.skip_d2_check:
            command.append("--skip-d2-check")
        return subprocess.run(command, env=env).returncode

    logger = Logger(args.output)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout_seconds)
    try:
        _run(
            logger,
            started,
            realization_name=realization_name,
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


def _timeout_handler(_signum, _frame) -> None:
    raise TimeoutExpired("wall-clock timeout reached")


def _wrapper_module_name(realization_name: RealizationName) -> str:
    return {
        "geometric": "computations.run_affine_a2_geometric_012_euler_trace_2_0",
        "geometric_3d": "computations.run_affine_a2_geometric_3d_012_euler_trace_2_0",
        "kac_moody_6d": "computations.run_affine_a2_kac_moody_6d_012_euler_trace_2_0",
    }[realization_name]


def _run(
    logger: Logger,
    started: float,
    *,
    realization_name: RealizationName,
    timeout_seconds: int,
    validate: bool,
    check_d2: bool,
) -> None:
    label = {
        "geometric": "geometric 2D",
        "geometric_3d": "geometric 3D",
        "kac_moody_6d": "6D universal Kac-Moody",
    }[realization_name]
    logger.log(f"Affine A2 {label} s0s1s2 Euler-trace 2.0 computation")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log(f"Timeout seconds: {timeout_seconds}")
    logger.log(f"Validate minimization pipeline: {validate}")
    logger.log(f"Check d^2=0 before Ext computations: {check_d2}")
    logger.log("")

    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    braid = ((0, "+"), (1, "+"), (2, "+"))

    section_start = time.perf_counter()
    realization = _realization(diagram, realization_name)
    logger.log_timing(f"setup {label} realization", section_start)
    logger.log(f"diagram vertices: {diagram.vertices}")
    logger.log(f"diagram edges: {sorted(diagram.edges)}")
    logger.log(f"braid: {braid}")
    logger.log(f"realization dimension: {realization.dim}")
    if realization_name == "geometric":
        logger.log("input matrices interpreted as V action; R uses the contragredient V^* action")
    elif realization_name == "geometric_3d":
        logger.log("basis: alpha_0, alpha_1, alpha_2")
    else:
        logger.log("basis: alpha_0, alpha_1, alpha_2, Lambda_0, Lambda_1, Lambda_2")
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
    logger.log_timing("build Rouquier complex as free left R-modules", section_start)
    logger.log(f"R variables: {rouquier_free.r_variables}")
    logger.log(f"Rouquier degrees: {rouquier_free.rouquier.degrees}")
    logger.log(f"number of Rouquier terms: {len(rouquier_free.rouquier.terms)}")
    logger.log(f"number of Rouquier arrows: {len(rouquier_free.rouquier.arrows)}")
    if check_d2:
        check_start = time.perf_counter()
        logger.log(f"d^2=0 on free-left-R Rouquier complex: {rouquier_free.check_d_squared()}")
        logger.log_timing("check d^2=0", check_start)
    else:
        logger.log("d^2=0 check skipped")
    logger.log("")

    section_start = time.perf_counter()
    ring = _polynomial_ring(rouquier_free.r_variables)
    logger.log_timing("construct left polynomial ring", section_start)
    logger.log(f"ring: {ring}")
    logger.log("")

    total_trace = sp.Integer(0)
    term_data: dict[tuple[int, ...], EulerTrace20TermData] = {}
    totals = {
        "koszul": 0.0,
        "field": 0.0,
        "minimal": 0.0,
        "free_split": 0.0,
        "grobner": 0.0,
    }

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
        totals["koszul"] += logger.log_elapsed("  Koszul complex", koszul_start)

        field_start = time.perf_counter()
        field_splitting = koszul_field_splitting(
            koszul,
            rouquier_free.r_variables,
            validate=validate,
            context=f"affine A2 {label} s0s1s2 Euler trace 2.0 term {choices}",
        )
        totals["field"] += logger.log_elapsed("  step 1 field B+H+S splitting", field_start)
        logger.log(
            "    field homology ranks by Ext degree: "
            f"{ {degree: len(data.homology_q_degrees) for degree, data in field_splitting.items()} }"
        )
        logger.log(
            "    field source ranks by Ext degree: "
            f"{ {degree: len(data.source_q_degrees) for degree, data in field_splitting.items()} }"
        )

        minimal_start = time.perf_counter()
        minimal = minimal_koszul_complex_from_splitting_data(
            koszul,
            rouquier_free.r_variables,
            field_splitting,
            validate=validate,
            context=f"affine A2 {label} s0s1s2 Euler trace 2.0 term {choices}",
        )
        totals["minimal"] += logger.log_elapsed(
            "  step 2 lifted cancellation/minimal complex",
            minimal_start,
        )
        logger.log(
            "    minimal ranks by Ext degree: "
            f"{ {degree: len(qs) for degree, qs in minimal.complex.q_degrees.items()} }"
        )
        logger.log(f"    cancelled S->B pairs by source degree: {minimal.cancelled_pairs_by_degree}")

        split_start = time.perf_counter()
        split = split_certified_free_summands(
            minimal.complex,
            rouquier_free.r_variables,
            validate=validate,
            context=f"affine A2 {label} s0s1s2 Euler trace 2.0 term {choices}",
        )
        totals["free_split"] += logger.log_elapsed(
            "  step 3 certified free homology splitting",
            split_start,
        )
        logger.log(f"    split free Q-degrees by Ext degree: {split.free_q_degrees}")
        logger.log(
            "    remaining ranks by Ext degree: "
            f"{ {degree: len(qs) for degree, qs in split.complex.q_degrees.items()} }"
        )

        grobner_start = time.perf_counter()
        remaining_hilbert = koszul_ext_hilbert_series_by_degree(
            split.complex,
            ring,
            rouquier_free.r_variables,
            DEFAULT_SHIFTS.variable_q_degree,
        )
        totals["grobner"] += logger.log_elapsed(
            "  step 4 Groebner Hilbert series on remaining complex",
            grobner_start,
        )

        free_hilbert = _free_split_hilbert_series(
            split.free_q_degrees,
            rouquier_free.r_variables,
            DEFAULT_SHIFTS.variable_q_degree,
        )
        hilbert_series = {
            degree: sp.cancel(
                remaining_hilbert.get(degree, sp.Integer(0))
                + free_hilbert.get(degree, sp.Integer(0))
            )
            for degree in sorted(set(remaining_hilbert) | set(free_hilbert))
        }
        sign = -1 if model.term.degree % 2 else 1
        for ext_degree, series in hilbert_series.items():
            total_trace += sign * A**ext_degree * series
            logger.log(f"  Ext^{ext_degree}: Hilb_Q={series}")
            logger.log(f"    remaining contribution={remaining_hilbert.get(ext_degree, sp.Integer(0))}")
            logger.log(f"    split-free contribution={free_hilbert.get(ext_degree, sp.Integer(0))}")

        term_data[choices] = EulerTrace20TermData(
            koszul=koszul,
            minimal_complex=minimal.complex,
            remaining_complex=split.complex,
            cancelled_pairs_by_degree=minimal.cancelled_pairs_by_degree,
            split_free_q_degrees=split.free_q_degrees,
            remaining_hilbert_series=remaining_hilbert,
            hilbert_series=hilbert_series,
        )
        logger.log_timing(f"term {choices} total", term_start)
        logger.log("")

    section_start = time.perf_counter()
    simplified = sp.factor(sp.cancel(total_trace))
    logger.log_timing("simplify final Euler trace", section_start)
    logger.log("")
    logger.log(f"Final affine A2 {label} s0s1s2 Euler trace 2.0:")
    logger.log(str(simplified))
    logger.log("")
    logger.log("Timing summary:")
    for key, timing_label in (
        ("koszul", "Koszul complex total"),
        ("field", "step 1 field B+H+S total"),
        ("minimal", "step 2 lifted cancellation/minimal total"),
        ("free_split", "step 3 certified free split total"),
        ("grobner", "step 4 Groebner Hilbert-series total"),
    ):
        logger.log(f"  {timing_label}: {totals[key]:.6f}s")
    logger.log(f"  Overall wall time: {time.perf_counter() - started:.6f}s")
    logger.log(f"  Recorded term data entries: {len(term_data)}")


def _realization(diagram: DynkinDiagram, realization_name: RealizationName) -> Realization:
    if realization_name == "kac_moody_6d":
        return kac_moody_affine_a2_universal_realization(diagram)
    if realization_name == "geometric_3d":
        action_on_dual = {}
        identity = sp.eye(3)
        for reflection in diagram.vertices:
            matrix = identity.copy()
            row = diagram.vertex_index(reflection)
            for column_vertex in diagram.vertices:
                column = diagram.vertex_index(column_vertex)
                kronecker = sp.Integer(1) if reflection == column_vertex else sp.Integer(0)
                matrix[row, column] = kronecker - diagram.cartan_entry(
                    reflection,
                    column_vertex,
                )
            action_on_dual[reflection] = matrix
        return Realization.from_matrices(
            diagram,
            action_on_dual,
            dual=True,
            validate=True,
        )
    matrices_on_v = {
        0: sp.Matrix([[-1, 0], [-1, 1]]),
        1: sp.Matrix([[0, 1], [1, 0]]),
        2: sp.Matrix([[1, -1], [0, -1]]),
    }
    return Realization.from_matrices(
        diagram,
        matrices_on_v,
        dual=False,
        validate=True,
    )
