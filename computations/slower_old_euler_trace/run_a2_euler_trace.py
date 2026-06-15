"""Run the type A2 Euler-trace computation with timing logs.

This script computes the Euler trace for the braid

    (1,+), (2,+), (1,+), (2,+)

in the finite type A2 Coxeter graph with edge 12, using the standard
reflection representation.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import sympy as sp

from computations.slower_old_euler_trace.euler_trace import (
    EulerTraceTermData,
    koszul_ext_hilbert_series_by_degree,
)
from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    DynkinDiagram,
    Realization,
)
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules


DEFAULT_OUTPUT = Path("computations/euler_trace_outputs/a2_1212_euler_trace_output.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="text file receiving progress logs and the final Euler trace",
    )
    parser.add_argument(
        "--caffeinate",
        action="store_true",
        help="on macOS, re-run under caffeinate so the system stays awake",
    )
    args = parser.parse_args()
    if args.caffeinate and not os.environ.get("A2_EULER_TRACE_CAFFEINATED"):
        env = os.environ.copy()
        env["A2_EULER_TRACE_CAFFEINATED"] = "1"
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            sys.executable,
            "-m",
            "computations.slower_old_euler_trace.run_a2_euler_trace",
            "--output",
            str(args.output),
        ]
        raise SystemExit(subprocess.run(command, env=env).returncode)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        logger = Logger(handle)
        logger.log("Type A2 Euler-trace computation")
        logger.log(f"Python: {sys.version.split()[0]}")
        logger.log(f"Platform: {platform.platform()}")
        logger.log(f"Output file: {args.output.resolve()}")
        logger.log("")

        section_start = time.perf_counter()
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        braid = ((1, "+"), (2, "+"), (1, "+"), (2, "+"))
        realization = Realization.standard(diagram)
        logger.log_timing("setup diagram and standard realization", section_start)
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
        logger.log_timing("build Rouquier complex as free left R-modules", section_start)
        logger.log(f"R variables: {rouquier_free.r_variables}")
        logger.log(f"Rouquier degrees: {rouquier_free.rouquier.degrees}")
        logger.log(f"number of Rouquier terms: {len(rouquier_free.rouquier.terms)}")
        logger.log(f"number of Rouquier arrows: {len(rouquier_free.rouquier.arrows)}")
        logger.log(f"d^2=0 on free-left-R Rouquier complex: {rouquier_free.check_d_squared()}")
        logger.log("")

        section_start = time.perf_counter()
        ring = _polynomial_ring(rouquier_free.r_variables)
        logger.log_timing("construct left polynomial ring", section_start)
        logger.log(f"ring: {ring}")
        logger.log("")

        total_trace = sp.Integer(0)
        term_data: dict[tuple[int, ...], EulerTraceTermData] = {}
        hilbert_total_time = 0.0

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
            koszul_elapsed = time.perf_counter() - koszul_start
            logger.log(f"  Koszul complex: {koszul_elapsed:.6f}s")

            sign = -1 if model.term.degree % 2 else 1
            hilbert_start = time.perf_counter()
            hilbert_series = koszul_ext_hilbert_series_by_degree(
                koszul,
                ring,
                rouquier_free.r_variables,
                DEFAULT_SHIFTS.variable_q_degree,
            )
            hilbert_elapsed = time.perf_counter() - hilbert_start
            hilbert_total_time += hilbert_elapsed
            logger.log(f"  Ext Hilbert series: {hilbert_elapsed:.6f}s")
            for ext_degree, series in hilbert_series.items():
                total_trace += sign * A**ext_degree * series
                logger.log(f"  Ext^{ext_degree}: Hilb_Q={series}")

            term_data[choices] = EulerTraceTermData(hilbert_series=hilbert_series)
            logger.log_timing(f"term {choices} total", term_start)
            logger.log("")

        section_start = time.perf_counter()
        simplified = sp.factor(sp.cancel(total_trace))
        logger.log_timing("simplify final Euler trace", section_start)
        logger.log("")
        logger.log("Final Euler trace:")
        logger.log(str(simplified))
        logger.log("")
        logger.log("Timing summary:")
        logger.log(f"  Ext Hilbert-series total: {hilbert_total_time:.6f}s")
        logger.log(f"  Overall wall time: {time.perf_counter() - started:.6f}s")
        logger.log("")
        logger.log("Reminder: this script can be wrapped in macOS caffeinate to prevent system sleep.")
        logger.log("Example:")
        logger.log(
            "  python3 -m computations.slower_old_euler_trace.run_a2_euler_trace --caffeinate "
            f"--output {args.output}"
        )


class Logger:
    def __init__(self, handle):
        self.handle = handle

    def log(self, message: str) -> None:
        print(message)
        self.handle.write(message + "\n")
        self.handle.flush()

    def log_timing(self, label: str, start: float) -> None:
        self.log(f"{label}: {time.perf_counter() - start:.6f}s")


if __name__ == "__main__":
    main()
