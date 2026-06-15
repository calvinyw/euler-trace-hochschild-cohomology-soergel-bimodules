"""Run the type A2 Euler-trace 2.0 computation with detailed timing logs.

This script computes the Euler trace for the braid

    s1 s2 s1 s2

in finite type A2, using the standard reflection representation.  For each
Bott--Samelson summand it times the 2.0 stages:

* field splitting of ``k tensor C`` into ``B + H + S``;
* lifted cancellation to the minimal free ``R`` complex;
* certified free-homology splitting via ``J`` and ``P``;
* final Groebner-basis Hilbert-series calculation on the remaining complex.
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

from computations.euler_trace_2_0 import (
    EulerTrace20TermData,
    FreeSplitData,
    koszul_field_splitting,
    minimal_koszul_complex_from_splitting_data,
    split_certified_free_summands,
)
from computations.khovanov_rozansky import A, DEFAULT_SHIFTS, DynkinDiagram, Realization
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.slower_old_euler_trace.euler_trace import koszul_ext_hilbert_series_by_degree


DEFAULT_OUTPUT = Path("computations/euler_trace_outputs/a2_1212_euler_trace_2_0_output.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip validation checks in the minimization pipeline",
    )
    args = parser.parse_args()

    if args.caffeinate and not os.environ.get("A2_EULER_TRACE_2_0_CAFFEINATED"):
        env = os.environ.copy()
        env["A2_EULER_TRACE_2_0_CAFFEINATED"] = "1"
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            sys.executable,
            "-m",
            "computations.run_a2_euler_trace_2_0",
            "--output",
            str(args.output),
        ]
        if args.no_validate:
            command.append("--no-validate")
        raise SystemExit(subprocess.run(command, env=env).returncode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        logger = Logger(handle)
        _run(
            logger,
            started,
            output=args.output,
            validate=not args.no_validate,
        )


def _run(
    logger: "Logger",
    started: float,
    *,
    output: Path,
    validate: bool,
) -> None:
    logger.log("Type A2 Euler-trace 2.0 computation")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"Output file: {output.resolve()}")
    logger.log(f"Validate minimization pipeline: {validate}")
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
            context=f"A2 1212 Euler trace 2.0 term {choices}",
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
            context=f"A2 1212 Euler trace 2.0 term {choices}",
        )
        totals["minimal"] += logger.log_elapsed("  step 2 lifted cancellation/minimal complex", minimal_start)
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
            context=f"A2 1212 Euler trace 2.0 term {choices}",
        )
        totals["free_split"] += logger.log_elapsed("  step 3 certified free homology splitting", split_start)
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
        grobner_elapsed = logger.log_elapsed("  step 4 Groebner Hilbert series on remaining complex", grobner_start)
        totals["grobner"] += grobner_elapsed

        free_hilbert = _free_split_hilbert_series(
            split,
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
    logger.log("Final Euler trace 2.0:")
    logger.log(str(simplified))
    logger.log("")
    logger.log("Timing summary:")
    for key, label in (
        ("koszul", "Koszul complex total"),
        ("field", "step 1 field B+H+S total"),
        ("minimal", "step 2 lifted cancellation/minimal total"),
        ("free_split", "step 3 certified free split total"),
        ("grobner", "step 4 Groebner Hilbert-series total"),
    ):
        logger.log(f"  {label}: {totals[key]:.6f}s")
    logger.log(f"  Overall wall time: {time.perf_counter() - started:.6f}s")
    logger.log(f"  Recorded term data entries: {len(term_data)}")


def _free_split_hilbert_series(
    split: FreeSplitData,
    variables,
    variable_q_degree: int,
) -> dict[int, sp.Expr]:
    denominator = (1 - sp.Symbol("Q") ** variable_q_degree) ** len(variables)
    return {
        degree: sp.cancel(
            sum((sp.Symbol("Q") ** q_degree for q_degree in q_degrees), sp.Integer(0))
            / denominator
        )
        for degree, q_degrees in split.free_q_degrees.items()
        if q_degrees
    }


class Logger:
    def __init__(self, handle) -> None:
        self.handle = handle

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


if __name__ == "__main__":
    main()
