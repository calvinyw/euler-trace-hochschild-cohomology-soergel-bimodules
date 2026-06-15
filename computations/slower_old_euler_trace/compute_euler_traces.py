#!/usr/bin/env python3
"""Compute Euler traces for selected type-A braids and save timed outputs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from computations.slower_old_euler_trace.euler_trace import (
    EulerTraceTermData,
    EulerTraceResult,
    koszul_ext_hilbert_series_by_degree,
)
from computations.khovanov_rozansky import A, DEFAULT_SHIFTS, DynkinDiagram
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules

OUTPUT_DIR = Path(__file__).with_name("euler_trace_outputs")


@dataclass(frozen=True)
class BraidCase:
    name: str
    filename: str
    diagram: DynkinDiagram
    braid: tuple[tuple[int, str], ...]
    braid_label: str


CASES = (
    BraidCase(
        name="A2",
        filename="A2_s1s2s1s2.txt",
        diagram=DynkinDiagram.from_data([1, 2], [(1, 2)]),
        braid=((1, "+"), (2, "+"), (1, "+"), (2, "+")),
        braid_label="s1 s2 s1 s2",
    ),
    BraidCase(
        name="A3",
        filename="A3_s1s2s3s1s2s3.txt",
        diagram=DynkinDiagram.from_data([1, 2, 3], [(1, 2), (2, 3)]),
        braid=((1, "+"), (2, "+"), (3, "+"), (1, "+"), (2, "+"), (3, "+")),
        braid_label="s1 s2 s3 s1 s2 s3",
    ),
)


@dataclass
class TimedEulerTrace:
    result: EulerTraceResult
    timings: dict[str, float]
    term_count: int


def compute_timed_euler_trace(case: BraidCase) -> TimedEulerTrace:
    """Compute the Euler trace with per-stage wall-clock timings."""

    timings: dict[str, float] = {}
    shifts = DEFAULT_SHIFTS

    t0 = time.perf_counter()
    rouquier_free = rouquier_complex_as_free_left_r_modules(
        case.diagram,
        case.braid,
        shifts=shifts,
    )
    timings["build_rouquier_complex"] = time.perf_counter() - t0

    ring = _polynomial_ring(rouquier_free.r_variables)
    term_data: dict = {}
    trace = sp.Integer(0)

    t0 = time.perf_counter()
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model, shifts=shifts)
        hilbert_series = koszul_ext_hilbert_series_by_degree(
            koszul,
            ring,
            rouquier_free.r_variables,
            shifts.variable_q_degree,
        )
        sign = -1 if model.term.degree % 2 else 1
        for degree, series in hilbert_series.items():
            trace += sign * A**degree * series

        term_data[choices] = EulerTraceTermData(hilbert_series=hilbert_series)
    timings["termwise_koszul_ext_hilbert"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    polynomial = sp.factor(sp.cancel(trace))
    result = EulerTraceResult(polynomial=polynomial, term_data=term_data, ring=ring)
    timings["assemble_euler_trace"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())

    return TimedEulerTrace(
        result=result,
        timings=timings,
        term_count=len(term_data),
    )


def _format_case_output(case: BraidCase, timed: TimedEulerTrace) -> str:
    lines = [
        f"Type {case.name} Euler trace",
        f"Braid: {case.braid_label}",
        f"Diagram vertices: {case.diagram.vertices}",
        f"Diagram edges: {sorted(tuple(sorted(edge)) for edge in case.diagram.edges)}",
        "",
        "Definition:",
        "  sum_a A^a sum_j (-1)^j Hilb_Q Ext^a_{R-R}(R, C^j)",
        "  (equivalently T = -1 in the unreduced A,Q,T Euler characteristic)",
        "",
        f"Left polynomial ring R = {timed.result.ring}",
        f"Rouquier terms: {timed.term_count}",
        "",
        "Timing (seconds):",
        f"  build Rouquier complex:        {timed.timings['build_rouquier_complex']:.3f}",
        f"  termwise Koszul/Ext/Hilb:      {timed.timings['termwise_koszul_ext_hilbert']:.3f}",
        f"  assemble Euler trace:          {timed.timings['assemble_euler_trace']:.3f}",
        f"  total:                         {timed.timings['total']:.3f}",
        "",
        "Euler trace:",
        str(timed.result.polynomial),
        "",
        "Expanded:",
        str(sp.expand(timed.result.polynomial)),
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "Euler trace batch run",
        f"Output directory: {OUTPUT_DIR.name}/",
        "",
    ]

    for case in CASES:
        print(f"Computing {case.name} ({case.braid_label})...", flush=True)
        timed = compute_timed_euler_trace(case)
        output_path = OUTPUT_DIR / case.filename
        text = _format_case_output(case, timed)
        output_path.write_text(text)
        print(
            f"  saved {output_path.name} in {timed.timings['total']:.1f}s",
            flush=True,
        )
        summary_lines.extend(
            [
                f"{case.name} ({case.braid_label}):",
                f"  file: {case.filename}",
                f"  result: {timed.result.polynomial}",
                f"  total time: {timed.timings['total']:.3f}s",
                "",
            ]
        )

    (OUTPUT_DIR / "README.txt").write_text("\n".join(summary_lines))
    print(f"Wrote summary to {OUTPUT_DIR / 'README.txt'}", flush=True)


if __name__ == "__main__":
    main()
