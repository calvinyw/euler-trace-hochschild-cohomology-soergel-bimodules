"""Parallel Euler-trace computation under the Ext-free assumption.

This module computes the same polynomial as
``computations.euler_trace_extfree``:

    sum_a A^a sum_j (-1)^j Hilb_Q Ext^a_{R-R}(R, C^j).

The computation is termwise in the Bott-Samelson summands of the Rouquier
complex.  Each worker computes

    sum_a A^a Hilb_Q Ext^a_{R-R}(R, B)

for one Bott-Samelson term ``B``; the parent process then applies the
Rouquier-degree sign and adds the contributions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Literal

import sympy as sp

from computations.euler_trace_extfree import (
    ExtFreeEulerTraceResult,
    ExtFreeEulerTraceTermData,
    extfree_generator_q_degrees_by_degree,
    extfree_hilbert_series_from_generators,
)
from computations.khovanov_rozansky import (
    A,
    DEFAULT_SHIFTS,
    BraidLetter,
    DynkinDiagram,
    Realization,
    ShiftConvention,
    rouquier_complex,
)
from computations.light_leaves import BottSamelsonFreeLeftRModel
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _polynomial_ring,
    free_r_koszul_complex,
)


ExecutorKind = Literal["process", "thread"]


@dataclass(frozen=True)
class _TermJob:
    choices: tuple[int, ...]
    model: BottSamelsonFreeLeftRModel
    variables: tuple[sp.Symbol, ...]
    shifts: ShiftConvention
    validate: bool
    context_prefix: str


@dataclass(frozen=True)
class _TermResult:
    choices: tuple[int, ...]
    term_id: int
    term_degree: int
    free_generator_q_degrees: dict[int, list[int]]
    hilbert_series: dict[int, sp.Expr]
    unsigned_trace_summand: sp.Expr


def khovanov_rozansky_parallel_euler_trace_extfree(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
    max_workers: int | None = None,
    executor: ExecutorKind = "process",
    chunksize: int = 1,
    mp_start_method: str | None = None,
) -> ExtFreeEulerTraceResult:
    """Compute the Ext-free Euler trace termwise in parallel.

    Parameters are the same as ``khovanov_rozansky_euler_trace_extfree``, with
    parallelism controls added:

    ``max_workers``
        Number of worker processes or threads.  ``None`` uses the smaller of
        the CPU count and the number of Bott-Samelson terms.  ``1`` runs the
        same term worker serially, which is useful for debugging.

    ``executor``
        ``"process"`` uses ``ProcessPoolExecutor`` and is usually better for
        CPU-heavy SymPy work.  ``"thread"`` avoids multiprocessing setup and is
        useful in restricted environments.

    ``mp_start_method``
        Optional multiprocessing start method, such as ``"spawn"`` or
        ``"fork"``.  Leave as ``None`` to use Python's platform default.
    """

    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    rouquier = rouquier_complex(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
    )
    variables = tuple(
        sp.Symbol(f"x_{position}") for position in range(rouquier.realization.dim)
    )
    ring = _polynomial_ring(variables)
    models = {
        choices: BottSamelsonFreeLeftRModel(
            rouquier.realization,
            term,
            shifts=shifts,
            r_variables=variables,
        )
        for choices, term in rouquier.terms.items()
    }

    jobs = tuple(
        _TermJob(
            choices=choices,
            model=model,
            variables=variables,
            shifts=shifts,
            validate=validate,
            context_prefix="Parallel Euler trace Ext-free",
        )
        for choices, model in sorted(
            models.items(),
            key=lambda item: item[1].term.term_id,
        )
    )
    results = _run_term_jobs(
        jobs,
        max_workers=max_workers,
        executor=executor,
        chunksize=chunksize,
        mp_start_method=mp_start_method,
    )

    trace = sum(
        (
            _rouquier_degree_sign(result.term_degree) * result.unsigned_trace_summand
            for result in results
        ),
        sp.Integer(0),
    )
    term_data = {
        result.choices: ExtFreeEulerTraceTermData(
            free_generator_q_degrees=result.free_generator_q_degrees,
            hilbert_series=result.hilbert_series,
        )
        for result in sorted(results, key=lambda result: result.term_id)
    }
    return ExtFreeEulerTraceResult(
        polynomial=sp.factor(sp.cancel(trace)),
        term_data=term_data,
        ring=ring,
    )


def compute_parallel_euler_trace_extfree(
    diagram: DynkinDiagram,
    braid: Iterable[BraidLetter],
    *,
    shifts: ShiftConvention = DEFAULT_SHIFTS,
    realization: Realization | None = None,
    validate: bool = True,
    max_workers: int | None = None,
    executor: ExecutorKind = "process",
    chunksize: int = 1,
    mp_start_method: str | None = None,
) -> ExtFreeEulerTraceResult:
    """Short alias for ``khovanov_rozansky_parallel_euler_trace_extfree``."""

    return khovanov_rozansky_parallel_euler_trace_extfree(
        diagram,
        braid,
        shifts=shifts,
        realization=realization,
        validate=validate,
        max_workers=max_workers,
        executor=executor,
        chunksize=chunksize,
        mp_start_method=mp_start_method,
    )


parallel_euler_trace_extfree = compute_parallel_euler_trace_extfree


def _run_term_jobs(
    jobs: Sequence[_TermJob],
    *,
    max_workers: int | None,
    executor: ExecutorKind,
    chunksize: int,
    mp_start_method: str | None,
) -> list[_TermResult]:
    if executor not in ("process", "thread"):
        raise ValueError("executor must be 'process' or 'thread'")

    worker_count = _worker_count(max_workers, len(jobs))
    if worker_count == 1:
        return [_compute_term(job) for job in jobs]

    if executor == "thread":
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(_compute_term, jobs))

    context = mp.get_context(mp_start_method) if mp_start_method is not None else None
    kwargs = {"max_workers": worker_count}
    if context is not None:
        kwargs["mp_context"] = context
    with ProcessPoolExecutor(**kwargs) as pool:
        return list(pool.map(_compute_term, jobs, chunksize=chunksize))


def _worker_count(max_workers: int | None, job_count: int) -> int:
    if job_count < 1:
        return 1
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        return min(max_workers, job_count)
    return min(os.cpu_count() or 1, job_count)


def _compute_term(job: _TermJob) -> _TermResult:
    koszul = free_r_koszul_complex(job.model, shifts=job.shifts)
    generator_degrees = extfree_generator_q_degrees_by_degree(
        koszul,
        job.variables,
        validate=job.validate,
        context=f"{job.context_prefix} term {job.choices}",
    )
    hilbert_series = extfree_hilbert_series_from_generators(
        generator_degrees,
        job.variables,
        job.shifts.variable_q_degree,
    )
    unsigned_summand = sum(
        (
            A**ext_degree * series
            for ext_degree, series in hilbert_series.items()
        ),
        sp.Integer(0),
    )
    return _TermResult(
        choices=job.choices,
        term_id=job.model.term.term_id,
        term_degree=job.model.term.degree,
        free_generator_q_degrees=generator_degrees,
        hilbert_series=hilbert_series,
        unsigned_trace_summand=unsigned_summand,
    )


def _rouquier_degree_sign(term_degree: int) -> int:
    return -1 if term_degree % 2 else 1


__all__ = [
    "khovanov_rozansky_parallel_euler_trace_extfree",
    "compute_parallel_euler_trace_extfree",
    "parallel_euler_trace_extfree",
]
