"""Run Khovanov--Rozansky HHH 2.0 computations with timing and Ext storage."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sympy as sp

from computations.euler_trace_2_0 import (
    koszul_field_splitting,
    minimal_koszul_complex_from_splitting_data,
)
from computations.khovanov_rozansky import (
    DEFAULT_SHIFTS,
    DynkinDiagram,
    Realization,
)
from computations.khovanov_rozansky_2_0 import (
    KR20TermData,
    _homology_hilbert_polynomials,
    _kr20_chain_groups,
    _kr20_horizontal_maps,
    _minimal_block_sizes,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules
from computations.run_affine_a2_012_euler_trace_2_0 import _realization
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    _free_module_homology,
    _horizontal_homology,
    _module_element_to_exprs,
    _polynomial_ring,
    free_r_koszul_complex,
)


CaseName = Literal["a2_1212", "affine_a2_geometric_3d_012012"]

DEFAULT_OUTPUTS: dict[CaseName, Path] = {
    "a2_1212": Path("computations/HHH_outputs/a2_1212_hhh_2_0_output.txt"),
    "affine_a2_geometric_3d_012012": Path(
        "computations/HHH_outputs/affine_a2_geometric_3d_012012_hhh_2_0_output.txt"
    ),
}
DEFAULT_SEQUENCE_DIRS: dict[CaseName, Path] = {
    "a2_1212": Path("computations/stored_ext_sequences/a2_1212_hhh_2_0"),
    "affine_a2_geometric_3d_012012": Path(
        "computations/stored_ext_sequences/affine_a2_geometric_3d_012012_hhh_2_0"
    ),
}


@dataclass(frozen=True)
class RunCase:
    name: CaseName
    title: str
    diagram: DynkinDiagram
    braid: tuple[tuple[int, str], ...]
    realization: Realization
    realization_note: str


class TimeoutExpired(RuntimeError):
    """Raised when a long HHH computation reaches its wall-clock timeout."""


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

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


def main_for_case(case_name: CaseName) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUTS[case_name],
        help="file receiving progress logs, timings, and the final HHH polynomial",
    )
    parser.add_argument(
        "--ext-sequence-dir",
        type=Path,
        default=DEFAULT_SEQUENCE_DIRS[case_name],
        help="directory receiving Ext^i(R,C^*) sequence JSON files",
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

    env_var = f"HHH_2_0_{case_name.upper()}_CAFFEINATED"
    if args.caffeinate and not os.environ.get(env_var):
        env = os.environ.copy()
        env[env_var] = "1"
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            sys.executable,
            "-m",
            _wrapper_module_name(case_name),
            "--output",
            str(args.output),
            "--ext-sequence-dir",
            str(args.ext_sequence_dir),
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
            case=_case(case_name),
            ext_sequence_dir=args.ext_sequence_dir,
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


def _wrapper_module_name(case_name: CaseName) -> str:
    return {
        "a2_1212": "computations.run_a2_hhh_2_0",
        "affine_a2_geometric_3d_012012": (
            "computations.run_affine_a2_geometric_3d_012012_hhh_2_0"
        ),
    }[case_name]


def _case(case_name: CaseName) -> RunCase:
    if case_name == "a2_1212":
        diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
        return RunCase(
            name=case_name,
            title="Type A2 s1s2s1s2 HHH 2.0 computation",
            diagram=diagram,
            braid=((1, "+"), (2, "+"), (1, "+"), (2, "+")),
            realization=Realization.standard(diagram),
            realization_note="standard finite type A2 reflection representation",
        )
    if case_name == "affine_a2_geometric_3d_012012":
        diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
        return RunCase(
            name=case_name,
            title="Affine A2 geometric 3D s0s1s2s0s1s2 HHH 2.0 computation",
            diagram=diagram,
            braid=((0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+")),
            realization=_realization(diagram, "geometric_3d"),
            realization_note="geometric 3D basis: alpha_0, alpha_1, alpha_2",
        )
    raise ValueError(f"unknown HHH 2.0 case {case_name!r}")


def _run(
    logger: Logger,
    started: float,
    *,
    case: RunCase,
    ext_sequence_dir: Path,
    timeout_seconds: int,
    validate: bool,
    check_d2: bool,
) -> None:
    logger.log(case.title)
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"Output file: {logger.path.resolve()}")
    logger.log(f"Ext sequence directory: {ext_sequence_dir.resolve()}")
    logger.log(f"Timeout seconds: {timeout_seconds}")
    logger.log(f"Validate minimization pipeline: {validate}")
    logger.log(f"Check d^2=0 before Ext computations: {check_d2}")
    logger.log("")

    section_start = time.perf_counter()
    diagram = case.diagram
    braid = case.braid
    realization = case.realization
    logger.log_timing("setup diagram and realization", section_start)
    logger.log(f"diagram vertices: {diagram.vertices}")
    logger.log(f"diagram edges: {sorted(diagram.edges)}")
    logger.log(f"braid: {braid}")
    logger.log(f"realization: {case.realization_note}")
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

    term_data: dict[tuple[int, ...], KR20TermData] = {}
    rank = rouquier_free.rouquier.realization.dim
    totals = {
        "koszul": 0.0,
        "field": 0.0,
        "minimal": 0.0,
        "termwise_ext": 0.0,
        "chain_groups": 0.0,
        "horizontal_maps": 0.0,
        "store_ext_sequences": 0.0,
        "horizontal_homology": 0.0,
        "hilbert_polynomials": 0.0,
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
            context=f"{case.name} HHH 2.0 term {choices}",
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
            context=f"{case.name} HHH 2.0 term {choices}",
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

        ext_start = time.perf_counter()
        ext = {}
        for degree in range(rank + 1):
            degree_start = time.perf_counter()
            ext[degree] = _free_module_homology(
                minimal.complex.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(minimal.complex.q_degrees[degree]), 0),
                minimal.complex.differentials[degree],
                ring,
                minimal.complex.q_degrees[degree],
                rouquier_free.r_variables,
                DEFAULT_SHIFTS.variable_q_degree,
                degree=degree,
            )
            logger.log_elapsed(f"    termwise Ext^{degree} Groebner presentation", degree_start)
            logger.log(
                f"      Ext^{degree} kernel gens={len(ext[degree].kernel.gens)}, "
                f"image gens={len(ext[degree].image.gens)}, zero={ext[degree].module.is_zero()}"
            )
        totals["termwise_ext"] += logger.log_elapsed(
            "  step 3 termwise Ext Groebner presentations",
            ext_start,
        )

        term_data[choices] = KR20TermData(
            choices=choices,
            model=model,
            koszul=koszul,
            minimal=minimal,
            block_sizes=_minimal_block_sizes(minimal),
            ext=ext,
        )
        logger.log_timing(f"term {choices} total", term_start)
        logger.log("")

    section_start = time.perf_counter()
    ext_chain_groups = _kr20_chain_groups(rouquier_free, term_data, ring)
    totals["chain_groups"] += logger.log_elapsed(
        "step 4 assemble Ext^i(R,C^*) chain groups",
        section_start,
    )
    logger.log(f"  chain groups: {len(ext_chain_groups)}")

    section_start = time.perf_counter()
    horizontal_maps = _kr20_horizontal_maps(
        rouquier_free,
        term_data,
        ext_chain_groups,
        validate=validate,
    )
    totals["horizontal_maps"] += logger.log_elapsed(
        "step 5 induced Rouquier differential on Ext sequences",
        section_start,
    )
    logger.log(f"  horizontal maps: {len(horizontal_maps)}")

    section_start = time.perf_counter()
    written = _write_ext_sequences(
        ext_sequence_dir,
        case=case,
        variables=rouquier_free.r_variables,
        term_data=term_data,
        ext_chain_groups=ext_chain_groups,
        horizontal_maps=horizontal_maps,
    )
    totals["store_ext_sequences"] += logger.log_elapsed(
        "step 6 write stored Ext sequence files",
        section_start,
    )
    for path in written:
        logger.log(f"  wrote {path}")

    section_start = time.perf_counter()
    horizontal_homology = _horizontal_homology(
        rouquier_free,
        ext_chain_groups,
        horizontal_maps,
        ring,
        DEFAULT_SHIFTS,
    )
    totals["horizontal_homology"] += logger.log_elapsed(
        "step 7 horizontal homology by Groebner subquotients",
        section_start,
    )
    logger.log(
        "  horizontal homology nonzero keys: "
        f"{[key for key, homology in sorted(horizontal_homology.items()) if not homology.module.is_zero()]}"
    )

    section_start = time.perf_counter()
    polynomial, euler_trace = _homology_hilbert_polynomials(
        horizontal_homology,
        rouquier_free.r_variables,
        DEFAULT_SHIFTS.variable_q_degree,
    )
    totals["hilbert_polynomials"] += logger.log_elapsed(
        "step 8 final Hilbert-series polynomial extraction",
        section_start,
    )
    logger.log("")
    logger.log("Final HHH 2.0 polynomial:")
    logger.log(str(polynomial))
    logger.log("")
    logger.log("Euler trace from HHH 2.0:")
    logger.log(str(euler_trace))
    logger.log("")
    logger.log("Timing summary:")
    for key, timing_label in (
        ("koszul", "Koszul complex total"),
        ("field", "step 1 field B+H+S total"),
        ("minimal", "step 2 lifted cancellation/minimal total"),
        ("termwise_ext", "step 3 termwise Ext Groebner total"),
        ("chain_groups", "step 4 assemble Ext sequences total"),
        ("horizontal_maps", "step 5 induced differential total"),
        ("store_ext_sequences", "step 6 write Ext sequence files total"),
        ("horizontal_homology", "step 7 horizontal homology Groebner total"),
        ("hilbert_polynomials", "step 8 Hilbert polynomial total"),
    ):
        logger.log(f"  {timing_label}: {totals[key]:.6f}s")
    logger.log(f"  Overall wall time: {time.perf_counter() - started:.6f}s")
    logger.log(f"  Recorded term data entries: {len(term_data)}")


def _write_ext_sequences(
    directory: Path,
    *,
    case: RunCase,
    variables,
    term_data: dict[tuple[int, ...], KR20TermData],
    ext_chain_groups,
    horizontal_maps: dict[tuple[int, int], sp.Matrix],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rank = case.realization.dim
    variable_names = [str(variable) for variable in variables]
    manifest = {
        "case": case.name,
        "title": case.title,
        "braid": [[generator, sign] for generator, sign in case.braid],
        "realization_note": case.realization_note,
        "variables": variable_names,
        "variable_q_degree": DEFAULT_SHIFTS.variable_q_degree,
        "ext_degrees": list(range(rank + 1)),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    written = [manifest_path]
    for ext_degree in range(rank + 1):
        data = {
            **manifest,
            "ext_degree": ext_degree,
            "chain_groups": [],
            "horizontal_maps": [],
        }
        for (_degree, rouquier_degree), group in sorted(ext_chain_groups.items()):
            if _degree != ext_degree:
                continue
            group_data = {
                "rouquier_degree": rouquier_degree,
                "term_keys": [list(key) for key in group.term_keys],
                "offsets": {str(key): offset for key, offset in sorted(group.offsets.items())},
                "ambient_q_degrees": list(group.ambient_q_degrees),
                "kernel_generators": _module_generators(group.kernel.gens, variables),
                "image_generators": _module_generators(group.image.gens, variables),
                "terms": [],
            }
            for key in group.term_keys:
                ext = term_data[key].ext[ext_degree]
                group_data["terms"].append(
                    {
                        "choices": list(key),
                        "word": list(term_data[key].model.term.word),
                        "rouquier_degree": term_data[key].model.term.degree,
                        "offset": group.offsets[key],
                        "ambient_q_degrees": list(ext.ambient_q_degrees),
                        "kernel_generator_q_degrees": list(ext.kernel_generator_q_degrees),
                        "kernel_generators": _module_generators(ext.kernel.gens, variables),
                        "image_generators": _module_generators(ext.image.gens, variables),
                        "module_is_zero": ext.module.is_zero(),
                    }
                )
            data["chain_groups"].append(group_data)

        for (degree, rouquier_degree), matrix in sorted(horizontal_maps.items()):
            if degree != ext_degree:
                continue
            data["horizontal_maps"].append(
                {
                    "source_rouquier_degree": rouquier_degree,
                    "target_rouquier_degree": rouquier_degree + 1,
                    "shape": list(matrix.shape),
                    "matrix": _matrix_entries(matrix),
                }
            )

        path = directory / f"Ext_{ext_degree}.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _matrix_entries(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(sp.expand(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _module_generators(generators, variables) -> list[list[str]]:
    return [
        [str(sp.expand(entry)) for entry in _module_element_to_exprs(generator, variables)]
        for generator in generators
    ]
