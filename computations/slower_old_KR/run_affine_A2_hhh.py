#!/usr/bin/env python3
"""Compute unreduced HHH (free-R module form) for affine A^2 braid."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import sympy as sp

from computations.khovanov_rozansky import DynkinDiagram, Q, Realization
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    FreeRExtTermData,
    RModuleHomology,
    _ext_chain_groups,
    _free_module_homology,
    _horizontal_ext_maps,
    _image_of_submodule,
    _module_vector_q_degree,
    _polynomial_ring,
    _preimage_submodule,
    _union_submodules,
    free_r_koszul_complex,
)
from computations.light_leaves import rouquier_complex_as_free_left_r_modules

OUT_PATH = Path(__file__).with_name("hhh_unreduced_affine_A2_braid.txt")


def _horizontal_homology_with_progress(rouquier_free, ext_chain_groups, horizontal_maps, ring):
    homology: dict[tuple[int, int], RModuleHomology] = {}
    rank = rouquier_free.rouquier.realization.dim
    shifts = rouquier_free.rouquier.shifts
    variables = rouquier_free.r_variables

    for ext_degree in range(rank + 1):
        for rouquier_degree in rouquier_free.rouquier.degrees:
            t0 = time.time()
            group = ext_chain_groups[(ext_degree, rouquier_degree)]
            next_matrix = horizontal_maps.get((ext_degree, rouquier_degree))
            next_group = ext_chain_groups.get((ext_degree, rouquier_degree + 1))
            if next_matrix is None or next_group is None:
                kernel = group.kernel
            else:
                kernel = _preimage_submodule(
                    next_matrix,
                    group.kernel,
                    next_group.image,
                    ring,
                    variables,
                )

            previous_matrix = horizontal_maps.get((ext_degree, rouquier_degree - 1))
            previous_group = ext_chain_groups.get((ext_degree, rouquier_degree - 1))
            previous_image = (
                _image_of_submodule(
                    previous_matrix,
                    previous_group.kernel,
                    ring,
                    variables,
                )
                if previous_matrix is not None and previous_group is not None
                else kernel.submodule()
            )
            image = _union_submodules(group.image, previous_image)
            module = kernel.quotient_module(image)
            homology[(ext_degree, rouquier_degree)] = RModuleHomology(
                degree=rouquier_degree,
                ambient_q_degrees=group.ambient_q_degrees,
                kernel=kernel,
                image=image,
                module=module,
                kernel_generator_q_degrees=[
                    _module_vector_q_degree(
                        generator,
                        group.ambient_q_degrees,
                        variables,
                        shifts.variable_q_degree,
                    )
                    for generator in kernel.gens
                ],
            )
            elapsed = time.time() - t0
            label = "zero" if homology[(ext_degree, rouquier_degree)].is_zero else str(module)
            print(
                f"  HHH^({ext_degree},{rouquier_degree}): {elapsed:.1f}s -> {label[:100]}",
                flush=True,
            )
    return homology


def main() -> None:
    diagram = DynkinDiagram.from_data([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    braid = [(0, "+"), (1, "+"), (2, "+"), (0, "+"), (1, "+"), (2, "+")]

    V = Realization.from_matrices(
        diagram,
        {
            0: sp.Matrix([[-1, 0], [-1, 1]]),
            1: sp.Matrix([[0, 1], [1, 0]]),
            2: sp.Matrix([[1, -1], [0, -1]]),
        },
    )

    total_t0 = time.time()

    print("Building Rouquier complex as free left R-modules...", flush=True)
    t0 = time.time()
    rouquier_free = rouquier_complex_as_free_left_r_modules(diagram, braid, realization=V)
    ring = _polynomial_ring(rouquier_free.r_variables)
    print(
        f"  {len(rouquier_free.models)} terms in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("Computing termwise Koszul Ext...", flush=True)
    t0 = time.time()
    term_data: dict[tuple[int, ...], FreeRExtTermData] = {}
    for choices, model in sorted(
        rouquier_free.models.items(),
        key=lambda item: item[1].term.term_id,
    ):
        koszul = free_r_koszul_complex(model)
        ext = {
            degree: _free_module_homology(
                koszul.differentials[degree - 1]
                if degree > 0
                else sp.zeros(len(koszul.basis[degree]), 0),
                koszul.differentials[degree],
                ring,
                koszul.q_degrees[degree],
                rouquier_free.r_variables,
                rouquier_free.rouquier.shifts.variable_q_degree,
                degree=degree,
            )
            for degree in range(V.dim + 1)
        }
        term_data[choices] = FreeRExtTermData(choices, model, koszul, ext)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print("Assembling Ext chain groups...", flush=True)
    t0 = time.time()
    ext_chain_groups = _ext_chain_groups(rouquier_free, term_data, ring)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print("Building horizontal Ext maps...", flush=True)
    t0 = time.time()
    horizontal_maps = _horizontal_ext_maps(rouquier_free, term_data, ext_chain_groups)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print("Computing horizontal homology over R...", flush=True)
    t0 = time.time()
    horizontal_homology = _horizontal_homology_with_progress(
        rouquier_free,
        ext_chain_groups,
        horizontal_maps,
        ring,
    )
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - total_t0
    r_vars = rouquier_free.r_variables
    hilbert_factor = sp.Integer(1) / (1 - Q**2) ** V.dim

    lines = [
        "Unreduced HHH for affine A^2 (triangle diagram 01, 12, 20)",
        "Computed via khovanov_rozansky_free_r.py (free left R-module model)",
        "",
        "Representation V: dim 2",
        "  s0 acts on V by [[-1, 0], [-1, 1]]",
        "  s1 acts on V by [[ 0, 1], [ 1, 0]]",
        "  s2 acts on V by [[ 1,-1], [ 0,-1]]",
        "",
        "Braid: (0,+), (1,+), (2,+), (0,+), (1,+), (2,+)",
        "",
        f"Left polynomial ring R = QQ[{', '.join(str(v) for v in r_vars)}]",
        f"Hilb(R) = {hilbert_factor}",
        f"Rouquier degrees: {rouquier_free.rouquier.degrees}",
        f"Computation time: {elapsed:.1f} seconds",
        "",
        "Horizontal homology HHH^{a,r} as R-modules (a = Ext degree, r = Rouquier degree):",
        "",
    ]

    for (ext_degree, rouquier_degree), homology in sorted(horizontal_homology.items()):
        if homology.is_zero:
            lines.append(f"  HHH^{{{ext_degree},{rouquier_degree}}} = 0")
        else:
            lines.append(f"  HHH^{{{ext_degree},{rouquier_degree}}} = {homology.module}")

    lines.extend(["", "Nonzero summary:"])
    for (ext_degree, rouquier_degree), homology in sorted(horizontal_homology.items()):
        if not homology.is_zero:
            lines.append(
                f"  (a,r)=({ext_degree},{rouquier_degree}): "
                f"{len(homology.module.gens)} generator(s)"
            )

    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(f"Saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
