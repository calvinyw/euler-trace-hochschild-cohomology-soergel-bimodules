# Khovanov-Rozansky computations

This folder contains a small pure-Python/SymPy implementation of the Soergel
bimodule construction of Khovanov-Rozansky cohomology for small examples.

## Current state of the art

The state-of-the-art files in this folder are:

* `euler_trace_extfree.py` for Euler-trace computations.
* `khovanov_rozansky_extfree.py` for Khovanov-Rozansky homology computations.

The Ext-free computations are conditional: they assume the termwise Ext
modules are free over the left polynomial ring.  In type A this is true but in affine A2 it seems this is not.  The Ext-free computations will not work correctly if the representation does not satisfy those assumptions.

* `euler_trace_2_0.py` for Euler-trace computations.
* `khovanov_rozansky_2_0.py` for Khovanov-Rozansky homology computations.

The 2.0 computations are the honest general Groebner-basis implementations.
They do not assume termwise Ext-freeness.  Before the Groebner step, each
termwise Koszul complex is shrunk by field splitting over
`k = R/(x_0, ..., x_n)`: acyclic free summands are cancelled over `R`, and
certified free homology summands are removed when a unit-pivot splitting is
found.  The final Groebner calculation is unchanged in principle, but it
operates on smaller matrices than the direct implementations in
`slower_old_euler_trace/euler_trace.py` and
`slower_old_KR/khovanov_rozansky_free_r.py`.  Use these when the Ext-free
shortcut is unavailable or untrusted, for example in affine type.

The slower/older implementations have been moved out of the main computation
surface:

* `slower_old_euler_trace/` contains the non-Ext-free and bounded Euler-trace
  implementations, along with their older run scripts and tests.
* `slower_old_KR/` contains the non-Ext-free KR implementations, the free-`R`
  model, the reduced-old implementation, the bounded implementation, and the
  Ext-free bounded implementation.

For historical comparison, the original direct KR functions are now in
`slower_old_KR/khovanov_rozansky.py`:

* `rouquier_complex(diagram, braid)` builds the tensor product Rouquier complex.
* `koszul_ext_complex(rouquier, reduced=True)` applies the finite reduced
  Koszul resolution for the diagonal ideal `(x_i - y_i)` termwise and records
  the induced maps.
* `khovanov_rozansky_cohomology(diagram, braid)` computes the unreduced Hilbert
  series in `A,Q,T`.
* `khovanov_rozansky_cohomology(diagram, braid, reduced=True)` computes the
  reduced triply graded Laurent polynomial.

The public API defaults to `reduced=False`.  This computes the **unreduced**
invariant honestly, *without* imposing the relations `z_{0,i} = 0`.  The
Bott-Samelson coordinate rings are then free of finite rank over the leftmost
polynomial ring `R = QQ[z_{0,i}]`, so the graded pieces are infinite
dimensional and the answer is a Hilbert series rather than a Laurent
polynomial:

```text
HHH^{a,r}(Q) = K_{a,r}(Q) / (1 - Q^2)^rank.
```

Every differential and induced map in the construction is degree `0`, so the
homology splits as a direct sum over the internal `Q`-degree.  The graded
dimension `dim HHH^{a,r}_q` is therefore the homology of a *finite* complex of
`QQ`-vector spaces and can be computed exactly for every `q` below a cutoff;
the numerators `K_{a,r}` are reconstructed from those dimensions.  The
total-degree cutoff is raised automatically until the numerators stabilize
(use `--max-degree` / `max_total_degree=` to fix it).  This is an honest
computation and does **not** assume the freeness shortcut
`HHH_unreduced = HHH_reduced / (1 - Q^2)^rank`; in general the two differ.

Pass `reduced=True` to use the finite-dimensional specialization directly; it
sets the leftmost polynomial variables to zero after forming each
Bott-Samelson term.

On the command line, add `--reduced` to print this reduced Laurent polynomial
instead of the unreduced Hilbert series.

There is also a bounded linear-algebra-only implementation in
`slower_old_KR/khovanov_rozansky_bounded.py`.  Its entry point
`khovanov_rozansky_bounded_cohomology(diagram, braid, max_q_degree=...)` keeps
the variables `z_{0,i}` and computes the finite truncation
`sum_{q <= max_q_degree} dim HHH^{a,r,q} A^a Q^q T^r` by replacing each
homogeneous quotient degree with an explicit vector-space quotient.  This path
does not use Groebner bases.

## Light leaves

`light_leaves.py` enumerates the `{0,1}^n` subexpression basis labels for a
Bott-Samelson word.  For each subexpression it records the endpoint, the
light-leaves `U/D` decoration, the left-`R` module degree, and the light-leaf
defect `#U0 - #D0`.

```bash
python3 -m computations.light_leaves \
  --vertices 0,1 \
  --edges 0-1 \
  --word 0,1,0
```

In Python:

```python
from computations.khovanov_rozansky import DynkinDiagram
from computations.light_leaves import bott_samelson_light_leaves

diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
basis = bott_samelson_light_leaves(diagram, [0, 1, 0])

print(basis.graded_rank())                         # R-module grading
print(basis.graded_rank_by_endpoint(degree="light_leaf"))
```

The same module can expand a Rouquier complex as a complex of free left
`R`-modules, using the same positive/negative local complexes and shifts as
the KR implementation.  Each chain group is decomposed into the
`{0,1}^k` basis of its Bott-Samelson summands, and every differential is a
SymPy matrix with entries in the leftmost polynomial ring.

```bash
python3 -m computations.light_leaves \
  --vertices 0,1 \
  --edges 0-1 \
  --braid '0:+,1:-'
```

In Python:

```python
from computations.khovanov_rozansky import DynkinDiagram
from computations.light_leaves import rouquier_complex_as_free_left_r_modules

diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
complex_ = rouquier_complex_as_free_left_r_modules(diagram, [(0, "+"), (1, "-")])

print(complex_.r_variables)
print(complex_.basis(-1))
print(complex_.differential(-1))
print(complex_.check_d_squared())
```

`slower_old_KR/khovanov_rozansky_free_r.py` integrates this free-left-`R` model into the
KR computation.  It builds the Koszul complexes for
`Ext^a_{R-R}(R, term)` over `R`, records the Rouquier maps on those complexes,
and computes horizontal homology over `R` as finitely presented SymPy AGCA
subquotient modules.  In particular, torsion modules such as `R/(x_0)` are
kept visible.

```python
from computations.khovanov_rozansky import DynkinDiagram
from computations.slower_old_KR.khovanov_rozansky_free_r import (
    khovanov_rozansky_free_r_homology,
)

diagram = DynkinDiagram.from_data([0], [])
result = khovanov_rozansky_free_r_homology(diagram, [(0, "+")])

print(result.term_data[(1,)].koszul.differentials[0])
print(result.horizontal_homology[(0, 0)].module)  # <[1] + <[2*x_0]>>
```

`euler_trace_extfree.py` is the current Euler-trace implementation.  It computes
the same termwise `Ext^a_{R-R}(R, C^j)` modules over the left polynomial ring
under the Ext-freeness assumption, but it does not compute horizontal homology.
Instead it returns

```text
sum_a A^a sum_j (-1)^j Hilb_Q Ext^a_{R-R}(R, C^j),
```

which is the same as setting `T = -1` in the unreduced `A,Q,T` Euler
characteristic.  The older non-Ext-free version in
`slower_old_euler_trace/euler_trace.py` computes the same
termwise `Ext^a_{R-R}(R, C^j)` modules over the left polynomial ring, but it
uses the slower general subquotient machinery.

```python
from computations.khovanov_rozansky import DynkinDiagram
from computations.euler_trace_extfree import khovanov_rozansky_euler_trace_extfree

diagram = DynkinDiagram.from_data([1, 2], [(1, 2)])
result = khovanov_rozansky_euler_trace_extfree(
    diagram,
    [(1, "+"), (2, "+"), (1, "+"), (2, "+")],
)

print(result.polynomial)  # A + Q**4 + 1
```

For larger examples, `parallel_euler_trace.py` parallelizes the same Ext-free
Euler trace across Bott-Samelson terms:

```python
from computations.parallel_euler_trace import (
    khovanov_rozansky_parallel_euler_trace_extfree,
)

result = khovanov_rozansky_parallel_euler_trace_extfree(
    diagram,
    [(1, "+"), (2, "+"), (1, "+"), (2, "+")],
    max_workers=4,
)
```

## Example

```bash
python3 -m computations.slower_old_KR.khovanov_rozansky \
  --vertices 0,1 \
  --edges 0-1 \
  --braid '0:+,1:+,0:+'
```

The braid is a comma-separated list of `generator:sign` pairs.  Signs may be
`+` or `-`.

In Python:

```python
from computations.slower_old_KR.khovanov_rozansky import (
    DynkinDiagram,
    khovanov_rozansky_cohomology,
)

diagram = DynkinDiagram.from_data([0, 1], [(0, 1)])
result = khovanov_rozansky_cohomology(
    diagram,
    [(0, "+"), (1, "+"), (0, "+")],
)

print(result.polynomial)
```
