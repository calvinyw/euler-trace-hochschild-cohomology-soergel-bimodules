# Euler trace of Hochschild Cohomology of Soergel Bimodules

Pure-Python/SymPy code for Khovanov–Rozansky cohomology and Euler-trace computations on small Soergel bimodule examples.

There is a type A code based on the freeness of Ext groups (the extfree files) and code that computes these invariants in general.

More information is in the ReadMe in computations.

## Setup

```bash
pip install -r requirements.txt
```

Run scripts from the repository root:

```bash
python -m computations.run_a2_euler_trace_2_0
python -m pytest computations/
```

## Package documentation

See [computations/README.md](computations/README.md) for module overview, current state of the art, and usage examples.
