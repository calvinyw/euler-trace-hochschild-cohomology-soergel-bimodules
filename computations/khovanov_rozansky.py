"""Compatibility layer for shared KR/Soergel data structures.

The slower direct Khovanov--Rozansky implementation now lives in
``computations.slower_old_KR.khovanov_rozansky``. New KR computations should
use ``computations.khovanov_rozansky_extfree``.
"""

from computations.slower_old_KR import khovanov_rozansky as _legacy

globals().update(
    {
        name: value
        for name, value in vars(_legacy).items()
        if not name.startswith("__")
    }
)

__all__ = [
    name
    for name in vars(_legacy)
    if not name.startswith("__")
]
