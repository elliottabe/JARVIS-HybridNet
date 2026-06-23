"""NumPy 2.0 compatibility shim for `imgaug`.

`imgaug` is unmaintained and references several NumPy aliases/functions that
were removed in NumPy 2.0 (``np.sctypes``, ``np.product``, ``np.float_`` …).
This restores exactly those names so imgaug imports and runs unchanged under
NumPy >= 2 (e.g. the 3d_tracking env, Py3.12 / numpy 2.2). It is a no-op on
NumPy 1.x (every name is guarded by ``hasattr``), so importing it is safe in
both the legacy `jarvis` env and the `3d_tracking` env.

Deliberately does NOT restore ``np.object``/``np.str``/``np.bool``/``np.int``:
NumPy intercepts those with a FutureWarning and imgaug does not need them.

Import this BEFORE importing ``imgaug`` (see jarvis/dataset/dataset2D.py).
"""
import numpy as np


def apply():
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void],
        }
    for fn, target in [("product", "prod"), ("cumproduct", "cumprod"),
                       ("round_", "round"), ("alltrue", "all"),
                       ("sometrue", "any")]:
        if not hasattr(np, fn):
            setattr(np, fn, getattr(np, target))
    for name, val in [("bool8", np.bool_), ("float_", np.float64),
                      ("complex_", np.complex128), ("unicode_", np.str_),
                      ("int0", np.intp), ("uint0", np.uintp)]:
        if not hasattr(np, name):
            setattr(np, name, val)


apply()
