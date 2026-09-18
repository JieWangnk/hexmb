"""Write an ordered surface grid ``(NL, NT, 3)`` as a binary STL file.

This turns the endocardial blood-pool surface (e.g. from ``hexmb.io.cap``) into a
standard triangulated CAD surface that meshers, CAD tools, and viewers all read.
Each grid quad (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) becomes two triangles; the
circumferential direction wraps (column NT joins column 0). No apex/base caps are
added -- the surface is the open endocardial wall (the CFD wall patch).
"""
import struct
from pathlib import Path

import numpy as np


def write_surface_stl(grid, path, scale=1.0, close_circumferential=True):
    """Write ``grid`` (NL x NT x 3, mm) to ``path`` as binary STL.

    ``scale`` multiplies coordinates (use 1.0 to keep millimetres, 1e-3 for metres).
    Returns the number of triangles written.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    G = np.asarray(grid, float) * scale
    NL, NT, _ = G.shape
    jmax = NT if close_circumferential else NT - 1
    tris = []
    for i in range(NL - 1):
        for j in range(jmax):
            j1 = (j + 1) % NT
            a, b, c, d = G[i, j], G[i, j1], G[i + 1, j1], G[i + 1, j]
            tris.append((a, b, c))
            tris.append((a, c, d))

    with open(path, "wb") as f:
        f.write(b"\0" * 80)                      # 80-byte header
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))        # attribute byte count
    return len(tris)
