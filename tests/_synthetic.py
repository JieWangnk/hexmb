"""A synthetic prolate-spheroid LV endocardial grid for tests.

Self-contained (numpy only, no data files, no private project), so the LV
templates can be exercised anywhere. Shape: tapered finite apex, wide body,
open base ring -- the ordered ``(NL, NT, 3)`` grid the LV mesher expects.
"""
import numpy as np


def synthetic_lv(NL=40, NT=56, length=80.0, radius=25.0):
    axis = np.array([0.0, 0.0, 1.0])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    fracs = np.linspace(0.0, 1.0, NL)
    ang = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    G = np.zeros((NL, NT, 3))
    for i, f in enumerate(fracs):
        z = f * length
        r = radius * np.sqrt(max(1e-3, 1.0 - (1.0 - f) ** 2))     # ~0 at apex, radius at base
        r = max(r, 2.0)                                            # keep the apex finite
        G[i, :, 0] = r * np.cos(ang)
        G[i, :, 1] = r * np.sin(ang)
        G[i, :, 2] = z
    return G, axis, e1, e2, G.reshape(-1, 3).mean(0)
