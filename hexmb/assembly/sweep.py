"""Sweep helpers: rotation-minimizing frames along a path and circular rings.

These turn a centreline path + radius into the (rings, e1s, e2s) that
`build_ogrid_from_rings` sweeps, so a straight pipe, a bent U-bend and a tapered
tube are the same operation with different paths/radii.
"""
from __future__ import annotations

import numpy as np

from .ogrid import build_ogrid_from_rings


def parallel_transport_frames(path):
    """Rotation-minimizing (double-reflection) frames along a polyline path
    (ns,3). Returns (tangents, e1s, e2s), each (ns,3). Twist-free: avoids the
    Frenet-normal flip through a bend."""
    path = np.asarray(path, float)
    ns = len(path)
    t = np.zeros((ns, 3))
    t[1:-1] = path[2:] - path[:-2]
    t[0] = path[1] - path[0]
    t[-1] = path[-1] - path[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-15
    # seed frame
    a = np.array([1.0, 0.0, 0.0])
    if abs(t[0] @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ t[0]) * t[0]
    e1 /= np.linalg.norm(e1)
    e1s = np.zeros((ns, 3))
    e2s = np.zeros((ns, 3))
    e1s[0] = e1
    e2s[0] = np.cross(t[0], e1)
    for i in range(1, ns):
        # double-reflection transport of e1 from i-1 to i
        v1 = path[i] - path[i - 1]
        c1 = v1 @ v1 + 1e-15
        rL = e1s[i - 1] - (2.0 / c1) * (v1 @ e1s[i - 1]) * v1
        tL = t[i - 1] - (2.0 / c1) * (v1 @ t[i - 1]) * v1
        v2 = t[i] - tL
        c2 = v2 @ v2 + 1e-15
        e1i = rL - (2.0 / c2) * (v2 @ rL) * v2
        e1i -= (e1i @ t[i]) * t[i]
        e1i /= np.linalg.norm(e1i) + 1e-15
        e1s[i] = e1i
        e2s[i] = np.cross(t[i], e1i)
    return t, e1s, e2s


def circular_rings(path, radius, NT: int, e1s=None, e2s=None):
    """Circle of `radius` (scalar or (ns,)) at each path station in its
    rotation-minimizing plane. Returns rings (ns,NT,3), e1s, e2s."""
    path = np.asarray(path, float)
    ns = len(path)
    if e1s is None:
        _, e1s, e2s = parallel_transport_frames(path)
    radius = np.broadcast_to(np.asarray(radius, float), (ns,))
    ang = 2 * np.pi * np.arange(NT) / NT
    rings = np.zeros((ns, NT, 3))
    for s in range(ns):
        rings[s] = path[s] + radius[s] * (
            np.outer(np.cos(ang), e1s[s]) + np.outer(np.sin(ang), e2s[s])
        )
    return rings, e1s, e2s


def sweep_ogrid(path, radius, NT: int = 32, NR: int = 6, core_frac: float = 0.42,
                radial_dist=None):
    """Build an O-grid tube by sweeping a circular section along a path.
    Returns a MultiBlockMesh (not assembled, no patches)."""
    rings, e1s, e2s = circular_rings(path, radius, NT)
    return build_ogrid_from_rings(rings, e1s, e2s, NR=NR, core_frac=core_frac,
                                  radial_dist=radial_dist)
