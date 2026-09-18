"""Transfinite interpolation (Coons patches).

`tfi2d` fills a quad from its 4 boundary edges (the bilinear blend used by the
2D butterfly in `cfd/ogrid_demo.py`). `tfi3d` is the Boolean-sum generalisation
that fills a hex block interior from its 6 boundary faces. Both are affine-exact:
a block whose faces are an affine image of the unit cube is reproduced exactly,
which matches the biharmonic RBF's affine reproduction.
"""
from __future__ import annotations

import numpy as np


def _dist(n: int, grade=None) -> np.ndarray:
    """1D parameter samples in [0,1]; `grade` may supply a custom monotonic array."""
    if grade is None:
        return np.linspace(0.0, 1.0, n)
    g = np.asarray(grade, float)
    if g.shape != (n,):
        raise ValueError(f"grade must have shape ({n},), got {g.shape}")
    return g


def tfi2d(edges: dict, su=None, sv=None) -> np.ndarray:
    """Bilinear Coons patch from 4 edges.

    edges: dict with keys 'u0','u1' (v-varying edges at u=0,1, shape (nv,3)) and
    'v0','v1' (u-varying edges at v=0,1, shape (nu,3)). Corners must agree.
    Returns (nu, nv, 3).
    """
    v0 = np.asarray(edges["v0"], float)  # (nu,3) at v=0
    v1 = np.asarray(edges["v1"], float)  # (nu,3) at v=1
    u0 = np.asarray(edges["u0"], float)  # (nv,3) at u=0
    u1 = np.asarray(edges["u1"], float)  # (nv,3) at u=1
    nu, nv = len(v0), len(u0)
    u = _dist(nu, su)[:, None, None]
    v = _dist(nv, sv)[None, :, None]
    # linear projectors
    Pu = (1 - u) * u0[None, :, :] + u * u1[None, :, :]
    Pv = (1 - v) * v0[:, None, :] + v * v1[:, None, :]
    # bilinear corner term
    c00, c01 = v0[0], v1[0]
    c10, c11 = v0[-1], v1[-1]
    Puv = (
        (1 - u) * (1 - v) * c00
        + (1 - u) * v * c01
        + u * (1 - v) * c10
        + u * v * c11
    )
    return Pu + Pv - Puv


def tfi3d(faces: dict, su=None, sv=None, sw=None) -> np.ndarray:
    """Boolean-sum (Coons) trilinear interpolation from 6 faces.

    faces: dict with keys
        'i0','i1' : (nj, nk, 3)   faces at i=0, i=ni-1
        'j0','j1' : (ni, nk, 3)   faces at j=0, j=nj-1
        'k0','k1' : (ni, nj, 3)   faces at k=0, k=nk-1
    Shared edges/corners must agree between neighbouring faces.
    Returns (ni, nj, nk, 3).
    """
    i0, i1 = np.asarray(faces["i0"], float), np.asarray(faces["i1"], float)
    j0, j1 = np.asarray(faces["j0"], float), np.asarray(faces["j1"], float)
    k0, k1 = np.asarray(faces["k0"], float), np.asarray(faces["k1"], float)
    ni = j0.shape[0]
    nj = i0.shape[0]
    nk = i0.shape[1]
    if i0.shape != (nj, nk, 3) or i1.shape != (nj, nk, 3):
        raise ValueError("i-faces must be (nj,nk,3)")
    if j0.shape != (ni, nk, 3) or j1.shape != (ni, nk, 3):
        raise ValueError("j-faces must be (ni,nk,3)")
    if k0.shape != (ni, nj, 3) or k1.shape != (ni, nj, 3):
        raise ValueError("k-faces must be (ni,nj,3)")

    u = _dist(ni, su).reshape(ni, 1, 1, 1)
    v = _dist(nj, sv).reshape(1, nj, 1, 1)
    w = _dist(nk, sw).reshape(1, 1, nk, 1)

    # single-axis linear projectors
    U = (1 - u) * i0[None, :, :, :] + u * i1[None, :, :, :]      # (ni,nj,nk,3)
    V = (1 - v) * j0[:, None, :, :] + v * j1[:, None, :, :]
    W = (1 - w) * k0[:, :, None, :] + w * k1[:, :, None, :]

    # edges (shared by pairs of faces), extracted from any owning face
    # edge along k at (i-var? ) -- build the 4-corner + 12-edge blend explicitly.
    # pairwise terms
    # UV: edges varying in k -> take from i-face at its j-boundary (== j-face at its i-boundary)
    e_i0j0 = i0[0, :, :]   # (nk,3) i=0,j=0
    e_i0j1 = i0[-1, :, :]  # i=0,j=nj-1
    e_i1j0 = i1[0, :, :]
    e_i1j1 = i1[-1, :, :]
    UV = (
        (1 - u) * (1 - v) * e_i0j0[None, None, :, :]
        + (1 - u) * v * e_i0j1[None, None, :, :]
        + u * (1 - v) * e_i1j0[None, None, :, :]
        + u * v * e_i1j1[None, None, :, :]
    )
    # UW: edges varying in j -> i-face at its k-boundary
    e_i0k0 = i0[:, 0, :]
    e_i0k1 = i0[:, -1, :]
    e_i1k0 = i1[:, 0, :]
    e_i1k1 = i1[:, -1, :]
    UW = (
        (1 - u) * (1 - w) * e_i0k0[None, :, None, :]
        + (1 - u) * w * e_i0k1[None, :, None, :]
        + u * (1 - w) * e_i1k0[None, :, None, :]
        + u * w * e_i1k1[None, :, None, :]
    )
    # VW: edges varying in i -> j-face at its k-boundary
    e_j0k0 = j0[:, 0, :]
    e_j0k1 = j0[:, -1, :]
    e_j1k0 = j1[:, 0, :]
    e_j1k1 = j1[:, -1, :]
    VW = (
        (1 - v) * (1 - w) * e_j0k0[:, None, None, :]
        + (1 - v) * w * e_j0k1[:, None, None, :]
        + v * (1 - w) * e_j1k0[:, None, None, :]
        + v * w * e_j1k1[:, None, None, :]
    )
    # corners
    c000 = i0[0, 0]
    c001 = i0[0, -1]
    c010 = i0[-1, 0]
    c011 = i0[-1, -1]
    c100 = i1[0, 0]
    c101 = i1[0, -1]
    c110 = i1[-1, 0]
    c111 = i1[-1, -1]
    UVW = (
        (1 - u) * (1 - v) * (1 - w) * c000
        + (1 - u) * (1 - v) * w * c001
        + (1 - u) * v * (1 - w) * c010
        + (1 - u) * v * w * c011
        + u * (1 - v) * (1 - w) * c100
        + u * (1 - v) * w * c101
        + u * v * (1 - w) * c110
        + u * v * w * c111
    )
    return U + V + W - UV - UW - VW + UVW
