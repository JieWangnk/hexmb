#!/usr/bin/env python3
"""Radial-basis-function (RBF) mesh morphing for large-deformation moving-mesh CFD.

Given a set of CONTROL points x_c (e.g. moving-wall nodes) with known
displacements d_c, RBF morphing produces a smooth displacement field s(x) that
is evaluated at EVERY mesh node to move the whole mesh. Unlike Laplacian mesh
smoothing (which diffuses boundary motion and inverts cells under large
deformation), an RBF field with a linear polynomial term reproduces affine
motion exactly and keeps interior cells well-shaped.

Math
----
Interpolant for one displacement component:
    s(x) = sum_i gamma_i * phi(||x - x_c_i||)  +  a0 + a . x
phi = radial kernel. The polynomial term p(x)=a0+a.x guarantees exact
reproduction of translations/rotations/stretch (rigid + affine), which is why
the interior does not shear. Coefficients solve the saddle-point system
    [ Phi   P ] [gamma]   [ d_c ]
    [ P^T   0 ] [ a   ] = [  0  ]
with Phi_ij = phi(||x_ci - x_cj||), P = [1, x_c] (n_c x 4 in 3D). The bottom
block P^T gamma = 0 enforces orthogonality so the polynomial carries the
affine part. Solved once (shared matrix) for all three components.

Kernels
-------
- 'biharmonic'  phi(r) = r            (global, exact, small control sets)
- 'triharmonic' phi(r) = r^3          (smoother, stiffer interior)
- 'tps'         phi(r) = r^2 log r    (thin-plate spline)
- 'wendland'    phi(r) = (1-r/R)_+^4 (4r/R+1)  (compact support R, sparse/scalable)

Control-point reduction: fitting to a decimated subset of wall nodes (a few
hundred) makes the dense solve trivial while the field stays smooth; the full
wall motion is still applied exactly at the wall via a final snap if desired.
"""
from __future__ import annotations
import numpy as np


def _phi(r, kernel, R):
    if kernel == "biharmonic":
        return r
    if kernel == "triharmonic":
        return r ** 3
    if kernel == "tps":
        out = np.where(r > 0, r * r * np.log(np.where(r > 0, r, 1.0)), 0.0)
        return out
    if kernel == "wendland":
        t = np.clip(r / R, 0, 1)
        return (1 - t) ** 4 * (4 * t + 1)
    raise ValueError(kernel)


class RBFMorpher:
    def __init__(self, kernel: str = "triharmonic", support: float | None = None):
        self.kernel = kernel
        self.R = support

    def fit(self, ctrl_pts: np.ndarray, ctrl_disp: np.ndarray) -> "RBFMorpher":
        """ctrl_pts (n,3), ctrl_disp (n,3). Solves the RBF+polynomial system."""
        x = np.asarray(ctrl_pts, float); d = np.asarray(ctrl_disp, float)
        n = len(x)
        R = self.R or (np.ptp(x, axis=0).max())     # default support = bbox size
        dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        Phi = _phi(dist, self.kernel, R)
        P = np.hstack([np.ones((n, 1)), x])          # (n,4)
        A = np.block([[Phi, P], [P.T, np.zeros((4, 4))]])
        rhs = np.vstack([d, np.zeros((4, 3))])
        sol = np.linalg.solve(A, rhs)
        self.gamma = sol[:n]; self.poly = sol[n:]    # (n,3),(4,3)
        self.x = x; self.R = R
        return self

    def displacement(self, pts: np.ndarray, block: int = 20000) -> np.ndarray:
        """Evaluate s(x) at query points (m,3) -> (m,3), in blocks for memory."""
        pts = np.asarray(pts, float); out = np.empty_like(pts)
        for s in range(0, len(pts), block):
            q = pts[s:s + block]
            dist = np.linalg.norm(q[:, None, :] - self.x[None, :, :], axis=2)
            Phi = _phi(dist, self.kernel, self.R)
            P = np.hstack([np.ones((len(q), 1)), q])
            out[s:s + block] = Phi @ self.gamma + P @ self.poly
        return out


def select_control_points(pts: np.ndarray, n_target: int, seed: int = 0) -> np.ndarray:
    """Farthest-point-ish subsample of boundary nodes -> even control coverage."""
    rng = np.random.default_rng(seed)
    if len(pts) <= n_target:
        return np.arange(len(pts))
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(n_target - 1):
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(idx)
