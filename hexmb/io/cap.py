"""Read a Cardiac Atlas Project (SCD) left-ventricle finite-element model and
return the endocardial blood-pool surface as an ordered grid hexmb can mesh.

The CAP Sunnybrook models are 3D LV *wall* models: hexahedral elements,
bicubic-Hermite in the circumferential + longitudinal directions and linear
transmurally, written in prolate-spheroidal coordinates (lambda + derivatives,
mu, theta; one focus per model). The 40 nodes are two transmural layers
(endocardium + epicardium), each 4 circumferential x 5 longitudinal.

For CFD we want the *endocardial* surface (the blood pool), so we take the inner
(smaller-lambda) layer and interpolate it densely. Interpolating the node
*positions* in Cartesian space avoids the cosh(lambda) blow-up a naive Hermite
evaluation in lambda suffers, and taking one layer removes the endo/epi wobble.

Data source (public domain, CC0): Cardiac Atlas Project - Sunnybrook Cardiac
Data, https://www.cardiacatlas.org/sunnybrook-cardiac-data/ . See docs/DATA.md.
"""
import re

import numpy as np


def _parse_exnode(path):
    txt = open(path).read()
    focus = float(re.search(r"focus=\s*([\d.]+)", txt).group(1))
    lam, d1, d2, d12, mu, th = ([] for _ in range(6))
    for nb in re.split(r"Node:\s*\d+\s*\n", txt)[1:]:
        rows = [r for r in nb.strip().splitlines() if r.strip()]
        if len(rows) < 3:
            continue
        v = [float(x) for x in rows[0].split()]
        lam.append(v[0]); d1.append(v[1]); d2.append(v[2]); d12.append(v[3])
        mu.append(float(rows[1].split()[0])); th.append(float(rows[2].split()[0]))
    return [np.array(a) for a in (lam, d1, d2, d12, mu, th)], focus


def _prolate_to_cart(lam, mu, th, focus):
    return np.stack([focus * np.cosh(lam) * np.cos(mu),
                     focus * np.sinh(lam) * np.sin(mu) * np.cos(th),
                     focus * np.sinh(lam) * np.sin(mu) * np.sin(th)], axis=-1)


def cap_endo_surface(exnode_path, NL=40, NT=56):
    """Return an ``(NL, NT, 3)`` endocardial surface (mm), apex at index 0.

    Takes the inner (endo) transmural layer of nodes, converts each to Cartesian,
    and interpolates there: periodic cubic spline circumferentially, cubic spline
    longitudinally. Needs scipy.
    """
    from scipy.interpolate import CubicSpline, interp1d
    (lam, d1, d2, d12, mu, th), focus = _parse_exnode(exnode_path)
    half = len(lam) // 2
    sel = slice(half, None) if lam[half:][-4:].mean() < lam[:half][-4:].mean() else slice(0, half)
    lam, mu, th = (a[sel] for a in (lam, mu, th))          # endo layer only

    ncirc = len(np.unique(np.round(th, 3)))
    nlong = len(lam) // ncirc
    P = _prolate_to_cart(lam, mu, th, focus).reshape(nlong, ncirc, 3)
    P = P[np.argsort(mu.reshape(nlong, ncirc).mean(1))]   # apex (mu~0) first

    thc = np.array(sorted(np.unique(np.round(th, 3))))
    te = np.append(thc, thc[0] + 2 * np.pi)
    tt = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    dense = np.zeros((nlong, NT, 3))
    for r in range(nlong):
        ext = np.vstack([P[r], P[r, :1]])
        for k in range(3):
            dense[r, :, k] = CubicSpline(te, ext[:, k], bc_type="periodic")(tt)
    kind = "cubic" if nlong >= 4 else "linear"
    return interp1d(np.linspace(0, 1, nlong), dense, axis=0, kind=kind)(np.linspace(0, 1, NL))


def surface_frame(G):
    """apex->base axis, base-plane e1/e2, centroid mu -- the frame the LV mesher expects."""
    apex = G[0].mean(0); base = G[-1].mean(0)
    axis = base - apex; axis /= np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0])
    e1 = tmp - axis * (tmp @ axis); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return axis, e1, e2, G.reshape(-1, 3).mean(0)
