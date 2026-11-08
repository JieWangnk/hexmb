"""Read a Cardiac Atlas Project (CAP) left-ventricle finite-element model and
return the endocardial blood-pool surface as an ordered grid hexmb can mesh.

The CAP Sunnybrook models are 3D LV *wall* models: 16 hexahedral elements
(4 circumferential x 4 longitudinal) with two transmural layers of 20 nodes,
in prolate-spheroidal coordinates with one ``focus`` per model.  Per element,
``lambda`` is bicubic Hermite in (xi1, xi2) with the scale factors the
``.exelem`` lists (all 1 in these models) and linear transmurally; ``mu`` is
trilinear Lagrange; ``theta`` is linear in xi1 and decreases along it.

This reader evaluates that definition as written, so the 40 nodes are
reproduced exactly (1e-4 mm).  It samples the inner (endocardial) layer on
``NL`` rows at uniform *global xi2* (the model's own longitudinal parameter:
row 0 is the apex, a single point; row ``NL-1`` the base ring) and ``NT``
columns at uniform ``theta``.  Every row is a closed curve that winds once
about the model axis (radius ``focus*sinh(lambda)*sin(mu)`` is positive
wherever ``lambda > 0``), so the O-grid builder's petals cannot cross on it.
hexmb 0.1-0.2.0 instead interpolated the node *positions* in Cartesian space,
which bent the first rings at the apex into crescents and under-read the
cavity by 3-30 % (median 8 %, most at end-systole).

The element file is the same template for every Sunnybrook subject apart from
the focus (which the ``.exnode`` also carries), so a copy is bundled and used
when no ``.exelem`` sits beside the ``.exnode``.

Data source (public domain, CC0): Cardiac Atlas Project - Sunnybrook Cardiac
Data, https://www.cardiacatlas.org/sunnybrook-cardiac-data/ . See DATA.md.
"""
import re
import warnings
from pathlib import Path

import numpy as np

__all__ = ["cap_endo_surface", "surface_frame", "parse_exnode", "parse_exelem", "find_exelem"]


def parse_exnode(path):
    """``({node_id: {lam, d1, d2, d12, mu, th}}, focus)`` from a CAP ``.exnode``."""
    txt = Path(path).read_text()
    m = re.search(r"focus=\s*([\d.]+)", txt)
    if m is None:
        raise ValueError(f"{path}: no 'focus=' in the file; not a CAP prolate-spheroidal model")
    focus = float(m.group(1))
    nodes = {}
    for m in re.finditer(r"Node:\s*(\d+)\s*\n(.*?)(?=\n\s*Node:|\Z)", txt, re.S):
        rows = [r for r in m.group(2).strip().splitlines() if r.strip()]
        if len(rows) < 3:
            continue
        v = [float(x) for x in rows[0].split()]
        if len(v) < 4:
            raise ValueError(f"{path}: node {m.group(1)} has {len(v)} lambda values, expected 4 (value + 3 derivatives)")
        nodes[int(m.group(1))] = dict(lam=v[0], d1=v[1], d2=v[2], d12=v[3],
                                      mu=float(rows[1].split()[0]), th=float(rows[2].split()[0]))
    if not nodes:
        raise ValueError(f"{path}: no nodes found")
    return nodes, focus


def parse_exelem(path):
    """3D elements of a CAP ``.exelem``: list of ``(node_ids[8], scale_factors (8, 4))``
    in CMISS local order (xi1 fastest, then xi2, then xi3)."""
    txt = Path(path).read_text()
    if "Dimension=3" not in txt:
        raise ValueError(f"{path}: no 3D elements")
    sec = txt.split("Dimension=3", 1)[1]
    elems = []
    for m in re.finditer(r"Element:\s*(\d+) 0 0\s*\n(.*?)(?=\n\s*Element:|\Z)", sec, re.S):
        body = m.group(2)
        mn = re.search(r"Nodes:\s*\n([^\n]+)", body)
        ms = re.search(r"Scale factors:\s*\n(.*)", body, re.S)
        if mn is None or ms is None:
            raise ValueError(f"{path}: element {m.group(1)} has no "
                             f"{'Nodes' if mn is None else 'Scale factors'} block (truncated or non-CAP .exelem?)")
        nodes = [int(x) for x in mn.group(1).split()]
        sf = [float(x) for x in ms.group(1).split()]
        if len(nodes) != 8 or len(sf) < 32:
            raise ValueError(f"{path}: element {m.group(1)} has {len(nodes)} nodes / {len(sf)} scale factors; "
                             "expected 8 nodes with 32 Hermite scale factors")
        elems.append((nodes, np.array(sf[:32]).reshape(8, 4)))
    if not elems:
        raise ValueError(f"{path}: no 3D elements parsed")
    return elems


def find_exelem(exnode_path):
    """The ``.exelem`` beside ``exnode_path`` (same directory), else the bundled
    Sunnybrook template (identical for every subject apart from the focus)."""
    p = Path(exnode_path)
    found = sorted(p.parent.glob("*.exelem"))
    if found:
        return str(found[0])
    from importlib import resources
    return str(resources.files("hexmb") / "data" / "cap" / "GlobalHermiteParam.exelem")


def _hermite(x):
    """1-D cubic Hermite basis on [0, 1]: (value at 0, slope at 0, value at 1, slope at 1)."""
    return (1 - 3 * x ** 2 + 2 * x ** 3, x * (x - 1) ** 2, x ** 2 * (3 - 2 * x), x ** 2 * (x - 1))


def _prolate_to_cart(lam, mu, th, focus):
    return np.stack([focus * np.cosh(lam) * np.cos(mu),
                     focus * np.sinh(lam) * np.sin(mu) * np.cos(th),
                     focus * np.sinh(lam) * np.sin(mu) * np.sin(th)], axis=-1)


def cap_endo_surface(exnode_path, NL=40, NT=56, exelem=None, layer="endo"):
    """Return an ``(NL, NT, 3)`` endocardial surface (mm), apex at index 0.

    ``exelem`` names the element file; by default the one beside the
    ``.exnode`` or the bundled template.  ``layer="epi"`` gives the outer
    (epicardial) layer instead.  Warns when the model crosses its own long
    axis (``lambda <= 0`` somewhere): the rows named in the warning cannot be
    meshed.  :func:`hexmb.services.motion.build_lv_motion` finds the same rows
    itself (:func:`hexmb.templates.conformal_lv.bad_rings`) and trims them at a
    rim or fills the phase from its neighbours.
    """
    nodes, focus = parse_exnode(exnode_path)
    exelem = exelem or find_exelem(exnode_path)
    elems = parse_exelem(exelem)
    missing = sorted({n for e, _ in elems for n in e} - set(nodes))
    if missing:
        raise ValueError(f"{exelem} refers to node(s) {missing[:6]} that {exnode_path} does not hold; "
                         "pass the model's own .exelem with exelem=")
    lam_lo = np.mean([nodes[n]["lam"] for e, _ in elems for n in e[:4]])
    lam_hi = np.mean([nodes[n]["lam"] for e, _ in elems for n in e[4:]])
    inner_lo = lam_lo < lam_hi                      # which xi3 face is the inner (endo) layer
    if layer == "epi":
        inner_lo = not inner_lo
    elif layer != "endo":
        raise ValueError(f"layer must be 'endo' or 'epi', got {layer!r}")

    patches = []
    for e, sf in elems:
        ids = e[:4] if inner_lo else e[4:]
        s = sf[:4] if inner_lo else sf[4:]
        N = [nodes[n] for n in ids]
        th0 = N[0]["th"]
        dth = (th0 - N[1]["th"]) % (2 * np.pi)      # theta decreases along xi1
        if dth < 1e-9:
            dth = 2 * np.pi
        patches.append(dict(N=N, ids=list(ids), s=s, th0=th0, dth=dth))

    # longitudinal bands by connectivity: the apex band holds the mu = 0 nodes and each
    # next band's lower edge is the previous band's upper edge
    band0 = [p for p in patches if p["N"][0]["mu"] < 1e-6 and p["N"][1]["mu"] < 1e-6]
    if not band0:
        raise ValueError(f"{exnode_path}: no element has its lower edge at the apex (mu = 0)")
    bands, left = [band0], [p for p in patches if p not in band0]
    while left:
        top = {i for p in bands[-1] for i in p["ids"][2:]}
        nxt = [p for p in left if set(p["ids"][:2]) <= top]
        if not nxt:
            raise ValueError(f"{exelem}: element bands do not chain from the apex to the base")
        bands.append(nxt)
        left = [p for p in left if p not in nxt]

    xi2g = np.linspace(0.0, len(bands), NL)         # global longitudinal parameter, apex 0
    ths = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    lam = np.full((NL, NT), np.nan)
    mu = np.full((NL, NT), np.nan)
    for b, band in enumerate(bands):
        rows = np.where((xi2g >= b - 1e-9) & (xi2g <= b + 1 + 1e-9))[0]
        xi2 = np.clip(xi2g[rows] - b, 0.0, 1.0)
        h2 = _hermite(xi2[:, None])
        X2 = xi2[:, None]
        for p in band:
            N, s = p["N"], p["s"]
            a = (p["th0"] - ths) % (2 * np.pi)
            cols = np.where(a <= p["dth"] + 1e-9)[0]
            xi1 = a[cols] / p["dth"]
            h1 = _hermite(xi1[None, :])
            X1 = xi1[None, :]
            val = np.zeros((len(rows), len(cols)))
            for k, (i1, i2) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                v = np.array([N[k]["lam"], N[k]["d1"], N[k]["d2"], N[k]["d12"]]) * s[k]
                val += (h1[2 * i1] * h2[2 * i2] * v[0] + h1[2 * i1 + 1] * h2[2 * i2] * v[1]
                        + h1[2 * i1] * h2[2 * i2 + 1] * v[2] + h1[2 * i1 + 1] * h2[2 * i2 + 1] * v[3])
            lam[np.ix_(rows, cols)] = val
            mu[np.ix_(rows, cols)] = ((1 - X2) * ((1 - X1) * N[0]["mu"] + X1 * N[1]["mu"])
                                      + X2 * ((1 - X1) * N[2]["mu"] + X1 * N[3]["mu"]))
    if np.isnan(lam).any():
        raise ValueError(f"{exelem}: {int(np.isnan(lam).sum())} grid points fall outside every element "
                         "(the elements do not tile theta 0-360 deg)")
    cross = lam[1:] <= 0                            # row 0 is the apex point, lambda there is irrelevant
    if cross.any():
        r = np.where(cross.any(1))[0] + 1
        warnings.warn(f"{Path(exnode_path).name}: the model crosses its own long axis (lambda <= 0) at "
                      f"{int(cross.sum())} grid points in rows {r.min()}-{r.max()} of {NL} (0 apex, {NL - 1} base); "
                      "those rows cannot be meshed (the moving-mesh builder trims them at a rim or fills the "
                      "phase from its neighbours when auto_trim / fill_failed are on)", stacklevel=2)
    return _prolate_to_cart(lam, mu, ths[None, :], focus)


def surface_frame(G):
    """apex->base axis, base-plane e1/e2, centroid mu -- the frame the LV mesher expects."""
    from .surface import frame_from_axis
    apex = G[0].mean(0); base = G[-1].mean(0)
    axis = base - apex; axis /= np.linalg.norm(axis)
    e1, e2 = frame_from_axis(axis)        # picks x unless the axis is (nearly) along x, then y
    return axis, e1, e2, G.reshape(-1, 3).mean(0)
