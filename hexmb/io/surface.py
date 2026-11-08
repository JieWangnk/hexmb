"""Bring your own ventricle: a closed or open endocardial surface mesh (STL, OBJ,
PLY, OFF, VTK/VTP via meshio), a segmentation mask (NIfTI) or a saved grid
(``.npz``) -> the ordered ``(NL, NT, 3)`` ring grid every LV template takes.

The grid is built by ray-casting: ``NL`` stations along the apex->base axis,
``NT`` rays per station cast outward in a FIXED in-plane frame, each ray's first
wall hit is a grid point.  The station origins are then recentred on the ring
centroids and the cast repeated, so the stations follow a curved ventricle.  A
fixed frame (not a rotation-minimising one) is deliberate: it gives the same
angular parameterisation on every phase of a beating heart, so grids cast from
per-phase surfaces correspond point-for-point and can drive a moving mesh.

Landmarks (``apex``, ``base`` = centre of the valve plane) can be given; if not
they are detected: an OPEN surface has its base at the centroid of its largest
boundary loop and its apex at the vertex farthest from it (a small apical hole
is fine); a CLOSED surface uses the
longest principal axis, with the narrower end as the apex.

A segmentation mask whose slices are much thicker than its pixels (a short-axis
cine stack: 1-2 mm pixels, 6-10 mm slices) is read slice by slice instead
(:func:`grid_from_slices`): the outline of the blood pool in every slice becomes
one ring about that slice's own centre, so an eccentric basal slice or a flat
top cannot make the rays miss, and the volume is Simpson's rule, which is how
the same stack is measured clinically.  An isotropic mask (CT) goes through
marching cubes and the ray cast.

Install the optional readers with ``pip install "hexmb[surface]"`` (trimesh +
rtree for ray casting, meshio for VTK) and ``"hexmb[image]"`` (nibabel +
scikit-image for masks).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "load_surface_mesh",
    "auto_landmarks",
    "endo_grid_from_mesh",
    "grid_from_mask",
    "grid_from_slices",
    "grid_from_file",
    "write_surface_npz",
    "frame_from_axis",
]

_MESH_EXT = (".stl", ".obj", ".ply", ".off", ".glb", ".gltf")
_VTK_EXT = (".vtk", ".vtp", ".vtu")
_MASK_EXT = (".nii", ".nii.gz")


# --------------------------------------------------------------------------- #
#  readers                                                                     #
# --------------------------------------------------------------------------- #
def load_surface_mesh(path):
    """Load a triangulated surface as a ``trimesh.Trimesh`` (coordinates as stored,
    expected in millimetres).  STL/OBJ/PLY/OFF go through trimesh; VTK/VTP/VTU
    through meshio (quads are split into triangles)."""
    import trimesh
    p = Path(path)
    name = p.name.lower()
    ext = p.suffix.lower()
    if ext in _VTK_EXT:
        try:
            import meshio
        except ImportError as e:
            raise ImportError("reading VTK surfaces needs meshio: pip install meshio") from e
        m = meshio.read(str(p))
        tris = []
        for blk in m.cells:
            if blk.type == "triangle":
                tris.append(np.asarray(blk.data))
            elif blk.type == "quad":
                q = np.asarray(blk.data)
                tris.append(np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
            elif blk.type == "polygon":
                for poly in blk.data:
                    for k in range(1, len(poly) - 1):
                        tris.append(np.array([[poly[0], poly[k], poly[k + 1]]]))
        if not tris:
            raise ValueError(f"{p}: no triangle/quad/polygon cells found")
        return trimesh.Trimesh(np.asarray(m.points, float), np.vstack(tris), process=True)
    if ext in _MESH_EXT or name.endswith(_MESH_EXT):
        return trimesh.load(str(p), force="mesh")
    raise ValueError(f"{p}: unsupported surface format {ext!r} "
                     f"(use {', '.join(_MESH_EXT + _VTK_EXT)})")


def frame_from_axis(axis):
    """Orthonormal in-plane frame ``(e1, e2)`` for a unit ``axis``: ``e1`` is the
    global x (or y) axis projected perpendicular to ``axis``; ``e2 = axis x e1``."""
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(tmp @ axis) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - axis * (tmp @ axis); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def _boundary_loops(mesh):
    """Vertex-index arrays of the open boundary loops (edges used by one face),
    one array per connected loop, largest perimeter first."""
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges, require_count=1)
    if len(groups) == 0:
        return []
    be = edges[groups]
    n = len(mesh.vertices)
    adj = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    V = np.asarray(mesh.vertices, float)
    loops = []
    for l in np.unique(lab[be.ravel()]):
        sel = be[(lab[be[:, 0]] == l)]
        perim = float(np.linalg.norm(V[sel[:, 0]] - V[sel[:, 1]], axis=1).sum())
        loops.append((perim, np.unique(sel.ravel())))
    loops.sort(key=lambda t: -t[0])
    return [v for _, v in loops]


def auto_landmarks(mesh):
    """Return ``(apex, base_centre, kind)`` in the mesh's units.  ``kind`` is
    ``"open"`` (base = centroid of the boundary loop, apex = farthest vertex) or
    ``"closed"`` (ends of the longest principal axis; the narrower end is the
    apex, the base centre is the centroid of the far 3 % slab)."""
    V = np.asarray(mesh.vertices, float)
    loops = _boundary_loops(mesh)
    if loops and len(loops[0]) >= 3:
        base = V[loops[0]].mean(0)                     # the largest opening is the valve plane
        apex = V[np.argmax(np.linalg.norm(V - base, axis=1))]
        return apex, base, "open"
    c = V.mean(0)
    ax = np.linalg.svd(V - c, full_matrices=False)[2][0]
    s = (V - c) @ ax
    lo, hi = s.min(), s.max()
    L = hi - lo

    def end_radius(sel):
        pts = V[sel]
        cc = pts.mean(0)
        r = pts - cc
        r -= np.outer(r @ ax, ax)
        return float(np.linalg.norm(r, axis=1).mean())

    lo_slab = s < lo + 0.15 * L
    hi_slab = s > hi - 0.15 * L
    if end_radius(lo_slab) <= end_radius(hi_slab):
        apex_sel, base_sel, sign = s < lo + 0.01 * L, s > hi - 0.03 * L, 1.0
    else:
        apex_sel, base_sel, sign = s > hi - 0.01 * L, s < lo + 0.03 * L, -1.0
    apex = V[apex_sel].mean(0)
    base = V[base_sel].mean(0)
    # put the base centre on the axis through the apex (keeps the cast stations centred)
    return apex, base, "closed"


def _smooth_path(path, iters=2):
    p = np.asarray(path, float).copy()
    for _ in range(iters):
        q = p.copy()
        q[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
        p = q
    return p


def _ring_area_vectors(G):
    """Vector area of every ring polygon about its centroid, shape ``(NL, 3)``."""
    c = G.mean(axis=1, keepdims=True)
    P = G - c
    return 0.5 * np.cross(P, np.roll(P, -1, axis=1)).sum(axis=1)


def _smooth_rings(G, iters=1):
    """Circumferential 3-point smoothing that keeps every ring's area.

    A plain moving average pulls each point towards the chord of its
    neighbours, so a ring of ``NT`` points shrinks by about ``1 - cos(2 pi / NT)``
    in radius per pass (0.9 % in area for ``NT = 56``), and the mesh built on
    it under-reads the cavity by that much at every phase.  After each pass
    every ring is rescaled about its centroid so its polygon area is what it
    was before the pass: the noise is still averaged out, the volume is kept.
    """
    G = G.copy()
    for _ in range(iters):
        a0 = np.linalg.norm(_ring_area_vectors(G), axis=1)
        S = (np.roll(G, 1, axis=1) + G + np.roll(G, -1, axis=1)) / 3.0
        a1 = np.linalg.norm(_ring_area_vectors(S), axis=1)
        c = S.mean(axis=1, keepdims=True)
        f = np.sqrt(np.where(a1 > 0, a0 / np.maximum(a1, 1e-300), 1.0))[:, None, None]
        G = c + (S - c) * f
    return G


def endo_grid_from_mesh(mesh, apex=None, base=None, NL=40, NT=56, recentre=2,
                        apex_offset=0.03, base_offset=0.02, smooth=1, clamp=2.5, frame=None):
    """Ray-cast an ``(NL, NT, 3)`` endocardial grid from a surface mesh.

    Row 0 is the apex point itself (repeated ``NT`` times, as the Cardiac Atlas
    and mask readers return it); rows 1 and 2 are cast at a third and two
    thirds of ``apex_offset`` so the dome at the tip is resolved (a straight
    loft from the tip to the first station would make a cone there, whose
    narrow cap disc gives skewed wall cells after the builder's apex trim);
    rows ``3 .. NL-2`` are cast at uniform stations from ``apex_offset`` to
    ``1 - base_offset`` of the apex->base length; the last row is the last
    cast ring moved along the axis onto the base plane (the rim of an open
    surface, the flat end of a closed one), so the grid spans the whole cavity
    from tip to valve plane.  Before hexmb 0.2.0 the grid stopped
    ``apex_offset`` short of the tip and ``base_offset`` short of the base and
    the mesh lost 2-3 % of the cavity there.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The endocardial (blood-pool) surface, open at the base or closed.
    apex, base : (3,) arrays, optional
        Apex tip and valve-plane centre; detected by :func:`auto_landmarks` if omitted.
    NL, NT : int
        Levels apex->base and rays per level (``NT`` divisible by 4).
    recentre : int
        Passes that move each station origin to its ring centroid and re-cast.
    apex_offset, base_offset : float
        First/last station as fractions of the apex->base length inside the ends
        (the tip itself is a point; the rim may be open).
    smooth : int
        Circumferential 3-point smoothing passes on the final grid.
    clamp : float
        Reject a ray whose hit is farther than ``clamp`` x the station's median
        radius (a ray that escaped through an opening); it is interpolated.
    frame : (axis, e1, e2), optional
        Use this frame instead of deriving one from the landmarks.  Pass the
        reference phase's frame when casting the other phases of a beating heart
        so the grids correspond point-for-point.

    Returns ``(G, axis, e1, e2, mu, info)``; ``info`` carries the landmarks, the
    axis length and the count of interpolated rays per level.
    """
    from ..project.stl_tube import rings_by_raycast
    if NT % 4:
        raise ValueError(f"NT={NT} must be divisible by 4")
    kind = "given"
    if apex is None or base is None:
        a_auto, b_auto, kind = auto_landmarks(mesh)
        apex = a_auto if apex is None else np.asarray(apex, float)
        base = b_auto if base is None else np.asarray(base, float)
    apex = np.asarray(apex, float); base = np.asarray(base, float)
    if frame is None:
        axis = base - apex
        L = float(np.linalg.norm(axis))
        axis = axis / L
        e1, e2 = frame_from_axis(axis)
    else:
        axis, e1, e2 = (np.asarray(v, float) for v in frame)
        axis = axis / np.linalg.norm(axis)
        L = float((base - apex) @ axis)
    if L <= 0:
        raise ValueError("base must lie beyond the apex along the axis")
    if L < 5.0:
        raise ValueError(f"apex->base length is {L:.3g}: hexmb works in millimetres (a ventricle is "
                         "60-120 mm long); scale the surface by 1000 if it is in metres")

    if NL < 6:
        raise ValueError(f"NL={NL}: at least 6 rows (apex point, two tip stations, two cast stations, base ring)")
    t = np.concatenate([[apex_offset / 3.0, 2.0 * apex_offset / 3.0],           # the dome at the tip
                        np.linspace(apex_offset, 1.0 - base_offset, NL - 4)])    # uniform cast stations
    origins = apex + np.outer(t * L, axis)
    E1 = np.tile(e1, (NL - 2, 1)); E2 = np.tile(e2, (NL - 2, 1))
    rings = miss = None
    for _ in range(recentre + 1):
        rings, miss = rings_by_raycast(mesh, origins, NT=NT, frames=(None, E1, E2), clamp=clamp)
        dead = np.isnan(rings[:, :, 0]).all(axis=1)
        if dead.any():
            lv = np.where(dead)[0]
            raise ValueError(f"every ray missed at level(s) {lv.tolist()} (fractions "
                             f"{np.round(t[lv], 3).tolist()}): the station lies outside the surface; "
                             f"check the landmarks or raise apex_offset/base_offset")
        cen = rings.mean(axis=1)
        shift = cen - origins
        shift -= np.outer(shift @ axis, axis)          # move stations in-plane only
        origins = _smooth_path(origins + shift, iters=2)
    too_many = np.where(miss > NT // 4)[0]
    if too_many.size:
        raise ValueError(f"more than NT/4 rays had to be interpolated at level(s) {too_many.tolist()} "
                         f"(fractions {np.round(t[too_many], 3).tolist()}, up to {int(miss.max())} of {NT}): "
                         "the surface is not star-shaped about the axis there (a figure-8 slice, an "
                         "escaped ray through an opening) or the landmarks are off; check --apex/--base, "
                         "close the surface, or lower NL near that end")
    rings = _smooth_rings(rings, smooth) if smooth else rings
    base_ring = rings[-1] + (((base - rings[-1].mean(0)) @ axis)) * axis     # onto the base plane
    G = np.concatenate([np.repeat(apex[None], NT, 0)[None], rings, base_ring[None]])
    mu = G.reshape(-1, 3).mean(0)
    miss = np.concatenate([[0], miss.astype(int), [0]])
    info = {"apex": apex, "base": base, "landmarks": kind, "axis_length": L,
            "interpolated_rays_per_level": miss, "origins": origins}
    return G, axis, e1, e2, mu, info


def grid_from_mask(path, label=None, NL=40, NT=56, taubin_iters=10, closing=1, keep_largest=True,
                   method="auto", **kw):
    """Segmentation mask (NIfTI) -> ring grid.

    ``label`` selects one label value (default: every nonzero voxel).  The mask
    is cleaned (largest connected component, binary closing).  ``method``:

    * ``"slices"`` -- one ring per slice from the blood-pool outline about the
      slice's own centre (:func:`grid_from_slices`); the route for short-axis
      cine stacks with thick slices;
    * ``"marching"`` -- marching cubes at the voxel size, Taubin smoothing
      (``taubin_iters``), then the ray cast of :func:`endo_grid_from_mesh`; the
      route for isotropic masks (CT);
    * ``"auto"`` (default) -- ``"slices"`` when the slice spacing is at least
      2.5 times the pixel size, else ``"marching"``.

    Extra keywords go to the chosen reader.  Returns ``(G, axis, e1, e2, mu, info)``
    with ``info["method"]`` and ``info["mask_volume_mL"]``; the marching route
    also sets ``info["mesh"]``.
    """
    try:
        import nibabel as nib
        from skimage import measure
        from scipy import ndimage as ndi
    except ImportError as e:
        raise ImportError("mask input needs nibabel, scikit-image and scipy: pip install 'hexmb[image]'") from e
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if label is None:
        vals = np.unique(data[data != 0])
        if len(vals) > 1:
            raise ValueError(f"{path} holds {len(vals)} labels {vals.tolist()}: choose the blood-pool one "
                             "with label=<value> (--label), otherwise every label would be merged")
    mask = (data == label) if label is not None else (data > 0)
    if keep_largest:
        lab, n = ndi.label(mask)
        if n > 1:
            mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if closing:
        mask = ndi.binary_closing(mask, iterations=closing)
    if mask.sum() == 0:
        raise ValueError(f"{path}: mask is empty (label {label})")
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    if method == "auto":
        method = "slices" if zooms.max() >= 2.5 * np.sort(zooms)[1] else "marching"
    if method == "slices":
        G, axis, e1, e2, mu, info = grid_from_slices(mask, img.affine, zooms, NL=NL, NT=NT, **kw)
        info["mask_volume_mL"] = float(mask.sum() * np.prod(zooms) * 1e-3)
        return G, axis, e1, e2, mu, info
    if method != "marching":
        raise ValueError(f"method must be 'auto', 'slices' or 'marching', got {method!r}")
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("the marching-cubes route needs trimesh: pip install 'hexmb[surface]'") from e
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=tuple(zooms))
    idx = verts / zooms                                   # back to voxel indices
    A = np.asarray(img.affine, float)
    world = idx @ A[:3, :3].T + A[:3, 3]
    mesh = trimesh.Trimesh(world, faces, process=True)
    if taubin_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iters)
    G, axis, e1, e2, mu, info = endo_grid_from_mesh(mesh, NL=NL, NT=NT, **kw)
    info["mesh"] = mesh
    info["method"] = "marching"
    info["mask_volume_mL"] = float(mask.sum() * np.prod(zooms) * 1e-3)
    return G, axis, e1, e2, mu, info


def _slice_ring(outline, centre, e1, e2, NT):
    """Resample a closed outline (world points) onto ``NT`` uniform polar angles
    about ``centre`` in the ``e1``/``e2`` frame; the offsets are interpolated as
    3-D vectors so the ring stays in the slice plane."""
    d = outline - centre
    ang = np.arctan2(d @ e2, d @ e1) % (2 * np.pi)
    order = np.argsort(ang)
    ang, d = ang[order], d[order]
    ang_p = np.concatenate([ang - 2 * np.pi, ang, ang + 2 * np.pi])     # periodic padding
    d_p = np.vstack([d, d, d])
    th = 2 * np.pi * np.arange(NT) / NT
    ring = np.stack([np.interp(th, ang_p, d_p[:, c]) for c in range(3)], axis=1)
    return centre + ring


def grid_from_slices(mask, affine, zooms, NL=40, NT=56, frame=None, apex=None, base=None, smooth=1,
                     min_pixels=4, interp="pchip", **kw):
    """Short-axis mask stack -> ``(NL, NT, 3)`` ring grid, one ring per slice.

    The slice axis is the voxel axis with the largest spacing.  In every slice
    that holds the label, the largest 2-D region is kept and its holes filled
    (papillary muscles belong to the cavity volume, as in the clinical rule),
    its outline is traced at the half-pixel level and resampled on ``NT``
    uniform angles about the slice's own centroid in the fixed ``e1``/``e2``
    frame.  The apical end is the end with the smaller outline; the apex point
    sits half a slice beyond the apical slice's centre, and the base ring is
    the basal outline moved half a slice outward, so the enclosed volume is the
    stack's Simpson's-rule volume.  The ring stack is then placed on ``NL``
    rows uniformly along the apex->base axis.

    ``frame`` (``axis, e1, e2``) fixes the angular parameterisation across the
    phases of a beating heart; by default the axis runs from the apical to the
    basal slice centroid.  ``apex`` / ``base`` landmarks are accepted for the
    signature's sake and ignored (the slices define the ends); ``smooth`` is
    the number of circumferential 3-point smoothing passes.  ``interp`` places
    the rows between slices: ``"pchip"`` (default) runs a monotone cubic through
    the slice rings, so the wall bulges between slices as a ventricle does
    instead of following straight chords (a chord loft under-reads a convex
    cavity by several percent with 10 mm slices); ``"linear"`` keeps the chords.

    Returns ``(G, axis, e1, e2, mu, info)``; ``info`` carries the slice count,
    spacing, per-slice areas (mm^2) and the apex/base points used.
    """
    from skimage import measure
    from scipy import ndimage as ndi
    from ..templates.conformal_lv import resample_grid_axial
    if NT % 4:
        raise ValueError(f"NT={NT} must be divisible by 4")
    A = np.asarray(affine, float); zooms = np.asarray(zooms, float)
    k_ax = int(np.argmax(zooms))
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    pix = float(np.sqrt(np.prod(np.delete(zooms, k_ax))))

    def to_world(idx):                                   # (n, 3) voxel indices -> world mm
        return idx @ A[:3, :3].T + A[:3, 3]

    rings_raw, centres, areas, ks = [], [], [], []
    for k in range(mask.shape[k_ax]):
        sl = np.take(mask, k, axis=k_ax)
        if sl.sum() < min_pixels:
            continue
        lab, n = ndi.label(sl)
        if n > 1:
            sl = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        sl = ndi.binary_fill_holes(sl)
        cont = measure.find_contours(sl.astype(float), 0.5)
        if not cont:
            continue
        c2 = max(cont, key=len)                           # (m, 2) in the slice's own (row, col) index space
        ij = np.nonzero(sl)
        idx_c = np.zeros(3); idx_c[k_ax] = k
        others = [a for a in range(3) if a != k_ax]
        idx_c[others[0]], idx_c[others[1]] = ij[0].mean(), ij[1].mean()
        idx_o = np.zeros((len(c2), 3)); idx_o[:, k_ax] = k
        idx_o[:, others[0]], idx_o[:, others[1]] = c2[:, 0], c2[:, 1]
        rings_raw.append(to_world(idx_o)); centres.append(to_world(idx_c[None])[0])
        areas.append(float(sl.sum()) * pix ** 2); ks.append(k)
    if len(rings_raw) < 3:
        raise ValueError(f"only {len(rings_raw)} slice(s) hold the label: a ring stack needs at least 3")
    centres = np.asarray(centres); areas = np.asarray(areas)
    # apical end = the end with the smaller outline (mean of the two end slices)
    n_end = max(1, min(2, len(areas) // 3))
    if areas[:n_end].mean() > areas[-n_end:].mean():
        rings_raw, centres, areas, ks = rings_raw[::-1], centres[::-1], areas[::-1], ks[::-1]
    spacing = float(zooms[k_ax])
    stack_dir = centres[-1] - centres[0]
    stack_dir /= np.linalg.norm(stack_dir) + 1e-30
    apex_pt = centres[0] - 0.5 * spacing * stack_dir
    base_c = centres[-1] + 0.5 * spacing * stack_dir
    if frame is None:
        axis = base_c - apex_pt
        L = float(np.linalg.norm(axis)); axis = axis / L
        e1, e2 = frame_from_axis(axis)
    else:
        axis, e1, e2 = (np.asarray(v, float) for v in frame)
        axis = axis / np.linalg.norm(axis)
        L = float((base_c - apex_pt) @ axis)
    if L < 5.0:
        raise ValueError(f"apex->base length is {L:.3g} mm: too short for a ventricle (check the affine / spacing)")
    rings = [_slice_ring(r, c, e1, e2, NT) for r, c in zip(rings_raw, centres)]
    base_ring = rings[-1] + 0.5 * spacing * stack_dir
    stack = np.stack([np.repeat(apex_pt[None], NT, 0)] + rings + [base_ring])
    if interp == "pchip":
        from scipy.interpolate import PchipInterpolator
        s_rows = (stack.mean(1) - apex_pt) @ axis                 # axial position of every row centre
        if np.any(np.diff(s_rows) <= 0):
            raise ValueError("slice centres do not advance monotonically along the axis")
        G = PchipInterpolator(s_rows, stack, axis=0)(np.linspace(s_rows[0], s_rows[-1], NL))
    elif interp == "linear":
        G = resample_grid_axial(stack, 0.0, 1.0, NL, axis=axis)
    else:
        raise ValueError(f"interp must be 'pchip' or 'linear', got {interp!r}")
    if smooth:
        G = np.concatenate([G[:1], _smooth_rings(G[1:], smooth)])
    mu = G.reshape(-1, 3).mean(0)
    info = {"method": "slices", "apex": apex_pt, "base": base_c, "landmarks": "slices", "axis_length": L,
            "n_slices": len(rings), "slice_spacing_mm": spacing, "slice_indices": ks,
            "slice_areas_mm2": areas, "interpolated_rays_per_level": np.zeros(NL, int)}
    return G, axis, e1, e2, mu, info


def grid_from_file(path, NL=40, NT=56, apex=None, base=None, frame=None, label=None, **kw):
    """One entry point for every surface source.  By extension:

    * ``.npz`` -- a saved grid (``grid``, optional ``axis``/``e1``/``e2``/``mu``);
    * ``.exnode`` -- a Cardiac Atlas Project LV model (:mod:`hexmb.io.cap`);
    * ``.stl .obj .ply .off .vtk .vtp .vtu`` -- a surface mesh (ray-cast);
    * ``.nii .nii.gz`` -- a segmentation mask (marching cubes, then ray-cast).

    ``apex``/``base`` landmarks and ``frame`` apply to mesh and mask inputs (the
    ray-cast sources); ``.npz`` and ``.exnode`` carry their own parametrisation,
    so passing landmarks for them is an error rather than a silent no-op.
    ``label`` applies to masks only.  Returns ``(G, axis, e1, e2, mu, info)``.
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith((".npz", ".exnode")) and (apex is not None or base is not None):
        raise ValueError(f"{p.name}: --apex/--base landmarks apply to mesh and mask inputs only")
    if name.endswith(".npz"):
        fs = np.load(str(p))
        G = np.asarray(fs["grid"], float)
        if "axis" in fs and "e1" in fs and "e2" in fs:
            axis, e1, e2 = (np.asarray(fs[k], float) for k in ("axis", "e1", "e2"))
            mu = np.asarray(fs["mu"], float) if "mu" in fs else G.reshape(-1, 3).mean(0)
        else:
            from .cap import surface_frame
            axis, e1, e2, mu = surface_frame(G)
        return G, axis, e1, e2, mu, {"source": "npz"}
    if name.endswith(".exnode"):
        from .cap import cap_endo_surface, surface_frame
        G = cap_endo_surface(str(p), NL=NL, NT=NT)
        axis, e1, e2, mu = surface_frame(G)
        return G, axis, e1, e2, mu, {"source": "cap"}
    if name.endswith(_MASK_EXT):
        return grid_from_mask(str(p), label=label, NL=NL, NT=NT, apex=apex, base=base, frame=frame, **kw)
    if label is not None:
        raise ValueError(f"{p.name}: --label applies to mask inputs only")
    mesh = load_surface_mesh(p)
    G, axis, e1, e2, mu, info = endo_grid_from_mesh(mesh, apex=apex, base=base, NL=NL, NT=NT,
                                                    frame=frame, **kw)
    info["source"] = "mesh"
    return G, axis, e1, e2, mu, info


def write_surface_npz(path, G, axis, e1, e2, mu=None):
    """Save a grid + frame in the form ``hexmb lv --surface x.npz`` reads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(G, float).reshape(-1, 3).mean(0) if mu is None else np.asarray(mu, float)
    np.savez(str(path), grid=np.asarray(G, float), axis=np.asarray(axis, float),
             e1=np.asarray(e1, float), e2=np.asarray(e2, float), mu=mu)
