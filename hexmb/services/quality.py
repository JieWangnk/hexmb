"""Cell quality: the per-hex scaled Jacobian (min over the 8 corner tetrahedra;
<= 0 means an inverted/degenerate cell; vertex order matches the OpenFOAM hex), and
``polymesh_metrics``, checkMesh's non-orthogonality, skewness, aspect ratio and cell
volumes for any polyMesh with OpenFOAM 12's own definitions."""
from __future__ import annotations

import numpy as np

# corner-tet list keyed to the fixed HEX_OFFSETS vertex order
_CORNERS = [
    (0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3),
]


def scaled_jac(points: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Return the per-cell minimum-corner scaled Jacobian in [-1,1].
    <= 0 marks an inverted (degenerate/tangled) cell."""
    h = points[hexes]  # (ncell, 8, 3)
    mn = np.full(len(hexes), np.inf)
    for a, b, c, e in _CORNERS:
        u = h[:, b] - h[:, a]
        v = h[:, c] - h[:, a]
        w = h[:, e] - h[:, a]
        det = np.einsum("ij,ij->i", u, np.cross(v, w))
        nrm = (
            np.linalg.norm(u, axis=1)
            * np.linalg.norm(v, axis=1)
            * np.linalg.norm(w, axis=1)
            + 1e-12
        )
        mn = np.minimum(mn, det / nrm)
    return mn


# --------------------------------------------------------------------------- #
#  checkMesh-style face and cell metrics on a general polyMesh                 #
# --------------------------------------------------------------------------- #
def _face_geometry(points, faces):
    """OpenFOAM's face centres and area vectors for polygon faces: a triangle
    fan about the vertex mean, area-weighted centroid (``primitiveMeshFaceCentresAndAreas``)."""
    P = np.asarray(points, float)
    n = len(faces)
    fc = np.zeros((n, 3)); sf = np.zeros((n, 3))
    by_len: dict[int, list] = {}
    for i, f in enumerate(faces):
        by_len.setdefault(len(f), []).append(i)
    for k, idx in by_len.items():
        idx = np.asarray(idx, int)
        V = P[np.array([faces[i] for i in idx], int)]            # (m, k, 3)
        if k == 3:
            fc[idx] = V.mean(1)
            sf[idx] = 0.5 * np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
            continue
        c0 = V.mean(1)
        sumN = np.zeros((len(idx), 3)); sumA = np.zeros(len(idx)); sumAc = np.zeros((len(idx), 3))
        for j in range(k):
            a, b = V[:, j], V[:, (j + 1) % k]
            nrm = np.cross(b - a, c0 - a)
            ar = np.linalg.norm(nrm, axis=1)
            sumN += nrm; sumA += ar; sumAc += ar[:, None] * (a + b + c0)
        fc[idx] = sumAc / (3.0 * sumA[:, None] + 1e-300)
        sf[idx] = 0.5 * sumN
    return fc, sf


def _cell_geometry(fc, sf, owner, neighbour, n_cells):
    """OpenFOAM's cell centres and volumes from face centres and area vectors
    (``primitiveMeshCellCentresAndVols``): pyramids about the mean of the face centres."""
    n_int = len(neighbour)
    cEst = np.zeros((n_cells, 3)); cnt = np.zeros(n_cells)
    np.add.at(cEst, owner, fc); np.add.at(cnt, owner, 1.0)
    np.add.at(cEst, neighbour, fc[:n_int]); np.add.at(cnt, neighbour, 1.0)
    cEst /= cnt[:, None]
    pyr_o = np.einsum("ij,ij->i", sf, fc - cEst[owner])                     # 3 x pyramid volume, owner side
    pyr_n = -np.einsum("ij,ij->i", sf[:n_int], fc[:n_int] - cEst[neighbour])
    pc_o = 0.75 * fc + 0.25 * cEst[owner]
    pc_n = 0.75 * fc[:n_int] + 0.25 * cEst[neighbour]
    cc = np.zeros((n_cells, 3)); v3 = np.zeros(n_cells)
    np.add.at(cc, owner, pyr_o[:, None] * pc_o); np.add.at(v3, owner, pyr_o)
    np.add.at(cc, neighbour, pyr_n[:, None] * pc_n); np.add.at(v3, neighbour, pyr_n)
    cc /= np.where(np.abs(v3) > 1e-300, v3, 1e-300)[:, None]
    return cc, v3 / 3.0


def polymesh_metrics(points, faces, owner, neighbour, n_cells=None):
    """``checkMesh``'s face and cell metrics, computed here with the definitions
    of OpenFOAM 12 ``meshCheck`` so that the numbers agree with ``checkMesh``:

    * ``non_orthogonality_deg`` -- per internal face, the angle between the face
      area vector and the owner-to-neighbour centre vector (OpenFOAM warns above
      70 degrees and fails above 90);
    * ``skewness`` -- per face (internal and boundary), the distance from the face
      centre to where the owner-neighbour line crosses the face, normalised by
      the face size in that direction (OpenFOAM warns above 4);
    * ``aspect_ratio`` -- per cell, the larger of the component ratio of the
      summed face-area magnitudes and ``cmptSum(sumMag) / (6 V^(2/3))``;
    * ``volume`` and ``openness`` per cell.

    ``points`` in any unit (volumes come back in that unit cubed).  ``faces`` is a
    list of point-id lists, internal faces first, as in ``constant/polyMesh``.
    Returns a dict with these arrays and a ``summary`` dict (max / mean values,
    counts over the checkMesh thresholds, cell and face counts).
    """
    owner = np.asarray(owner, int); neighbour = np.asarray(neighbour, int)
    n_int = len(neighbour)
    if n_cells is None:
        n_cells = int(max(owner.max(), neighbour.max() if n_int else -1)) + 1
    P = np.asarray(points, float)
    fc, sf = _face_geometry(P, faces)
    cc, vol = _cell_geometry(fc, sf, owner, neighbour, n_cells)
    mag_sf = np.linalg.norm(sf, axis=1)

    # non-orthogonality (internal faces)
    d = cc[neighbour] - cc[owner[:n_int]]
    cosang = np.einsum("ij,ij->i", d, sf[:n_int]) / (np.linalg.norm(d, axis=1) * mag_sf[:n_int] + 1e-300)
    nonorth = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    # skewness: internal faces
    Cpf = fc[:n_int] - cc[owner[:n_int]]
    sv = Cpf - (np.einsum("ij,ij->i", sf[:n_int], Cpf)
                / (np.einsum("ij,ij->i", sf[:n_int], d) + 1e-300))[:, None] * d
    fd = 0.2 * np.linalg.norm(d, axis=1) + 1e-300
    # boundary faces: d is the normal projection of Cpf
    Cpf_b = fc[n_int:] - cc[owner[n_int:]]
    nb = sf[n_int:] / (mag_sf[n_int:, None] + 1e-300)
    d_b = nb * np.einsum("ij,ij->i", nb, Cpf_b)[:, None]
    sv_b = Cpf_b - (np.einsum("ij,ij->i", sf[n_int:], Cpf_b)
                    / (np.einsum("ij,ij->i", sf[n_int:], d_b) + 1e-300))[:, None] * d_b
    fd_b = 0.4 * np.linalg.norm(d_b, axis=1) + 1e-300
    sv_all = np.vstack([sv, sv_b]); fd_all = np.concatenate([fd, fd_b])
    svhat = sv_all / (np.linalg.norm(sv_all, axis=1)[:, None] + 1e-300)
    for i, f in enumerate(faces):                     # fd = max(fd, |svHat . (p - fC)|) over the face's points
        proj = np.abs((P[f] - fc[i]) @ svhat[i])
        m = proj.max()
        if m > fd_all[i]:
            fd_all[i] = m
    skew = np.linalg.norm(sv_all, axis=1) / fd_all

    # cell closedness: aspect ratio and openness
    sum_closed = np.zeros((n_cells, 3)); sum_mag = np.zeros((n_cells, 3))
    np.add.at(sum_closed, owner, sf); np.add.at(sum_closed, neighbour, -sf[:n_int])
    np.add.at(sum_mag, owner, np.abs(sf)); np.add.at(sum_mag, neighbour, np.abs(sf[:n_int]))
    # checkMesh's openness: the worst component of |sum Sf| / sum |Sf| (primitiveMeshCheck.C)
    openness = (np.abs(sum_closed) / (sum_mag + 1e-300)).max(1)
    v = np.maximum(np.abs(vol), 1e-300)
    aspect = np.maximum(sum_mag.max(1) / (sum_mag.min(1) + 1e-300),
                        sum_mag.sum(1) / (6.0 * v ** (2.0 / 3.0)))

    summary = dict(
        n_cells=int(n_cells), n_faces=len(faces), n_internal_faces=int(n_int), n_points=len(P),
        max_non_orthogonality_deg=float(nonorth.max()) if n_int else 0.0,
        mean_non_orthogonality_deg=float(nonorth.mean()) if n_int else 0.0,
        # checkMesh's 'average' is the angle whose cosine is the mean cosine
        average_non_orthogonality_deg=float(np.degrees(np.arccos(min(1.0, float(cosang.mean()))))) if n_int else 0.0,
        n_non_orthogonality_gt70=int((nonorth > 70).sum()), n_non_orthogonality_gt90=int((nonorth > 90).sum()),
        max_skewness=float(skew.max()), max_skewness_internal=float(skew[:n_int].max()) if n_int else 0.0,
        max_skewness_boundary=float(skew[n_int:].max()) if len(faces) > n_int else 0.0,
        n_skewness_gt4=int((skew > 4).sum()),
        max_aspect_ratio=float(aspect.max()), max_openness=float(openness.max()),
        min_volume=float(vol.min()), max_volume=float(vol.max()), total_volume=float(vol.sum()),
        n_negative_volume=int((vol <= 0).sum()),
    )
    return dict(non_orthogonality_deg=nonorth, skewness=skew, aspect_ratio=aspect, volume=vol,
                openness=openness, face_centres=fc, face_areas=sf, cell_centres=cc, summary=summary)


def polymesh_metrics_of(mesh):
    """:func:`polymesh_metrics` of a :class:`hexmb.assembly.splithex.PolyMesh`
    (points in mm, so volumes in mm^3), using the same face topology the writer emits."""
    faces, owner, neigh, _ = mesh.topology()
    return polymesh_metrics(mesh.points, faces, owner, neigh, n_cells=len(mesh.cells))


def metrics_line(summary: dict) -> str:
    """One printable line of the checkMesh-style summary."""
    return (f"non-orthogonality max {summary['max_non_orthogonality_deg']:.1f} deg "
            f"(average {summary['average_non_orthogonality_deg']:.1f}, {summary['n_non_orthogonality_gt70']} faces > 70), "
            f"skewness max {summary['max_skewness']:.2f} ({summary['n_skewness_gt4']} faces > 4), "
            f"aspect ratio max {summary['max_aspect_ratio']:.1f}, "
            f"{summary['n_negative_volume']} cells with non-positive volume")
