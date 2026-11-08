"""Matplotlib-only pictures of what the LV tools produced (no VTK/ParaView
needed, so they work in a notebook or on a headless box)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_surface_grid", "render_motion", "slice_segments", "render_case_slices"]


def _view_basis(G, elev=15.0, azim=-60.0):
    """Screen axes (u right, w up) for an oblique view of a ring grid: the long
    axis (apex -> base) points up, the camera sits ``elev`` degrees above the
    equator at azimuth ``azim``.  Plain 2D projection, no 3D toolkit needed."""
    G = np.asarray(G, float)
    axis = G[-1].mean(0) - G[0].mean(0)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    t = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    el, az = np.radians(elev), np.radians(azim)
    view = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * axis
    u = np.cross(axis, view); u /= (np.linalg.norm(u) + 1e-30)
    w = np.cross(view, u)
    return u, w


def plot_surface_grid(ax, G, color="k", lw=0.4, every=1, elev=15.0, azim=-60.0):
    """Wireframe of an ``(NL, NT, 3)`` grid on an ordinary 2D axes, drawn as an
    oblique projection (see :func:`_view_basis`).  Rings and meridians are
    plotted every ``every`` lines."""
    G = np.asarray(G, float)
    NL, NT, _ = G.shape
    u, w = _view_basis(G, elev, azim)
    X, Y = G @ u, G @ w
    for i in range(0, NL, every):
        ax.plot(np.append(X[i], X[i, 0]), np.append(Y[i], Y[i, 0]), color=color, lw=lw)
    for j in range(0, NT, every):
        ax.plot(X[:, j], Y[:, j], color=color, lw=lw)
    ax.set_aspect("equal")
    ax.set_axis_off()


def render_motion(grids, res, times, out_png):
    """Three panels: the surface at the largest and smallest cavity, and the
    cavity-volume curve with the per-phase quality gate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vols = np.asarray(res["volumes_mL"])
    ied, ies = int(vols.argmax()), int(vols.argmin())
    fig = plt.figure(figsize=(12, 4.2))
    for k, (idx, title) in enumerate([(ied, "largest cavity (ED)"), (ies, "smallest cavity (ES)")]):
        ax = fig.add_subplot(1, 3, k + 1)
        plot_surface_grid(ax, grids[idx], color="0.25", every=2)
        ax.set_title(f"phase {idx}: {title}\n{vols[idx]:.0f} mL", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(times, vols, "o-", color="#660099")
    bad = [r["phase"] for r in res["quality"] if r["n_nonpositive"] > 0]
    if bad:
        ax.plot(np.asarray(times)[bad], vols[bad], "rx", ms=10, label="non-positive cells")
        ax.legend(frameon=False)
    ax.axvline(times[res["ref"]], color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("cavity volume (mL)")
    ax.set_title(f"EF {100 * (1 - vols.min() / vols.max()):.0f} %, every phase "
                 f"{'valid' if res['valid'] else 'INVALID'}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def slice_segments(points, faces, origin, normal, e_u, e_v):
    """Cut a face list with a plane: every face crossing the plane gives one
    segment, returned as ``(n, 2, 2)`` coordinates in the plane's ``(e_u, e_v)``
    frame.  Faces are grouped by vertex count so the work is vectorised."""
    points = np.asarray(points, float)
    d = (points - origin) @ normal
    by_len = {}
    for f in faces:
        by_len.setdefault(len(f), []).append(f)
    segs = []
    for k, fl in by_len.items():
        F = np.asarray(fl, int)
        dv = d[F]
        nxt = np.roll(np.arange(k), -1)
        cross = (dv * dv[:, nxt]) < 0
        rows = np.nonzero(cross.sum(1) == 2)[0]
        for row in rows:
            P = []
            for e in np.nonzero(cross[row])[0]:
                a, b = F[row, e], F[row, nxt[e]]
                t = d[a] / (d[a] - d[b])
                P.append(points[a] + t * (points[b] - points[a]))
            segs.append([((p - origin) @ e_u, (p - origin) @ e_v) for p in P])
    return np.asarray(segs, float).reshape(-1, 2, 2)


def render_case_slices(case_dir, out_gif=None, out_png=None, n_panels=4, duration_ms=120,
                       field="wantedDisplacement"):
    """Long-axis cut of a hexmb moving-mesh case through one cycle: a GIF (one
    frame per table time from ``startTime``) and/or a PNG strip of ``n_panels``
    frames.  The cut plane holds the mesh's first two principal directions
    (PCA of the reference points), so no frame file is needed.  Returns a list of
    ``(time_s, cavity_mL)`` rows.  Needs matplotlib (and Pillow for the GIF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from ..io.foam import read_polymesh, read_point_field, controldict_value, case_time_dirs

    case_dir = Path(case_dir)
    pts0, faces, owner, neigh = read_polymesh(case_dir / "constant" / "polyMesh")
    pts0 = pts0 * 1e3                                             # m -> mm
    nb = len(neigh)
    t_ref = controldict_value(case_dir, "startTime")
    tdirs = [d for d in case_time_dirs(case_dir, field) if float(d.name) >= t_ref - 1e-9]
    if not tdirs:
        raise ValueError(f"{case_dir}: no time folders with {field} at or after startTime {t_ref}")
    c = pts0.mean(0)
    w, V = np.linalg.eigh(np.cov((pts0 - c).T))
    axis, e1, nrm = V[:, 2], V[:, 1], V[:, 0]
    # all phases on common axes
    disp = [read_point_field(d / field) * 1e3 for d in tdirs]
    allp = np.concatenate([pts0 + dd for dd in disp])
    u, v = (allp - c) @ e1, (allp - c) @ axis
    lim = ((u.min() - 3, u.max() + 3), (v.min() - 3, v.max() + 3))

    def cavity_mL(pts):
        tot = 0.0
        for k in {len(f) for f in faces}:
            F = np.asarray([f for f in faces[nb:] if len(f) == k], int)
            if not len(F):
                continue
            q = pts[F]; fc = q.mean(1)
            sf = 0.5 * sum(np.cross(q[:, i] - fc, q[:, (i + 1) % k] - fc) for i in range(k))
            tot += np.einsum("ij,ij->i", fc, sf).sum() / 3.0
        return abs(tot) * 1e-3

    frames, rows = [], []
    for d, dd in zip(tdirs, disp):
        pts = pts0 + dd
        segs = slice_segments(pts, faces, c, nrm, e1, axis)
        vol = cavity_mL(pts)
        rows.append((float(d.name), vol))
        fig, ax = plt.subplots(figsize=(4.2, 5.6))
        ax.add_collection(LineCollection(segs, colors="0.2", lw=0.45))
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"t = {float(d.name) - t_ref:.2f} s   cavity {vol:.0f} mL", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    if out_gif:
        from PIL import Image
        ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
        Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
        ims[0].save(out_gif, save_all=True, append_images=ims[1:], duration=duration_ms, loop=0)
    if out_png:
        sel = [int(round(q * (len(frames) - 1))) for q in np.linspace(0, 1, n_panels, endpoint=False)]
        fig, axs = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 4.4))
        for ax, k in zip(np.atleast_1d(axs), sel):
            ax.imshow(frames[k]); ax.set_axis_off()
        fig.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
    return rows
