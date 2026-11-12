"""Static cavity command: explicit units, fail-closed quality and audit report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..templates.cavity import BoxCavityTemplate, RingCavityTemplate
from .export_foam import build_faces, write_polymesh
from .quality import polymesh_metrics, metrics_line


def cmd_cavity(a):
    from ..cli import _write_min_system

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} is not empty; choose a new output directory")
    for name in ("max_nonorth", "max_skew"):
        value = getattr(a, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if a.max_nonorth >= 90:
        raise ValueError("max_nonorth must be below 90 degrees")
    factor = {"mm": 1., "cm": 10., "m": 1000.}[a.units]
    source = {}
    if a.size is not None:
        if a.start != 0 or a.end != 1 or a.ends != "closed" or a.apex or a.base:
            raise ValueError("start/end/ends/apex/base apply to surface cavities only")
        template = BoxCavityTemplate(np.asarray(a.size) * factor, a.cells, a.stretch, a.two_d)
        source = dict(kind="box", size=a.size, cells=a.cells, stretch=a.stretch, two_d=a.two_d)
    else:
        if a.two_d or a.cells != [24, 24, 24] or a.stretch != [0., 0., 0.]:
            raise ValueError("two-d/cells/stretch apply to box cavities only")
        from ..io.surface import grid_from_file, load_surface_mesh, endo_grid_from_mesh
        from ..cli import _xyz
        path = Path(a.surface)
        if a.NL < 6 or a.NT < 4 or a.NT % 4:
            raise ValueError("NL must be >= 6 and NT >= 4 divisible by 4")
        source = dict(kind="surface", path=str(path.resolve()), ends=a.ends,
                      retained_axial_fractions=[a.start, a.end],
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      requested_sampling=dict(NL=a.NL, NT=a.NT),
                      meshing=dict(NR=a.NR, core_frac=a.core_frac, wall_ratio=a.wall_ratio))
        if path.suffix.lower() == ".npz":
            G, axis, e1, e2, _, info = grid_from_file(path, apex=_xyz(a.apex), base=_xyz(a.base))
            G = G * factor
        else:
            surface = load_surface_mesh(path)
            surface.apply_scale(factor)
            if not np.isfinite(surface.vertices).all() or len(surface.faces) == 0:
                raise ValueError("surface is empty or has non-finite vertices")
            if len(surface.split(only_watertight=False)) != 1:
                raise ValueError("surface must have exactly one connected component")
            if surface.is_watertight and surface.is_winding_consistent:
                source["input_enclosed_volume_m3"] = float(abs(surface.volume) * 1e-9)
            ap = None if a.apex is None else _xyz(a.apex) * factor
            ba = None if a.base is None else _xyz(a.base) * factor
            G, axis, e1, e2, _, info = endo_grid_from_mesh(surface, apex=ap, base=ba, NL=a.NL, NT=a.NT, check_lv_scale=False)
        source["sampling"] = info
        template = RingCavityTemplate(G, axis, e1, e2, NR=a.NR, core_frac=a.core_frac,
                                      wall_ratio=a.wall_ratio, ends=a.ends, start=a.start, end=a.end)
    mesh = template.build()
    sj = mesh.scaled_jacobian()
    if not np.isfinite(mesh.points).all() or not np.isfinite(sj).all():
        raise ValueError("non-finite mesh coordinates or scaled Jacobians")
    faces, owner, neighbour, boundary = build_faces(mesh.points, mesh.hexes)
    faces += [face for _, face, _ in boundary]
    owner += [own for own, _, _ in boundary]
    # Use metres in the report, independently of the source coordinate unit.
    metrics = polymesh_metrics(mesh.points * 1e-3, faces, owner, neighbour)
    q = metrics["summary"]
    valid = bool(np.all(sj > 0) and all(np.isfinite(v) for v in q.values())
                 and q["n_negative_volume"] == 0 and q["max_openness"] <= 1e-6
                 and np.all(np.linalg.norm(metrics["face_areas"], axis=1) > 0))
    accepted = bool(valid and q["max_non_orthogonality_deg"] <= a.max_nonorth
                    and q["max_skewness"] <= a.max_skew)
    report = dict(schema_version=1, input_units=a.units, output_units="m", source=source,
                  geometrically_valid=valid, accepted=accepted,
                  limits=dict(max_nonorth_deg=a.max_nonorth, max_skewness=a.max_skew),
                  min_scaled_jacobian=float(sj.min()), metrics=q,
                  patch_types=template.patch_types,
                  scope="Static mesh only; no flow solution or mesh-independence claim.")
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    # Strict JSON is also a guard against non-finite results in the audit record.
    report_text = json.dumps(report, indent=2, default=serial, allow_nan=False) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh_report.json").write_text(report_text)
    print(f"cavity: {len(mesh.hexes)} cells, min scaled-Jac {sj.min():.4g}")
    print("  " + metrics_line(q))
    if not accepted:
        print(f"  REJECTED: no polyMesh written; see {out / 'mesh_report.json'}")
        return 1
    patch_of, _ = mesh.patch_key_map()
    write_polymesh(out / "constant" / "polyMesh", mesh.points, mesh.hexes,
                   patch_of, template.patch_types)
    _write_min_system(out)
    if a.vtk:
        from ..io.vtk import case_to_vtk
        case_to_vtk(out)
    print(f"  ACCEPTED: wrote {out}/constant/polyMesh and mesh_report.json (coordinates in metres)")
    return 0
