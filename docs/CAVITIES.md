# Static cavity meshing

`hexmb cavity` separates cavity geometry from the LV valve convention. It writes
an all-hexahedral static OpenFOAM mesh, a machine-readable `mesh_report.json`,
and optionally ParaView files. It does not prescribe velocity, pressure,
viscosity or a solver. The `lid` patch is a wall patch, not a velocity condition.

## Rectangular cavities

```bash
# 3D box; dimensions in metres; counts are CELLS in x, y, z
hexmb cavity --size 1 1 1 --units m --cells 32 32 32 -o cavity_3d

# 2D lid-driven-cavity mesh, one cell through thickness
hexmb cavity --size 1 1 0.01 --units m --cells 64 64 1 --two-d \
  --stretch 1.5 1.5 0 --vtk -o cavity_2d
hexmb verify -o cavity_2d
```

The box starts at the origin. `left/right` are x-min/x-max, `bottom/lid` are
y-min/y-max, and `front/back` are z-min/z-max. All six patches are `wall` in 3D;
`front/back` become `empty` with `--two-d`. Set matching `empty` conditions in
your solver fields. The other fields and solver dictionaries must be supplied
before running CFD; the generated controlDict is only a minimal mesh utility
scaffold. Do not use the cardiac `hexmb flow` command on these cases.

`--stretch BX BY BZ` clusters nodes towards both walls on each axis with a
symmetric tanh distribution. Zero is uniform; allowed values are 0–5. Clustering
changes cell sizes without changing the box or the number of cells. Inspect
aspect ratio in the report, especially for thin 3D domains.

For a refinement study, generate separate cases with proportionally increased
cell counts, holding size and stretch fixed. Then assess convergence of the
flow quantity of interest; passing a mesh gate is not a grid-convergence result.

## Smooth cavities

```bash
# Sac surface in millimetres, explicitly truncate the pointed tip at 5% of length
hexmb cavity --surface sac.stl --units mm --start 0.05 --ends open --vtk -o sac_mesh

# Ordered rings with a finite first and last section; close both caps
hexmb cavity --surface rings.npz --units mm --ends closed -o closed_mesh
```

The surface reader supports STL, OBJ, PLY, OFF and supported VTK surface formats;
install `hexmb[surface]`. It currently reuses the sac-oriented ray caster: use a
single smooth elongated sac with a meaningful start/end axis, not arbitrary CAD.
NIfTI and CAP inputs remain in the existing `lv` workflow. For a tube with two
open ends, use ordered rings or the existing `pipe`/`elbow`/`stl` commands.

The NPZ contains `grid` of shape `(stations, angles, 3)`, preferably with explicit
unit vectors `axis`, `e1`, `e2`. Angles must be divisible by four. Each ring must
be planar, star-shaped about its centre and consistently ordered; ring centres
must advance along the supplied axis. Frames must be orthonormal and
right-handed. Non-planar sections, folded rows, disconnected components and
collapsed cap rings are rejected. This is not a complete surface-intersection
test; inspect the reconstructed boundary as well as its mesh quality.

| End rule | First cap | Last cap | Side |
|---|---|---|---|
| `closed` | wall | wall | wall |
| `open` | wall | opening | wall |
| `through` | inlet | outlet | wall |

Caps are flat discs at the retained end sections. The command does not create
anatomical valve sectors. `--start` and `--end` explicitly select retained axial
fractions; default 0 and 1 preserve the supplied extent. A point at a pole cannot
be represented by this single O-grid and must be explicitly truncated. This
removes volume and changes the physical geometry. Check that effect for your
application rather than choosing a trim only to pass the quality gate.

Surface reconstruction still performs the existing ray interpolation and ring
smoothing; it is not an exact CAD projection. Sampling diagnostics, including
interpolated-ray counts, are recorded. Optional `--apex x,y,z --base x,y,z`
landmarks are in the source units. Automatic landmarks are only an estimate.
The report's sampling coordinates and axis length use internal millimetres.
The `--NL/--NT` settings apply to surface sampling; an NPZ uses its stored grid.

## Acceptance and reproducibility

Source units are required (`mm`, `cm`, `m`); all exported coordinates are metres.
The JSON report stores mesh metrics in metres (volumes in m³), input options,
source SHA-256 for file inputs, the retained fractions, and boundary types.
When a triangulated source is watertight and consistently wound, its enclosed
volume is also recorded for comparison with the reconstructed mesh volume.
An open surface has no unambiguous enclosed input volume.

Acceptance requires finite geometry/metrics, positive corner scaled Jacobians,
positive cell volumes, nonzero face areas, closed cells, and the requested
non-orthogonality/skewness limits. Defaults are 70° and 4. They are engineering
screening settings, not a guarantee for a particular solver. Adjust with
`--max-nonorth` (strictly below 90°) and `--max-skew` only with a justified reason.
Aspect ratio is reported but not limited.

A completed quality assessment that fails writes the report and returns exit 1
without exporting a polyMesh. Invalid inputs fail before report creation. A
non-empty output directory is refused, so a failed rerun cannot leave an old
mesh appearing to be a new result. Use a new directory for each refinement.
`hexmb verify` now returns failure on readback errors, invalid built-in geometry
checks, and unsuccessful external `checkMesh` runs. If OpenFOAM is absent, it
explicitly reports that the external check was skipped.

The general cavity command is static. The existing `motion` workflow retains
its LV-specific behaviour. Branched cavities, internal obstacles, handles,
re-entrant sections and arbitrary sharp CAD features need other block layouts
or another mesher. Rectangular corners are supported by the box template only.
