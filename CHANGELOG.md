# Changelog

## 0.2.0 (unreleased, November 2026)

- **VTK and Abaqus export** (`hexmb.io.vtk`, `hexmb export`, `--vtk` on `lv`
  / `motion` / `demo`): any written case becomes `mesh_<time>.vtu` per phase
  (hexahedra and the split-hex transition cells as VTK polyhedra, point
  displacement and cell volume attached), one `boundary_<patch>.vtp` per
  patch and a `motion.pvd` time series, so the moving mesh opens in ParaView
  with no OpenFOAM reader; base64 binary (86 MB, 30 s for the 20-phase demo)
  or `--ascii`. All-hex meshes also write an Abaqus / CalculiX `.inp` (C3D8,
  node sets per patch) for a solid or FSI solver; the split-hex template is
  refused there with a message, since C3D8 cannot hold a nine-face cell. No
  VTK library is needed.
- **Built-in checkMesh-style metrics** (`hexmb.services.quality.polymesh_metrics`):
  non-orthogonality (max, average, faces > 70 deg), skewness (OpenFOAM's
  internal and boundary definitions, faces > 4), aspect ratio and signed
  volumes, computed with OpenFOAM's own face-centre and cell-centre
  constructions, so they match `checkMesh` to the printed precision without
  OpenFOAM installed. `hexmb verify` prints them for any case; `--metrics` on
  `motion` / `demo` / `lv` adds them per phase to `motion_quality.csv` (the
  face topology is built once and reused, about 0.3 s per phase at the
  default size).
- **Wall-driven flow case** (`hexmb.services.flow`, `hexmb flow`): a
  moving-mesh case becomes a runnable `foamRun -solver incompressibleFluid`
  case for a diastolic (mitral open) or systolic (aortic open) window, with
  the standard prescribed-motion boundary conditions, blood viscosity, a
  Courant-limited PIMPLE setup tuned for the split-hex transition, and
  function objects for the valve flow rate, peak speed, kinetic energy, wall
  shear stress and `checkMesh` at write times; `Allrun` decomposes, runs in
  parallel and reconstructs.  The decomposition uses `-time '0:'` so every
  motion-table folder reaches the processors: a plain `decomposePar` copies
  only the start time, the `interpolator` mover then finds no later
  `wantedDisplacement` and the mesh never moves (Courant number 0, zero valve
  flow, no error message).  `Allrun` now checks the processor time folders and
  the final Courant number and exits with a CHECK line when the mesh did not move.
- **Motion table interpolated by a spline.** The `interpolator` mover now gets
  `interpolationScheme spline` (Catmull-Rom through the phase table) instead of
  `linear`.  Linear interpolation holds the wall velocity constant between
  phases and jumps it at every table time, so the valve flow of a 20-phase
  table was a staircase with steps of up to 97 mL/s and the peak speed spiked
  after each jump; the spline gives a continuous wall velocity, the same
  filled volume, and a smooth flow curve (`docs/VERIFICATION.md`).
  `--table-interpolation linear` on `motion` / `demo` restores the old
  behaviour (`interpolation=` in `build_lv_motion`).
- **Default apex trim 0.01 -> 0.02.** At 1 % the disc that caps the trimmed
  tip was narrower than the first axial layer is thick, the cells on its rim
  were flat, and 12 wall faces of the bundled heart had skewness above 4 at
  every phase (`checkMesh` reported "highly skew faces"). At 2 % the disc is
  about as wide as the layer: max skewness 2.3-2.5 and max non-orthogonality
  58 deg instead of 4.7-4.9 and 64 deg, for 0.1 mL (0.08 %) less cavity.
  The demo now passes `checkMesh` with no warning at end-systole.
- **Surface readers read the whole cavity and keep its volume.** The
  ray-cast reader (STL / OBJ / VTK / marching-cubes masks) stopped 3 % short
  of the apex and 2 % short of the base; its grid now starts at the apex
  point, casts two stations inside the first 3 % so the dome at the tip is
  resolved (a straight loft from the tip made a cone there and skewed the
  apex cells), and ends with the last ring moved onto the base plane, as the
  Cardiac Atlas and mask readers already did. The per-phase STL route on the
  bundled heart now gives the same volumes as the model route (EDV 173.3
  against 173.1 mL; it gave 168.9 before). The circumferential 3-point
  smoothing of every reader shrank each ring by about 0.9 % in area per
  pass; it now rescales each ring about its centroid to the area it had, so
  the noise is averaged and the volume kept. On a 1 mm isotropic synthetic
  mask of the bundled heart the mesh went from 4.4 % to 2.0 % below the exact
  cavity, on a 10 mm short-axis stack from 1.2 % to 0.4 % at end-diastole.
- **Slice-by-slice mask reader** (`hexmb.io.surface.grid_from_slices`,
  `--mask-method slices|marching|auto`): a short-axis cine stack with thick
  slices (the clinical case: 1.4 mm pixels, 8-10 mm slices) gives one ring
  per slice from the blood-pool outline about its own centroid, holes filled
  (papillary muscles count as cavity, the clinical rule), the apex point half
  a slice beyond the apical slice and the base ring half a slice beyond the
  basal one, so the enclosed volume is the stack's Simpson's-rule volume, and
  a monotone cubic (PCHIP) through the rings for the rows between slices.
  Marching cubes + ray cast on such a stack used to fail where a ring crossed
  the staircase; it remains the route for isotropic masks (CT). `auto` picks
  `slices` when the slice spacing is at least 2.5 x the pixel.
- **The builder reports what its trims cost**: when it moves `base_cut` or
  `apex_trim`, every quality row gains `trim_removed_mL` (the surface volume
  above the cut and below the trim) and `motion_adjustments.txt` states the
  removed volume and its fraction at the largest and smallest phase, so a
  cohort table can carry the cut and its cost side by side (`DATA.md`).
- **Verification study** (`docs/VERIFICATION.md`, `scripts/verification_study.py`,
  `scripts/verification_figures.py`, `scripts/synthetic_masks.py`,
  `scripts/synthetic_mask_figures.py`, `scripts/flow_slice_render.py`): three
  meshes of the bundled heart refined 1.5x per direction, exact cavity volume
  from the analytic Atlas surface, Richardson extrapolation and grid-convergence
  indices for the volume and for a wall-driven flow solve on each mesh
  (valve flow, peak speed, kinetic energy), checkMesh at every write time,
  and a synthetic-mask test of the readers at three scanner resolutions.
- **Fixes from the second code review (8 Nov 2026).** (1) A filled reference
  phase of the `single` builder was written with its un-filled (failed) points
  while every displacement was measured from the filled ones, so no phase landed
  on its surface; both builders now copy the filled points into the mesh before
  writing. (2) The reference phase was chosen among all phases, so with the
  fill off a failed end-systolic mesh could become `constant/polyMesh`; it is
  now chosen among the phases that passed or were filled. (3) The valve-plane
  classifier froze the cap centre at build time; it now re-reads it from the
  current points when the boundary is classified (`classify.prepare()`), which
  on the bundled heart moves 11 cap faces between the mitral and aortic patches
  when a mesh built at end-diastole is written with end-systolic points.
  (4) A one-layer coarse zone between two finer zones split only one of its two
  end faces; `emit_ogrid_cell` now emits the 12-face both-split cell. (5) The
  "faces > 70 deg" column of the verification study was a constant 0; the count
  is now carried per phase (`n_non_orthogonality_gt70`) and read by the study
  (the bundled heart has none at any level, max 58-67 deg). (6) The cell
  openness metric now follows `checkMesh` (worst component, so an open cell
  reads 1.0 where it read 0.2). (7) `hex_volumes` is signed again (a mirrored
  hex is negative). (8) Zero-area faces are counted once, not once per cell.
  (9) `hexmb motion` / `hexmb demo` honour `--allow-invalid`: a case with a
  failed phase is no longer written without it (`write_invalid=` in
  `build_lv_motion`); `hexmb lv` reports and refuses a surface whose rings
  cross the axis away from the rims. (10) A malformed `--apex`/`--base`, a
  truncated `.exelem` and a case path with a space in `hexmb verify` all end
  in a one-line error (OpenFOAM itself refuses a path with a space; `verify`
  now says so).
- Tests: 74 (`tests/test_export_flow.py`: metrics against hand-computed cells,
  openness and the > 70 deg count, VTU round trip, Abaqus writer, flow-case
  windows; a slice-reader test on a synthetic thick-slice stack; the filled
  reference phase for both builders, the both-split cell, the classifier on
  replaced points, the CLI `--allow-invalid` path).

- **Conformal split-hex telescoping LV** (`hexmb.assembly.splithex`,
  `hexmb.templates.conformal_lv`): coarse apex cap, 2x-finer body and coarse
  valve cap joined by conformal 2:1 transitions, so the cavity is ONE
  flux-conservative region. This is the mesh behind the validated moving-ventricle
  CFD runs; it was research code outside the package until now.
- **Bring your own surface** (`hexmb.io.surface`): STL/OBJ/PLY/OFF, VTK/VTP/VTU,
  NIfTI segmentation masks, Cardiac Atlas models and saved grids all become the
  `(NL, NT, 3)` ring grid by recentred ray-casting with a fixed frame, with
  automatic apex/base landmarks. `hexmb lv --surface <file>`.
- **Moving mesh** (`hexmb.services.motion`, `hexmb motion`): per-phase surfaces
  -> the same mesh rebuilt at every phase, gated, written as an OpenFOAM 12
  `interpolator` case (`wantedDisplacement` per phase time, zero at the reference).
- **Bundled CC0 heart + `hexmb demo`**: the 20 phases of Cardiac Atlas subject
  SCD0000101 ship inside the package (170 KB), so the LV demo runs with no download.
- `hexmb verify` reads polyhedral meshes too; package renamed `hexmb` on PyPI
  (was `hex-multiblock`); scipy is now a core dependency.
- Vectorised polyhedral cell volumes (20x faster gate on split-hex meshes).
- **Gate fixed after an independent code review (6 Nov 2026).** The split-hex
  cell volume was unsigned (faces were flipped outward before summing), so a
  mirrored or folded cell passed; `checkMesh` cannot catch that either once the
  writer has oriented the faces. Cells are now emitted with outward faces (the
  contour direction is normalised once per build) and the gate is the signed
  face-pyramid volume of every face (`PolyMesh.pyramid_volumes`,
  `min_pyramid_per_cell`). A folded band of rings on the bundled heart now fails
  with 5 184 bad cells where it passed before.
- The valve-plane patches come from faces tagged at build time
  (`PolyMesh.top_faces`), not from a 3 mm depth below the highest point, so a
  tilted base ring keeps its whole cap.
- The single O-grid builder in `hexmb motion` now gates on the scaled Jacobian
  and trims a point apex (`apex_trim`, 2 % by default) like the conformal one.
- Input checks that used to fail silently now raise with a hint: a surface in
  metres, a multi-label mask without `--label`, more than a quarter of rays
  interpolated at a level, `--apex/--base` or `--label` for a source they do not
  apply to, out-of-order axial zones, a face shared by three cells, a single
  phase. `hexmb motion` refuses to delete an output directory that is not a case.
- `hexmb render`: long-axis cut of a moving-mesh case through the cycle as GIF
  and PNG strip (`hexmb.services.render.render_case_slices`), with
  `hexmb.io.foam` readers for what hexmb writes.
- CLI: `--core-frac` no longer overrides the conformal default; the single
  template uses the same valve sectors and wall ratio as the conformal one;
  errors print one line instead of a traceback. Tests: 50.
- **Cardiac Atlas reader rewritten to evaluate the model exactly**
  (`hexmb.io.cap`): bicubic Hermite `lambda` per element with the `.exelem`
  scale factors, bilinear `mu`, sampled on the model's own longitudinal
  parameter and uniform `theta`. The 40 nodes are reproduced to 1e-4 mm and
  every ring winds once about the model axis. The old reader interpolated node
  positions in Cartesian space, which bent the first apex rings into crescents
  (the cause of the apex failures in the cohort) and under-read the cavity by
  3-30 % (median 8 %, most at end-systole); the bundled heart's EDV/ESV move
  from 163.7/113.1 to 173.3/123.2 mL. The element template ships in the package
  (`hexmb/data/cap/GlobalHermiteParam.exelem`). The reader warns when a model
  crosses its own long axis (`lambda <= 0`), naming the rows to drop.
- The AMI telescoping template trims 4 apex rings and uses 24 apex levels (was
  2 and 40): the exact surface has a flatter apex dome, on which the old
  defaults inverted 40 tip cells of the worked example.
- **Levels are now spaced uniformly along the long axis** in the conformal and
  single builders (`resample_grid_axial(..., axis=)`), not by the surface's row
  index. A dome sampled at uniform polar angle (every Cardiac Atlas model)
  crowds its rows at the tip, so the first apex layer used to be a flat wedge a
  tenth of a millimetre thick that the smoother could fold; `apex_frac`,
  `base_frac`, `apex_trim` and `base_cut` are fractions of the axial length as
  their docstrings always said. Default `apex_trim` was 0.01 (about 1 mm) at this
  point; it is 0.02 since the apex-trim entry above. On the
  bundled heart the worst cell's pyramid volume goes from 0.011 to 0.08 mm^3
  with the same cavity volume. Rows beyond a rim that folds back along the axis
  are dropped with a warning instead of producing inverted cells.
- **The moving-mesh builder screens the rings and trims the rims itself**
  (`bad_rings`, `build_phases`; `auto_trim` and `fill_failed` in
  `build_lv_motion`, `--no-auto-trim` / `--no-fill` on the CLI). A ring that
  passes through the long axis, overlaps itself, runs the wrong way or collapses
  to a point cannot be meshed: a block of such rows at the base or apex moves
  `base_cut` / `apex_trim` to the last good row for every phase; cells that
  fail the gate at a rim trim it by one step and rebuild every phase (at most
  three times); a phase whose surface is degenerate away from the rims is
  replaced by the linear interpolation in time of its nearest valid neighbours
  and gated again. The quality table gains `status` (`ok` / `filled` / `failed`)
  and `note` columns and every change is written to `motion_adjustments.txt`.
  `hexmb lv` runs the same screen on a single surface; the single template
  takes `base_cut` too. On the Sunnybrook cohort this replaces the hand-set
  knobs: 40 of 45 subjects build exactly as given, 4 are trimmed by the builder
  (2-8 % of the base), 1 has one phase filled, 45 of 45 cases are valid.
- Docs: `docs/DEMO.md` step-by-step walkthrough; `docs/OPEN_LV_DATA.md` open
  per-phase LV geometry sources; the 45-subject Sunnybrook cohort in `DATA.md`
  with what the builder did per subject and a figure of the filled phase.

## 0.1.0 (October 2026)

- O-grid engine, pipe/elbow/STL tube templates, single and AMI-telescoping LV
  templates, Cardiac Atlas reader, OpenFOAM polyMesh export, RBF morphing.
