# hexmb — Methods

A formal description of the mesh methods in `hexmb`: the structured O-grid
engine, the sweep that reuses it across shapes, the moving-mesh morph, and the
conformal split-hex transition that lets one mesh carry two resolutions in a
single conservative region. This document states the method, the exactness
conditions it depends on, how it is verified, and its measured limits.

Status: `v0.1.0`, research prototype with code. Tubes and single chambers are
validated (checkMesh-clean); the split-hex/telescoping path is a working
prototype with documented residual skewness on strongly-curved petals.

---

## 1. Scope and design goal

`hexmb` builds all-hexahedral (and, for the split-hex path, hex-dominant)
meshes for CFD and writes them directly as an OpenFOAM `constant/polyMesh`. The
target is single-lumen tubes and single cavities whose cross-sections are
star-shaped about a centreline: pipes, bends, tapered vessels, and a heart
chamber. The engine builds the block topology once and applies geometry
templates on top, so the same code produces a straight pipe and a left
ventricle.

The design goal that sets every choice below: a mesh that can be **morphed
through large deformation as one fixed-topology mesh, without remeshing**, while
staying valid (all cell volumes positive) and keeping wall-shear-stress
accuracy. That rules out a mesh with a singular axis, and it motivates the
O-grid.

---

## 2. The O-grid (butterfly) topology

A circular or star-shaped cross-section is divided into five blocks: a square
**core** (an "H" block) and four boundary-fitted **petals** around it. All four
petals meet the core edges; the wall is the outer edge of the petals.

The property that matters: there is **no polar axis**. A naive structured cavity
mesh places a ring of cells around a central line, so every cell collapses onto
that line at the centre — a polar singularity. Under end-systolic contraction
those central cells invert (negative volume). The O-grid core replaces the
singular line with a finite square block, so the centre is meshed by ordinary
hexes and nothing collapses.

Convention: `NT` is the circumferential node count (must be divisible by 4, one
quarter per petal), `Q = NT/4` is the per-petal tangential count, `NR` is the
radial node count from core edge to wall, and `core_frac` sets the core size as
a fraction of the local radius.

---

## 3. The sweep: one operation, many shapes

A 3-D O-grid is the 2-D butterfly section lofted along a path. `hexmb` sweeps a
butterfly section along a stack of rings using rotation-minimizing
(parallel-transport) frames, which avoid the twist a Frenet frame introduces
through a bend. The same operation produces:

- **pipe** — straight path, constant ring;
- **elbow / U-bend** — arc path, constant ring;
- **tapered vessel / aorta segment** — curved centreline, varying ring;
- **LV cavity** — per-level endocardial rings from a surface, closed at the apex.

Templates (`hexmb.templates`) are thin wrappers over this sweep. A new shape is a
new way to supply the rings, not new topology code.

Boundary layers are available by grading the radial distribution
(`graded_radial(NR, wall_ratio)`), which clusters petal cells toward the wall.

---

## 4. Cross-section construction and exact subdivision

Two cross-section builders exist, for two purposes.

**General (TFI).** For the all-hex engine, each petal is filled by transfinite
interpolation (a Coons patch from its four bounding curves), and the core by a
bilinear map. This gives smooth, well-graded sections for arbitrary star-shaped
rings.

**Affine (exactly subdivisible).** For the split-hex transition (Section 7), the
section must subdivide exactly. `ogrid_section(ring, core_frac, radial)` builds
the core **bilinear** and each petal **linear** in (radius, tangent):

```
core(u,v) = S0 (1-u)(1-v) + S1 u(1-v) + S2 uv + S3 (1-u)v
petal(r,t) = (1-r)·inner(t) + r·outer(t),   inner(t) linear between core corners
```

Both maps are affine in their parameters. Therefore, if the boundary ring is
refined by inserting edge midpoints, every interior node of the refined section
is the exact average of the coarse nodes it sits between, and the coarse section
nodes are an exact subset of the fine section nodes. This exactness is what a
conformal coarse-fine join would rely on; it follows from the affine section maps.

---

## 5. Morphing an O-grid through large deformation

The topology (cells, faces, owner/neighbour, patches) is built once at a
reference shape and reused for every deformed shape; only the point positions
change. For the LV, each cardiac phase is produced by:

1. move the endocardial **surface** grid by a biharmonic RBF fit of the measured
   wall displacement (control points subsampled from the wall);
2. rebuild the O-grid interior by TFI from the moved surface (the "resurface"
   scheme).

Two schemes were compared. Displacing the interior nodes directly by the wall
RBF ("move-points") inverts cells at end-systole. Rebuilding the interior by TFI
from each phase's own surface ("resurface") stays valid across the whole volume
swing. The resurface scheme is also reference-independent: building the mesh at
end-diastole, mid-cycle, or end-systole gives the same per-phase quality.

Result on the validated LV: 0 inverted cells at every phase across a ~65% volume
swing, checkMesh clean at each sampled phase.

---

## 6. The apex resolution ceiling (a topological limit)

A single morphing O-grid LV inverts cells at the apex beyond roughly 56 cells
across the short axis. Two experiments show this is a geometry limit set by the
pinched end-systolic tip, not a tunable parameter:

- a construction sweep (bigger butterfly core, apex trim, axial coarsening) only
  moves the worst-cycle minimum scaled-Jacobian in the wrong direction;
- building at the contracted (ES) reference instead of ED does not raise it (the
  resurface scheme is reference-independent, Section 5).

Routes past this ceiling change the scheme: interval remeshing, an apex tet
patch, immersed boundary, or **carrying a second resolution in the same mesh** —
which is what Sections 7–8 do.

---

## 7. Telescoping transition: AMI (shipped) and the split-hex alternative

To give one LV mesh two resolutions — a fine body, coarse apex and base caps —
neighbouring zones must be joined across a coarse↔fine plane. The shipped path
(Section 8, `templates/telescoping_lv` + `assembly/ami`) joins them with a
**non-conformal AMI** (OpenFOAM `nonConformalCyclic` via
`createNonConformalCouples`). The rest of this section describes a **conformal
split-hex** transition that was prototyped as a flux-conserving alternative; it is
**not part of this release** and is kept here as method background.

### 7.1 Why a conservative alternative was explored

An AMI **interpolates** the flux across the interface, so mass is not conserved
across it exactly. On a moving LV, diastolic filling drives flow *through* that
interface (the mitral E-wave), and the non-conservative flux can stall the
pressure/continuity solve — the timestep collapses and the run halts. Ejection,
with a weaker gradient across the interface, solves. Two alternative explanations
were tested and refuted: refining the mesh does not fix it, and matching the
interface cell sizes (4× → 1.6× mismatch) does not reduce the pressure iterations.
The cause is the non-conservative interpolation itself — so gate a moving
telescoping run on whether the *filling* phase solves, not only ejection.

### 7.2 The split-hex construction (prototype, not shipped)

Where a coarse cell meets four finer cells across a plane, split the coarse
cell's interface face into a 2×2 grid using its edge midpoints and centre:

```
coarse face ABCD  →  midpoints AB, BC, CD, DA and centre O
                     four sub-quads  (A,AB,O,DA) (AB,B,BC,O) (O,BC,C,CD) (DA,O,CD,D)
```

The coarse cell's four side faces become pentagons (each gains one midpoint),
so the coarse cell is a 9-face **split-hex polyhedron**; the four fine cells
stay hexes. Because the section is affine and the fine ring is the coarse ring
subdivided (Section 4), the geometric split points (`AB`, `O`, …) coincide
exactly with the fine nodes. The two sides therefore **share faces** — the join
is conformal, and flux is conserved across it with no interpolation.

(This prototype lived in an `experimental.splithex` module that has been removed
from the release; the construction is recorded here for completeness.)

### 7.3 Requirements

- fine circumferential `NT_f = 2·NT_c`;
- fine radial `NR_f = 2·NR_c − 1`;
- `NT` divisible by 4;
- the fine ring is the coarse ring subdivided (`subdivide_ring`), and the
  section is affine (`ogrid_section`).

### 7.4 Validation (prototype)

The split-hex prototype produced an O-grid disk transition and a straight
telescoping tube as **one region, checkMesh Mesh OK**, openness ~1e-16, positive
volumes. A straight telescoping tube (7344 hex + 216 split-hex) had max skewness
0.45 and max non-orthogonality 24.8. On the LV the transition cells in the curved
petals carried skewness ~5–6 (non-orthogonality up to 77°); a light interior
Laplacian relaxation reduced the non-orthogonality but not the residual skew.

---

## 8. The telescoping O-grid (shipped, AMI)

`templates/telescoping_lv` stacks three swept O-grid zones — a **coarse apex cap**,
a **fine body**, a **coarse base cap** — and joins neighbouring zones by a
non-conformal AMI (`assembly/ami.HybridAssembler`: `mergeMeshes` combines the zone
polyMeshes, `createNonConformalCouples` adds the AMI on the touching cap/body
faces). The body scales to ~1M cells while each cap ring stays coarse, so neither
cap pinches through the volume swing (the apex ceiling of Section 6).

`build_telescoping_lv(G, axis, e1, e2, cfg, mu)` builds the three zones from an
endocardial ring grid and tags patches (base-plane faces → `inlet` mitral /
`outlet1` aortic by angular sector, everything else `wall`);
`assemble_telescoping_lv(zones, out, cfg)` does the merge + AMI couple and needs
OpenFOAM on `PATH`. `TelescopingLVConfig` carries the ~1M defaults (`body_NT=340`,
`body_NR=24`) and the valve-plane fixes: a Fourier low-pass **de-spike** of the
noisy annulus contour and a **base cut** at the annulus (`base_cut=0.98`) rather
than the flared tip. The AMI flux caveat of Section 7.1 applies: ejection solves,
filling can stall — gate on the filling phase.

### 8.1 Multi-level (nested) telescoping

The three-zone stack generalises to more axial zones at higher refinement (apex
`NT` → body `2·NT` → mid-body `4·NT` → back), neighbouring zones differing by at
most one level, so the mid-body can step up a further factor of two while the apex
ring stays coarse.

Why the apex must stay coarse, and the cost of the extra level, are both measured
on the contracting LV (`NT_c=44`, so apex 44 / body 88 / mid-body 176):

- **apex-pinch is avoided.** A uniform `NT88` mesh reaches skew 165 at
  end-systole because its apex ring (88 across) pinches to near-zero cell size in
  the contracted cavity. Holding the apex at 44 while the body refines to 176
  keeps end-systole skew at **10.2**, the same as the single-level grid that runs
  the full cycle.
- **the added level skews at the dilated end.** The mid-body level-2 transition
  cells sit in the curved petals; when the cavity dilates toward end-diastole they
  stretch and the mesh reaches skew **~26**. So the nested level is safe only
  across a bounded dilation (for example the early-diastolic E-wave window), not
  the whole ES→ED swing. Restricting level-2 to the near-core petals, or relaxing
  those transitions harder, is future work.

The single-level three-zone stack therefore stays the default for a whole-cycle
moving mesh; the nested form is for refining a region of interest over a limited
part of the cycle.

---

## 9. Polyhedral export and verification

`PolyMesh` writes a general polyhedral `constant/polyMesh`:

- points de-duplicated by rounded coordinate, so shared faces between cells
  collapse to one internal face;
- each face oriented so its area vector points from owner to neighbour (owner is
  the lower cell index); boundary faces oriented outward;
- internal faces first, sorted by (owner, neighbour); boundary faces grouped
  contiguously per patch;
- points scaled mm → m on write.

Verification ladder, applied to every mesh:

1. `cell_volumes()` (divergence theorem over polyhedral faces) — **all positive**
   is the mandatory gate; for all-hex meshes `scaled_jacobian().min() > 0`.
2. `checkMesh` — expect **Number of regions: 1** and no negative-volume cells;
   skewness/non-orthogonality reported and judged against FV-scheme tolerances.
3. For a moving mesh, the volume check is applied at every phase across the
   deformation.

---

## 10. Honest limits and applicability domain

In scope, validated:

- single-lumen tube or single cavity (genus 0), cross-sections star-shaped about
  the centreline, `NT` divisible by 4, gentle bends;
- boundary layers via `wall_ratio`;
- morphing through large deformation with the resurface scheme;
- the AMI telescoping transition (Section 8), with the filling caveat below.

Out of scope or with caveats:

- branches, junctions, and multi-lumen trees are out of scope — conforming
  meshing through a junction is unsolved here (earlier bifurcation/tree/snappy
  prototypes never reached a single flow-verified region through a junction and
  have been removed);
- the LV apex inverts beyond ~56 cells across in a single-resolution O-grid
  (Section 6); the telescoping route addresses resolution but the apex cap stays
  coarse;
- the AMI coupling interpolates flux, so it is not exactly conservative: a moving
  telescoping LV solves ejection but can stall in diastolic filling (Section 7.1) —
  gate on the filling phase;
- the all-hex assembly is vectorised and reaches millions of cells.

---

## 11. Reproducing the results / API map

General engine (all-hex):

- `hexmb.templates.tube` — `PipeTemplate`, `ElbowTemplate`;
- `hexmb.templates.stl` — `TubeSTLTemplate`;
- `hexmb.templates.lv` — `LVChamberTemplate` (single-region cavity);
- `hexmb.templates.telescoping_lv` — `TelescopingLVConfig`, `build_telescoping_lv`,
  `assemble_telescoping_lv` (multi-resolution, AMI);
- `hexmb.assembly.ami.HybridAssembler` — the non-conformal AMI assembler;
- `hexmb.services.export_foam.write_polymesh`, `hexmb.services.quality.scaled_jac`,
  `hexmb.services.morph.RBFMorpher`;
- CLI: `hexmb pipe|elbow|cap|lv|stl|verify`.

Data input (the public demo):

- `hexmb.io.cap` — read a Cardiac Atlas Project LV `.exnode` model into an
  endocardial ring grid (`cap_endo_surface`, `surface_frame`); see `docs/DATA.md`;
- `hexmb.io.stl_surface.write_surface_stl` — write a ring grid as an STL/CAD file;
- `scripts/cap_to_mesh.py` — the end-to-end worked example (CAP model → surface →
  STL → hex mesh + quality gate).
