# Getting the demo data

hexmb is demonstrated on **public-domain** left-ventricle geometry so anyone can
reproduce the pipeline. No private patient data is shipped in this repository.

## Cardiac Atlas Project — Sunnybrook Cardiac Data (SCD)

- **What it is:** 45 cardiac-MRI studies across mixed pathologies (healthy,
  hypertrophy, heart failure with and without infarction), each with expert
  short-axis contours and a **3D left-ventricle finite-element model** per
  cardiac phase (~20 phases per subject). Also known as the 2009 Cardiac-MR LV
  Segmentation Challenge data.
- **Licence:** Public Domain (**CC0 1.0 Universal**) — free to use for any
  purpose, no request or authorisation needed.
- **Download:** https://www.cardiacatlas.org/sunnybrook-cardiac-data/
  (project home: https://www.cardiacatlas.org/). The LV finite-element models are
  provided alongside the images; the CAP Client on the site imports them.

### File format

Each phase is a pair of CMISS/`cmgui` export files:

| file | holds |
|---|---|
| `*.model.exnode` | node values: prolate-spheroidal `lambda` (+ derivatives), `mu`, `theta`, and one `focus` per model |
| `*.model.exelem` | element topology (bicubic-Hermite circumferential x longitudinal, linear transmural) |

hexmb reads the `.exnode` directly (`hexmb.io.cap`); the `.exelem` topology is
not needed because the reader interpolates the endocardial layer itself.

## Run the demo

```bash
# one LV model -> endocardial surface -> STL/CAD -> telescoping hex mesh + quality gate
python scripts/cap_to_mesh.py /path/to/SCD0000101_1.model.exnode -o out/lv --stl out/lv_endo.stl
```
```text
[1] read CAP model: .../SCD0000101_1.model.exnode
    endocardial surface (40, 56, 3)  mid-radius ~28 mm
[2] wrote CAD surface: out/lv_endo.stl  (4368 triangles, mm)
[3] mesh with hexmb (telescoping: coarse apex / fine body / coarse base)
    zone apex :    12617 cells  min scaled-Jac +0.391  (valid)
    zone body :    26796 cells  min scaled-Jac +0.612  (valid)
    zone base :    31900 cells  min scaled-Jac +0.616  (valid)
[4] wrote telescoping polyMesh under out/lv  (AMI-coupled)
```

`hexmb cap --exnode /path/to/SCD0000101_1.model.exnode -o out/lv --stl out/lv_endo.stl`
does the same thing from the CLI.

What it creates:

- **`out/lv_endo.stl`** — the endocardial blood-pool surface as a CAD file (mm).
- **`out/lv/`** — the telescoping case: zone meshes `apex/`, `body/`, `base/`, with
  `out/lv/apex/` the merged master carrying the AMI couplings (open `out/lv/apex` in
  ParaView). Patches are `wall` / `inlet` (mitral) / `outlet1` (aortic).

`checkMesh -constant` on the assembled mesh reports **Cell volumes OK** (zero
negative-volume cells); a face or two may sit just over the skewness flag at the
curved zone transition, which the FV skewness-corrected schemes tolerate.

## A note on the private data

An earlier version of this engine was developed against a cine-MRI case that is
**not public**, so it has been removed from this repository. The method is the
same; only the demonstration geometry changed to the CC0 CAP data above.
