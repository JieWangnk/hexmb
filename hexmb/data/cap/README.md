# Bundled demo heart: Cardiac Atlas Project, Sunnybrook subject SCD0000101

`SCD0000101/SCD0000101_<phase>.model.exnode`, phases 1 to 20, are the 3D
left-ventricle finite-element models of one subject from the **Sunnybrook
Cardiac Data** distributed by the Cardiac Atlas Project
(https://www.cardiacatlas.org/sunnybrook-cardiac-data/).

- **Licence:** Public Domain, CC0 1.0 Universal. The files are redistributed
  here unchanged so the LV demo runs without any download.
- **What a file holds:** 40 nodes in prolate-spheroidal coordinates (two
  transmural layers, 4 circumferential x 5 longitudinal) with Hermite
  derivatives; one `focus` per model.
- **`GlobalHermiteParam.exelem`** (this directory) is the element file of the
  same subject: 16 elements, bicubic Hermite `lambda`, unit scale factors. It is
  the same template for every Sunnybrook subject apart from the focus, so
  `hexmb.io.cap` uses it whenever no `.exelem` sits beside a `.exnode`, and
  evaluates the inner (endocardial) layer exactly from it.
- **Timing:** the Sunnybrook models are phases of one cardiac cycle, phase 1 at
  end-diastole. The files carry no RR interval; the demo spaces the 20 phases
  uniformly over `--period` seconds (default 1.0 s).

Run the demo:

```bash
hexmb demo -o out/demo            # mesh at every phase + moving-mesh case, no OpenFOAM needed
```
