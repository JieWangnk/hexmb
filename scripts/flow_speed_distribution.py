"""Where is the fastest cell, and how is the speed distributed?

    pvbatch scripts/flow_speed_distribution.py CASE TIME

Reads the written field closest to TIME from a hexmb flow case and prints the
cell of maximum |U| with its size, its distance to the open-valve plane, to the
nearest open-valve patch point and to the nearest wall point, and the 99th and
99.9th percentiles and the volume-weighted mean of |U| over the cavity.  Used in
docs/VERIFICATION.md to show that the pointwise maximum sits in the rim cell of
the open valve and grows with refinement, while the distribution measures are
mesh-independent.  The open valve is taken as the patch named ``inlet`` by
default; pass the patch name as a third argument for a systolic case (``outlet1``).
"""

import sys, numpy as np
from paraview.simple import *
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy
case, t = sys.argv[1], float(sys.argv[2])
valve = sys.argv[3] if len(sys.argv) > 3 else "inlet"
r = OpenFOAMReader(FileName=case + "/case.foam"); r.MeshRegions = ["internalMesh", f"patch/{valve}", "patch/wall"]
r.CellArrays = ["U"]; r.ListtimestepsaccordingtocontrolDict = 1; r.UpdatePipelineInformation()
ts = np.array(r.TimestepValues); tt = ts[np.argmin(np.abs(ts - t))]
r.UpdatePipeline(tt)
mb = sm.Fetch(r)
def block(name):
    it = mb.NewIterator(); it.UnRegister(None); it.InitTraversal()
    while not it.IsDoneWithTraversal():
        if it.GetCurrentMetaData().Get(mb.NAME()) == name:
            return it.GetCurrentDataObject()
        it.GoToNextItem()
    # nested: search all leaves
    import vtk
    def walk(obj):
        if isinstance(obj, vtk.vtkMultiBlockDataSet):
            for i in range(obj.GetNumberOfBlocks()):
                n = obj.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) if obj.HasMetaData(i) else None
                if n == name: return obj.GetBlock(i)
                res = walk(obj.GetBlock(i))
                if res is not None: return res
        return None
    return walk(mb)
im = block("Block 0_internalMesh") or block("internalMesh"); inlet = block(valve); wall = block("wall")
import vtk
f = vtk.vtkCellCenters(); f.SetInputData(im); f.Update()
C = vtk_to_numpy(f.GetOutput().GetPoints().GetData()); U = vtk_to_numpy(im.GetCellData().GetArray("U"))
mag = np.linalg.norm(U, axis=1); i = int(np.argmax(mag))
Pin = vtk_to_numpy(inlet.GetPoints().GetData()); Pw = vtk_to_numpy(wall.GetPoints().GetData())
cin = Pin.mean(0); n = np.linalg.svd(Pin - cin)[2][2]
d_plane = float(abs((C[i] - cin) @ n)); d_cent = float(np.linalg.norm(C[i] - cin))
r_in = float(np.linalg.norm(Pin - cin, axis=1).max())
d_wall = float(np.min(np.linalg.norm(Pw - C[i], axis=1)))
d_rim = float(np.min(np.linalg.norm(Pin - C[i], axis=1)))
vol = vtk.vtkCellSizeFilter(); vol.SetInputData(im); vol.ComputeVolumeOn(); vol.Update()
V = vtk_to_numpy(vol.GetOutput().GetCellData().GetArray("Volume"))
print(f"t {tt:g}: max|U| {mag[i]:.3f} m/s in cell {i} (h ~ {V[i]**(1/3)*1e3:.2f} mm); distance to valve plane {d_plane*1e3:.1f} mm, "
      f"to valve centroid {d_cent*1e3:.1f} mm (valve radius {r_in*1e3:.1f} mm), to nearest valve-patch point {d_rim*1e3:.1f} mm, to nearest wall point {d_wall*1e3:.1f} mm; "
      f"99th pct |U| {np.percentile(mag,99):.3f}, 99.9th {np.percentile(mag,99.9):.3f}, volume-weighted mean {np.sum(mag*V)/V.sum():.3f}")
