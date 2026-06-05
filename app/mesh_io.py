"""Format-agnostic mesh I/O.

`pv.read()` already dispatches on extension and supports STL/OBJ/PLY, but a
few formats (notably OBJ with multiple groups) come back as a MultiBlock and
PLY meshes can carry vertex colors / scalar arrays that we don't want bleeding
into downstream stages or saved project artifacts. This helper normalises
everything to a clean single PolyData.
"""
import os
import contextlib
import numpy as np
import pyvista as pv
import vtk


SUPPORTED_EXTS = (".stl", ".obj", ".ply")
FILE_DIALOG_FILTER = "Mesh (*.stl *.obj *.ply)"


@contextlib.contextmanager
def _silence_vtk_warnings():
    """Temporarily route VTK warnings to a null output window.

    Large OBJ files (exocad-style dental scans) trigger thousands of
    `vtkOBJReader: unexpected data at end of line` warnings — each printed
    individually to stderr, which freezes the Qt UI thread for tens of
    seconds. Suppressing them speeds reads from ~150s to a few seconds.
    """
    old = vtk.vtkOutputWindow.GetInstance()
    null_window = vtk.vtkFileOutputWindow()
    null_window.SetFileName(os.devnull)
    vtk.vtkOutputWindow.SetInstance(null_window)
    try:
        yield
    finally:
        vtk.vtkOutputWindow.SetInstance(old)


def read_mesh(path):
    """Load a mesh from STL / OBJ / PLY and return a single PolyData.

    Raises ValueError if the extension is unsupported or the file produces
    no triangles.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported mesh format: {ext} (expected one of {SUPPORTED_EXTS})")

    with _silence_vtk_warnings():
        mesh = pv.read(path)

    # OBJ files with multiple `o`/`g` groups load as a MultiBlock — flatten it.
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine().extract_surface()

    # Drop any incoming point/cell scalars (PLY can ship them) so they don't
    # propagate into shell/trim/refine outputs or get baked into saved STLs.
    # We re-attach OBJ vertex colors below as a dedicated "RGB" array.
    try:
        for name in list(mesh.point_data.keys()):
            del mesh.point_data[name]
        for name in list(mesh.cell_data.keys()):
            del mesh.cell_data[name]
    except Exception:
        pass

    # Exocad/MeshLab OBJs encode per-vertex RGB as extra columns on `v` lines
    # (`v x y z r g b`). VTK's OBJ reader ignores those columns and warns —
    # we parse them ourselves so the user can switch to a "realistic" render.
    if ext == ".obj":
        rgb = _parse_obj_vertex_colors(path)
        if rgb is not None and len(rgb) == mesh.n_points:
            mesh.point_data["RGB"] = rgb

    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh has no geometry: {path}")

    return mesh


def _parse_obj_vertex_colors(path):
    """Return an (N, 3) uint8 array of per-vertex RGB from an OBJ file, in
    vertex-declaration order. Returns None if the file has no `r g b` columns
    on its `v` lines (or if parsing fails partway through).
    """
    colors = []
    has_color = False
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        colors.append((int(parts[4]), int(parts[5]), int(parts[6])))
                        has_color = True
                    except ValueError:
                        return None
                elif has_color:
                    # Mixed file (some verts colored, some not) — bail.
                    return None
    except OSError:
        return None
    if not has_color:
        return None
    return np.asarray(colors, dtype=np.uint8)
