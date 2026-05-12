"""Save and load .ndent project files.

A .ndent file is a zip bundle:

    my_case.ndent
    ├── project.json         metadata + per-stage UI state
    ├── jaw.stl              imported prep mesh
    ├── crown.stl            placed crown
    ├── shell_inner.stl      inner offset shell
    ├── trimmed.stl          margin-trimmed crown
    ├── final.stl            solidified watertight crown
    └── margin.npy           margin points

Only the artifacts that exist at save time are written. project.json's `artifacts`
map tells the loader which keys to look up.
"""
import datetime
import io
import json
import os
import tempfile
import zipfile

import numpy as np
import pyvista as pv


SCHEMA_VERSION = 1
PROJECT_EXT = ".ndent"
PROJECT_FILTER = "NuDent Project (*.ndent)"


def _mesh_to_bytes(mesh, suffix=".stl"):
    """PyVista's .save() requires a path, so we round-trip through a temp file."""
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        mesh.save(tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try: os.remove(tmp)
        except Exception: pass


def _bytes_to_mesh(data, suffix=".stl"):
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        return pv.read(tmp)
    finally:
        try: os.remove(tmp)
        except Exception: pass


def save_project(path, app):
    """Write the current app state to `path` as a .ndent file. Atomic via tmp+rename."""
    state = app.state
    now = datetime.datetime.now().isoformat(timespec="seconds")

    stages_data = {}
    for stage in app.stages:
        stages_data[stage.name.lower()] = stage.serialize()

    artifacts = {}
    payloads = {}

    if state.jaw_mesh is not None:
        artifacts["jaw"] = "jaw.stl"
        payloads["jaw.stl"] = _mesh_to_bytes(state.jaw_mesh)
    if state.margin_points:
        artifacts["margin"] = "margin.npy"
        buf = io.BytesIO()
        np.save(buf, np.array(state.margin_points))
        payloads["margin.npy"] = buf.getvalue()
    if state.crown is not None:
        artifacts["crown"] = "crown.stl"
        payloads["crown.stl"] = _mesh_to_bytes(state.crown)
    if state.shell_inner is not None:
        artifacts["shell_inner"] = "shell_inner.stl"
        payloads["shell_inner.stl"] = _mesh_to_bytes(state.shell_inner)
    if state.trimmed_crown is not None:
        artifacts["trimmed"] = "trimmed.stl"
        payloads["trimmed.stl"] = _mesh_to_bytes(state.trimmed_crown)
    if state.final_crown is not None:
        artifacts["final"] = "final.stl"
        payloads["final.stl"] = _mesh_to_bytes(state.final_crown)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "modified_at": now,
        "current_stage": app.current_stage_idx,
        "jaw_filename": os.path.basename(state.jaw_path) if state.jaw_path else None,
        "margin_loop_closed": state.margin_loop_closed,
        "stages": stages_data,
        "artifacts": artifacts,
    }

    tmp_path = path + ".tmp"
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(metadata, indent=2))
        for name, data in payloads.items():
            z.writestr(name, data)
    os.replace(tmp_path, path)


def load_project(path, app):
    """Read a .ndent file into app.state and return the metadata dict.

    Does not call stage.restore() — the caller (MainWindow) does that after
    refreshing visualization scaffolding (jaw_actor, details panel, etc.).
    """
    with zipfile.ZipFile(path, "r") as z:
        meta = json.loads(z.read("project.json").decode("utf-8"))
        artifacts = meta.get("artifacts", {})

        state = app.state
        # Hard reset before populating
        state.jaw_mesh = None
        state.jaw_path = None
        state.margin_points = []
        state.margin_loop_closed = False
        state.crown = None
        state.shell_outer = None
        state.shell_inner = None
        state.trimmed_crown = None
        state.final_crown = None

        if "jaw" in artifacts:
            state.jaw_mesh = _bytes_to_mesh(z.read(artifacts["jaw"]))
            state.jaw_path = meta.get("jaw_filename") or "jaw.stl"
        if "margin" in artifacts:
            arr = np.load(io.BytesIO(z.read(artifacts["margin"])))
            state.margin_points = [np.array(p) for p in arr]
            state.margin_loop_closed = bool(meta.get("margin_loop_closed", False))
        if "crown" in artifacts:
            state.crown = _bytes_to_mesh(z.read(artifacts["crown"]))
            state.shell_outer = state.crown  # outer surface IS the placed crown
        if "shell_inner" in artifacts:
            state.shell_inner = _bytes_to_mesh(z.read(artifacts["shell_inner"]))
        if "trimmed" in artifacts:
            state.trimmed_crown = _bytes_to_mesh(z.read(artifacts["trimmed"]))
        if "final" in artifacts:
            state.final_crown = _bytes_to_mesh(z.read(artifacts["final"]))

    return meta
