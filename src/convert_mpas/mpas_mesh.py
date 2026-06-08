from __future__ import annotations

import numpy as np

from .models import MPASMesh
from .scan_input import (
    scan_input_close,
    scan_input_for_field,
    scan_input_free_field,
    scan_input_open,
    scan_input_read_field,
)

REQUIRED_MESH_FIELDS = (
    "nEdgesOnCell",
    "cellsOnCell",
    "verticesOnCell",
    "cellsOnVertex",
    "edgesOnCell",
    "cellsOnEdge",
    "latCell",
    "lonCell",
    "latVertex",
    "lonVertex",
    "latEdge",
    "lonEdge",
)


def _read_field_array(handle, name: str):
    field = scan_input_for_field(handle, name)
    scan_input_read_field(field)
    return field


def mpas_mesh_setup(mesh_filename: str, mesh: MPASMesh | None = None) -> MPASMesh:
    mesh = mesh or MPASMesh()
    handle, _ = scan_input_open(mesh_filename)
    try:
        missing = [name for name in REQUIRED_MESH_FIELDS if name not in handle.dataset.variables]
        if missing:
            raise RuntimeError(
                "input file is missing required MPAS mesh variables: " + ", ".join(missing)
            )

        field = _read_field_array(handle, "nEdgesOnCell")
        mesh.nCells = int(field.dimlens[0])
        mesh.nEdgesOnCell = np.asarray(field.array1i, dtype=np.int32)
        scan_input_free_field(field)

        field = _read_field_array(handle, "cellsOnCell")
        mesh.maxEdges = int(field.dimlens[1])
        mesh.cellsOnCell = np.asarray(field.array2i, dtype=np.int32).T
        scan_input_free_field(field)

        field = _read_field_array(handle, "verticesOnCell")
        mesh.verticesOnCell = np.asarray(field.array2i, dtype=np.int32).T
        scan_input_free_field(field)

        field = _read_field_array(handle, "cellsOnVertex")
        mesh.nVertices = int(field.dimlens[0])
        mesh.cellsOnVertex = np.asarray(field.array2i, dtype=np.int32).T
        scan_input_free_field(field)

        field = _read_field_array(handle, "edgesOnCell")
        mesh.edgesOnCell = np.asarray(field.array2i, dtype=np.int32).T
        scan_input_free_field(field)

        field = _read_field_array(handle, "cellsOnEdge")
        mesh.nEdges = int(field.dimlens[0])
        mesh.cellsOnEdge = np.asarray(field.array2i, dtype=np.int32).T
        scan_input_free_field(field)

        field = _read_field_array(handle, "latCell")
        mesh.latCell = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)

        field = _read_field_array(handle, "lonCell")
        mesh.lonCell = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)

        field = _read_field_array(handle, "latVertex")
        mesh.latVertex = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)

        field = _read_field_array(handle, "lonVertex")
        mesh.lonVertex = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)

        field = _read_field_array(handle, "latEdge")
        mesh.latEdge = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)

        field = _read_field_array(handle, "lonEdge")
        mesh.lonEdge = np.asarray(field.array1r if field.array1r is not None else field.array1d, dtype=np.float64)
        scan_input_free_field(field)
    finally:
        scan_input_close(handle)

    mesh.valid = True
    return mesh


def mpas_mesh_free(mesh: MPASMesh) -> None:
    mesh.valid = False
    mesh.nCells = 0
    mesh.nVertices = 0
    mesh.nEdges = 0
    mesh.maxEdges = 0
    mesh.nEdgesOnCell = None
    mesh.cellsOnCell = None
    mesh.verticesOnCell = None
    mesh.cellsOnVertex = None
    mesh.edgesOnCell = None
    mesh.cellsOnEdge = None
    mesh.latCell = None
    mesh.lonCell = None
    mesh.latVertex = None
    mesh.lonVertex = None
    mesh.latEdge = None
    mesh.lonEdge = None
