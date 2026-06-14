from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from .models import TargetMesh


def despace(string: str) -> str:
    return "".join(ch for ch in string if ch not in (" ", "\t"))


def target_mesh_setup(mesh: TargetMesh, lat2d: np.ndarray | None = None, lon2d: np.ndarray | None = None) -> TargetMesh:
    if lat2d is not None and lon2d is not None:
        mesh.irank = 1
        mesh.nlat = int(lat2d.shape[1])
        mesh.nlon = int(lon2d.shape[0])
        mesh.lats = lat2d
        mesh.lons = lon2d
        mesh.valid = True
        return mesh

    path = Path("target_domain")
    if path.exists():
        mesh.startlat = -90.0
        mesh.endlat = 90.0
        mesh.startlon = -180.0
        mesh.endlon = 180.0
        mesh.nlat = 360
        mesh.nlon = 720
        for raw in path.read_text().splitlines():
            spec = despace(raw)
            if not spec:
                continue
            if "=" not in spec:
                raise ValueError(f"Syntax error in target_domain: {raw!r}")
            key, value = spec.split("=", 1)
            key = key.lower()
            if key == "nlat":
                mesh.nlat = int(value)
            elif key == "nlon":
                mesh.nlon = int(value)
            elif key == "startlat":
                mesh.startlat = float(value)
            elif key == "endlat":
                mesh.endlat = float(value)
            elif key == "startlon":
                mesh.startlon = float(value)
            elif key == "endlon":
                mesh.endlon = float(value)
            else:
                raise ValueError(f"Unrecognized target_domain keyword: {key}")
    else:
        mesh.startlat = -90.0
        mesh.endlat = 90.0
        mesh.startlon = -180.0
        mesh.endlon = 180.0
        mesh.nlat = 360
        mesh.nlon = 720

    mesh.lats = np.empty((1, mesh.nlat), dtype=np.float64)
    mesh.lons = np.empty((mesh.nlon, 1), dtype=np.float64)
    pi_const = 2.0 * math.asin(1.0)

    delta = (mesh.endlat - mesh.startlat) / float(mesh.nlat)
    for i in range(mesh.nlat):
        mesh.lats[0, i] = (mesh.startlat + 0.5 * delta + float(i) * delta) * pi_const / 180.0

    delta = (mesh.endlon - mesh.startlon) / float(mesh.nlon)
    for i in range(mesh.nlon):
        mesh.lons[i, 0] = (mesh.startlon + 0.5 * delta + float(i) * delta) * pi_const / 180.0

    mesh.valid = True
    return mesh


def target_mesh_free(mesh: TargetMesh) -> None:
    mesh.valid = False
    mesh.nlat = 0
    mesh.nlon = 0
    mesh.startlat = 0.0
    mesh.endlat = 0.0
    mesh.startlon = 0.0
    mesh.endlon = 0.0
    if mesh.irank == 0:
        mesh.lats = None
        mesh.lons = None
