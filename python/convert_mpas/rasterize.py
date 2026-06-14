from __future__ import annotations

import math
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import uxarray as ux
from dask.distributed import Client, LocalCluster
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio import open as rio_open

from .models import TargetMesh
from .target_mesh import target_mesh_setup

RASTERIZE_VARS = (
    ("t2m", "T2m"),
    ("mslp", "mslp"),
    ("q2", "humidity_2m"),
)


def _to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_time_metadata(dataset) -> str:
    if "xtime" in dataset:
        value = dataset["xtime"].isel(Time=0).compute().values
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"S", "U"}:
                text = b"".join(value.tolist()).decode("utf-8", errors="ignore") if value.ndim else _to_str(value)
            else:
                text = _to_str(value)
        else:
            text = _to_str(value)
        return text.strip("\x00").strip()
    if "Time" in dataset.coords:
        return _to_str(dataset["Time"].isel(Time=0).compute().values)
    return ""


def _make_cluster() -> tuple[LocalCluster, Client]:
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=max(2, os.cpu_count() or 1),
        processes=False,
        dashboard_address=None,
    )
    client = Client(cluster)
    return cluster, client


def _destination_grid(mesh: TargetMesh) -> ux.Grid:
    lats = np.rad2deg(np.asarray(mesh.lats[0, :], dtype=np.float64))
    lons = np.rad2deg(np.asarray(mesh.lons[:, 0], dtype=np.float64))
    return ux.Grid.from_structured(lon=lons, lat=lats)


def _reshape_face_data(data: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 1 or arr.size != nlat * nlon:
        raise RuntimeError(f"unexpected rasterized data shape {arr.shape}; expected {(nlat * nlon,)}")
    return arr.reshape((nlat, nlon))


def rasterize_to_geotiff(mesh_filename: str, data_filename: str, output_filename: str = "latlon.tif") -> str:
    cluster, client = _make_cluster()
    try:
        source = ux.open_dataset(mesh_filename, data_filename, chunks="auto", chunk_grid=False)
        try:
            mesh = target_mesh_setup(TargetMesh())
            dest_grid = _destination_grid(mesh)

            missing = [src for src, _ in RASTERIZE_VARS if src not in source]
            if missing:
                raise RuntimeError("input file is missing rasterize variables: " + ", ".join(missing))

            selected = source[[src for src, _ in RASTERIZE_VARS]].isel(Time=0)
            time_value = _extract_time_metadata(source)
            remapped = selected.remap.nearest_neighbor(dest_grid, remap_to="faces")
            remapped = remapped.compute()

            transform = from_bounds(
                mesh.startlon,
                mesh.startlat,
                mesh.endlon,
                mesh.endlat,
                mesh.nlon,
                mesh.nlat,
            )
            profile = {
                "driver": "GTiff",
                "height": mesh.nlat,
                "width": mesh.nlon,
                "count": len(RASTERIZE_VARS),
                "dtype": "float32",
                "crs": CRS.from_epsg(4326),
                "transform": transform,
                "tiled": True,
                "compress": "deflate",
                "nodata": np.nan,
            }

            with rio_open(output_filename, "w", **profile) as dst:
                for band_index, (src_name, band_name) in enumerate(RASTERIZE_VARS, start=1):
                    values = np.asarray(remapped[src_name].values, dtype=np.float32)
                    band = _reshape_face_data(values, mesh.nlat, mesh.nlon)
                    dst.write(np.flipud(band), band_index)
                    dst.set_band_description(band_index, band_name)
                dst.update_tags(
                    source_mesh=Path(mesh_filename).name,
                    source_data=Path(data_filename).name,
                    model_time=time_value,
                    raster_format="GeoTIFF",
                )
            return output_filename
        finally:
            with suppress(Exception):
                source.close()
    finally:
        with suppress(Exception):
            client.close()
        with suppress(Exception):
            cluster.close()
