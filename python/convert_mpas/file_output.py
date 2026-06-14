from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset, default_fillvals

from .models import (
    FIELD_TYPE_DOUBLE,
    FIELD_TYPE_INTEGER,
    FIELD_TYPE_REAL,
    OutputHandle,
    TargetField,
)

FILE_MODE_CLOBBER = 1
FILE_MODE_APPEND = 2


def _dtype_and_fill(xtype: int) -> tuple[np.dtype, object]:
    if xtype == FIELD_TYPE_REAL:
        return np.dtype("f4"), default_fillvals["f4"]
    if xtype == FIELD_TYPE_DOUBLE:
        return np.dtype("f8"), default_fillvals["f8"]
    if xtype == FIELD_TYPE_INTEGER:
        return np.dtype("i4"), default_fillvals["i4"]
    raise RuntimeError(f"unsupported field type: {xtype}")


def file_output_open(filename: str, mode: int = FILE_MODE_CLOBBER) -> tuple[OutputHandle, int]:
    path = Path(filename)
    if path.exists() and mode == FILE_MODE_CLOBBER:
        dataset = Dataset(filename, "w", format="NETCDF4")
        dataset.createDimension("Time", None)
        handle = OutputHandle(dataset=dataset, unlimited_id="Time", in_define_mode=True, current_frame=0)
        return handle, 0

    if not path.exists():
        dataset = Dataset(filename, "w", format="NETCDF4")
        dataset.createDimension("Time", None)
        handle = OutputHandle(dataset=dataset, unlimited_id="Time", in_define_mode=True, current_frame=0)
        return handle, 0

    if mode == FILE_MODE_APPEND:
        dataset = Dataset(filename, "a")
        unlimited_id = next((name for name, dim in dataset.dimensions.items() if dim.isunlimited()), None)
        if unlimited_id is None:
            raise RuntimeError("output file has no unlimited dimension")
        current_frame = len(dataset.dimensions[unlimited_id])
        handle = OutputHandle(dataset=dataset, unlimited_id=unlimited_id, in_define_mode=False, current_frame=current_frame)
        return handle, current_frame

    raise RuntimeError(f"unsupported output mode: {mode}")


def file_output_close(handle: OutputHandle) -> None:
    handle.dataset.close()
    handle.in_define_mode = True
    handle.current_frame = 0


def file_output_register_field(handle: OutputHandle, field: TargetField) -> None:
    dims: list[str] = []
    if field.dimnames is None or field.dimlens is None:
        raise RuntimeError(f"field {field.name!r} is missing dimension metadata")
    for idx, dimname in enumerate(field.dimnames):
        dimname = str(dimname)
        if dimname not in handle.dataset.dimensions:
            handle.dataset.createDimension(dimname, int(field.dimlens[idx]))
        dims.append(dimname)
    if field.isTimeDependent:
        dims.append("Time")
        if "Time" not in handle.dataset.dimensions:
            handle.dataset.createDimension("Time", None)

    if field.name in handle.dataset.variables:
        return
    if handle.current_frame > 0:
        raise RuntimeError(f"cannot define new variable {field.name!r} in existing file")

    dtype, fill = _dtype_and_fill(field.xtype)
    handle.dataset.createVariable(field.name, dtype, tuple(dims), fill_value=fill)


def file_output_write_field(handle: OutputHandle, field: TargetField, frame: int | None = None) -> None:
    if field.name not in handle.dataset.variables:
        raise KeyError(field.name)
    var = handle.dataset.variables[field.name]
    data = _target_field_data(field)
    if field.isTimeDependent:
        if frame is None:
            raise RuntimeError("time-dependent field write requires a frame number")
        var[..., frame - 1] = data
        return
    var[:] = data


def _target_field_data(field: TargetField) -> np.ndarray:
    if field.xtype == FIELD_TYPE_REAL:
        if field.ndims == 1:
            return np.asarray(field.array1r, dtype=np.float32)
        if field.ndims == 2:
            return np.asarray(field.array2r, dtype=np.float32)
        if field.ndims == 3:
            return np.asarray(field.array3r, dtype=np.float32)
        if field.ndims == 4:
            return np.asarray(field.array4r, dtype=np.float32)
        return np.asarray(field.array0r, dtype=np.float32)
    if field.xtype == FIELD_TYPE_DOUBLE:
        if field.ndims == 1:
            return np.asarray(field.array1d, dtype=np.float64)
        if field.ndims == 2:
            return np.asarray(field.array2d, dtype=np.float64)
        if field.ndims == 3:
            return np.asarray(field.array3d, dtype=np.float64)
        if field.ndims == 4:
            return np.asarray(field.array4d, dtype=np.float64)
        return np.asarray(field.array0d, dtype=np.float64)
    if field.xtype == FIELD_TYPE_INTEGER:
        if field.ndims == 1:
            return np.asarray(field.array1i, dtype=np.int32)
        if field.ndims == 2:
            return np.asarray(field.array2i, dtype=np.int32)
        if field.ndims == 3:
            return np.asarray(field.array3i, dtype=np.int32)
        if field.ndims == 4:
            return np.asarray(field.array4i, dtype=np.int32)
        return np.asarray(field.array0i, dtype=np.int32)
    raise RuntimeError(f"unsupported field type for {field.name}")
