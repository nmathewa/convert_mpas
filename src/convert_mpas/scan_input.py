from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from .models import (
    FIELD_TYPE_CHARACTER,
    FIELD_TYPE_DOUBLE,
    FIELD_TYPE_INTEGER,
    FIELD_TYPE_REAL,
    FIELD_TYPE_UNSUPPORTED,
    InputField,
    InputHandle,
)


def _xtype_from_var(var: object) -> int:
    dtype = np.dtype(var.dtype)
    if dtype.kind == "f" and dtype.itemsize == 4:
        return FIELD_TYPE_REAL
    if dtype.kind == "f" and dtype.itemsize == 8:
        return FIELD_TYPE_DOUBLE
    if dtype.kind in {"i", "u"}:
        return FIELD_TYPE_INTEGER
    if dtype.kind in {"S", "U", "O"}:
        return FIELD_TYPE_CHARACTER
    return FIELD_TYPE_UNSUPPORTED


def _build_field(handle: InputHandle, varname: str) -> InputField:
    var = handle.dataset.variables[varname]
    field = InputField(
        name=varname,
        varname=varname,
        xtype=_xtype_from_var(var),
        ndims=len(var.dimensions),
        dimnames=np.array(var.dimensions, dtype=object),
        dimlens=np.array([len(handle.dataset.dimensions[d]) for d in var.dimensions], dtype=np.int64),
        dimids=np.arange(len(var.dimensions), dtype=np.int64),
        file_handle=handle,
    )
    if handle.unlimited_dim_name is not None and handle.unlimited_dim_name in var.dimensions:
        field.isTimeDependent = True
    return field


def scan_input_open(filename: str) -> tuple[InputHandle, int]:
    dataset = Dataset(filename, "r")
    varnames = list(dataset.variables.keys())
    unlimited_dim_name = next((name for name, dim in dataset.dimensions.items() if dim.isunlimited()), None)
    handle = InputHandle(
        dataset=dataset,
        num_vars=len(varnames),
        current_var=0,
        varnames=varnames,
        unlimited_dim_name=unlimited_dim_name,
    )
    if unlimited_dim_name is not None:
        n_records = len(dataset.dimensions[unlimited_dim_name])
    else:
        n_records = 0
        if handle.num_vars > 0:
            n_records = 1
    return handle, n_records


def scan_input_close(handle: InputHandle) -> None:
    handle.dataset.close()
    handle.current_var = 0


def scan_input_rewind(handle: InputHandle) -> None:
    handle.current_var = 0


def scan_input_next_field(handle: InputHandle) -> InputField | None:
    if handle.current_var < 0 or handle.current_var >= handle.num_vars:
        return None
    field = _build_field(handle, handle.varnames[handle.current_var])
    handle.current_var += 1
    return field


def scan_input_for_field(handle: InputHandle, fieldname: str) -> InputField:
    if fieldname not in handle.dataset.variables:
        raise KeyError(fieldname)
    return _build_field(handle, fieldname)


def _filled(data: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled())
    return np.asarray(data)


def scan_input_read_field(field: InputField, frame: int | None = None) -> None:
    if field.file_handle is None:
        raise RuntimeError("field is not associated with a file handle")
    var = field.file_handle.dataset.variables[field.varname]
    data = var[...]
    if field.isTimeDependent:
        if field.dimnames is None:
            raise RuntimeError(f"time-dependent field {field.name!r} is missing dimension metadata")
        try:
            time_axis = list(field.dimnames).index("Time")
        except ValueError as exc:
            raise RuntimeError(f"time-dependent field {field.name!r} has no Time dimension") from exc
        if frame is None:
            frame = 1
        data = np.take(data, frame - 1, axis=time_axis)
    data = _filled(data)

    if field.xtype == FIELD_TYPE_REAL:
        data = np.asarray(data, dtype=np.float32)
    elif field.xtype == FIELD_TYPE_DOUBLE:
        data = np.asarray(data, dtype=np.float64)
    elif field.xtype == FIELD_TYPE_INTEGER:
        data = np.asarray(data, dtype=np.int32)
    else:
        raise RuntimeError(f"unsupported field type for read: {field.name}")

    if data.ndim == 0:
        if field.xtype == FIELD_TYPE_REAL:
            field.array0r = float(data)
        elif field.xtype == FIELD_TYPE_DOUBLE:
            field.array0d = float(data)
        else:
            field.array0i = int(data)
        return

    if data.ndim == 1:
        if field.xtype == FIELD_TYPE_REAL:
            field.array1r = data
        elif field.xtype == FIELD_TYPE_DOUBLE:
            field.array1d = data
        else:
            field.array1i = data
    elif data.ndim == 2:
        if field.xtype == FIELD_TYPE_REAL:
            field.array2r = data
        elif field.xtype == FIELD_TYPE_DOUBLE:
            field.array2d = data
        else:
            field.array2i = data
    elif data.ndim == 3:
        if field.xtype == FIELD_TYPE_REAL:
            field.array3r = data
        elif field.xtype == FIELD_TYPE_DOUBLE:
            field.array3d = data
        else:
            field.array3i = data
    elif data.ndim == 4:
        if field.xtype == FIELD_TYPE_REAL:
            field.array4r = data
        elif field.xtype == FIELD_TYPE_DOUBLE:
            field.array4d = data
        else:
            raise RuntimeError(f"unsupported integer rank for field {field.name}")
    else:
        raise RuntimeError(f"unsupported rank for field {field.name}")


def scan_input_free_field(field: InputField) -> None:
    field.dimids = None
    field.dimlens = None
    field.dimnames = None
    field.array1r = None
    field.array2r = None
    field.array3r = None
    field.array4r = None
    field.array1d = None
    field.array2d = None
    field.array3d = None
    field.array4d = None
    field.array1i = None
    field.array2i = None
    field.array3i = None
    field.file_handle = None
