from __future__ import annotations

from .models import OutputHandle, TargetField


def copy_field_atts(handle, field, output_handle: OutputHandle, target_field: TargetField) -> None:
    src = handle.dataset.variables[field.varname]
    dst = output_handle.dataset.variables[target_field.name]
    for attname in src.ncattrs():
        if attname == "_FillValue":
            continue
        dst.setncattr(attname, src.getncattr(attname))


def add_latlon_atts(handle: OutputHandle) -> None:
    lat = handle.dataset.variables["latitude"]
    lat.setncattr("units", "degree_north")
    lat.setncattr("long_name", "latitude")
    lat.setncattr("standard_name", "latitude")

    lon = handle.dataset.variables["longitude"]
    lon.setncattr("units", "degree_east")
    lon.setncattr("long_name", "longitude")
    lon.setncattr("standard_name", "longitude")
