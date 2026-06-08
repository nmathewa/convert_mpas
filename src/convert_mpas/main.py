from __future__ import annotations

import sys
from contextlib import suppress

from .copy_atts import add_latlon_atts, copy_field_atts
from .field_list import field_list_finalize, field_list_init, should_remap_field
from .file_output import (
    FILE_MODE_APPEND,
    file_output_close,
    file_output_open,
    file_output_register_field,
    file_output_write_field,
)
from .models import OutputHandle, TargetMesh
from .mpas_mesh import mpas_mesh_free, mpas_mesh_setup
from .remapper import (
    can_remap_field,
    free_target_field,
    remap_field,
    remap_field_dryrun,
    remap_get_target_latitudes,
    remap_get_target_longitudes,
    remap_info_free,
    remap_info_setup,
)
from .scan_input import (
    scan_input_close,
    scan_input_free_field,
    scan_input_next_field,
    scan_input_open,
    scan_input_read_field,
    scan_input_rewind,
)
from .target_mesh import target_mesh_free, target_mesh_setup
from .timer import Timer, timer_start, timer_stop, timer_time


def _usage() -> None:
    print(" ", file=sys.stderr)
    print("Usage: convert_mpas mesh-file [data-files]", file=sys.stderr)
    print(" ", file=sys.stderr)
    print("If only one file argument is given, both the MPAS mesh information and", file=sys.stderr)
    print("the fields will be read from the specified file.", file=sys.stderr)
    print("If two or more file arguments are given, the MPAS mesh information will", file=sys.stderr)
    print("be read from the first file and fields to be remapped will be read from", file=sys.stderr)
    print("the subsequent files.", file=sys.stderr)
    print("All time records from input files will be processed and appended to", file=sys.stderr)
    print("the output file.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 1:
        _usage()
        return 1

    total_timer = Timer()
    read_timer = Timer()
    remap_timer = Timer()
    write_timer = Timer()

    source_mesh = None
    destination_mesh = None
    remap_info = None
    output_handle: OutputHandle | None = None
    include_field_list = None
    exclude_field_list = None
    input_handle = None
    try:
        timer_start(total_timer)
        mesh_filename = args[0]
        data_files = [args[0]] if len(args) == 1 else args[1:]

        print(f"Reading MPAS mesh information from file '{mesh_filename}'", file=sys.stderr)
        destination_mesh = target_mesh_setup(TargetMesh())
        source_mesh = mpas_mesh_setup(mesh_filename)

        print(" ", file=sys.stderr)
        print("Computing remapping weights", file=sys.stderr)
        timer_start(remap_timer)
        remap_info = remap_info_setup(source_mesh, destination_mesh)
        timer_stop(remap_timer)
        print(f"    Time to compute remap weights: {timer_time(remap_timer):10.6f} s", file=sys.stderr)

        output_handle, n_records_out = file_output_open("latlon.nc", mode=FILE_MODE_APPEND)
        if n_records_out != 0:
            print(f"Existing output file has {n_records_out} records", file=sys.stderr)
        else:
            print("Created a new output file", file=sys.stderr)

        include_field_list, exclude_field_list = field_list_init()

        for file_index, data_filename in enumerate(data_files):
            print(f"Remapping MPAS fields from file '{data_filename}'", file=sys.stderr)
            input_handle, n_records_in = scan_input_open(data_filename)
            try:
                print(f"Input file has {n_records_in} records", file=sys.stderr)
                if n_records_out == 0:
                    print(" ", file=sys.stderr)
                    print("Defining fields in output file", file=sys.stderr)
                    lat_field = remap_get_target_latitudes(remap_info)
                    file_output_register_field(output_handle, lat_field)
                    free_target_field(lat_field)

                    lon_field = remap_get_target_longitudes(remap_info)
                    file_output_register_field(output_handle, lon_field)
                    free_target_field(lon_field)

                    field = scan_input_next_field(input_handle)
                    while field is not None:
                        if can_remap_field(field) and should_remap_field(field, include_field_list, exclude_field_list):
                            target_field = remap_field_dryrun(remap_info, field)
                            file_output_register_field(output_handle, target_field)
                            copy_field_atts(input_handle, field, output_handle, target_field)
                            free_target_field(target_field)
                        scan_input_free_field(field)
                        field = scan_input_next_field(input_handle)

                    lat_field = remap_get_target_latitudes(remap_info)
                    file_output_write_field(output_handle, lat_field)
                    free_target_field(lat_field)

                    lon_field = remap_get_target_longitudes(remap_info)
                    file_output_write_field(output_handle, lon_field)
                    free_target_field(lon_field)
                    add_latlon_atts(output_handle)

                for irec in range(1, n_records_in + 1):
                    scan_input_rewind(input_handle)
                    field = scan_input_next_field(input_handle)
                    while field is not None:
                        if can_remap_field(field) and should_remap_field(field, include_field_list, exclude_field_list):
                            print(f"Remapping field {field.name}, frame {irec}", file=sys.stderr)
                            timer_start(read_timer)
                            scan_input_read_field(field, frame=irec)
                            timer_stop(read_timer)
                            print(f"    read:  {timer_time(read_timer):10.6f} s", file=sys.stderr)

                            timer_start(remap_timer)
                            target_field = remap_field(remap_info, field)
                            timer_stop(remap_timer)
                            print(f"    remap: {timer_time(remap_timer):10.6f} s", file=sys.stderr)

                            timer_start(write_timer)
                            file_output_write_field(output_handle, target_field, frame=n_records_out + irec)
                            timer_stop(write_timer)
                            print(f"    write: {timer_time(write_timer):10.6f} s", file=sys.stderr)
                            free_target_field(target_field)
                        scan_input_free_field(field)
                        field = scan_input_next_field(input_handle)

                n_records_out += n_records_in
            finally:
                scan_input_close(input_handle)
                input_handle = None

        file_output_close(output_handle)
        output_handle = None
        mpas_mesh_free(source_mesh)
        source_mesh = None
        target_mesh_free(destination_mesh)
        destination_mesh = None
        remap_info_free(remap_info)
        remap_info = None
        field_list_finalize(include_field_list, exclude_field_list)
        include_field_list = None
        exclude_field_list = None
        timer_stop(total_timer)
        print(" ", file=sys.stderr)
        print(f"Total runtime: {timer_time(total_timer):10.6f}", file=sys.stderr)
        print(" ", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        with suppress(Exception):
            if input_handle is not None:
                scan_input_close(input_handle)
        with suppress(Exception):
            if output_handle is not None:
                file_output_close(output_handle)
        if source_mesh is not None:
            mpas_mesh_free(source_mesh)
        if destination_mesh is not None:
            target_mesh_free(destination_mesh)
        if remap_info is not None:
            remap_info_free(remap_info)
        if include_field_list is not None and exclude_field_list is not None:
            field_list_finalize(include_field_list, exclude_field_list)


if __name__ == "__main__":
    raise SystemExit(main())
