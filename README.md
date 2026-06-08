convert_mpas
============

The `convert_mpas` project provides a Python module for mapping native MPAS
output to a regular lat/lon grid.

## Installing:

The package uses `numpy`, `netCDF4`, `xarray`, `uxarray`, `dask`, and
`rasterio`. After creating an environment, install the project in editable
mode:

```bash
pip install -e .
```

## Running:

Run the CLI entry point with one or more command-line arguments:

- If only one argument is given, both the MPAS mesh information and the fields
  will be read from the specified file.
- If two or more file arguments are given, the MPAS mesh information will be
  read from the first file and fields to be remapped will be read from the
  subsequent files.

All time records from input files are processed and appended to `latlon.nc`.
Running the command with no arguments prints a usage summary.

Add `--rasterize` to write a GeoTIFF (`latlon.tif`) with `T2m`, `mslp`, and
`humidity_2m` bands from the first time step:

```bash
python -m convert_mpas --rasterize mesh.nc data.nc
```

By default, the 'convert_mpas' will remap all integer, real, or double-precision
fields that it finds in the input data file. However, by creating a list of
fields in a file named 'include_fields' in the run directory, with one field name 
per line, the 'convert_mpas' program will remap only those fields listed in 
the file. Alternatively, one can create a list of fields to be excluded from 
the output file; this list should be written to a file named 'exclude_fields'.
If both an 'include_fields' file and an 'exclude_fields' file are present in 
the run directory, only fields listed in the 'include_fields' file will be 
remapped, and the contents of the 'exclude_fields' file are ignored.

The target domain defaults to a 0.5x0.5-degree global lat-lon grid. However, one
may specify an alternate target domain using a file named 'target_domain' in 
the run directory. This file may contain lines assigning values to keywords, i.e.,

keyword = value

The following are available keywords for describing the target domain:
 - nlat : the number of latitude points in the grid (default value 360)
 - nlon : the number of longitude points in the grid (default value 720)
 - startlat : the starting latitude (default value -90.0)
 - startlon : the starting longitude (default value -180.0)
 - endlat : the ending latitude (default value 90.0)
 - endlon : the ending longitude (default value 180.0)

The actual points to which fields are interpolated are determined by dividing
the latitude and longitude ranges into the specified number of intervals, then
locating the interpolation points at the center of these intervals. For example,
specifying startlat=0, endlat=10, and nlat=10 would result in target latitudes
of 0.5, 1.5, ..., 8.5, and 9.5.

## Interpolation methodology:

Integer fields are remapped to the target grid using a nearest-neighbor scheme.
For all real-valued (single- or double-precision) fields, the 'convert_mpas' program
employs a barycentric interpolation, the output of which is C0 continuous. Cell-based
fields in the MPAS mesh are sampled from the three cell centers that form the vertices
of the Delaunay triangle containing the target point. Vertex- and edge-based fields
are sampled from the corners or faces, respectively, of the Voronoi cell containing
the target point.

## To-do:
- Transfer 'xtime' variable from input files to output file
- Ensure that, for cell fields, the interpolation location lies within the triangle 
  used for interpolation
- Make sure that, when dealing with existing output files, the target mesh matches
  what is found in the output file
- Experiment with OpenMP directives to speed up interpolation
- Allow locations of 'include_fields', 'exclude_fields', and 'target_domain' files
  to be specified with environment variables
- Decide what to do if input file contains no unlimited dimension
