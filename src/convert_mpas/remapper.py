from __future__ import annotations

import math

import numpy as np

from .models import (
    FIELD_TYPE_DOUBLE,
    FIELD_TYPE_INTEGER,
    FIELD_TYPE_REAL,
    InputField,
    MPASMesh,
    RemapInfo,
    TargetField,
    TargetMesh,
)

MPAS_CELL_FIELD = 0x01
MPAS_VTX_FIELD = 0x02
MPAS_EDGE_FIELD = 0x04
CAM_CELL_FIELD = 0x10
CAM_VTX_FIELD = 0x20
CAM_EDGE_FIELD = 0x40
UNSUPPORTED_FIELD = 0x00

MPAS_MASK = 0x07
CAM_MASK = 0x70
CELL_MASK = 0x11
VTX_MASK = 0x22
EDGE_MASK = 0x44


def index2d(irank: int, idx: int) -> int:
    return irank * (idx - 1) + 1


def convert_lx(lat: float, lon: float, radius: float) -> np.ndarray:
    return np.array(
        [
            radius * math.cos(lon) * math.cos(lat),
            radius * math.sin(lon) * math.cos(lat),
            radius * math.sin(lat),
        ],
        dtype=np.float64,
    )


def sphere_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> float:
    arg1 = math.sqrt(
        math.sin(0.5 * (lat2 - lat1)) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(0.5 * (lon2 - lon1)) ** 2
    )
    arg1 = min(1.0, max(0.0, arg1))
    return 2.0 * radius * math.asin(arg1)


def mpas_arc_length(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    cx = bx - ax
    cy = by - ay
    cz = bz - az
    r = math.sqrt(ax * ax + ay * ay + az * az)
    c = math.sqrt(cx * cx + cy * cy + cz * cz)
    return r * 2.0 * math.asin(min(1.0, c / (2.0 * r)))


def mpas_triangle_signed_area_sphere(a: np.ndarray, b: np.ndarray, c: np.ndarray, radius: float) -> float:
    ab = mpas_arc_length(a[0], a[1], a[2], b[0], b[1], b[2]) / radius
    bc = mpas_arc_length(b[0], b[1], b[2], c[0], c[1], c[2]) / radius
    ca = mpas_arc_length(c[0], c[1], c[2], a[0], a[1], a[2]) / radius
    semiperim = 0.5 * (ab + bc + ca)
    tanqe = math.sqrt(
        max(
            0.0,
            math.tan(0.5 * semiperim)
            * math.tan(0.5 * (semiperim - ab))
            * math.tan(0.5 * (semiperim - bc))
            * math.tan(0.5 * (semiperim - ca)),
        )
    )
    area = 4.0 * radius * radius * math.atan(tanqe)

    ablen = b - a
    aclen = c - a
    dlen = np.array(
        [
            (ablen[1] * aclen[2]) - (ablen[2] * aclen[1]),
            -((ablen[0] * aclen[2]) - (ablen[2] * aclen[0])),
            (ablen[0] * aclen[1]) - (ablen[1] * aclen[0]),
        ],
        dtype=np.float64,
    )
    if float(dlen[0] * a[0] + dlen[1] * a[1] + dlen[2] * a[2]) < 0.0:
        area = -area
    return area


def mpas_wachspress_coordinates(
    n_vertices: int, vert_coords: np.ndarray, point_interp: np.ndarray, area_bin: np.ndarray | None = None
) -> np.ndarray:
    radius_local = math.sqrt(float(np.sum(vert_coords[:, 0] ** 2)))
    area_b = np.zeros(n_vertices, dtype=np.float64)
    if area_bin is None:
        for i in range(n_vertices):
            im1 = (n_vertices + i - 1) % n_vertices
            i0 = i % n_vertices
            ip1 = (i + 1) % n_vertices
            area_b[i] = mpas_triangle_signed_area_sphere(
                vert_coords[:, im1], vert_coords[:, i0], vert_coords[:, ip1], radius_local
            )
    else:
        area_b[:] = area_bin[:]

    area_a = np.zeros(n_vertices, dtype=np.float64)
    for i in range(n_vertices):
        i0 = i % n_vertices
        ip1 = (i + 1) % n_vertices
        area_a[i0] = mpas_triangle_signed_area_sphere(point_interp, vert_coords[:, i0], vert_coords[:, ip1], radius_local)

    wach = np.zeros(n_vertices, dtype=np.float64)
    for i in range(n_vertices):
        wach[i] = area_b[i]
        for j in range(i + 1, i + n_vertices - 1):
            i0 = j % n_vertices
            wach[i] *= area_a[i0]

    total = float(np.sum(wach))
    if total == 0.0:
        return np.zeros(n_vertices, dtype=np.float64)
    return wach / total


def interior_element(
    neighbors_on_element: np.ndarray, n_neighbors: np.ndarray | None = None, n_neighbors_constant: int | None = None
) -> int:
    n_elements = neighbors_on_element.shape[1]
    if n_neighbors is not None:
        for i in range(n_elements):
            for j in range(int(n_neighbors[i])):
                if neighbors_on_element[j, i] == 0:
                    break
            else:
                return i + 1
    elif n_neighbors_constant is not None:
        for i in range(n_elements):
            for j in range(n_neighbors_constant):
                if neighbors_on_element[j, i] == 0:
                    break
            else:
                return i + 1
    return 0


def nearest_cell(
    target_lat: float,
    target_lon: float,
    start_cell: int,
    n_cells: int,
    max_edges: int,
    n_edges_on_cell: np.ndarray,
    cells_on_cell: np.ndarray,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
) -> int:
    nearest = start_cell
    current = -1
    while nearest != current:
        current = nearest
        current_distance = sphere_distance(lat_cell[current - 1], lon_cell[current - 1], target_lat, target_lon, 1.0)
        nearest = current
        nearest_distance = current_distance
        for i in range(int(n_edges_on_cell[current - 1])):
            neighbor = int(cells_on_cell[i, current - 1])
            if neighbor > 0 and neighbor <= n_cells:
                d = sphere_distance(lat_cell[neighbor - 1], lon_cell[neighbor - 1], target_lat, target_lon, 1.0)
                if d < nearest_distance:
                    nearest = neighbor
                    nearest_distance = d
            else:
                return 0
    return nearest


def nearest_vertex(
    target_lat: float,
    target_lon: float,
    start_vertex: int,
    n_cells: int,
    n_vertices: int,
    max_edges: int,
    vertex_degree: int,
    n_edges_on_cell: np.ndarray,
    vertices_on_cell: np.ndarray,
    cells_on_vertex: np.ndarray,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    lat_vertex: np.ndarray,
    lon_vertex: np.ndarray,
) -> int:
    nearest = start_vertex
    current = -1
    while nearest != current:
        current = nearest
        current_distance = sphere_distance(
            lat_vertex[current - 1], lon_vertex[current - 1], target_lat, target_lon, 1.0
        )
        nearest = current
        nearest_distance = current_distance
        cell1 = int(cells_on_vertex[0, current - 1])
        if cell1 <= 0:
            return 0
        cell1_dist = sphere_distance(lat_cell[cell1 - 1], lon_cell[cell1 - 1], target_lat, target_lon, 1.0)
        cell2 = int(cells_on_vertex[1, current - 1])
        if cell2 <= 0:
            return 0
        cell2_dist = sphere_distance(lat_cell[cell2 - 1], lon_cell[cell2 - 1], target_lat, target_lon, 1.0)
        if vertex_degree == 3:
            cell3 = int(cells_on_vertex[2, current - 1])
            if cell3 <= 0:
                return 0
            cell3_dist = sphere_distance(lat_cell[cell3 - 1], lon_cell[cell3 - 1], target_lat, target_lon, 1.0)
            if cell1_dist < cell2_dist:
                i_cell = cell1 if cell1_dist < cell3_dist else cell3
            else:
                i_cell = cell2 if cell2_dist < cell3_dist else cell3
        else:
            i_cell = cell1 if cell1_dist < cell2_dist else cell2
        for i in range(int(n_edges_on_cell[i_cell - 1])):
            neighbor = int(vertices_on_cell[i, i_cell - 1])
            d = sphere_distance(lat_vertex[neighbor - 1], lon_vertex[neighbor - 1], target_lat, target_lon, 1.0)
            if d < nearest_distance:
                nearest = neighbor
                nearest_distance = d
    return nearest


def field_class(dimnames: np.ndarray | list[str]) -> int:
    dims = [str(name).strip() for name in dimnames]
    if not dims:
        return UNSUPPORTED_FIELD
    decomp_dim = len(dims)
    if dims[-1] == "Time":
        decomp_dim -= 1
    if decomp_dim <= 0:
        return UNSUPPORTED_FIELD
    dim = dims[decomp_dim - 1]
    if dim == "nCells":
        return MPAS_CELL_FIELD
    if dim == "nVertices":
        return MPAS_VTX_FIELD
    if dim == "nEdges":
        return MPAS_EDGE_FIELD
    dim = dims[0]
    if dim in {"nCells", "ncol"}:
        return CAM_CELL_FIELD
    if dim == "nVertices":
        return CAM_VTX_FIELD
    if dim == "nEdges":
        return CAM_EDGE_FIELD
    return UNSUPPORTED_FIELD


def can_remap_field(field: InputField) -> bool:
    if field.xtype not in {FIELD_TYPE_INTEGER, FIELD_TYPE_REAL, FIELD_TYPE_DOUBLE}:
        return False
    if field.ndims == 0 or (field.ndims == 1 and field.isTimeDependent):
        return False
    fld_class = field_class(field.dimnames if field.dimnames is not None else [])
    return bool(fld_class & MPAS_MASK or fld_class & CAM_MASK)


def _horizontal_axis(fld_class: int) -> int:
    return -1 if fld_class & MPAS_MASK else 0


def _source_extras(src_array: np.ndarray, axis: int) -> tuple[int, ...]:
    if axis == -1:
        return src_array.shape[:-1]
    return src_array.shape[1:]


def _remap_array(
    src_array: np.ndarray,
    nearest_index: np.ndarray,
    source_nodes: np.ndarray,
    node_weights: np.ndarray,
    fld_class: int,
    fill_value: object,
) -> np.ndarray:
    axis = _horizontal_axis(fld_class)
    extras = _source_extras(src_array, axis)
    out = np.full((nearest_index.shape[0], nearest_index.shape[1]) + extras, fill_value, dtype=src_array.dtype)
    for iy in range(nearest_index.shape[1]):
        for ix in range(nearest_index.shape[0]):
            if np.sum(node_weights[:, ix, iy]) == 0.0:
                continue
            acc = None
            for j in range(source_nodes.shape[0]):
                src_idx = int(source_nodes[j, ix, iy])
                if src_idx <= 0:
                    continue
                slc = np.take(src_array, src_idx - 1, axis=axis)
                term = node_weights[j, ix, iy] * slc
                acc = term if acc is None else acc + term
            if acc is not None:
                out[ix, iy, ...] = acc
    return out


def remap_info_setup(src_mesh: MPASMesh, dst_mesh: TargetMesh) -> RemapInfo:
    remap_info = RemapInfo(method=1, src_mesh=src_mesh, dst_mesh=dst_mesh)
    remap_info.nearestCell = np.zeros((dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)
    remap_info.nearestVertex = np.zeros((dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)
    remap_info.nearestEdge = np.zeros((dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)

    irank = dst_mesh.irank

    last_idx = interior_element(src_mesh.cellsOnCell, n_neighbors=src_mesh.nEdgesOnCell)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_cell(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.maxEdges,
                src_mesh.nEdgesOnCell,
                src_mesh.cellsOnCell,
                src_mesh.latCell,
                src_mesh.lonCell,
            )
            remap_info.nearestCell[ix, iy] = idx
            if idx > 0:
                last_idx = idx

    last_idx = interior_element(src_mesh.cellsOnVertex, n_neighbors_constant=3)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_vertex(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.nVertices,
                src_mesh.maxEdges,
                3,
                src_mesh.nEdgesOnCell,
                src_mesh.verticesOnCell,
                src_mesh.cellsOnVertex,
                src_mesh.latCell,
                src_mesh.lonCell,
                src_mesh.latVertex,
                src_mesh.lonVertex,
            )
            remap_info.nearestVertex[ix, iy] = idx
            if idx > 0:
                last_idx = idx

    last_idx = interior_element(src_mesh.cellsOnEdge, n_neighbors_constant=2)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_vertex(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.nEdges,
                src_mesh.maxEdges,
                2,
                src_mesh.nEdgesOnCell,
                src_mesh.edgesOnCell,
                src_mesh.cellsOnEdge,
                src_mesh.latCell,
                src_mesh.lonCell,
                src_mesh.latEdge,
                src_mesh.lonEdge,
            )
            remap_info.nearestEdge[ix, iy] = idx
            if idx > 0:
                last_idx = idx

    remap_info.cellWeights = np.zeros((3, dst_mesh.nlon, dst_mesh.nlat), dtype=np.float64)
    remap_info.sourceCells = np.ones((3, dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)
    last_idx = interior_element(src_mesh.cellsOnCell, n_neighbors=src_mesh.nEdgesOnCell)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_vertex(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.nVertices,
                src_mesh.maxEdges,
                3,
                src_mesh.nEdgesOnCell,
                src_mesh.verticesOnCell,
                src_mesh.cellsOnVertex,
                src_mesh.latCell,
                src_mesh.lonCell,
                src_mesh.latVertex,
                src_mesh.lonVertex,
            )
            if idx > 0:
                remap_info.sourceCells[:, ix, iy] = src_mesh.cellsOnVertex[:, idx - 1]
                point_interp = convert_lx(lat, lon, 6371229.0)
                vert_coords = np.zeros((3, 3), dtype=np.float64)
                for j in range(3):
                    cell = int(src_mesh.cellsOnVertex[j, idx - 1])
                    vert_coords[:, j] = convert_lx(src_mesh.latCell[cell - 1], src_mesh.lonCell[cell - 1], 6371229.0)
                remap_info.cellWeights[:, ix, iy] = mpas_wachspress_coordinates(3, vert_coords, point_interp)
                last_idx = idx
            else:
                remap_info.cellWeights[:, ix, iy] = 0.0

    remap_info.vertexWeights = np.zeros((src_mesh.maxEdges, dst_mesh.nlon, dst_mesh.nlat), dtype=np.float64)
    remap_info.sourceVertices = np.ones((src_mesh.maxEdges, dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)
    last_idx = interior_element(src_mesh.cellsOnVertex, n_neighbors_constant=3)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_cell(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.maxEdges,
                src_mesh.nEdgesOnCell,
                src_mesh.cellsOnCell,
                src_mesh.latCell,
                src_mesh.lonCell,
            )
            if idx > 0:
                nn = int(src_mesh.nEdgesOnCell[idx - 1])
                remap_info.sourceVertices[:, ix, iy] = 1
                remap_info.sourceVertices[:nn, ix, iy] = src_mesh.verticesOnCell[:nn, idx - 1]
                point_interp = convert_lx(lat, lon, 6371229.0)
                vert_coords = np.zeros((3, nn), dtype=np.float64)
                for j in range(nn):
                    vtx = int(src_mesh.verticesOnCell[j, idx - 1])
                    vert_coords[:, j] = convert_lx(src_mesh.latVertex[vtx - 1], src_mesh.lonVertex[vtx - 1], 6371229.0)
                remap_info.vertexWeights[:nn, ix, iy] = mpas_wachspress_coordinates(nn, vert_coords, point_interp)
                last_idx = idx
            else:
                remap_info.vertexWeights[:, ix, iy] = 0.0

    remap_info.edgeWeights = np.zeros((src_mesh.maxEdges, dst_mesh.nlon, dst_mesh.nlat), dtype=np.float64)
    remap_info.sourceEdges = np.ones((src_mesh.maxEdges, dst_mesh.nlon, dst_mesh.nlat), dtype=np.int32)
    last_idx = interior_element(src_mesh.cellsOnEdge, n_neighbors_constant=2)
    for iy in range(dst_mesh.nlat):
        for ix in range(dst_mesh.nlon):
            lat = float(dst_mesh.lats[index2d(irank, ix + 1) - 1, iy])  # type: ignore[index]
            lon = float(dst_mesh.lons[ix, index2d(irank, iy + 1) - 1])  # type: ignore[index]
            idx = nearest_cell(
                lat,
                lon,
                last_idx,
                src_mesh.nCells,
                src_mesh.maxEdges,
                src_mesh.nEdgesOnCell,
                src_mesh.cellsOnCell,
                src_mesh.latCell,
                src_mesh.lonCell,
            )
            if idx > 0:
                nn = int(src_mesh.nEdgesOnCell[idx - 1])
                remap_info.sourceEdges[:, ix, iy] = 1
                remap_info.sourceEdges[:nn, ix, iy] = src_mesh.edgesOnCell[:nn, idx - 1]
                point_interp = convert_lx(lat, lon, 6371229.0)
                vert_coords = np.zeros((3, nn), dtype=np.float64)
                for j in range(nn):
                    edge = int(src_mesh.edgesOnCell[j, idx - 1])
                    vert_coords[:, j] = convert_lx(src_mesh.latEdge[edge - 1], src_mesh.lonEdge[edge - 1], 6371229.0)
                remap_info.edgeWeights[:nn, ix, iy] = mpas_wachspress_coordinates(nn, vert_coords, point_interp)
                last_idx = idx
            else:
                remap_info.edgeWeights[:, ix, iy] = 0.0

    return remap_info


def remap_info_free(remap_info: RemapInfo) -> None:
    remap_info.method = -1
    remap_info.src_mesh = None
    remap_info.dst_mesh = None
    remap_info.nearestCell = None
    remap_info.nearestVertex = None
    remap_info.nearestEdge = None
    remap_info.cellWeights = None
    remap_info.vertexWeights = None
    remap_info.edgeWeights = None
    remap_info.sourceCells = None
    remap_info.sourceVertices = None
    remap_info.sourceEdges = None


def remap_field_dryrun(remap_info: RemapInfo, src_field: InputField) -> TargetField:
    fld_class = field_class(src_field.dimnames if src_field.dimnames is not None else [])
    dst = TargetField(name=src_field.name, xtype=src_field.xtype, isTimeDependent=src_field.isTimeDependent)
    if fld_class & MPAS_MASK:
        if src_field.isTimeDependent:
            extra_names = list(src_field.dimnames[:-2])  # type: ignore[index]
            extra_lens = list(src_field.dimlens[:-2])  # type: ignore[index]
        else:
            extra_names = list(src_field.dimnames[:-1])  # type: ignore[index]
            extra_lens = list(src_field.dimlens[:-1])  # type: ignore[index]
    else:
        if src_field.isTimeDependent:
            extra_names = list(src_field.dimnames[1:-1])  # type: ignore[index]
            extra_lens = list(src_field.dimlens[1:-1])  # type: ignore[index]
        else:
            extra_names = list(src_field.dimnames[1:])  # type: ignore[index]
            extra_lens = list(src_field.dimlens[1:])  # type: ignore[index]
    dst.dimnames = np.array(["longitude", "latitude", *[str(name) for name in extra_names]], dtype=object)
    dst.dimlens = np.array([remap_info.dst_mesh.nlon, remap_info.dst_mesh.nlat, *extra_lens], dtype=np.int64)
    dst.ndims = len(dst.dimlens)
    return dst


def remap_field(remap_info: RemapInfo, src_field: InputField) -> TargetField:
    fld_class = field_class(src_field.dimnames if src_field.dimnames is not None else [])
    dst = remap_field_dryrun(remap_info, src_field)
    if src_field.xtype == FIELD_TYPE_REAL:
        if src_field.array1r is not None:
            src_array = np.asarray(src_field.array1r, dtype=np.float32)
        elif src_field.array2r is not None:
            src_array = np.asarray(src_field.array2r, dtype=np.float32)
        elif src_field.array3r is not None:
            src_array = np.asarray(src_field.array3r, dtype=np.float32)
        elif src_field.array4r is not None:
            src_array = np.asarray(src_field.array4r, dtype=np.float32)
        else:
            src_array = np.asarray(src_field.array0r, dtype=np.float32)
    elif src_field.xtype == FIELD_TYPE_DOUBLE:
        if src_field.array1d is not None:
            src_array = np.asarray(src_field.array1d, dtype=np.float64)
        elif src_field.array2d is not None:
            src_array = np.asarray(src_field.array2d, dtype=np.float64)
        elif src_field.array3d is not None:
            src_array = np.asarray(src_field.array3d, dtype=np.float64)
        elif src_field.array4d is not None:
            src_array = np.asarray(src_field.array4d, dtype=np.float64)
        else:
            src_array = np.asarray(src_field.array0d, dtype=np.float64)
    elif src_field.xtype == FIELD_TYPE_INTEGER:
        if src_field.array1i is not None:
            src_array = np.asarray(src_field.array1i, dtype=np.int32)
        elif src_field.array2i is not None:
            src_array = np.asarray(src_field.array2i, dtype=np.int32)
        elif src_field.array3i is not None:
            src_array = np.asarray(src_field.array3i, dtype=np.int32)
        else:
            src_array = np.asarray(src_field.array0i, dtype=np.int32)
    else:
        raise RuntimeError(f"unsupported field type: {src_field.xtype}")

    if fld_class & MPAS_MASK:
        nearest_index = remap_info.nearestCell if fld_class & CELL_MASK else remap_info.nearestVertex if fld_class & VTX_MASK else remap_info.nearestEdge
        source_nodes = remap_info.sourceCells if fld_class & CELL_MASK else remap_info.sourceVertices if fld_class & VTX_MASK else remap_info.sourceEdges
        node_weights = remap_info.cellWeights if fld_class & CELL_MASK else remap_info.vertexWeights if fld_class & VTX_MASK else remap_info.edgeWeights
    else:
        nearest_index = remap_info.nearestCell if fld_class & CELL_MASK else remap_info.nearestVertex if fld_class & VTX_MASK else remap_info.nearestEdge
        source_nodes = remap_info.sourceCells if fld_class & CELL_MASK else remap_info.sourceVertices if fld_class & VTX_MASK else remap_info.sourceEdges
        node_weights = remap_info.cellWeights if fld_class & CELL_MASK else remap_info.vertexWeights if fld_class & VTX_MASK else remap_info.edgeWeights

    if src_field.xtype == FIELD_TYPE_INTEGER:
        axis = _horizontal_axis(fld_class)
        extras = _source_extras(src_array, axis)
        out = np.full((nearest_index.shape[0], nearest_index.shape[1]) + extras, -2147483647, dtype=np.int32)
        for iy in range(nearest_index.shape[1]):
            for ix in range(nearest_index.shape[0]):
                idx = int(nearest_index[ix, iy])
                if idx <= 0:
                    continue
                out[ix, iy, ...] = np.take(src_array, idx - 1, axis=axis)
        if out.ndim == 2:
            dst.array2i = out
        elif out.ndim == 3:
            dst.array3i = out
        elif out.ndim == 4:
            dst.array4i = out
        else:
            dst.array1i = out
        return dst

    fill_value = np.float32(np.nan) if src_field.xtype == FIELD_TYPE_REAL else np.float64(np.nan)
    out = _remap_array(src_array, nearest_index, source_nodes, node_weights, fld_class, fill_value)
    if src_field.xtype == FIELD_TYPE_REAL:
        if out.ndim == 2:
            dst.array2r = out
        elif out.ndim == 3:
            dst.array3r = out
        elif out.ndim == 4:
            dst.array4r = out
        else:
            dst.array1r = out
    else:
        if out.ndim == 2:
            dst.array2d = out
        elif out.ndim == 3:
            dst.array3d = out
        elif out.ndim == 4:
            dst.array4d = out
        else:
            dst.array1d = out
    return dst


def remap_get_target_latitudes(remap_info: RemapInfo) -> TargetField:
    lat = TargetField(name="latitude", xtype=FIELD_TYPE_REAL, ndims=1, isTimeDependent=False)
    lat.dimnames = np.array(["latitude"], dtype=object)
    lat.dimlens = np.array([remap_info.dst_mesh.nlat], dtype=np.int64)
    lat.array1r = np.asarray(remap_info.dst_mesh.lats[0, :], dtype=np.float32) * (90.0 / math.asin(1.0))
    return lat


def remap_get_target_longitudes(remap_info: RemapInfo) -> TargetField:
    lon = TargetField(name="longitude", xtype=FIELD_TYPE_REAL, ndims=1, isTimeDependent=False)
    lon.dimnames = np.array(["longitude"], dtype=object)
    lon.dimlens = np.array([remap_info.dst_mesh.nlon], dtype=np.int64)
    lon.array1r = np.asarray(remap_info.dst_mesh.lons[:, 0], dtype=np.float32) * (90.0 / math.asin(1.0))
    return lon


def free_target_field(field: TargetField) -> None:
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
    field.array4i = None
