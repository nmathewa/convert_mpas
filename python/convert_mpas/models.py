from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from netCDF4 import Dataset

FIELD_TYPE_UNSUPPORTED = -1
FIELD_TYPE_REAL = 1
FIELD_TYPE_DOUBLE = 2
FIELD_TYPE_INTEGER = 3
FIELD_TYPE_CHARACTER = 4


@dataclass
class Timer:
    count_start: int = 0
    count_stop: int = 0
    count_rate: int = 1


@dataclass
class FieldList:
    num_fields: int = 0
    field_names: list[str] = field(default_factory=list)


@dataclass
class TargetMesh:
    valid: bool = False
    irank: int = 0
    nlat: int = 0
    nlon: int = 0
    startlat: float = 0.0
    endlat: float = 0.0
    startlon: float = 0.0
    endlon: float = 0.0
    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None


@dataclass
class MPASMesh:
    valid: bool = False
    nCells: int = 0
    nVertices: int = 0
    nEdges: int = 0
    maxEdges: int = 0
    nEdgesOnCell: Optional[np.ndarray] = None
    cellsOnCell: Optional[np.ndarray] = None
    verticesOnCell: Optional[np.ndarray] = None
    cellsOnVertex: Optional[np.ndarray] = None
    edgesOnCell: Optional[np.ndarray] = None
    cellsOnEdge: Optional[np.ndarray] = None
    latCell: Optional[np.ndarray] = None
    lonCell: Optional[np.ndarray] = None
    latVertex: Optional[np.ndarray] = None
    lonVertex: Optional[np.ndarray] = None
    latEdge: Optional[np.ndarray] = None
    lonEdge: Optional[np.ndarray] = None


@dataclass
class InputHandle:
    dataset: Dataset
    num_vars: int
    current_var: int
    varnames: list[str]
    unlimited_dim_name: Optional[str]


@dataclass
class InputField:
    name: str = ""
    isTimeDependent: bool = False
    varname: str = ""
    xtype: int = FIELD_TYPE_UNSUPPORTED
    ndims: int = -1
    dimnames: Optional[np.ndarray] = None
    dimlens: Optional[np.ndarray] = None
    dimids: Optional[np.ndarray] = None
    file_handle: Optional[InputHandle] = None
    array0r: float = 0.0
    array1r: Optional[np.ndarray] = None
    array2r: Optional[np.ndarray] = None
    array3r: Optional[np.ndarray] = None
    array4r: Optional[np.ndarray] = None
    array0d: float = 0.0
    array1d: Optional[np.ndarray] = None
    array2d: Optional[np.ndarray] = None
    array3d: Optional[np.ndarray] = None
    array4d: Optional[np.ndarray] = None
    array0i: int = 0
    array1i: Optional[np.ndarray] = None
    array2i: Optional[np.ndarray] = None
    array3i: Optional[np.ndarray] = None


@dataclass
class TargetField:
    name: str = ""
    ndims: int = -1
    xtype: int = FIELD_TYPE_UNSUPPORTED
    isTimeDependent: bool = False
    dimlens: Optional[np.ndarray] = None
    dimnames: Optional[np.ndarray] = None
    array0r: float = 0.0
    array1r: Optional[np.ndarray] = None
    array2r: Optional[np.ndarray] = None
    array3r: Optional[np.ndarray] = None
    array4r: Optional[np.ndarray] = None
    array0d: float = 0.0
    array1d: Optional[np.ndarray] = None
    array2d: Optional[np.ndarray] = None
    array3d: Optional[np.ndarray] = None
    array4d: Optional[np.ndarray] = None
    array0i: int = 0
    array1i: Optional[np.ndarray] = None
    array2i: Optional[np.ndarray] = None
    array3i: Optional[np.ndarray] = None
    array4i: Optional[np.ndarray] = None


@dataclass
class OutputHandle:
    dataset: Dataset
    unlimited_id: Optional[str]
    in_define_mode: bool = True
    current_frame: int = 0


@dataclass
class RemapInfo:
    method: int = -1
    src_mesh: Optional[MPASMesh] = None
    dst_mesh: Optional[TargetMesh] = None
    nearestCell: Optional[np.ndarray] = None
    nearestVertex: Optional[np.ndarray] = None
    nearestEdge: Optional[np.ndarray] = None
    cellWeights: Optional[np.ndarray] = None
    vertexWeights: Optional[np.ndarray] = None
    edgeWeights: Optional[np.ndarray] = None
    sourceCells: Optional[np.ndarray] = None
    sourceVertices: Optional[np.ndarray] = None
    sourceEdges: Optional[np.ndarray] = None
