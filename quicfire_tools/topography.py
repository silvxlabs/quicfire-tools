"""
QUIC-Fire Tools Topography Model
"""
from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from typing import Literal
from pydantic import BaseModel, PositiveInt, PositiveFloat, Field

class TopoSources(Enum):
    flat = 0
    gaussian_hill = 1
    hill_pass = 2
    slope_mesa = 3
    canyon = 4
    custom = 5
    half_circle = 6
    sinusoid = 7
    cos_hill = 8
    QP_elevation_bin = 9
    terrainOutput_txt = 10
    terrain_dat = 11

class TopoType(BaseModel):
    topo_flag: TopoSources

    def __str__(self):
        return (f"{self.topo_flag.value}\t\t! N/A, "
                f"topo flag: 0 = flat, 1 = Gaussian hill, "
                f"2 = hill pass, 3 = slope mesa, 4 = canyon, "
                f"5 = custom, 6 = half circle, 7 = sinusoid, "
                f"8 = cos hill, 9 = QP_elevation.inp, "
                f"10 = terrainOutput.txt (ARA), "
                f"11 = terrain.dat (firetec)\n")
    
class GaussianHillTopo(TopoType):
    topo_flag: TopoSources = 1
    x_hilltop: PositiveInt
    y_hilltop: PositiveInt
    elevation_max: PositiveInt
    elevation_std: PositiveFloat

class HillPassTopo(TopoType):
    max_height: PositiveInt
    location_param: PositiveFloat

class SlopeMesaTopo(TopoType):
    slope_axis: Literal[0,1]
    slope_value: PositiveFloat
    flat_fraction: float = Field(ge=0, le=1)

class CanyonTopo(TopoType):
    x_start: PositiveInt
    y_center: PositiveInt
    slope_value: PositiveFloat
    canyon_std: PositiveFloat
    vertical_offset: PositiveFloat

class HalfCircleTopo(TopoType):
    x_location: PositiveInt
    y_location: PositiveInt
    radius: PositiveFloat

class SinusoidTopo(TopoType):
    period: PositiveFloat
    amplitude: PositiveFloat

class CosHillTopo(TopoType):
    aspect: PositiveFloat
    height: PositiveInt