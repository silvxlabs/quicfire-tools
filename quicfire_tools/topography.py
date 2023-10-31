"""
QUIC-Fire Tools Topography Model
"""
from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, SerializeAsAny


class TopoSources(int, Enum):
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
    topo_flag: SerializeAsAny[TopoSources]

    def __str__(self):
        return (
            f"{self.topo_flag.value}\t\t! N/A, "
            f"topo flag: 0 = flat, 1 = Gaussian hill, "
            f"2 = hill pass, 3 = slope mesa, 4 = canyon, "
            f"5 = custom, 6 = half circle, 7 = sinusoid, "
            f"8 = cos hill, 9 = QP_elevation.inp, "
            f"10 = terrainOutput.txt (ARA), "
            f"11 = terrain.dat (firetec)"
        )


class GaussianHillTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(1)
    x_hilltop: PositiveInt
    y_hilltop: PositiveInt
    elevation_max: PositiveInt
    elevation_std: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.x_hilltop}\t! m, x-center\n"
            f"{self.y_hilltop}\t! m, y-center\n"
            f"{self.elevation_max}\t! m, max height\n"
            f"{self.elevation_std}\t! m, std"
        )
        return flag_line + params


class HillPassTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(2)
    max_height: PositiveInt
    location_param: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.max_height}\t! m, max height\n"
            f"{self.location_param}\t! m, location parameter"
        )
        return flag_line + params


class SlopeMesaTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(3)
    slope_axis: Literal[0, 1]
    slope_value: PositiveFloat
    flat_fraction: float = Field(ge=0, le=1)

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.slope_axis}\t! N/A, slope axis (0 = x, 1 = y)\n"
            f"{self.slope_value}\t! N/A, slope val (dh/dx)\n"
            f"{self.flat_fraction}\t! N/A, fraction of domain that is flat"
        )
        return flag_line + params


class CanyonTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(4)
    x_start: PositiveInt
    y_center: PositiveInt
    slope_value: PositiveFloat
    canyon_std: PositiveFloat
    vertical_offset: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.x_start}\t! m, x-start of canyon on slope\n"
            f"{self.y_center}\t! m, y-center of canyon"
            f"{self.slope_value}\t! N/A, slope\n"
            f"{self.canyon_std}\t! m, std\n"
            f"{self.vertical_offset}\t! m, height offset of the bottom of the canyon"
        )
        return flag_line + params


class HalfCircleTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(6)
    x_location: PositiveInt
    y_location: PositiveInt
    radius: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.x_location}\t! m, x-location\n"
            f"{self.y_location}\t! m, y-location\n"
            f"{self.radius}\t! m, radius of half circle"
        )
        return flag_line + params


class SinusoidTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(7)
    period: PositiveFloat
    amplitude: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = f"\n{self.period}\t! m, mean\n" f"{self.amplitude}\t! m, amplitude"
        return flag_line + params


class CosHillTopo(TopoType):
    topo_flag: SerializeAsAny[TopoSources] = TopoSources(8)
    aspect: PositiveFloat
    height: PositiveInt

    def __str__(self):
        flag_line = super().__str__()
        params = f"\n{self.aspect}\t! m, mean\n" f"{self.height}\t! m, height"
        return flag_line + params
