"""
QUIC-Fire Tools Topography Model
"""

from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from typing import Literal
from pydantic import BaseModel, Field, PositiveFloat, SerializeAsAny


class TopoFlags(int, Enum):
    """
    Enum class for all valid topography source options in QUIC-Fire.
    """

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


class Topography(BaseModel):
    """
    Base class for all topography types in QUIC-Fire. This class is used to
    provide a common string representation for all topography types.
    """

    topo_flag: SerializeAsAny[TopoFlags]

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


class GaussianHillTopo(Topography):
    """
    Creates a Gaussian hill topography in QUIC-Fire.

    Attributes
    ----------
    x_hilltop : PositiveFloat
        Gaussian hill top location x-direction [m]
    y_hilltop : PositiveFloat
        Gaussian hill top location y-direction [m]
    elevation_max : PositiveFloat
        Maximum elevation of the Gaussian hill [m]
    elevation_std : PositiveFloat
        Standard deviation of the Gaussian hill [m]
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(1)
    x_hilltop: PositiveFloat
    y_hilltop: PositiveFloat
    elevation_max: PositiveFloat
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


class HillPassTopo(Topography):
    """
    Creates a hill pass topography in QUIC-Fire.

    Attributes
    ----------
    max_height : PositiveFloat
        Maximum elevation of the hill pass [m]
    location_param : PositiveFloat
        Location parameter of the hill pass [m]
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(2)
    max_height: PositiveFloat
    location_param: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.max_height}\t! m, max height\n"
            f"{self.location_param}\t! m, location parameter"
        )
        return flag_line + params


class SlopeMesaTopo(Topography):
    """
    Creates a slope mesa topography in QUIC-Fire.

    Attributes
    ----------
    slope_axis : Literal[0, 1]
        Slope axis (0 = x, 1 = y)
    slope_value : float
        Slope in dh/dx or dh/dy
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(3)
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


class CanyonTopo(Topography):
    """
    Creates a canyon topography in QUIC-Fire.

    Attributes
    ----------
    x_location: PositiveFloat
        Canyon location in x-dir [m].
    y_location: PositiveFloat
        Canyon location in y-dir [m].
    slope_value: PositiveFloat
        Slope in dh/dx or dy/dy [-].
    canyon_std: PositiveFloat
        Canyon function standard deviation [m].
    vertical_offset: PositiveFloat
        Canyon vertical offset [m].
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(4)
    x_location: PositiveFloat
    y_location: PositiveFloat
    slope_value: PositiveFloat
    canyon_std: PositiveFloat
    vertical_offset: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.x_location}\t! m, x-start of canyon on slope\n"
            f"{self.y_location}\t! m, y-center of canyon"
            f"{self.slope_value}\t! N/A, slope\n"
            f"{self.canyon_std}\t! m, std\n"
            f"{self.vertical_offset}\t! m, height offset of the bottom of the canyon"
        )
        return flag_line + params


class HalfCircleTopo(Topography):
    """
    Creates a half-circle topography in QUIC-Fire.

    Attributes
    ----------
    x_location: PositiveFloat
        The x-coordinate of the center of the half-circle topography [m].
    y_location: PositiveFloat
        The y-coordinate of the center of the half-circle topography [m].
    radius: PositiveFloat
        The radius of the half-circle topography [m].
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(6)
    x_location: PositiveFloat
    y_location: PositiveFloat
    radius: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = (
            f"\n{self.x_location}\t! m, x-location\n"
            f"{self.y_location}\t! m, y-location\n"
            f"{self.radius}\t! m, radius of half circle"
        )
        return flag_line + params


class SinusoidTopo(Topography):
    """
    Creates a sinusoidal topography in QUIC-Fire.

    Attributes
    ----------
    period: PositiveFloat
        The period of the sinusoidal wave [m].
    amplitude: PositiveFloat
        The amplitude of the sinusoidal wave [m].
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(7)
    period: PositiveFloat
    amplitude: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = f"\n{self.period}\t! m, period\n" f"{self.amplitude}\t! m, amplitude"
        return flag_line + params


class CosHillTopo(Topography):
    """
    Creates a cosine-shaped hill topography in QUIC-Fire.

    Attributes
    ----------
    aspect: PositiveFloat
        The aspect (or orientation) of the hill [-].
    height: PositiveFloat
        The height of the hill [m].
    """

    topo_flag: SerializeAsAny[TopoFlags] = TopoFlags(8)
    aspect: PositiveFloat
    height: PositiveFloat

    def __str__(self):
        flag_line = super().__str__()
        params = f"\n{self.aspect}\t! [0], aspect\n" f"{self.height}\t! m, height"
        return flag_line + params


def serialize_topography(topo_data: dict):
    topo_flag = topo_data.get("topo_flag")
    if topo_flag == TopoFlags(1):
        return GaussianHillTopo(**topo_data)
    elif topo_flag == TopoFlags(2):
        return HillPassTopo(**topo_data)
    elif topo_flag == TopoFlags(3):
        return SlopeMesaTopo(**topo_data)
    elif topo_flag == TopoFlags(4):
        return CanyonTopo(**topo_data)
    elif topo_flag == TopoFlags(6):
        return HalfCircleTopo(**topo_data)
    elif topo_flag == TopoFlags(7):
        return SinusoidTopo(**topo_data)
    elif topo_flag == TopoFlags(8):
        return CosHillTopo(**topo_data)
    else:
        return Topography(**topo_data)
