"""
QUIC-Fire Tools Ignitions Module
"""

from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from pydantic import BaseModel


class IgnitionFlags(int, Enum):
    """
    Enum class for all valid ignition source options in QUIC-Fire.
    """

    rectangle = 1
    square_ring = 2
    circular_ring = 3
    ignite_dat_file = 7


class Ignition(BaseModel):
    """
    Base class for all ignition types in QUIC-Fire. This class is used to
    provide a common string representation for all ignition types.
    """

    ignition_flag: IgnitionFlags

    def __str__(self):
        return (
            f"{self.ignition_flag.value}\t! 1 = rectangle, "
            f"2 = square ring, 3 = circular ring, "
            f"4 = file (QF_Ignitions.inp), "
            f"5 = time-dependent ignitions (QF_IgnitionPattern.inp), "
            f"7 = ignite.dat (firetech)"
        )


class RectangleIgnition(Ignition):
    """
    Represents a rectangle ignition source in QUIC-Fire.

    Attributes
    ----------
    x_min : float
        South-west corner in the x-direction [m]
    y_min : float
        South-west corner in the y-direction [m]
    x_length : float
        Length in the x-direction [m]
    y_length : float
        Length in the y-direction [m]
    """

    ignition_flag: IgnitionFlags = IgnitionFlags(1)
    x_min: float
    y_min: float
    x_length: float
    y_length: float

    def __str__(self):
        flag_line = super().__str__()
        locations = (
            f"\n{self.x_min}\t! South-west corner in the x-direction\n"
            f"{self.y_min}\t! South-west corner in the y-direction\n"
            f"{self.x_length}\t! Length in the x-direction\n"
            f"{self.y_length}\t! Length in the y-direction"
        )
        return flag_line + locations


class SquareRingIgnition(Ignition):
    """
    Represents a square ring ignition source in QUIC-Fire.

    Attributes
    ----------
    x_min : float
        South-west corner in the x-direction [m]
    y_min : float
        South-west corner in the y-direction [m]
    x_length : float
        Length in the x-direction [m]
    y_length : float
        Length in the y-direction [m]
    x_width : float
        Width in the x-direction [m]
    y_width : float
        Width in the y-direction [m]
    """

    ignition_flag: IgnitionFlags = IgnitionFlags(2)
    x_min: float
    y_min: float
    x_length: float
    y_length: float
    x_width: float
    y_width: float

    def __str__(self):
        flag_line = super().__str__()
        locations = (
            f"\n{self.x_min}\t! South-west corner in the x-direction\n"
            f"{self.y_min}\t! South-west corner in the y-direction\n"
            f"{self.x_length}\t! Length in the x-direction\n"
            f"{self.y_length}\t! Length in the y-direction\n"
            f"{self.x_width}\t! Width in the x-direction\n"
            f"{self.y_width}\t! Width in the y-direction"
        )
        return flag_line + locations


class CircularRingIgnition(Ignition):
    """
    Represents a circular ring ignition source in QUIC-Fire.

    Attributes
    ----------
    x_min : float
        South-west corner in the x-direction [m]
    y_min : float
        South-west corner in the y-direction [m]
    x_length : float
        Length in the x-direction [m]
    y_length : float
        Length in the y-direction [m]
    ring_width : float
        Width of the ring [m]
    """

    ignition_flag: IgnitionFlags = IgnitionFlags(3)
    x_min: float
    y_min: float
    x_length: float
    y_length: float
    ring_width: float

    def __str__(self):
        flag_line = super().__str__()
        locations = (
            f"\n{self.x_min}\t! South-west corner in the x-direction\n"
            f"{self.y_min}\t! South-west corner in the y-direction\n"
            f"{self.x_length}\t! Length in the x-direction\n"
            f"{self.y_length}\t! Length in the y-direction\n"
            f"{self.ring_width}\t! Width of the ring"
        )
        return flag_line + locations


def default_line_ignition(nx, ny, wind_direction):
    width = 10
    if 45 <= wind_direction < 135:
        x_min = (0.9 * (nx * 2)) - width
        y_min = 0.1 * (ny * 2)
        x_length = width
        y_length = 0.8 * (nx * 2)
    elif 135 <= wind_direction < 225:
        x_min = 0.1 * (nx * 2)
        y_min = 0.1 * (ny * 2)
        x_length = 0.8 * (nx * 2)
        y_length = width
    elif 225 <= wind_direction < 315:
        x_min = 0.1 * (nx * 2)
        y_min = 0.1 * (ny * 2)
        x_length = width
        y_length = 0.8 * (nx * 2)
    else:
        x_min = 0.1 * (ny * 2)
        y_min = (0.9 * (nx * 2)) - width
        x_length = 0.8 * (nx * 2)
        y_length = width

    return RectangleIgnition(
        x_min=x_min, y_min=y_min, x_length=x_length, y_length=y_length
    )


def serialize_ignition(ignition_data: dict):
    ignition_flag = ignition_data.get("ignition_flag")
    if ignition_flag == IgnitionFlags(1):
        return RectangleIgnition(**ignition_data)
    elif ignition_flag == IgnitionFlags(2):
        return SquareRingIgnition(**ignition_data)
    elif ignition_flag == IgnitionFlags(3):
        return CircularRingIgnition(**ignition_data)
    else:
        return Ignition(**ignition_data)
