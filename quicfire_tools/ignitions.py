"""
QUIC-Fire Tools Ignitions Module
"""
from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from pydantic import BaseModel


class IgnitionSources(int, Enum):
    rectangle = 1
    square_ring = 2
    circular_ring = 3
    ignite_dat_file = 6


class IgnitionType(BaseModel):
    """
    Test docs
    """

    ignition_flag: IgnitionSources

    def __str__(self):
        return (
            f"{self.ignition_flag.value}\t! 1 = rectangle, "
            f"2 = square ring, 3 = circular ring, "
            f"4 = file (QF_Ignitions.inp), "
            f"5 = time-dependent ignitions (QF_IgnitionPattern.inp), "
            f"6 = ignite.dat (firetech)"
        )


class RectangleIgnition(IgnitionType):
    """
    Represents a rectangle ignition source in QUIC-Fire.

    Parameters
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

    ignition_flag: IgnitionSources = IgnitionSources(1)
    x_min: int
    y_min: int
    x_length: int
    y_length: int

    def __str__(self):
        flag_line = super().__str__()
        locations = (
            f"\n{self.x_min}\t! South-west corner in the x-direction\n"
            f"{self.y_min}\t! South-west corner in the y-direction\n"
            f"{self.x_length}\t! Length in the x-direction\n"
            f"{self.y_length}\t! Length in the y-direction"
        )
        return flag_line + locations


class SquareRingIgnition(IgnitionType):
    """
    Represents a square ring ignition source in QUIC-Fire.

    Parameters
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

    ignition_flag: IgnitionSources = IgnitionSources(2)
    x_min: int
    y_min: int
    x_length: int
    y_length: int
    x_width: int
    y_width: int

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


class CircularRingIgnition(IgnitionType):
    """
    Represents a circular ring ignition source in QUIC-Fire.

    Parameters
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

    ignition_flag: IgnitionSources = IgnitionSources(3)
    x_min: int
    y_min: int
    x_length: int
    y_length: int
    ring_width: int

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
