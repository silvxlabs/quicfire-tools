"""
QUIC-Fire Tools Ignitions Module
"""
from __future__ import annotations

# Core Imports
from enum import Enum

# External Imports
from pydantic import BaseModel, SerializeAsAny


class IgnitionSources(int, Enum):
    rectangle = 1
    square_ring = 2
    circular_ring = 3
    ignite_dat_file = 6


class IgnitionType(BaseModel):
    ignition_flag: SerializeAsAny[IgnitionSources]

    def __str__(self):
        return (
            f"{self.ignition_flag.value}\t! 1 = rectangle, "
            f"2 = square ring, 3 = circular ring, "
            f"4 = file (QF_Ignitions.inp), "
            f"5 = time-dependent ignitions (QF_IgnitionPattern.inp), "
            f"6 = ignite.dat (firetech)"
        )


class RectangleIgnition(IgnitionType):
    ignition_flag: SerializeAsAny[IgnitionSources] = IgnitionSources(1)
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


class SquareRingIgnition(IgnitionType):
    ignition_flag: SerializeAsAny[IgnitionSources] = IgnitionSources(2)
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


class CircularRingIgnition(IgnitionType):
    ignition_flag: SerializeAsAny[IgnitionSources] = IgnitionSources(3)
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
    if 135 <= wind_direction < 225:
        x_min = 0.1 * (nx * 2)
        y_min = 0.1 * (ny * 2)
        x_length = 0.8 * (nx * 2)
        y_length = width
    if 225 <= wind_direction < 315:
        x_min = 0.1 * (nx * 2)
        y_min = 0.1 * (ny * 2)
        x_length = width
        y_length = 0.8 * (nx * 2)
    else:
        x_min = 0.1 * (ny * 2)
        y_min = (0.9 * (nx * 2)) - width
        x_length = 0.8 * (nx * 2)
        y_length = width

    return x_min, y_min, x_length, y_length
