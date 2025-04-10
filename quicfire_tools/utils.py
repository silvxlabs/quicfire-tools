"""
Utility functions for quicfire-tools.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from numpy import ndarray
from scipy.io import FortranFile


def compute_parabolic_stretched_grid(
    dz_surf: float, n_surf_cell: int, n_tot_cells: int, domain_height: float
) -> np.ndarray:
    """
    Generate a vertical grid spacing array with parabolic stretching.

    Parameters
    ----------
    dz_surf : float
        Vertical cell spacing at the surface, typically about 1 m.
    n_surf_cell : int
        Number of cells with constant vertical cell spacing (dz_surf) at the
        surface.
    n_tot_cells : int
        Total number of vertical cells in the grid. Typically ranges between
        20 and 30.
    domain_height : float
        Total height of the grid. For flat terrain without smoke transport
        concerns, it is usually around 100 m. For terrain with features, it
        should be approximately 3 times the height of the tallest feature.

    Returns
    -------
    np.ndarray
        Array representing the vertical grid spacing (dz) values for the
        simulation.

    Notes
    -----
    The function computes a parabolic stretching of the vertical grid spacings.
    The grid consists of a specified number of cells with uniform spacing at
    the surface, followed by cells with spacings determined by a parabolic
    formula until the total domain height is achieved.
    """
    dzmax_high = domain_height - dz_surf * n_surf_cell
    dzmax_low = 0

    dz = np.ones(n_tot_cells) * dz_surf

    while True:
        dzmax = 0.5 * (dzmax_low + dzmax_high)

        c1 = (dzmax - dz_surf) / ((n_tot_cells - n_surf_cell) ** 2)
        c2 = -2.0 * c1 * n_surf_cell
        c3 = dz_surf + c1 * n_surf_cell**2

        # Apply parabolic formula for cells beyond the surface cells
        kreal = np.arange(n_surf_cell, n_tot_cells, dtype=float)
        dz[n_surf_cell:] = (c1 * kreal**2) + (c2 * kreal) + c3

        zmax_temp = np.sum(dz)

        if abs(domain_height - zmax_temp) < 0.001:
            break
        elif zmax_temp > domain_height:
            dzmax_high = dzmax
        else:
            dzmax_low = dzmax

    dz[-1] += domain_height - np.sum(dz)

    return dz


def read_dat_file(filename: Path | str, nx: int, ny: int, nz: int | None) -> ndarray:
    """
    Read in a .dat file as a numpy array. For 2D .dat files use nz = None.

    Parameters
    ----------
    filename : Path or str
        The path to the .dat file to read.
    nx : int
        Number of cells in the x-direction.
    ny : int
        Number of cells in the y-direction.
    nz : int | None
        Number of cells in the z-direction. Use None for a 2D array.

    Returns
    -------
    ndarray
        A numpy array representing the data in the .dat file with shape (nz, ny, nx)
    """
    if isinstance(filename, str):
        filename = Path(filename)

    shape = (nx, ny, nz) if nz else (nx, ny)

    with open(filename, "rb") as fin:
        arr = (
            FortranFile(fin).read_reals(dtype="float32").reshape(shape, order="F").T
        )  # read in column-major, then transpose

    return arr


def write_dat_file(array: ndarray, filename: Path | str, dtype: type = np.float32):
    """
    Write a numpy array into a .dat file

    Parameters
    ----------
    array : ndarray
        The array written to the .dat file
        The function uses Fortran row-major order to write
        an array of any dimension into linear form (see example)
    filename : Path or str
        The path to the .dat file to read.
    dtype : type
        The desired data type of the array written to the .dat file

    Example
    ----------
    If you have an array N x M:     | a11, a12, a13 |
                                    | a21, a22, a23 |

    Then this function will write it as:
                                    [ a11, a12, a13, a21, a22, a23 ]

    In order to modify the values of a 3D array of N x M x Z by loading from an existing dat file,
    Use the read_dat_file() with the shape argument as a 1D array of length = (N)x(M)x(Z)
    Then modify the numpy array and write back to .dat
    This will preserve the order of the data when it is read and then written


    """
    if isinstance(filename, str):
        filename = Path(filename)
    array = array.astype(dtype)

    with FortranFile(filename, "w") as f:
        f.write_record(array)  # this will write in row-major order


def list_default_factory():
    return []
