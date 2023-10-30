"""
Utility functions for quicfire-tools.
"""

from pathlib import Path
import numpy as np
from scipy.io import FortranFile

from quicfire_tools.inputs import QU_TopoInputs


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


def read_topo_dat(
    topo_path: Path | str, filename: str, x_dim: int, y_dim: int, order: str = "C"
):
    """
    Read in a topo.dat file as a numpy array.
    """
    if isinstance(topo_path, str):
        topo_path = Path(topo_path)

    full_path = topo_path / filename

    with open(full_path, "rb") as fin:
        arr = (
            FortranFile(fin)
            .read_reals(dtype="float32")
            .reshape((y_dim, x_dim), order=order)
        )

    return arr


def calc_quic_height(
    qu_topo: QU_TopoInputs,
    fire_nz,
    fire_dz,
    nx,
    ny,
    topo_path: Path | str,
) -> int:
    """
    Calculate the QUIC domain height from the tallest fuels and the maximum elevation.

    Parameters
    ----------
    topo_path : Path | str
        Path to directory where topo.dat file is saved.
    qu_topo : QU_TopoInputs
        QU_TpopoInputs InputFile class instance
    """
    fire_height = fire_nz * fire_dz
    if qu_topo.topo_flag.value == 0:
        topo_height = 0
    elif qu_topo.topo_flag.value == 1:
        topo_height = qu_topo.topo_type.elevation_max
    elif qu_topo.topo_flag.value == 2:
        topo_height = qu_topo.topo_type.max_height
    elif qu_topo.topo_flag.value == 3:
        # something with slope_value and flat_fraction
        pass
    elif qu_topo.topo_flag.value == 4:
        # don't know how to do this one either
        pass
    elif qu_topo.topo_flag.value == 6:
        # maybe something with radius?
        pass
    elif qu_topo.topo_flag.value == 7:
        topo_height = 2 * qu_topo.topo_type.amplitude  # probably?
    elif qu_topo.topo_flag.value == 8:
        topo_height = qu_topo.topo_type.height
    elif qu_topo.topo_flag.value == 5:
        topo_dat = read_topo_dat(topo_path, qu_topo.filename, nx, ny)
        topo_height = np.max(topo_dat) - np.min(topo_dat)

    return int((topo_height + fire_height) * 3)
