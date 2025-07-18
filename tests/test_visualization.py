"""
Test module for the visualization module of the quicfire_tools package.
"""

# Core imports
from __future__ import annotations
from pathlib import Path

# External imports
import pytest
from matplotlib import pyplot as plt

# Internal imports
from quicfire_tools import SimulationOutputs
from quicfire_tools.visualization import plot_fuel_density

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data" / "test-visualization"
TMP_DIR = TEST_DIR / "tmp"


def get_test_sim_outputs():
    sim_outputs = SimulationOutputs(TEST_DATA_DIR, fire_nz=56, ny=117, nx=116)
    return sim_outputs


class TestPlotFuelDensity:
    def test_surface_fire(self):
        sim_outputs = get_test_sim_outputs()
        plot_fuel_density(
            simulation_outputs=sim_outputs,
            save_directory=TMP_DIR,
            png=False,
            gif=True,
            z_layers=0,
        )

    def test_canopy_slice(self):
        sim_outputs = get_test_sim_outputs()
        plot_fuel_density(
            simulation_outputs=sim_outputs,
            save_directory=TMP_DIR,
            png=False,
            gif=True,
            z_layers=2,
        )

    def test_integrated_canopy(self):
        sim_outputs = get_test_sim_outputs()
        plot_fuel_density(
            simulation_outputs=sim_outputs,
            save_directory=TMP_DIR,
            png=False,
            gif=True,
            z_layers=list(range(1, sim_outputs.fire_nz)),
            integrated=True,
        )


def plot_array(x):
    plt.figure(2)
    plt.set_cmap("viridis")
    plt.imshow(x, origin="lower")
    plt.colorbar()
    plt.show()


# if __name__ == "__main__":
#     inputs_directory = Path(__file__).parent
#     outputs_directory = inputs_directory / "Output"
#     save_directory = inputs_directory / "viz2d_test"
#     plot_fuel_density(
#         outputs_directory=outputs_directory,
#         save_directory=save_directory,
#         png=False,
#         gif=True,
#         z_layers=0,
#     )
