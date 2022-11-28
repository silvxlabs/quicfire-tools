# Core Imports
import sys

# Internal Imports
sys.path.append("../quicfire_tools")
from quicfire_tools.inputs import InputModule

TEST_PARAMS = {
    "nx": 100,
    "ny": 100,
    "nz": 1,
    "dx": 1.,
    "dy": 1.,
    "dz": 1.,
    "wind_speed": 4.,
    "wind_direction": 265,
    "sim_time": 60,
    "auto_kill": 1,
    "num_cpus": 1,
    "fuel_flag": 3,
    "ignition_flag": 1,
    "output_time": 10,
}

SUT = InputModule("pytest-simulation")


def test_write_line_fire_ignition():
    ignition_line_params = TEST_PARAMS.copy()
    ignition_line_params["ignition_flag"] = 1

    # Define default ignition line locations
    north_wind = "\n10\n89\n80\n1\n100"
    east_wind = "\n89\n10\n1\n80\n100"
    south_wind = "\n10\n9\n80\n1\n100"
    west_wind = "\n9\n10\n1\n80\n100"

    # Test north wind
    ignition_line_params["wind_direction"] = 0
    assert SUT._write_line_fire_ignition(ignition_line_params) == north_wind

    # Test east wind
    ignition_line_params["wind_direction"] = 90
    assert SUT._write_line_fire_ignition(ignition_line_params) == east_wind

    # Test south wind
    ignition_line_params["wind_direction"] = 180
    assert SUT._write_line_fire_ignition(ignition_line_params) == east_wind

    # Test west wind
    ignition_line_params["wind_direction"] = 270
    assert SUT._write_line_fire_ignition(ignition_line_params) == west_wind

    # Test invalid wind direction
    ignition_line_params["wind_direction"] = "invalid"

