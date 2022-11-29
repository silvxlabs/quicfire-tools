# Core Imports
import sys

# Internal Imports
sys.path.append("../quicfire_tools")
from quicfire_tools.inputs import InputModule

# External Imports
import pytest

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


def test_write_fuel_data():
    # Test fuel flag 1
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 1
    fuel_data_params["fuel_density"] = 1.25
    fuel_data_params["fuel_moisture"] = 0.75
    fuel_data_params["fuel_height"] = 0.5
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("\n1.25", "\n0.75", "\n0.5")

    # Test fuel flag 1
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 1
    fuel_data_params["fuel_density"] = 0.0
    fuel_data_params["fuel_moisture"] = 0.0
    fuel_data_params["fuel_height"] = 0.0
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("\n0.0", "\n0.0", "\n0.0")

    # Test fuel flag 1
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 1
    with pytest.raises(KeyError):
        fuel_params = SUT._write_fuel(fuel_data_params)

    # Test fuel flag 3
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 3
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("", "", "")

    # Test fuel flag 3 with fuel data
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 3
    fuel_data_params["fuel_density"] = 1.25
    fuel_data_params["fuel_moisture"] = 0.75
    fuel_data_params["fuel_height"] = 0.5
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("", "", "")

    # Test fuel flag 4
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 4
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("", "", "")


def test_write_ignition_locations():
    # Test custom ignition flag 6
    ignition_custom_params = TEST_PARAMS.copy()
    ignition_custom_params["ignition_flag"] = 6
    assert SUT._write_ignition_locations(ignition_custom_params) == ""

    # Test invalid ignition flag
    ignition_custom_params = TEST_PARAMS.copy()
    ignition_custom_params["ignition_flag"] = 0
    with pytest.raises(ValueError):
        SUT._write_ignition_locations(ignition_custom_params)


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
    assert SUT._write_line_fire_ignition(ignition_line_params) == south_wind

    # Test west wind
    ignition_line_params["wind_direction"] = 270
    assert SUT._write_line_fire_ignition(ignition_line_params) == west_wind

    # Test 45 degree wind
    ignition_line_params["wind_direction"] = 45
    assert SUT._write_line_fire_ignition(ignition_line_params) == north_wind

    # Test 45.01 degree wind
    ignition_line_params["wind_direction"] = 45.01
    assert SUT._write_line_fire_ignition(ignition_line_params) == east_wind

    # Test -45 degree wind
    ignition_line_params["wind_direction"] = -45
    assert SUT._write_line_fire_ignition(ignition_line_params) == north_wind

    # Test -45.01 degree wind
    ignition_line_params["wind_direction"] = -45.01
    assert SUT._write_line_fire_ignition(ignition_line_params) == west_wind

    # Test 134.99 degree wind
    ignition_line_params["wind_direction"] = 134.99
    assert SUT._write_line_fire_ignition(ignition_line_params) == east_wind

    # Test 135 degree wind
    ignition_line_params["wind_direction"] = 135
    assert SUT._write_line_fire_ignition(ignition_line_params) == south_wind
