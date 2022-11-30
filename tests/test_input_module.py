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


def test_invalid_input_parameters():
    # Test missing nx parameter
    test_custom_params = TEST_PARAMS.copy()
    _ = test_custom_params.pop("nx")
    with pytest.raises(KeyError):
        SUT.setup_input_files(test_custom_params)

    # Test non-integer nx parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["nx"] = 1.0
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test string nx parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["nx"] = "10"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test nx parameter less than 1
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["nx"] = 0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test string dx parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["dx"] = "1.0"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test dx parameter less than 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["dx"] = -1.0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test dx parameter equal to 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["dx"] = 0.0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test missing wind_speed parameter
    test_custom_params = TEST_PARAMS.copy()
    _ = test_custom_params.pop("wind_speed")
    with pytest.raises(KeyError):
        SUT.setup_input_files(test_custom_params)

    # Test string wind_speed parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["wind_speed"] = "4.0"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test wind_speed parameter less than 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["wind_speed"] = -4.0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test wind_direction string parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["wind_direction"] = "265"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test wind_direction parameter less than 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["wind_direction"] = -265
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test wind_direction parameter greater than 360
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["wind_direction"] = 365
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test missing sim_time parameter
    test_custom_params = TEST_PARAMS.copy()
    _ = test_custom_params.pop("sim_time")
    with pytest.raises(KeyError):
        SUT.setup_input_files(test_custom_params)

    # Test string sim_time parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["sim_time"] = "60"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test float sim_time parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["sim_time"] = 60.0
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test sim_time parameter less than 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["sim_time"] = -60
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test sim_time parameter equal to 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["sim_time"] = 0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test string auto_kill parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["auto_kill"] = "1"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test auto_kill parameter not 0 or 1
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["auto_kill"] = 2
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test string num_cpus parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["num_cpus"] = "1"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test num_cpus parameter less than 1
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["num_cpus"] = 0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test non-integer output_time parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["output_time"] = 1.0
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test string output_time parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["output_time"] = "1"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test output_time parameter less than 1
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["output_time"] = 0
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test non-integer fuel flag parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["fuel_flag"] = 1.0
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test non-supported fuel flag parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["fuel_flag"] = 2
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test fuel flag 1 has fuel_density, fuel_moisture, fuel_height parameters
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["fuel_flag"] = 1
    with pytest.raises(KeyError):
        SUT.setup_input_files(test_custom_params)

    # Test fuel flag 1 string fuel_density parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["fuel_flag"] = 1
    test_custom_params["fuel_density"] = "1.0"
    test_custom_params["fuel_moisture"] = 0.5
    test_custom_params["fuel_height"] = 0.5
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test fuel flag 1 fuel_density parameter less than 0
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["fuel_flag"] = 1
    test_custom_params["fuel_density"] = -1.0
    test_custom_params["fuel_moisture"] = 0.5
    test_custom_params["fuel_height"] = 0.5
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)

    # Test string ignition_flag parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["ignition_flag"] = "1"
    with pytest.raises(TypeError):
        SUT.setup_input_files(test_custom_params)

    # Test non-supported ignition_flag parameter
    test_custom_params = TEST_PARAMS.copy()
    test_custom_params["ignition_flag"] = 2
    with pytest.raises(ValueError):
        SUT.setup_input_files(test_custom_params)


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

    # Test fuel flag 3
    fuel_data_params = TEST_PARAMS.copy()
    fuel_data_params["fuel_flag"] = 3
    fuel_params = SUT._write_fuel(fuel_data_params)
    assert fuel_params == ("", "", "")

    # Test fuel flag 3 with spurious fuel data
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
