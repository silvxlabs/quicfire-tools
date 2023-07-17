"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations

# Core Imports
import time
from pathlib import Path
from string import Template

# Internal Imports
from quicfire_tools.parameters import SimulationParameters


class InputModule:
    """
    Input Module
    """

    def __init__(self, directory: Path | str):
        if isinstance(directory, str):
            path = Path(directory)
            directory = path.resolve()

        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

    def setup_input_files(self, params):
        """
        Populates input file templates with user defined parameters

        Parameters
        ----------
        params: SimulationParameters
            Dictionary of user defined parameters.

        Returns
        -------
        None:
            Sets up simulation files in the simulation directory.

        """
        # Get current unix time
        params["timenow"] = int(time.time())

        # Write fuels data
        (params["fuel_density"], params["fuel_moisture"],
         params["fuel_height_flag"]) = self._write_fuel(params)

        # Write ignition data
        params["ignition_locations"] = self._write_ignition_locations(params)

        # Write input template files
        try:
            version = params["version"]
        except KeyError:
            version = "latest"
        template_files_path = Path(
            __file__).parent / "input-templates" / version
        template_files_list = template_files_path.glob("*")
        for fname in template_files_list:
            self._fill_form_with_dict(fname, params)

    @staticmethod
    def _write_fuel(params: dict) -> tuple[str, str, str]:
        """
        Writes fuel data to the QUIC_fire.inp input file. This function

        Parameters
        ----------
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        None:
            Writes fuel data to the fuel_data.inp file.

        """
        # Get data from params
        fuel_flag = params["fuel_flag"]

        # Uniform fuel properties
        if fuel_flag == 1:
            fuel_density = "\n" + str(params["fuel_density"])
            fuel_moisture = "\n" + str(params["fuel_moisture"])
            fuel_height = (f"\n{fuel_flag}    ! fuel height flag: 1 = uniform; "
                           f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
                           f" files for quic grid, 4 = Firetech files for "
                           f"different grid (need interpolation)"
                           f"\n{str(params['fuel_height'])}")

        # Custom fuel .dat files (fuel flags 3 or 4)
        else:
            fuel_density, fuel_moisture, fuel_height = "", "", ""

        return fuel_density, fuel_moisture, fuel_height

    def _write_ignition_locations(self, params: dict) -> str:
        """
        Writes ignition locations file to simulation directory if ignition flag
        is 6. Otherwise, it writes the ignition locations to the QUIC_fire.inp
        input file.

        Parameters
        ----------
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        ignition_locations: str
            Data string to write to the QUIC_fire.inp input file.

        """
        if params["ignition_flag"] == 6 or params["ignition_flag"] == 7:
            return ""

        else:  # Ignition flag 1
            return self._write_line_fire_ignition(params)

    @staticmethod
    def _write_line_fire_ignition(params: dict) -> str:
        """
        Writes line fire ignition locations to the QUIC_fire.inp input file.

        Line fire ignition takes the following parameters:
        fire_source_x0: south-west corner in the x-direction
        fire_source_y0: south-west corner in the y-direction
        fire_source_xlen: length in the x-direction
        fire_source_ylen: length in the y-direction
        num_ignition_cells: number of ignition cells constant at 100

        Parameters
        ----------
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        ignition_locations: str
            Data string to write to the QUIC_fire.inp input file.
        """
        # Get data from params
        wind_direction = params["wind_direction"]
        nx, ny = params["nx"], params["ny"]
        dx, dy = params["dx"], params["dy"]

        # Hold number ignitions constant at 100 per documentation
        # num_ignition_cells = 100

        # Compute cardinal direction from wind direction
        dirs = ["N", "E", "S", "W"]
        ix = round(wind_direction / (360. / len(dirs)))
        cardinal_direction = dirs[ix % len(dirs)]

        # Ignition strip on top border
        if cardinal_direction == "N":
            fire_source_x0 = int(0.1 * nx * dx)
            fire_source_y0 = int(0.9 * ny * dy) - 1
            fire_source_xlen = int(0.8 * nx * dx)
            fire_source_ylen = 1

        # Ignition strip on left border
        elif cardinal_direction == "W":
            fire_source_x0 = int(0.1 * nx * dx) - 1
            fire_source_y0 = int(0.1 * ny * dy)
            fire_source_xlen = 1
            fire_source_ylen = int(0.8 * ny * dy)

        # Ignition strip on bottom border
        elif cardinal_direction == "S":
            fire_source_x0 = int(0.1 * nx * dx)
            fire_source_y0 = int(0.1 * ny * dy) - 1
            fire_source_xlen = int(0.8 * nx * dx)
            fire_source_ylen = 1

        # Ignition strip on right border
        else:
            fire_source_x0 = int(0.9 * nx * dx) - 1
            fire_source_y0 = int(0.1 * ny * dy)
            fire_source_xlen = 1
            fire_source_ylen = int(0.8 * ny * dy)

        ignition_list = ["", fire_source_x0, fire_source_y0, fire_source_xlen,
                         fire_source_ylen]
        return "\n".join((str(x) for x in ignition_list))

    def _fill_form_with_dict(self, template_file_path: Path, params: dict):
        """
        Fills a form with a dictionary of values.

        Parameters
        ----------
        template_file_path: Path
            Path to the template file
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        None:
            Writes parameter data to the template file
        """
        output_file_path = self.directory / template_file_path.name
        with open(template_file_path, "r") as ftemp:
            with open(output_file_path, "w") as fout:
                src = Template(ftemp.read())
                result = src.substitute(params)
                fout.write(result)

        """
        Validates the params dictionary.

        Parameters
        ----------
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        None:
            Raises ValueError if any parameters are invalid.
        """
        # Check for required parameters
        required_params = ["nx", "ny", "nz", "dx", "dy", "dz", "wind_speed",
                           "wind_direction", "sim_time", "num_cpus",
                           "fuel_flag", "ignition_flag", "output_time",
                           "topo_flag"]
        for param in required_params:
            if param not in params:
                raise KeyError(f"Parameter {param} is required.")

        # nx, ny, and nz must be integers greater than 0
        for param in ("nx", "ny", "nz"):
            if not isinstance(params[param], int):
                raise TypeError(f"Parameter {param} must be an integer.")
            if params[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0.")

        # dx, dy, and dz must be numbers greater than 0
        for param in ("dx", "dy", "dz"):
            if not isinstance(params[param], (int, float)):
                raise TypeError(f"Parameter {param} must be a number.")
            if params[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0.")

        # wind_speed must be a number greater than or equal to 0
        if not isinstance(params["wind_speed"], (int, float)):
            raise TypeError("Parameter wind_speed must be a number.")
        if params["wind_speed"] < 0:
            raise ValueError("Parameter wind_speed must be greater than or "
                             "equal to 0.")

        # wind_direction must be a number between 0 and 360
        if not isinstance(params["wind_direction"], (int, float)):
            raise TypeError("Parameter wind_direction must be a number.")
        if params["wind_direction"] < 0 or params["wind_direction"] > 360:
            raise ValueError("Parameter wind_direction must be between 0 and "
                             "360.")

        # sim_time must be an integer greater than 0
        if not isinstance(params["sim_time"], int):
            raise TypeError("Parameter sim_time must be an integer.")
        if params["sim_time"] <= 0:
            raise ValueError("Parameter sim_time must be greater than 0.")

        # auto_kill must be an integer equal to 0 or 1
        if not isinstance(params["auto_kill"], int):
            raise TypeError("Parameter auto_kill must be an integer.")
        if params["auto_kill"] not in (0, 1):
            raise ValueError("Parameter auto_kill must be 0 or 1.")

        # num_cpus must be an integer greater than 0
        if not isinstance(params["num_cpus"], int):
            raise TypeError("Parameter num_cpus must be an integer.")
        if params["num_cpus"] <= 0:
            raise ValueError("Parameter num_cpus must be greater than 0.")

        # output_time must be an integer greater than 0
        if not isinstance(params["output_time"], int):
            raise TypeError("Parameter output_time must be an integer.")
        if params["output_time"] <= 0:
            raise ValueError("Parameter output_time must be greater than 0.")

        # topo_flag must be an integer equal to 0 or 5
        if not isinstance(params["topo_flag"], int):
            raise TypeError("Parameter topo_flag must be an integer.")
        if params["topo_flag"] not in (0, 5):
            raise ValueError("Parameter topo_flag must be 0 or 5.")

        # Fuel flag must be an integer
        if not isinstance(params["fuel_flag"], int):
            raise TypeError("Parameter fuel_flag must be an integer.")

        # Fuel flags 1, 3, 4 are currently supported
        if params["fuel_flag"] not in (1, 3, 4, 5):
            raise ValueError("Parameter fuel_flag must be 1, 3, 4, or 5. Future"
                             "versions of this package will support more.")

        # If fuel_flag is 1, then the user must provide fuel_density,
        # fuel_moisture, and fuel_height parameters
        if params["fuel_flag"] == 1:
            if "fuel_density" not in params:
                raise KeyError("Parameter fuel_density is required when "
                               "fuel_flag is 1.")
            if "fuel_moisture" not in params:
                raise KeyError("Parameter fuel_moisture is required when "
                               "fuel_flag is 1.")
            if "fuel_height" not in params:
                raise KeyError("Parameter fuel_height is required when "
                               "fuel_flag is 1.")

        # fuel_density, fuel_moisture, and fuel_height must be numbers greater
        # or equal to 0
        for param in ("fuel_density", "fuel_moisture", "fuel_height"):
            if param in params:
                if not isinstance(params[param], (int, float)):
                    raise TypeError(f"Parameter {param} must be a number.")
                if params[param] < 0:
                    raise ValueError(f"Parameter {param} must be greater than"
                                     " or equal to 0.")

        # Ignition flag must be an integer
        if not isinstance(params["ignition_flag"], int):
            raise TypeError("Parameter ignition_flag must be an integer.")

        return params.copy()


if __name__ == '__main__':
    test_params = SimulationParameters(
        nx=100,
        ny=100,
        nz=1,
        dx=1.,
        dy=1.,
        dz=1.,
        wind_speed=4.,
        wind_direction=270,
        sim_time=60,
        auto_kill=1,
        num_cpus=1,
        fuel_flag=3,
        ignition_flag=1,
        output_time=10,
        topo_flag=0,
    )

    sim_test = InputModule("../tests/test-simulation/")
    sim_test.setup_input_files(test_params)
