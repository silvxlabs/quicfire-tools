"""
QUIC-Fire Tools Simulation Input Module
"""

# Core Imports
import time
from pathlib import Path
from string import Template


class InputModule:
    """
    Input Module
    """

    def __init__(self, directory: Path | str):
        if type(directory) == str:
            path = Path(directory)
            directory = path.resolve()

        directory.mkdir(exist_ok=True)
        self.directory = directory

    def setup_input_files(self, params):
        """
        Populates input file templates with user defined parameters

        Parameters
        ----------
        params: dict
            Dictionary of user defined parameters.

        Returns
        -------
        None:
            Sets up simulation files in the simulation directory.

        """
        # Get current unix time
        params["timenow"] = int(time.time())

        # Write fuels data
        params["fuel_density"], params["fuel_moisture"], params["fuel_height"] \
            = self._write_fuel_data(params)

        # Write ignition data
        params["ignition_locations"] = self._write_ignition_locations(params)

        # Write input template files
        template_files_path = Path(__file__).parent / "input-templates"
        template_files_list = template_files_path.glob("*")
        for fname in template_files_list:
            self._fill_form_with_dict(fname, params)

    def _write_fuel_data(self, params: dict) -> tuple[str, str, str]:
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
            try:
                fuel_density = "\n" + str(params["fuel_density"])
                fuel_moisture = "\n" + str(params["fuel_moisture"])
                fuel_height = "\n" + str(params["fuel_height"])
            except KeyError as e:
                raise KeyError(f"{e.args[0]} value must be defined if fuel flag"
                               f" is 1")

        # Custom fuel .dat files
        elif fuel_flag in (3, 4):
            fuel_density, fuel_moisture, fuel_height = "", "", ""

        # Unsupported fuel flag
        else:
            raise ValueError("Invalid fuel flag. Only fuel flags 1, 3, and 4 "
                             "are supported.")

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
        if params["ignition_flag"] == 6:
            return ""
        elif params["ignition_flag"] == 1:
            return self._write_line_fire_ignition(params)
        else:
            raise ValueError("Invalid ignition flag. Only ignition flags 1 "
                             "(rectangular) and 6 (file) are supported.")

    def _write_line_fire_ignition(self, params: dict) -> str:
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
        num_ignition_cells = 100

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
                         fire_source_ylen, num_ignition_cells]
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


if __name__ == '__main__':
    test_params = {
        "nx": 100,
        "ny": 100,
        "nz": 1,
        "dx": 1.,
        "dy": 1.,
        "dz": 1.,
        "wind_speed": 4.,
        "wind_direction": 270,
        "sim_time": 60,
        "auto_kill": 1,
        "num_cpus": 1,
        "fuel_flag": 3,
        "ignition_flag": 1,
        "output_time": 10,
    }

    sim_test = InputModule("../tests/test-simulation/")
    sim_test.setup_input_files(test_params)
