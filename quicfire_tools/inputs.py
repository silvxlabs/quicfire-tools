"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations

# Core Imports
import json
import time
import importlib.resources
from pathlib import Path
from string import Template

# Internal Imports
from quicfire_tools.parameters import SimulationParameters

DOCS_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("documentation")
TEMPLATES_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("templates")


class SimulationInputs:
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
        # Convert params to dict
        params = params.to_dict()

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
            __file__).parent / "templates" / version
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


class InputFile:
    """Base class representing an input file."""

    def __init__(self, filename):
        self.filename = filename
        with open(DOCS_PATH / f"{self.filename}.json", "r") as f:
            self.param_info = json.load(f)

    def list_params(self):
        """List all parameters in the input file."""
        return list(self.param_info.keys())

    def get_param_info(self, param_name):
        """Retrieve documentation for a parameter."""
        return self.param_info.get(param_name, {})

    def print_param_info(self, param_name):
        """Print documentation for a parameter."""
        info = self.get_param_info(param_name)

        if info:
            print(f"Documentation for {param_name}:")
            try:
                print(f"- Line: {info['line']}")
            except KeyError:
                pass
            print(f"- Values accepted: {info['values_accepted']}")
            print(f"- Tested values range: {info['tested_values_range']}")
            print(f"- Description: {info['description']}")
            print(f"- Units: {info['units']}")
            print(f"- Variable name (Fortran): {info['variable_name_fortran']}")
        else:
            print(f"No documentation found for {param_name}")

    def to_dict(self):
        """
        Convert the object to a dictionary, excluding attributes that start with an underscore.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return {attr: value for attr, value in self.__dict__.items()
                if not attr.startswith('_')}

    def to_file(self, directory: Path, version: str = "latest"):
        if isinstance(directory, str):
            directory = Path(directory)

        template_file_path = TEMPLATES_PATH / version / f"{self.filename}"
        with open(template_file_path, "r") as ftemp:
            src = Template(ftemp.read())

        result = src.substitute(self.to_dict())

        output_file_path = directory / self.filename
        with open(output_file_path, "w") as fout:
            fout.write(result)


# Print method same as before

class Gridlist(InputFile):
    def __init__(self, n: int, m: int, l: int, dx: float, dy: float, dz: float,
                 aa1: float):
        """
        Initialize the Gridlist class to manage fuel-grid related data.

        Parameters
        ----------
        n : int
            Number of cells in the x-direction, must be greater than 0 (ft%nx).
        m : int
            Number of cells in the y-direction, must be greater than 0 (ft%ny).
        l : int
            Number of cells in the z-direction, must be greater than 0 (ft%nz).
        dx : float
            Cell size in the x-direction in meters, must be greater than 0.
            Recommended value: 2 m (ft%dx).
        dy : float
            Cells siz in the y-direction in meters, must be greater than 0.
            Recommended value: 2 m (ft%dy).
        dz : float
            Cell size in the z-direction in meters, must be greater than 0 (zb).
        aa1 : float
            Stretching factor for the vertical grid spacing, must be greater
            than 0 (aa1).
        """
        self._validate_inputs(n, m, l, dx, dy, dz, aa1)
        super().__init__("gridlist")
        self.n = n
        self.m = m
        self.l = l
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.aa1 = aa1

    @staticmethod
    def _validate_inputs(n, m, l, dx, dy, dz, aa1):
        for val in (n, m, l):
            if not isinstance(val, int):
                raise TypeError(f"{val} must be an integer")
            if val <= 0:
                raise ValueError(f"{val} must be greater than 0")
        for val in (dx, dy, dz, aa1):
            if not isinstance(val, float) and not isinstance(val, int):
                raise TypeError(f"{val} must be a real number")
            if val <= 0:
                raise ValueError(f"{val} must be greater than 0")


class RasterOrigin(InputFile):
    def __init__(self, utm_x: float = 0., utm_y: float = 0.):
        """
        Initialize the RasterOrigin class to manage raster origin related data.

        Parameters
        ----------
        utm_x : float
            South-west corner of the domain in UTM (ft%utm_x).
        utm_y : float
            South-west corner of the domain in UTM (ft%utm_y).
        """
        self._validate_inputs(utm_x, utm_y)
        super().__init__("rasterorigin.txt")
        self.utm_x = utm_x
        self.utm_y = utm_y

    @staticmethod
    def _validate_inputs(utm_x, utm_y):
        for val in (utm_x, utm_y):
            if not isinstance(val, float) and not isinstance(val, int):
                raise TypeError(f"{val} must be a real number")
            if val < 0:
                raise ValueError(f"{val} must be greater than 0")


class QU_Buildings(InputFile):
    def __init__(self, wall_roughness_length: float = 0.1,
                 number_of_buildings: int = 0,
                 number_of_polygon_nodes: int = 0):
        """
        Initialize the QU_Buildings class to manage building-related data.

        Parameters
        ----------
        wall_roughness_length : float
            Wall roughness length in meters, must be greater than 0 (bld%zo).
        number_of_buildings : int
            Number of buildings, must be greater than 0.
            Recommended value: 0 (building algorithms are not part of QUIC-Fire) (bld%number).
        number_of_polygon_nodes : int
            Number of polygon building nodes, must be greater than 0.
            Recommended value: 0 (inumpolygon).
        """
        self._validate_inputs(wall_roughness_length, number_of_buildings,
                              number_of_polygon_nodes)
        super().__init__("QU_buildings.inp")
        self.wall_roughness_length = wall_roughness_length
        self.number_of_buildings = number_of_buildings
        self.number_of_polygon_nodes = number_of_polygon_nodes

    @staticmethod
    def _validate_inputs(wall_roughness_length, number_of_buildings,
                         number_of_polygon_nodes):
        if not isinstance(wall_roughness_length, float) and not isinstance(
                wall_roughness_length, int):
            raise TypeError(f"{wall_roughness_length} must be a real number")
        if wall_roughness_length <= 0:
            raise ValueError(f"{wall_roughness_length} must be greater than 0")

        for val in (number_of_buildings, number_of_polygon_nodes):
            if not isinstance(val, int):
                raise TypeError(f"{val} must be an integer")
            if val < 0:
                raise ValueError(f"{val} must be greater than or equal to 0")

class QU_Fileoptions(InputFile):
    def __init__(self,
                 output_data_file_format_flag: int = 2,
                 non_mass_conserved_initial_field_flag: int = 0,
                 initial_sensor_velocity_field_flag: int = 0,
                 qu_staggered_velocity_file_flag: int = 0,
                 generate_wind_startup_files_flag: int = 0):
        """
        Initialize the QU_Fileoptions class to manage file-related options.

        Parameters
        ----------
        output_data_file_format_flag : int
            Output data file format flag, values accepted [0 3],
            recommended value 2.
        non_mass_conserved_initial_field_flag : int
            Flag to write out non-mass conserved initial field (uofield.dat), values accepted [0 1], recommended value 0.
        initial_sensor_velocity_field_flag : int
            Flag to write out the file uosensorfield.dat, values accepted [0 1], recommended value 0.
        qu_staggered_velocity_file_flag : int
            Flag to write out the file QU_staggered_velocity.bin, values accepted [0 1], recommended value 0.
        generate_wind_startup_files_flag : int
            Generate wind startup files for ensemble simulations, values accepted [0 1].
        """
        self._validate_inputs(output_data_file_format_flag,
                              non_mass_conserved_initial_field_flag,
                              initial_sensor_velocity_field_flag,
                              qu_staggered_velocity_file_flag,
                              generate_wind_startup_files_flag)
        super().__init__("QU_fileoptions")
        self.output_data_file_format_flag = output_data_file_format_flag
        self.non_mass_conserved_initial_field_flag = non_mass_conserved_initial_field_flag
        self.initial_sensor_velocity_field_flag = initial_sensor_velocity_field_flag
        self.qu_staggered_velocity_file_flag = qu_staggered_velocity_file_flag
        self.generate_wind_startup_files_flag = generate_wind_startup_files_flag

    @staticmethod
    def _validate_inputs(output_data_file_format_flag, non_mass_conserved_initial_field_flag, initial_sensor_velocity_field_flag, QU_staggered_velocity_file_flag, generate_wind_startup_files_flag):
        for val in (output_data_file_format_flag,
                    non_mass_conserved_initial_field_flag,
                    initial_sensor_velocity_field_flag,
                    QU_staggered_velocity_file_flag,
                    generate_wind_startup_files_flag):
            if not isinstance(val, int):
                raise TypeError(f"{val} must be an integer")
            if val not in [0, 1] and val != output_data_file_format_flag:
                raise ValueError(f"{val} must be 0 or 1")
            if val not in range(0, 4) and val == output_data_file_format_flag:
                raise ValueError(f"{val} must be between 0 and 3")
