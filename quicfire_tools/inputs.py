"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations

# Internal Imports
from utils import compute_parabolic_stretched_grid

# Core Imports
import json
import time
import importlib.resources
from pathlib import Path
from string import Template

# External Imports
import numpy as np

# TODO: Multiple wind directions
# TODO: String for .dat files that exist

DOCS_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("documentation")
TEMPLATES_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("templates")


class InputFile:
    """
    Base class representing an input file.

    This base class provides a common interface for all input files in order to
    accomplish two main goals:
    1) Return documentation for each parameter in the input file.
    2) Provide a method to write the input file to a specified directory.
    """

    def __init__(self, filename):
        self.filename = filename
        with open(DOCS_PATH / f"{self.filename}.json", "r") as f:
            self.param_info = json.load(f)

    def list_parameters(self):
        """List all parameters in the input file."""
        return list(self.param_info.keys())

    def get_documentation(self, parameter: str = None):
        """
        Retrieve documentation for a parameter. If no parameter is specified,
        return documentation for all parameters.
        """
        if parameter:
            return self.param_info.get(parameter, {})
        else:
            return self.param_info

    def print_documentation(self, parameter: str = None):
        """
        Print documentation for a parameter. If no parameter is specified,
        print documentation for all parameters.
        """
        if parameter:
            info = self.get_documentation(parameter)
        else:
            info = self.get_documentation()
        for key, value in info.items():
            key = key.replace("_", " ").capitalize()
            print(f"- {key}: {value}")

    def to_dict(self):
        """
        Convert the object to a dictionary, excluding attributes that start
        with an underscore.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return {attr: value for attr, value in self.__dict__.items()
                if not attr.startswith('_')}

    def to_file(self, directory: Path, version: str = "latest"):
        """
        Write the input file to a specified directory.

        Parameters
        ----------
        directory : Path
            Directory to write the input file to.
        version : str
            Version of the input file to write. Default is "latest".
        """
        if isinstance(directory, str):
            directory = Path(directory)

        template_file_path = TEMPLATES_PATH / version / f"{self.filename}"
        with open(template_file_path, "r") as ftemp:
            src = Template(ftemp.read())

        result = src.substitute(self.to_dict())

        output_file_path = directory / self.filename
        with open(output_file_path, "w") as fout:
            fout.write(result)


class InputValidator:

    @classmethod
    def real_number(cls, variable_name, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"{variable_name} must be a real number")

    @classmethod
    def positive_real(cls, variable_name, value):
        cls.real_number(variable_name, value)
        if value <= 0:
            raise ValueError(f"{variable_name} must be greater than 0")

    @classmethod
    def integer(cls, variable_name, value):
        if not isinstance(value, int):
            raise TypeError(f"{variable_name} must be an integer")

    @classmethod
    def positive_integer(cls, variable_name, value):
        cls.integer(variable_name, value)
        if value <= 0:
            raise ValueError(f"{variable_name} must be greater than 0")

    @classmethod
    def non_negative_integer(cls, variable_name, value):
        cls.integer(variable_name, value)
        if value < 0:
            raise ValueError(
                f"{variable_name} must be greater than or equal to 0")

    @classmethod
    def in_list(cls, variable_name, value, valid_values):
        if value not in valid_values:
            raise ValueError(
                f"{variable_name} must be one of the following: {valid_values}")


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
        # Validate inputs
        InputValidator.positive_integer("n", n)
        InputValidator.positive_integer("m", m)
        InputValidator.positive_integer("l", l)
        InputValidator.positive_real("dx", dx)
        InputValidator.positive_real("dy", dy)
        InputValidator.positive_real("dz", dz)
        InputValidator.positive_real("aa1", aa1)

        # Initialize the class
        super().__init__("gridlist")
        self.n = n
        self.m = m
        self.l = l
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.aa1 = aa1


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
        # Validate inputs
        InputValidator.real_number("utm_x", utm_x)
        InputValidator.real_number("utm_y", utm_y)

        # Initialize the class
        super().__init__("rasterorigin.txt")
        self.utm_x = utm_x
        self.utm_y = utm_y


class QU_Buildings(InputFile):
    def __init__(self, wall_roughness_length: float = 0.1,
                 number_of_buildings: int = 0,
                 number_of_polygon_nodes: int = 0):
        """
        Initialize the QU_Buildings class to manage building-related data.

        Parameters
        ----------
        wall_roughness_length : float
            Wall roughness length in meters, must be greater than 0..
        number_of_buildings : int
            Number of buildings, must be greater than 0. (building algorithms
            not part of QUIC-Fire).
        number_of_polygon_nodes : int
            Number of polygon building nodes, must be greater than 0.
        """
        # Validate inputs
        InputValidator.positive_real("wall_roughness_length",
                                     wall_roughness_length)
        InputValidator.positive_integer("number_of_buildings",
                                        number_of_buildings)
        InputValidator.positive_integer("number_of_polygon_nodes",
                                        number_of_polygon_nodes)

        # Initialize the class
        super().__init__("QU_buildings.inp")
        self.wall_roughness_length = wall_roughness_length
        self.number_of_buildings = number_of_buildings
        self.number_of_polygon_nodes = number_of_polygon_nodes


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
            Flag to write out non-mass conserved initial field (uofield.dat),
            values accepted [0 1], recommended value 0.
        initial_sensor_velocity_field_flag : int
            Flag to write out the file uosensorfield.dat, values accepted [0 1]
            , recommended value 0.
        qu_staggered_velocity_file_flag : int
            Flag to write out the file QU_staggered_velocity.bin, values
            accepted [0 1], recommended value 0.
        generate_wind_startup_files_flag : int
            Generate wind startup files for ensemble simulations, values
            accepted [0 1].
        """
        # Validate inputs
        InputValidator.in_list("output_data_file_format_flag",
                               output_data_file_format_flag, [0, 3])
        InputValidator.in_list("non_mass_conserved_initial_field_flag",
                               non_mass_conserved_initial_field_flag, [0, 1])
        InputValidator.in_list("initial_sensor_velocity_field_flag",
                               initial_sensor_velocity_field_flag, [0, 1])
        InputValidator.in_list("QU_staggered_velocity_file_flag",
                               qu_staggered_velocity_file_flag, [0, 1])
        InputValidator.in_list("generate_wind_startup_files_flag",
                               generate_wind_startup_files_flag, [0, 1])

        # Initialize the class
        super().__init__("QU_fileoptions.inp")
        self.output_data_file_format_flag = output_data_file_format_flag
        self.non_mass_conserved_initial_field_flag = non_mass_conserved_initial_field_flag
        self.initial_sensor_velocity_field_flag = initial_sensor_velocity_field_flag
        self.qu_staggered_velocity_file_flag = qu_staggered_velocity_file_flag
        self.generate_wind_startup_files_flag = generate_wind_startup_files_flag


class QU_Simparams(InputFile):
    def __init__(self,
                 nx: int,
                 ny: int,
                 nz: int,
                 dx: float,
                 dy: float,
                 surface_vertical_cell_size: float = 1.,
                 number_surface_cells: int = 5,
                 stretch_grid_flag: int = 3,
                 dz_array: list[float] = None,
                 utc_offset: int = 0,
                 wind_step_times: list[int] = None,
                 sor_iter_max: int = 10,
                 sor_residual_reduction: int = 3,
                 use_diffusion_flag: int = 0,
                 number_diffusion_iterations: int = 10,
                 domain_rotation: float = 0.,
                 utm_x: float = 0.,
                 utm_y: float = 0.,
                 utm_zone_number: int = 1,
                 utm_zone_letter: int = 1,
                 quic_cfd_flag: int = 0,
                 explosive_bldg_flag: int = 0,
                 bldg_array_flag: int = 0,
                 ):
        """
        Initialize the QU_Simparams class to manage simulation parameters.

        Parameters
        ----------
        nx : int
            Number of cells in the x-direction [-]. Recommended value: > 100
        ny : int
            Number of cells in the y-direction [-]. Recommended value: > 100
        nz : int
            Number of cells in the z-direction [-].
        dx : float
            Cell size in the x-direction [m]. Recommended value: 2 m
        dy : float
            Cell size in the y-direction [m]]. Recommended value: 2 m
        surface_vertical_cell_size : float
            Surface vertical cell size [m]].
        stretch_grid_flag : int
            Vertical grid stretching flag, values accepted [0, 1, 2, 3, 4].
            Recommended value: 3
        utc_offset : int
            Hours difference from UTM [h]
        wind_step_times : list[int]
            List of times at which the winds are available in Unix Epoch time
            (integer seconds since 1970/1/1 00:00:00). These are UTC times.
            Defaults to [int(time.time())] if not provided.
        sor_iter_max : int
            Maximum number of iterations of the SOR wind solver.
            Recommended value: 10
        sor_residual_reduction : int
            Residual reduction to assess convergence of the SOR solver
            (orders of magnitude). Recommended value: 3
        use_diffusion_flag : int
            Use diffusion algorithm: 0 = off, 1 = on. Recommended value: 0
        domain_rotation : float
            Domain rotation relative to true north (clockwise is positive)
            [degrees]. Recommended value: 0 deg
        utm_x : float
            UTM-x coordinates of the south-west corner of domain [m]
        utm_y : float
            UTM-y coordinates of the south-west corner of domain [m]
        utm_zone_number : int
            UTM zone number [-]
        utm_zone_letter : int
            UTM zone letter (A=1, B=2,..) [-]
        quic_cfd_flag : int
            QUIC-CFD flag: 0 = off, 1 = on. Recommended value: 0
        explosive_bldg_flag : int
            Explosive building damage flag: 0 = off, 1 = on.
            Recommended value: 0
        bldg_array_flag : int
            Building array flag. 0 = off, 1 = on. Recommended value: 0
        """
        # Validate inputs
        InputValidator.positive_integer("nx", nx)
        InputValidator.positive_integer("ny", ny)
        InputValidator.positive_integer("nz", nz)
        InputValidator.positive_real("dx", dx)
        InputValidator.positive_real("dy", dy)
        InputValidator.positive_real("surface_vertical_cell_size",
                                     surface_vertical_cell_size)
        InputValidator.positive_integer("number_surface_cells",
                                        number_surface_cells)
        InputValidator.in_list("stretch_grid_flag", stretch_grid_flag,
                               [0, 1, 3])
        InputValidator.integer("utc_offset", utc_offset)
        if wind_step_times is not None:
            for time_value in wind_step_times:
                InputValidator.integer("wind_step_times", time_value)
        InputValidator.positive_integer("sor_iter_max", sor_iter_max)
        InputValidator.positive_integer("sor_residual_reduction",
                                        sor_residual_reduction)
        InputValidator.in_list("use_diffusion_flag", use_diffusion_flag,
                               [0, 1])
        InputValidator.positive_integer("number_diffusion_iterations",
                                        number_diffusion_iterations)
        InputValidator.real_number("domain_rotation", domain_rotation)
        InputValidator.real_number("utm_x", utm_x)
        InputValidator.real_number("utm_y", utm_y)
        InputValidator.positive_integer("utm_zone_number", utm_zone_number)
        InputValidator.positive_integer("utm_zone_letter", utm_zone_letter)
        InputValidator.in_list("quic_cfd_flag", quic_cfd_flag, [0, 1])
        InputValidator.in_list("explosive_bldg_flag", explosive_bldg_flag,
                               [0, 1])
        InputValidator.in_list("bldg_array_flag", bldg_array_flag, [0, 1])

        # Initialize the class
        super().__init__("QU_simparams.inp")
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.surface_vertical_cell_size = surface_vertical_cell_size
        self.number_surface_cells = number_surface_cells
        self.stretch_grid_flag = stretch_grid_flag
        self.dz_array = dz_array if dz_array else []
        self.vertical_grid = self._generate_vertical_grid()
        self.utc_offset = utc_offset
        self.wind_times = wind_step_times if wind_step_times else [
            int(time.time())]
        self.wind_lines = self._generate_wind_time_lines()
        self.sor_iter_max = sor_iter_max
        self.sor_residual_reduction = sor_residual_reduction
        self.use_diffusion_flag = use_diffusion_flag
        self.number_diffusion_iterations = number_diffusion_iterations
        self.domain_rotation = domain_rotation
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.utm_zone_number = utm_zone_number
        self.utm_zone_letter = utm_zone_letter
        self.quic_cfd_flag = quic_cfd_flag
        self.explosive_bldg_flag = explosive_bldg_flag
        self.bldg_array_flag = bldg_array_flag

    def _generate_vertical_grid(self):
        """
        Parses the vertical grid stretching flag and dz_array to generate the
        vertical grid as a string for the QU_simparams.inp file.

        Also modifies dz_array if stretch_grid_flag is not 1.
        """
        if self.stretch_grid_flag == 0:
            return self._stretch_grid_flag_0()
        elif self.stretch_grid_flag == 1:
            return self._stretch_grid_flag_1()
        elif self.stretch_grid_flag == 3:
            return self._stretch_grid_flag_3()
        else:
            raise ValueError("stretch_grid_flag must be 0, 1, or 3")

    def _stretch_grid_flag_0(self):
        """
        Generates a uniform vertical grid as a string for the QU_simparams.inp
        file. Adds the uniform grid to dz_array.
        """
        # Create a uniform dz_grid
        for i in range(self.nz):
            self.dz_array.append(self.surface_vertical_cell_size)

        # Create the lines for the uniform grid
        surface_dz_line = f"{float(self.surface_vertical_cell_size)}\t! Surface DZ [m]"
        number_surface_cells_line = f"{self.number_surface_cells}\t! Number of uniform surface cells"

        return f"{surface_dz_line}\n{number_surface_cells_line}"

    def _stretch_grid_flag_1(self):
        """
        Generates a custom vertical grid as a string for the QU_simparams.inp
        file.
        """
        # Verify that dz_array is not empty
        if not self.dz_array:
            raise ValueError("dz_array must not be empty if stretch_grid_flag "
                             "is 1. Please provide a dz_array with nz elements"
                             " or use a different stretch_grid_flag.")

        # Verify that nz is equal to the length of dz_array
        if self.nz != len(self.dz_array):
            raise ValueError(f"nz must be equal to the length of dz_array. "
                             f"{self.nz} != {len(self.dz_array)}")

        # Verify that the first number_surface_cells_line elements of dz_array
        # are equal to the surface_vertical_cell_size
        for dz in self.dz_array[:self.number_surface_cells]:
            if dz != self.surface_vertical_cell_size:
                raise ValueError("The first number_surface_cells_line "
                                 "elements of dz_array must be equal to "
                                 "surface_vertical_cell_size")

        # Write surface vertical cell size line
        surface_dz_line = (f"{float(self.surface_vertical_cell_size)}\t! "
                           f"Surface DZ [m]")

        # Write header line
        header_line = f"! DZ array [m]"

        # Write dz_array lines
        dz_array_lines_list = []
        for dz in self.dz_array:
            dz_array_lines_list.append(f"{float(dz)}")
        dz_array_lines = "\n".join(dz_array_lines_list)

        return f"{surface_dz_line}\n{header_line}\n{dz_array_lines}"

    def _stretch_grid_flag_3(self):
        """
        Generates a vertical grid for stretch_grid_flag 3 as a string for the
        QU_simparams.inp file. Stretch grid flag 3 is a stretching with
        parabolic vertical cell size. Adds the parabolic grid to dz_array.
        """
        # Write surface vertical cell size line
        surface_dz_line = (f"{float(self.surface_vertical_cell_size)}\t! "
                           f"Surface DZ [m]")

        # Write number of surface cells line
        number_surface_cells_line = (f"{self.number_surface_cells}\t! "
                                     f"Number of uniform surface cells")

        # Write header line
        header_line = f"! DZ array [m]"

        # Generate the parabolic grid
        self.dz_array = compute_parabolic_stretched_grid(
            self.surface_vertical_cell_size, self.number_surface_cells,
            self.nz, 300)

        # Write dz_array lines
        dz_lines = "\n".join([f"{float(dz)}" for dz in self.dz_array])

        return (f"{surface_dz_line}\n{number_surface_cells_line}\n{header_line}"
                f"\n{dz_lines}")

    def _generate_wind_time_lines(self):
        """
        Parses the utc_offset and wind_step_times to generate the wind times
        as a string for the QU_simparams.inp file.
        """
        # Verify that wind_step_times is not empty
        if not self.wind_times:
            raise ValueError("wind_step_times must not be empty. Please "
                             "provide a wind_step_times with num_wind_steps "
                             "elements or use a different num_wind_steps.")

        # Write number of time increments line
        number_time_increments_line = (f"{len(self.wind_times)}\t"
                                       f"! Number of time increments")

        # Write utc_offset line
        utc_offset_line = f"{self.utc_offset}\t! UTC offset [hours]"

        # Write header line
        header_line = f"! Wind step times [s]"

        # Write wind_step_times lines
        wind_step_times_lines_list = []
        for wind_time in self.wind_times:
            wind_step_times_lines_list.append(f"{wind_time}")
        wind_step_times_lines = "\n".join(wind_step_times_lines_list)

        return "\n".join([number_time_increments_line,
                          utc_offset_line,
                          header_line,
                          wind_step_times_lines])
