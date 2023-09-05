"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations

# Core Imports
import json
import importlib.resources
from pathlib import Path
from string import Template

# External Imports
import numpy as np

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


