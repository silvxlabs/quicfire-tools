"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations
from xml.sax.xmlreader import InputSource

# Internal Imports
from utils import compute_parabolic_stretched_grid
from parameters import SimulationParameters #HOW TO DEAL WITH SIM PARAMS

# Core Imports
import json
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

class SimulationInputs:
    outputs: list = []

    def to_json(self):
        for output in outputs:
            output.to_dict()

    @classmethod
    def from_json(cls, json):
        pass


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
    def string(cls, variable_name, value):
        if not isinstance(value, str):
            raise TypeError(f"{variable_name} must be a string")

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
    
    @classmethod
    def negative_one(cls, variable_name, value):
        cls.integer(variable_name, value)
        if value < -1 or value == 0:
            raise ValueError(
                f"{variable_name} must be -1 or greater than 0")
    
    @classmethod
    def binary_flag(cls, variable_name, value):
        cls.integer(variable_name,value)
        if value not in [0,1]:
            raise ValueError(
                f"{variable_name} must be either 0 or 1")
    
    @classmethod
    def list(cls, variable_name, value):
        if not isinstance(value, list):
            raise ValueError(
                f"{variable_name} must be a list")
    
    @classmethod
    def list_of_positive_ints(cls, variable_name, value):
        cls.list(value)
        for v in value():
            cls.non_negative_integer("All values of {}".format(variable_name), v)
    
    @classmethod
    def list_of_positive_floats(cls, variable_name, value):
        cls.list(value)
        for v in value():
            cls.positive_real("All values of {}".format(variable_name), v)


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
                 dz_array: list[float] = None):
        """
        Initialize the QU_Simparams class to manage simulation parameters.

        Parameters
        ----------
        nx : int
            Number of cells in the x-direction.
        ny : int
            Number of cells in the y-direction.
        nz : int
            Number of cells in the z-direction.
        dx : float
            Cell size in the x-direction (in meters).
        dy : float
            Cell size in the y-direction (in meters).
        dz : float
            Surface vertical cell size (in meters).
        stretchgridflag : int
            Vertical grid stretching flag, values accepted [0, 1, 2, 3, 4].
        """
        # self._validate_inputs()
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

    @staticmethod
    def _validate_inputs(nx, ny, nz, dx, dy, surface_vertical_cell_size,
                         number_surface_cells, stretch_grid_flag, dz_array):
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
                               [0, 1, 2, 3, 4])

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

        return ""

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
        dz_array = compute_parabolic_stretched_grid(
            self.surface_vertical_cell_size, self.number_surface_cells,
            self.nz, 300)
        print()

class QUIC_fire(InputFile):
    def __init__(self,
                 nx: int,
                 ny: int,
                 nz: int,
                 output_time: int,
                 time_now: int, #WHERE IS THIS CALCULATED
                 sim_time: int = SimulationParameters.sim_time, #HOW TO DEAL WITH SIM PARAMS
                 fire_flag: int = 1,
                 random_seed: int = -1,
                 fire_time_step: int = 1,
                 quic_time_step: int = 1,
                 stretch_grid_flag: int = 0,
                 file_path: str = "",
                 dz_array: list[float] = None,
                 fuel_flag: int = 3,
                 fuel_params: list[float] = None,
                 ignition_flag: int = 1,
                 ignition_params: list[int] = None,
                 ignitions_per_cell: int = 2,
                 firebrand_flag: int = 0,
                 auto_kill: int = 1,
                 # Output flags
                 eng_to_atm_out: int = 1,
                 react_rate_out: int = 0,
                 fuel_dens_out: int = 1,
                 QF_wind_out: int = 1,
                 QU_wind_inst_out: int = 1,
                 QU_wind_avg_out: int = 0,
                 fuel_moist_out: int = 1,
                 mass_burnt_out: int = 1,
                 firebrand_out: int = 0,
                 emissions_out: int = 0,
                 radiation_out: int = 0,
                 intensity_out: int = 0):
        """
        Initialize the QU_Simparams class to manage simulation parameters.

        Parameters
        ----------
        nx : int
            Number of cells in the x-direction.
        ny : int
            Number of cells in the y-direction.
        nz : int
            Number of fire grid cells in the z-direction.
        output_time : int
            After how many time steps to print out:
                - fire-related files (excluding emissions and radiation)
                - average emissions and radiation
            After how many quic updates to print out:
                - wind-related files
                - averaged wind-related files
            Use -1 to provide custom times in file QFire_ListOutputTimes.inp
        time_now : int
            When the fire is ignited in Unix Epoch time (integer seconds since 1970/1/1 00:00:00). Must be greater or equal to the time of the first wind
        sim_time : int
            Total simulation time for the fire [s]
        fire_flag : int
            Fire flag, 1 = run fire; 0 = no fire
        random_seed : int
            Random number generator, -1: use time and date, any other integer > 0 is used as the seed
        fire_time_step : int
            time step for the fire simulation [s]
        quic_time_step : int
            Number of fire time steps done before updating the quic wind field (integer, >= 1)
        stretch_grid_flag : int
            Vertical stretching flag: 0 = uniform dz, 1 = custom
        file_path : str
            Path to files defining fuels, ignitions, and topography, with file separator at the end. Defaults to "", indicating files are in the same directory as all other input files
        dz_array : list[float]
            custom dz, one dz per line must be specified, from the ground to the top of the domain
         fuel_flag : int
            Flag for fuel inputs:
                - density
                - moisture
                - height
            1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)
        fuel_params : list[float]
            List of fuel parameters for a uniform grid (fuel_flag = 1) in the order [density, moisture, height]. All must be real numbers 0-1
        ignition_flag : int
            1 = rectangle, 2 = square ring, 3 = circular ring, 4 = file (QF_Ignitions.inp), 5 = time-dependent ignitions (QF_IgnitionPattern.inp), 7 = ignite.dat (firetech)
        ignition_params: list[int]
            List of ignitions parameters to define locations for rectangle, square ring, and circular ring ignitions.
            For all ignition patterns, the following four parameters must be provided in order:
                - Southwest corner in the x-direction (m)
                - Southwest corner in the y-direction(m)
                - Length in the x-direction (m)
                - Length in the y-direction (m)             
            Additional paramters only for square ring pattern (ignition_flag = 2):
                - Width of the ring in the x-direction (m)
                - Width of the ring in the y-direction (m)
            Additional paramters only for circular ring pattern (ignition_flag = 3):
                - Width of the ring (m)
        ignitions_per_cell: int
            Number of ignition per cell of the fire model. Recommended max value of 100
        firebrand_flag : int
            Firebrand flag, 0 = off; 1 = on
            Recommended value = 0 ; firebrands are untested for small scale problems
        auto_kill : int
            Kill if the fire is out and there are no more ignitions or firebrands (0 = no, 1 = yes)
        eng_to_atm_out : int
            Output flag [0, 1]: gridded energy-to-atmosphere (3D fire grid + extra layers)
        react_rate_out : int
            Output flag [0, 1]: compressed array reaction rate (fire grid)
        fuel_dens_out : int
            Output flag [0, 1]: compressed array fuel density (fire grid)
        QF_wind_out : int
            Output flag [0, 1]: gridded wind (u,v,w,sigma) (3D fire grid)
        QU_wind_inst_out : int
            Output flag [0, 1]: gridded QU winds with fire effects, instantaneous (QUIC-URB grid)
        QU_wind_avg_out : int
            Output flag [0, 1]: gridded QU winds with fire effects, averaged (QUIC-URB grid)
        fuel_moist_out : int
            Output flag [0, 1]: compressed array fuel moisture (fire grid)
        mass_burnt_out : int
            Output flag [0, 1]: vertically-integrated % mass burnt (fire grid)
        firebrand_out : int
            Output flag [0, 1]: firebrand trajectories. Must be 0 when firebrand flag is 0
        emissions_out : int
            Output flag [0, 5]: compressed array emissions (fire grid):
                0 = do not output any emission related variables
                1 = output emissions files and simulate CO in QUIC-SMOKE
                2 = output emissions files and simulate PM2.5 in QUIC- SMOKE
                3 = output emissions files and simulate both CO and PM2.5 in QUIC-SMOKE
                4 = output emissions files but use library approach in QUIC-SMOKE
                5 = output emissions files and simulate both water in QUIC-SMOKE
        radiation_out : int
            Output flag [0, 1]: gridded thermal radiation (fire grid)
        intensity_out : int
            Output flag [0, 1]: surface fire intensity at every fire time step
        """
        InputValidator.positive_integer("nx", nx)
        InputValidator.positive_integer("ny", ny)
        InputValidator.positive_integer("nz", nz)
        InputValidator.negative_one("output_time", output_time)
        if output_time == -1:
            print("CAUTION: User must provide custom times in file QFire_ListOutputTimes.inp when output_time = -1")
        InputValidator.positive_integer("time_now", time_now)
        InputValidator.positive_integer("sim_time", sim_time)
        InputValidator.binary_flag("fire_flag", fire_flag)
        InputValidator.negative_one("random_seed", random_seed)
        InputValidator.positive_integer("fire_time_step", fire_time_step)
        InputValidator.positive_integer("quic_time_step", quic_time_step)
        InputValidator.binary_flag("stretch_grid_flag", stretch_grid_flag)
        InputValidator.string("file_path", file_path)
        if stretch_grid_flag == 1:
            InputValidator.list_of_positive_floats("dz_array", dz_array)
        InputValidator.in_list("fuel_flag", fuel_flag, [1,2,3,4])
        if fuel_flag == 1:
            InputValidator.list_of_positive_floats("fulel_params", fuel_params)
        InputValidator.in_list("ignition_flag", ignition_flag, [1,2,3,4,5,7])
        if ignition_flag in [1,2,3,4,5]:
            InputValidator.list_of_positive_ints("ignition_params", ignition_params)
        InputValidator.positive_integer("ignitions_per_cell", ignitions_per_cell)
        InputValidator.binary_flag("firebrand_flag", firebrand_flag)
        InputValidator.binary_flag("auto_kill", auto_kill)
        # Output flags
        InputValidator.binary_flag("eng_to_atm_out", eng_to_atm_out)
        InputValidator.binary_flag("react_rate_out", react_rate_out)
        InputValidator.binary_flag("fuel_dens_out", fuel_dens_out)
        InputValidator.binary_flag("QF_wind_out", QF_wind_out)
        InputValidator.binary_flag("QU_wind_inst_out", QU_wind_inst_out)
        InputValidator.binary_flag("QU_wind_avg_out", QU_wind_avg_out)
        InputValidator.binary_flag("fuel_moist_out", fuel_moist_out)
        InputValidator.binary_flag("mass_burnt_out", mass_burnt_out)
        InputValidator.binary_flag("firebrand_out", firebrand_out)
        if firebrand_out == 1 and firebrand_out == 0:
            raise ValueError("Firebrand trajectories cannot be output when firebrands are off")
        InputValidator.in_list("emissions_out", emissions_out, [0,1,2,3,4,5])
        InputValidator.binary_flag("radiation_out", radiation_out)
        InputValidator.binary_flag("intensity_out", intensity_out)

        super().__init__("QUIC_fire.inp")
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.output_time = output_time
        self.time_now = time_now
        self.sim_time = sim_time
        self.fire_flag = fire_flag
        self.random_seed = random_seed
        self.fire_time_step = fire_time_step
        self.quic_time_step = quic_time_step
        self.stretch_grid_flag = stretch_grid_flag
        self.stretch_grid_input = self._get_custom_stretch_grid()
        self.file_path = file_path
        self.dz_array = dz_array if dz_array else []
        self.fuel_flag = fuel_flag
        self.fuel_params = fuel_params if fuel_params else []
        self.fuel_density, self.fuel_moisture, self.fuel_height = self._get_fuel_inputs()
        self.ignition_flag = ignition_flag
        self.ignition_params = ignition_params if ignition_params else []
        self.ignition_locations = self._get_ignition_locations()
        self.ignitions_per_cell = ignitions_per_cell
        self.firebrand_flag = firebrand_flag
        self.auto_kill = auto_kill
        # Output flags
        self.eng_to_atm_out = eng_to_atm_out
        self.react_rate_out = react_rate_out
        self.fuel_dens_out = react_rate_out
        self.QF_wind_out = QF_wind_out
        self.QU_wind_inst_out = QU_wind_inst_out
        self.QU_wind_avg_out = QU_wind_avg_out
        self.fuel_moist_out = fuel_moist_out
        self.mass_burnt_out = mass_burnt_out
        self.emissions_out = emissions_out
        self.radiation_out = radiation_out
        self.intensity_out = intensity_out


    def _get_fuel_inputs(self):
        """
        Writes custom fuel inputs to QUIC_fire.inp, if provided.
        """
        if len(self.fuel_params) != 3:
                raise ValueError("fuel_params must have length of 3")
        # Uniform fuel properties
        if self.fuel_flag == 1:
            fuel_density = f"\n{str(self.fuel_params[0])}"
            fuel_moisture = f"\n{str(self.fuel_params[1])}"
            fuel_height = (f"\n{self.fuel_flag}\t! fuel height flag: 1 = uniform; "
                           f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
                           f" files for quic grid, 4 = Firetech files for "
                           f"different grid (need interpolation)"
                           f"\n{str(self.fuel_params[2])}")
        # Custom fuel .dat files (fuel flags 3 or 4)
        else:
            fuel_density, fuel_moisture, fuel_height = "", "", ""
        
        if self.fuel_flag == 2:
            print("CAUTION: User must provide fuel inputs in QF_FuelDensity.inp, QF_FuelMoisture.inp, and QF_FuelHeight.inp when fuel_flag = 2")

        return fuel_density, fuel_moisture, fuel_height
    
    def _get_ignition_locations(self):
        if self.ignition_flag == 1:
            self._get_ignitions_rect()
        elif self.ignition_flag == 2:
            self._get_ignitions_sq_ring()
        elif self.ignition_flag == 3:
            self._get_ignitions_cir_ring()

        if self.ignition_flag == 4:
            print("CAUTION: User must provide ignition locations in QF_Ignitions.inp when ignition_flag = 4")
        if self.ignition_flag == 5:
            print("CAUTION: User must provide time- and space-dependent ignition locations in QF_IgnitionPattern.inp when ignition_flag = 5")
        
        return ""
    
    def _get_ignitions_rect(self):
        if len(self.ignition_params) != 4:
            raise ValueError("ignition_params must have length of 4 when ignition_flag = 1 (rectangle ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]

        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
            raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)")
        
    def _get_ignitions_sq_ring(self):
        if len(self.ignition_params) != 6:
            raise ValueError("ignition_params must have length of 6 when ignition_flag = 2 (square ring ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]
        x_wid = self.ignition_params[4]
        y_wid = self.ignition_params[5]
    
        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
                raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)"
                f"\n{str(x_wid)}\t! Width of the ring in the x-direction (m)"
                f"\n{str(y_wid)}\t! Width of the ring in the y-direction (m)")
    
    def _get_ignitions_cir_ring(self):
        if len(self.ignition_params) != 5:
            raise ValueError("ignition_params must have length of 5 when ignition_flag = 3 (circular ring ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]
        wid = self.ignition_params[4]
    
        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
                raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)"
                f"\n{str(wid)}\t! Width of the ring (m)")


    def _get_custom_stretch_grid(self):
        """
        Writes a custom stretch grid to QUIC_fire.inp, if provided.
        """
        if self.stretch_grid_flag == 1:
            # Verify that dz_array is not empty
            if not self.dz_array:
                raise ValueError("dz_array must not be empty if stretch_grid_flag "
                                "is 1. Please provide a dz_array with nz elements"
                                " or use a different stretch_grid_flag.")

            # Verify that nz is equal to the length of dz_array
            if self.nz != len(self.dz_array):
                raise ValueError(f"nz must be equal to the length of dz_array. "
                                f"{self.nz} != {len(self.dz_array)}")

            # Write dz_array lines
            dz_array_lines_list = []
            for dz in self.dz_array:
                dz_array_lines_list.append(f"{float(dz)}")
            dz_array_lines = "\n".join(dz_array_lines_list)

            return f"{dz_array_lines}"
        else:
            return self.nz
        
        
