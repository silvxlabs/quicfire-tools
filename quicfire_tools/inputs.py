"""
QUIC-Fire Tools Simulation Input Module
"""
from __future__ import annotations

# Internal Imports
from quicfire_tools.utils import compute_parabolic_stretched_grid
from quicfire_tools.ignitions import (IgnitionType, RectangleIgnition,
                                      SquareRingIgnition,
                                      CircularRingIgnition)

# Core Imports
import json
import time
import importlib.resources
from pathlib import Path
from typing import Literal
from string import Template

# External Imports
import numpy as np
from pydantic import (BaseModel, Field, NonNegativeInt, PositiveInt,
                      PositiveFloat, NonNegativeFloat, computed_field, field_validator)

DOCS_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("documentation")
TEMPLATES_PATH = importlib.resources.files('quicfire_tools').joinpath(
    'inputs').joinpath("templates")


class InputFile(BaseModel, validate_assignment=True):
    """
    Base class representing an input file.

    This base class provides a common interface for all input files in order to
    accomplish two main goals:
    1) Return documentation for each parameter in the input file.
    2) Provide a method to write the input file to a specified directory.
    """
    filename: str

    @property
    def param_info(self):
        """
        Return a dictionary of parameter information for the input file.
        """
        with open(DOCS_PATH / f"{self.filename}.json", "r") as f:
            return json.load(f)

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
        # return {attr: value for attr, value in self.__dict__.items()
        #         if not attr.startswith('_')}
        return self.model_dump(exclude={"filename", "param_info"})

    def to_file(self, directory: Path, version: str = "latest"):
        """
        Write the input file to a specified directory.

        Attributes
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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class OutputTimes(BaseModel):
    out_time_fire: PositiveInt
    out_time_wind: PositiveInt
    out_time_emis_rad: PositiveInt
    out_time_wind_avg: PositiveInt

    def __str__(self):
        lines = (
            f"{self.out_time_fire}\t! After how many fire time steps to print out fire-related files (excluding emissions and radiation)\n"
            f"{self.out_time_wind}\t! After how many quic updates to print out wind-related files\n"
            f"{self.out_time_emis_rad}\t! After how many fire time steps to average emissions and radiation\n"
            f"{self.out_time_wind_avg}\t! After how many quic updates to print out averaged wind-related files")
        return lines


class FuelParams(BaseModel):
    fuel_density: PositiveFloat
    fuel_moisture: PositiveFloat
    fuel_height: PositiveFloat


class Gridlist(InputFile):
    """
    Class representing the gridlist.txt file. This file contains the grid
    information for the QUIC-Fire simulation when canopies are present.

    Attributes
    ----------
    filename : str
        Name of the file to write to. Default is "gridlist.txt".
    n : int
        Number of cells in the x-direction [-]
    m : int
        Number of cells in the y-direction [-]
    l : int
        Number of cells in the z-direction [-]
    dx : float
        Cell size in the x-direction [m]
    dy : float
        Cell size in the y-direction [m]
    dz : float
        Cell size in the z-direction [m]
    aa1 : float
        Stretching factor for the vertical grid spacing [-]
    """
    filename: str = Field("gridlist", allow_mutation=False)
    n: PositiveInt
    m: PositiveInt
    l: PositiveInt
    dx: PositiveFloat
    dy: PositiveFloat
    dz: PositiveFloat
    aa1: PositiveFloat


class RasterOrigin(InputFile):
    """
    Class representing the rasterorigin.txt file. This file contains the
    coordinates of the south-west corner of the domain in UTM coordinates.

    Attributes
    ----------
    filename : str
        Name of the file to write to. Default is "rasterorigin.txt".
    utm_x : float
        UTM-x coordinates of the south-west corner of domain [m]
    utm_y : float
        UTM-y coordinates of the south-west corner of domain [m]
    """
    filename: str = Field("rasterorigin.txt", allow_mutation=False)
    utm_x: NonNegativeFloat = 0.
    utm_y: NonNegativeFloat = 0.

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a RasterOrigin object from a directory containing a
        rasterorigin.txt file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "rasterorigin.txt", "r") as f:
            lines = f.readlines()
        utm_x = float(lines[0].split()[0])
        utm_y = float(lines[1].split()[0])
        return cls(utm_x=utm_x, utm_y=utm_y)


class QU_Buildings(InputFile):
    """
    Class representing the QU_buildings.inp file. This file contains the
    building-related data for the QUIC-Fire simulation. This class is not
    currently used in QUIC-Fire.

    Attributes
    ----------
    filename : str
        Name of the file to write to. Default is "QU_buildings.inp".
    wall_roughness_length : float
        Wall roughness length [m]. Must be greater than 0. Default is 0.1.
    number_of_buildings : int
        Number of buildings [-]. Default is 0. Not currently used in QUIC-Fire.
    number_of_polygon_nodes : int
        Number of polygon building nodes [-]. Default is 0. Not currently used
        in QUIC-Fire.
    """
    filename: str = Field("QU_buildings.inp", allow_mutation=False)
    wall_roughness_length: PositiveFloat = 0.1
    number_of_buildings: NonNegativeInt = 0
    number_of_polygon_nodes: NonNegativeInt = 0

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QU_Buildings object from a directory containing a
        QU_buildings.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_buildings.inp", "r") as f:
            lines = f.readlines()
        wall_roughness_length = float(lines[1].split()[0])
        number_of_buildings = int(lines[2].split()[0])
        number_of_polygon_nodes = int(lines[3].split()[0])
        return cls(wall_roughness_length=wall_roughness_length,
                   number_of_buildings=number_of_buildings,
                   number_of_polygon_nodes=number_of_polygon_nodes)


class QU_Fileoptions(InputFile):
    """
    Class representing the QU_fileoptions.inp file. This file contains
    file output-related options for the QUIC-Fire simulation.

    Attributes
    ----------
    filename : str
        Name of the file to write to. Default is "QU_fileoptions.inp".
    output_data_file_format_flag : int
        Output data file format flag. Values accepted are [1, 2, 3].
        Recommended value 2. 1 - binary, 2 - ASCII, 3 - both.
    non_mass_conserved_initial_field_flag : int
        Flag to write out non-mass conserved initial field file uofield.dat.
        Values accepted are [0, 1]. Recommended value 0. 0 - off, 1 - on.
    initial_sensor_velocity_field_flag : int
        Flag to write out the file uosensorfield.dat. Values accepted are
        [0, 1]. Recommended value 0. 0 - off, 1 - on.
    qu_staggered_velocity_file_flag : int
        Flag to write out the file QU_staggered_velocity.bin. Values accepted
        are [0, 1]. Recommended value 0. 0 - off, 1 - on.
    generate_wind_startup_files_flag : int
        Generate wind startup files for ensemble simulations. Values accepted
        are [0, 1]. Recommended value 0. 0 - off, 1 - on.
    """
    filename: str = Field("QU_fileoptions.inp", allow_mutation=False)
    output_data_file_format_flag: Literal[1, 2, 3] = 2
    non_mass_conserved_initial_field_flag: Literal[0, 1] = 0
    initial_sensor_velocity_field_flag: Literal[0, 1] = 0
    qu_staggered_velocity_file_flag: Literal[0, 1] = 0
    generate_wind_startup_files_flag: Literal[0, 1] = 0

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QU_Fileoptions object from a directory containing a
        QU_fileoptions.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_fileoptions.inp", "r") as f:
            lines = f.readlines()
        output_data_file_format_flag = int(lines[1].split()[0])
        non_mass_conserved_initial_field_flag = int(lines[2].split()[0])
        initial_sensor_velocity_field_flag = int(lines[3].split()[0])
        qu_staggered_velocity_file_flag = int(lines[4].split()[0])
        generate_wind_startup_files_flag = int(lines[5].split()[0])
        return cls(output_data_file_format_flag=output_data_file_format_flag,
                   non_mass_conserved_initial_field_flag=non_mass_conserved_initial_field_flag,
                   initial_sensor_velocity_field_flag=initial_sensor_velocity_field_flag,
                   qu_staggered_velocity_file_flag=qu_staggered_velocity_file_flag,
                   generate_wind_startup_files_flag=generate_wind_startup_files_flag)


class QU_Simparams(InputFile):
    """
    Class representing the QU_simparams.inp file. This file contains the
    simulation parameters for the QUIC-Fire simulation.

    filename : str
        Name of the file to write to. Default is "QU_simparams.inp".
    nx : int
        Number of cells in the x-direction [-]. Recommended value: > 100
    ny : int
        Number of cells in the y-direction [-]. Recommended value: > 100
    nz : int
        Number of cells in the z-direction [-].
    dx : float
        Cell size in the x-direction [m]. Recommended value: 2 m
    dy : float
        Cell size in the y-direction [m]. Recommended value: 2 m
    surface_vertical_cell_size : float
        Surface vertical cell size [m].
    number_surface_cells : int
        Number of uniform surface cells [-].
    stretch_grid_flag : int
        Vertical grid stretching flag, values accepted [0, 1, 3]. Default is 3.
        0 - uniform, 1 - custom, 3 - parabolic. If stretch_grid_flag is 0 or 3
        dz_array is computed from nz, surface_vertical_cell_size, and
        number_surface_cells. If stretch_grid_flag is 1, custom_dz_array must
        be provided.
    custom_dz_array : list[float]
        Vertical grid spacing array [m]. Must be provided if stretch_grid_flag
        is 1. If stretch_grid_flag is 0 or 3, dz_array is computed from nz,
        surface_vertical_cell_size, and number_surface_cells.
    utc_offset : int
        Hours difference from UTC (aka UTM) [h]
    wind_times : list[int]
        List of times at which the winds are available in Unix Epoch time
        (integer seconds since 1970/1/1 00:00:00). These are UTC times.
        Defaults to [int(time.time())] if not provided.
    sor_iter_max : int
        Maximum number of iterations of the SOR wind solver. Recommended value:
        10. Default is 10.
    sor_residual_reduction : int
        Residual reduction to assess convergence of the SOR solver (orders of
        magnitude). Recommended value: 3. Default is 3.
    use_diffusion_flag : int
        Use diffusion algorithm: 0 = off, 1 = on. Recommended value: 0.
        Default is 0.
    number_diffusion_iterations : int
        Number of diffusion iterations. Recommended value: 10. Default is 10.
    domain_rotation : float
        Domain rotation relative to true north (clockwise is positive)
        [degrees]. Recommended value: 0 deg. Default is 0.
    utm_x : float
        UTM-x coordinates of the south-west corner of domain [m]. Default is 0.
    utm_y : float
        UTM-y coordinates of the south-west corner of domain [m]. Default is 0.
    utm_zone_number : int
        UTM zone number [-]. Default is 1.
    utm_zone_letter : int
        UTM zone letter (A=1, B=2, ...) [-]. Default is 1.
    quic_cfd_flag : int
        QUIC-CFD flag: 0 = off, 1 = on. Recommended value: 0. Default is 0.
    explosive_bldg_flag : int
        Explosive building damage flag: 0 = off, 1 = on. Recommended value: 0.
        Default is 0.
    bldg_array_flag : int
        Building array flag. 0 = off, 1 = on. Recommended value: 0. Default is
        0.
    """
    filename: str = Field("QU_simparams.inp", allow_mutation=False)
    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt
    dx: PositiveFloat
    dy: PositiveFloat
    surface_vertical_cell_size: PositiveFloat = 1.
    number_surface_cells: PositiveInt = 5
    stretch_grid_flag: Literal[0, 1, 3] = 3
    custom_dz_array: list[PositiveFloat] = []
    utc_offset: int = 0
    wind_times: list[int] = [int(time.time())]
    sor_iter_max: PositiveInt = 10
    sor_residual_reduction: PositiveInt = 3
    use_diffusion_flag: Literal[0, 1] = 0
    number_diffusion_iterations: PositiveInt = 10
    domain_rotation: float = 0.
    utm_x: float = 0.
    utm_y: float = 0.
    utm_zone_number: PositiveInt = 1
    utm_zone_letter: PositiveInt = 1
    quic_cfd_flag: Literal[0, 1] = 0
    explosive_bldg_flag: Literal[0, 1] = 0
    bldg_array_flag: Literal[0, 1] = 0
    _from_file: bool = False
    _from_file_dz_array: list[PositiveFloat] = []

    @computed_field
    @property
    def dz_array(self) -> list[float]:
        if self._from_file:
            return self._from_file_dz_array
        elif self.stretch_grid_flag == 0:
            return [self.surface_vertical_cell_size] * self.nz
        elif self.stretch_grid_flag == 1:
            return self.custom_dz_array
        elif self.stretch_grid_flag == 3:
            return compute_parabolic_stretched_grid(
                self.surface_vertical_cell_size, self.number_surface_cells,
                self.nz, 300).tolist()

    @computed_field
    @property
    def vertical_grid_lines(self) -> str:
        """
        Parses the vertical grid stretching flag and dz_array to generate the
        vertical grid as a string for the QU_simparams.inp file.

        Also modifies dz_array if stretch_grid_flag is not 1.
        """
        stretch_grid_func_map = {
            0: self._stretch_grid_flag_0,
            1: self._stretch_grid_flag_1,
            3: self._stretch_grid_flag_3
        }
        return stretch_grid_func_map[self.stretch_grid_flag]()

    @computed_field
    @property
    def wind_time_lines(self) -> str:
        return self._generate_wind_time_lines()

    def _stretch_grid_flag_0(self):
        """
        Generates a uniform vertical grid as a string for the QU_simparams.inp
        file. Adds the uniform grid to dz_array.
        """
        # Create the lines for the uniform grid
        surface_dz_line = (f"{float(self.surface_vertical_cell_size)}\t"
                           f"! Surface DZ [m]")
        number_surface_cells_line = (f"{self.number_surface_cells}\t"
                                     f"! Number of uniform surface cells")

        return f"{surface_dz_line}\n{number_surface_cells_line}"

    def _stretch_grid_flag_1(self):
        """
        Generates a custom vertical grid as a string for the QU_simparams.inp
        file.
        """
        # Verify that dz_array is not empty
        if not self.dz_array:
            raise ValueError("dz_array must not be empty if stretch_grid_flag "
                             "is 1. Please provide a custom_dz_array with nz "
                             "elements or use a different stretch_grid_flag.")

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

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QU_Simparams object from a directory containing a
        QU_simparams.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QU_simparams.inp", "r") as f:
            lines = f.readlines()

        # Read QU grid parameters
        nx = int(lines[1].strip().split("!")[0])
        ny = int(lines[2].strip().split("!")[0])
        nz = int(lines[3].strip().split("!")[0])
        dx = float(lines[4].strip().split("!")[0])
        dy = float(lines[5].strip().split("!")[0])

        # Read stretch grid flag
        stretch_grid_flag = int(lines[6].strip().split("!")[0])

        # Read vertical grid lines as function of stretch grid flag
        _from_file_dz_array = []
        custom_dz_array = []
        if stretch_grid_flag == 0:
            surface_vertical_cell_size = float(lines[7].strip().split("!")[0])
            number_surface_cells = int(lines[8].strip().split("!")[0])
            current_line = 9
        elif stretch_grid_flag == 1:
            surface_vertical_cell_size = float(lines[7].strip().split("!")[0])
            number_surface_cells = 5
            for i in range(9, 9 + nz):
                custom_dz_array.append(float(lines[i].strip().split("!")[0]))
            current_line = 9 + nz
        elif stretch_grid_flag == 3:
            surface_vertical_cell_size = float(lines[7].strip().split("!")[0])
            number_surface_cells = int(lines[8].strip().split("!")[0])
            _header = lines[9].strip().split("!")[0]
            for i in range(10, 10 + nz):
                _from_file_dz_array.append(
                    float(lines[i].strip().split("!")[0]))
            current_line = 10 + nz
        else:
            raise ValueError("stretch_grid_flag must be 0, 1, or 3.")

        # Read QU wind parameters
        number_wind_steps = int(lines[current_line].strip().split("!")[0])
        utc_offset = int(lines[current_line + 1].strip().split("!")[0])
        _header = lines[current_line + 2].strip().split("!")[0]
        wind_times = []
        for i in range(current_line + 3, current_line + 3 + number_wind_steps):
            wind_times.append(int(lines[i].strip()))
        current_line = current_line + 3 + number_wind_steps

        # Skip not used parameters
        current_line += 9

        # Read remaining QU parameters
        sor_iter_max = int(lines[current_line].strip().split("!")[0])
        sor_residual_reduction = int(
            lines[current_line + 1].strip().split("!")[0])
        use_diffusion_flag = int(lines[current_line + 2].strip().split("!")[0])
        number_diffusion_iterations = int(
            lines[current_line + 3].strip().split("!")[0])
        domain_rotation = float(lines[current_line + 4].strip().split("!")[0])
        utm_x = float(lines[current_line + 5].strip().split("!")[0])
        utm_y = float(lines[current_line + 6].strip().split("!")[0])
        utm_zone_number = int(lines[current_line + 7].strip().split("!")[0])
        utm_zone_letter = int(lines[current_line + 8].strip().split("!")[0])
        quic_cfd_flag = int(lines[current_line + 9].strip().split("!")[0])
        explosive_bldg_flag = int(
            lines[current_line + 10].strip().split("!")[0])
        bldg_array_flag = int(lines[current_line + 11].strip().split("!")[0])

        return cls(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy,
                   surface_vertical_cell_size=surface_vertical_cell_size,
                   number_surface_cells=number_surface_cells,
                   stretch_grid_flag=stretch_grid_flag,
                   custom_dz_array=custom_dz_array,
                   utc_offset=utc_offset,
                   wind_times=wind_times,
                   sor_iter_max=sor_iter_max,
                   sor_residual_reduction=sor_residual_reduction,
                   use_diffusion_flag=use_diffusion_flag,
                   number_diffusion_iterations=number_diffusion_iterations,
                   domain_rotation=domain_rotation,
                   utm_x=utm_x,
                   utm_y=utm_y,
                   utm_zone_number=utm_zone_number,
                   utm_zone_letter=utm_zone_letter,
                   quic_cfd_flag=quic_cfd_flag,
                   explosive_bldg_flag=explosive_bldg_flag,
                   bldg_array_flag=bldg_array_flag,
                   _from_file=True,
                   _from_file_dz_array=_from_file_dz_array)


class QFire_Advanced_User_Inputs(InputFile):
    """
    Class representing the QFire_Advanced_User_Inputs.inp input file. This file
    contains advanced parameters related to firebrands.

    Attributes
    ----------
    fraction_cells_launch_firebrands : PositiveFloat
        Fraction of cells that could launch firebrand tracers from which
        firebrand tracers will actually be launched [-]. Higher value = more
        firebrand tracers. Recommended value: 0.05
    firebrand_radius_scale_factor : PositiveFloat
        Multiplicative factor used to relate the length scale of the mixing
        (firebrand distribution entrainment length scale) to the initial size
        of the distribution [-]. Higher value = higher growth rate or RT
        (firebrand distribution) with flight time. Recommended value: 40
    firebrand_trajectory_time_step : PositiveInt
        Time step used to determine the firebrand tracer trajectory [s].
        Higher value = less accurate trajectory. Recommended value: 1 s
    firebrand_launch_interval : PositiveInt
        Time interval between launching of firebrand tracers [s]. Higher value =
        less firebrand tracers launched. Recommended value: 10 s
    firebrands_per_deposition : PositiveInt
        Number of firebrand tracers that one firebrand tracer launched
        represents [-]. Recommended value: 500
    firebrand_area_ratio : PositiveFloat
        Multiplicative factor used to relate the cell area and fraction of cells
        from which tracers are launched to initial area represented by one
        firebrand [-].
    minimum_burn_rate_coefficient : PositiveFloat
        Multiplicative factor relating the minimum mass-loss rate that a
        firebrand tracer needs to have to justify continuing to track its
        trajectory to the energy associated with a new ignition [-].
    max_firebrand_thickness_fraction : PositiveFloat
        Multiplicative factor relating the thickness of launched firebrand
        tracer to maximum loftable firebrand thickness [-].
    firebrand_germination_delay : PositiveInt
        Time after a firebrand has landed at which a fire is started [s]
    vertical_velocity_scale_factor : PositiveFloat
        Maximum value of the multiplicative factor of the vertical velocity
        experienced by a firebrand = 1/(fraction of the QUIC-URB on fire) [-]
    minimum_firebrand_ignitions : PositiveInt
        Minimum number of ignitions to be sampled in a position where a
        firebrand lands [-]
    maximum_firebrand_ignitions : PositiveInt
        Maximum number of ignitions sampled at positions distributed within RT
        around where a firebrand tracer lands [-]
    minimum_landing_angle : PositiveFloat
        Minimum value considered for the angle between the trajectory of the
        firebrand when it hits the ground and horizontal [rad]
    maximum_firebrand_thickness : PositiveFloat
        Maximum firebrand's thickness [m]
    """
    filename: str = Field("QFire_Advanced_User_Inputs.inp",
                          allow_mutation=False)
    fraction_cells_launch_firebrands: PositiveFloat = Field(0.05, ge=0, lt=1)
    firebrand_radius_scale_factor: PositiveFloat = Field(40., ge=1)
    firebrand_trajectory_time_step: PositiveInt = 1
    firebrand_launch_interval: PositiveInt = 10
    firebrands_per_deposition: PositiveInt = 500
    firebrand_area_ratio: PositiveFloat = 20.0
    minimum_burn_rate_coefficient: PositiveFloat = 50.0
    max_firebrand_thickness_fraction: PositiveFloat = 0.75
    firebrand_germination_delay: PositiveInt = 180
    vertical_velocity_scale_factor: PositiveFloat = 5.0
    minimum_firebrand_ignitions: PositiveInt = 50
    maximum_firebrand_ignitions: PositiveInt = 100
    minimum_landing_angle: PositiveFloat = Field(0.523598, ge=0, le=np.pi / 2)
    maximum_firebrand_thickness: PositiveFloat = 0.03

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QFire_Advanced_User_Inputs object from a directory
        containing a QFire_Advanced_User_Inputs.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QFire_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()
        fraction_cells_launch_firebrands = float(lines[0].split()[0])
        firebrand_radius_scale_factor = float(lines[1].split()[0])
        firebrand_trajectory_time_step = int(lines[2].split()[0])
        firebrand_launch_interval = int(lines[3].split()[0])
        firebrands_per_deposition = int(lines[4].split()[0])
        firebrand_area_ratio = float(lines[5].split()[0])
        minimum_burn_rate_coefficient = float(lines[6].split()[0])
        max_firebrand_thickness_fraction = float(lines[7].split()[0])
        firebrand_germination_delay = int(lines[8].split()[0])
        vertical_velocity_scale_factor = float(lines[9].split()[0])
        minimum_firebrand_ignitions = int(lines[10].split()[0])
        maximum_firebrand_ignitions = int(lines[11].split()[0])
        minimum_landing_angle = float(lines[12].split()[0])
        maximum_firebrand_thickness = float(lines[13].split()[0])
        return cls(
            fraction_cells_launch_firebrands=fraction_cells_launch_firebrands,
            firebrand_radius_scale_factor=firebrand_radius_scale_factor,
            firebrand_trajectory_time_step=firebrand_trajectory_time_step,
            firebrand_launch_interval=firebrand_launch_interval,
            firebrands_per_deposition=firebrands_per_deposition,
            firebrand_area_ratio=firebrand_area_ratio,
            minimum_burn_rate_coefficient=minimum_burn_rate_coefficient,
            max_firebrand_thickness_fraction=max_firebrand_thickness_fraction,
            firebrand_germination_delay=firebrand_germination_delay,
            vertical_velocity_scale_factor=vertical_velocity_scale_factor,
            minimum_firebrand_ignitions=minimum_firebrand_ignitions,
            maximum_firebrand_ignitions=maximum_firebrand_ignitions,
            minimum_landing_angle=minimum_landing_angle,
            maximum_firebrand_thickness=maximum_firebrand_thickness)


class QUIC_fire(InputFile):
    """
    Class representing the QUIC_fire.inp input file. This file
    contains the parameters relating to the fire simulation and
    outputs.

    Parameters
    ----------
    nz : int
        Number of fire grid cells in the z-direction.
    output_times: int | OutputTimes
        @anthony
    time_now : int
        When the fire is ignited in Unix Epoch time (integer seconds since
        1970/1/1 00:00:00). Must be greater or equal to the time of the first
        wind
    sim_time : int
        Total simulation time for the fire [s]
    fire_flag : int
        Fire flag, 1 = run fire; 0 = no fire
    random_seed : int
        Random number generator, -1: use time and date, any other integer > 0
        is used as the seed
    fire_time_step : int
        time step for the fire simulation [s]
    quic_time_step : int
        Number of fire time steps done before updating the quic wind field
        (integer, >= 1)
    stretch_grid_flag : int
        Vertical stretching flag: 0 = uniform dz, 1 = custom
    file_path : str
        Path to files defining fuels, ignitions, and topography, with file
        separator at the end. Defaults to "", indicating files are in the same directory as all other input files
    dz : int
        Cell size in the z-direction [m] of the fire grid. Recommended value: 1m
    dz_array : list[float]
        custom dz, one dz per line must be specified, from the ground to the
        top of the domain
    fuel_flag : int
        Flag for fuel inputs:
            - density
            - moisture
            - height
        1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files
        for quic grid, 4 = Firetech files for different grid
        (need interpolation)
    fuel_params : FuelParams
        FuelParams class with named fuel parameters for a uniform grid.
        Defaults to None unless fuel_flag = 1
    ignition_type: IgnitionType
        Ignitions shape or source. @anthony
    ignitions_per_cell: int
        Number of ignition per cell of the fire model. Recommended max value
        of 100
    firebrand_flag : int
        Firebrand flag, 0 = off; 1 = on. Recommended value = 0 ; firebrands
        are untested for small scale problems
    auto_kill : int
        Kill if the fire is out and there are no more ignitions or firebrands
        (0 = no, 1 = yes)
    eng_to_atm_out : int
        Output flag [0, 1]: gridded energy-to-atmosphere
        (3D fire grid + extra layers)
    react_rate_out : int
        Output flag [0, 1]: compressed array reaction rate (fire grid)
    fuel_dens_out : int
        Output flag [0, 1]: compressed array fuel density (fire grid)
    QF_wind_out : int
        Output flag [0, 1]: gridded wind (u,v,w,sigma) (3D fire grid)
    QU_wind_inst_out : int
        Output flag [0, 1]: gridded QU winds with fire effects, instantaneous
        (QUIC-URB grid)
    QU_wind_avg_out : int
        Output flag [0, 1]: gridded QU winds with fire effects, averaged
        (QUIC-URB grid)
    fuel_moist_out : int
        Output flag [0, 1]: compressed array fuel moisture (fire grid)
    mass_burnt_out : int
        Output flag [0, 1]: vertically-integrated % mass burnt (fire grid)
    firebrand_out : int
        Output flag [0, 1]: firebrand trajectories. Must be 0 when firebrand
        flag is 0
    emissions_out : int
        Output flag [0, 5]: compressed array emissions (fire grid):
            0 = do not output any emission related variables
            1 = output emissions files and simulate CO in QUIC-SMOKE
            2 = output emissions files and simulate PM2.5 in QUIC- SMOKE
            3 = output emissions files and simulate both CO and PM2.5 in
                QUIC-SMOKE
            4 = output emissions files but use library approach in QUIC-SMOKE
            5 = output emissions files and simulate both water in QUIC-SMOKE
    radiation_out : int
        Output flag [0, 1]: gridded thermal radiation (fire grid)
    intensity_out : int
        Output flag [0, 1]: surface fire intensity at every fire time step
    """
    filename: str = Field("QUIC_fire.inp", allow_mutation=False)
    nz: PositiveInt
    time_now: PositiveInt
    sim_time: PositiveInt
    output_times: PositiveInt | OutputTimes
    ignition_type: IgnitionType = IgnitionType(ignition_flag=6)
    fire_flag: Literal[0, 1] = 1
    random_seed: int = Field(ge=-1, default = -1)
    fire_time_step: PositiveInt = 1
    quic_time_step: PositiveInt = 1
    stretch_grid_flag: Literal[0, 1] = 0
    dz: PositiveInt = 1
    dz_array: list[PositiveFloat] = []
    fuel_flag: Literal[1, 2, 3, 4] = 3
    fuel_params: FuelParams | None = None
    ignitions_per_cell: PositiveInt = 2
    firebrand_flag: Literal[0, 1] = 0
    auto_kill: Literal[0, 1] = 1
    # Output flags
    eng_to_atm_out: Literal[0, 1] = 1
    react_rate_out: Literal[0, 1] = 0
    fuel_dens_out: Literal[0, 1] = 1
    QF_wind_out: Literal[0, 1] = 1
    QU_wind_inst_out: Literal[0, 1] = 1
    QU_wind_avg_out: Literal[0, 1] = 0
    fuel_moist_out: Literal[0, 1] = 1
    mass_burnt_out: Literal[0, 1] = 1
    firebrand_out: Literal[0, 1] = 0
    emissions_out: Literal[0, 1, 2, 3, 4, 5] = 0
    radiation_out: Literal[0, 1] = 0
    intensity_out: Literal[0, 1] = 0

    @field_validator('random_seed')
    @classmethod
    def validate_smoothing(cls, v: int) -> int:
        if v == 0: raise ValueError(f"QUIC_fire.inp: random_seed must be not be 0")
        return v

    @computed_field
    @property
    def stretch_grid_input(self) -> str:
        """
        Writes a custom stretch grid to QUIC_fire.inp, if provided.
        """
        if self.stretch_grid_flag == 1:
            # Verify that dz_array is not empty
            if not self.dz_array:
                raise ValueError(
                    "dz_array must not be empty if stretch_grid_flag "
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

            return f"{dz_array_lines}\n"
        else:
            return str(self.dz)

    @computed_field
    @property
    def out_time_lines(self) -> str:
        if isinstance(self.output_times, int):
            self.output_times = OutputTimes(out_time_fire=self.output_times,
                                            out_time_wind=self.output_times,
                                            out_time_emis_rad=self.output_times,
                                            out_time_wind_avg=self.output_times)
        return str(self.output_times)

    @computed_field
    @property
    def ignition_lines(self) -> str:
        return str(self.ignition_type)

    @computed_field
    @property
    def fuel_lines(self) -> str:
        flag_line = (f" 1 = uniform; "
                     f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
                     f" files for quic grid, 4 = Firetech files for "
                     f"different grid (need interpolation)")
        fuel_density_flag_line = f"{self.fuel_flag}\t! fuel density flag:" + flag_line
        fuel_moist_flag_line = f"\n{self.fuel_flag}\t! fuel moisture flag:" + flag_line
        fuel_height_flag_line = f"\n{self.fuel_flag}\t! fuel height flag:" + flag_line
        if self.fuel_flag == 1:
            try:
                fuel_dens_line = f"\n{self.fuel_params.fuel_density}"
                fuel_moist_line = f"\n{self.fuel_params.fuel_moisture}"
                fuel_height_line = f"\n{self.fuel_params.fuel_height}"
            except IndexError:
                raise ValueError(
                    "fuel_params: FuelInputs class must have values for fuel_density, fuel_moisture, and fuel_height")
            return fuel_density_flag_line + fuel_dens_line + fuel_moist_flag_line + fuel_moist_line + fuel_height_flag_line + fuel_height_line
        return fuel_density_flag_line + fuel_moist_flag_line

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QUIC_fire object from a directory containing a
        QUIC_Fire.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QUIC_fire.inp", "r") as f:
            lines = f.readlines()

        # Read fire flag and random seed
        fire_flag = int(lines[0].strip().split("!")[0])
        random_seed = int(lines[1].strip().split("!")[0])

        # Read fire times
        time_now = int(lines[3].strip().split("!")[0])
        sim_time = int(lines[4].strip().split("!")[0])
        fire_time_step = int(lines[5].strip().split("!")[0])
        quic_time_step = int(lines[6].strip().split("!")[0])
        out_time_fire = int(lines[7].strip().split("!")[0])
        out_time_wind = int(lines[8].strip().split("!")[0])
        out_time_emis_rad = int(lines[9].strip().split("!")[0])
        out_time_wind_avg = int(lines[10].strip().split("!")[0])
        output_times = OutputTimes(out_time_fire=out_time_fire,
                                   out_time_wind=out_time_wind,
                                   out_time_emis_rad=out_time_emis_rad,
                                   out_time_wind_avg=out_time_wind_avg)

        # Read fire grid parameters
        nz = int(lines[12].strip().split("!")[0])
        stretch_grid_flag = int(lines[13].strip().split("!")[0])
        dz_array = []
        if stretch_grid_flag == 0:
            dz = int(lines[14].strip().split("!")[0])
            current_line = 15
        else:
            for i in range(14, 14 + len(nz)):
                try:
                    float(lines[i].strip())
                except ValueError:
                    print(
                        "QUIC_fire.inp: dz input value is not a float. Does the number of dz inputs match nz?")
                dz_array.append(float(lines[i].strip()))
            current_line = 15 + len(nz)

        current_line += 4  # skip unused lines

        # Read fuel inputs
        # current_line = ! FUEL
        current_line += 1  # header
        fuel_flag = int(lines[current_line].strip().split("!")[0])
        if fuel_flag == 1:
            fuel_density = float(lines[current_line + 1].strip())
            moisture_flag = int(lines[current_line + 2].strip().split("!")[0])
            fuel_moisture = float(lines[current_line + 3].strip())
            height_flag = int(lines[current_line + 4].strip().split("!")[0])
            fuel_height = float(lines[current_line + 5].strip())
            fuel_params = FuelParams(fuel_density, fuel_moisture, fuel_height)
            if moisture_flag != fuel_flag or height_flag != fuel_flag:
                raise ValueError(
                    "QUIC_fire.inp: Fuel moisture and fue height flags must match fuel density flag")
            current_line += 6
        else:
            fuel_params = None
            current_line += 2

            # Read ignition inputs
        # current_line = ! IGNITION LOCATIONS
        current_line += 1  # header
        ignition_flag = int(lines[current_line].strip().split("!")[0])
        add_lines = {1: 4, 2: 6, 3: 5, 4: 0, 5: 0, 6: 0, 7: 0}
        add = add_lines.get(ignition_flag)
        ignition_params = []
        current_line += 1
        for i in range(current_line, current_line + add):
            ignition_params.append(int(lines[i].strip().split("!")[0]))
        if ignition_flag == 1:
            x_min, y_min, x_length, y_length = ignition_params
            ignition_type = RectangleIgnition(x_min, y_min, x_length, y_length)
        elif ignition_flag == 2:
            x_min, y_min, x_length, y_length, x_width, y_width = ignition_params
            ignition_type = SquareRingIgnition(x_min, y_min, x_length, y_length,
                                               x_width, y_width)
        elif ignition_flag == 3:
            x_min, y_min, x_length, y_length, ring_width = ignition_params
            ignition_type = CircularRingIgnition(x_min, y_min, x_length,
                                                 y_length, ring_width)
        elif ignition_flag == 6:
            ignition_type = IgnitionType(ignition_flag=6)
        current_line += add
        ignitions_per_cell = int(lines[current_line].strip().split("!")[0])
        current_line += 1

        # Read firebrands
        # current_line = ! FIREBRANDS
        current_line += 1  # header
        firebrand_flag = int(lines[current_line].strip().split("!")[0])
        current_line += 1

        # Read output flags
        # current_line = !OUTPUT_FILES
        eng_to_atm_out = int(lines[current_line + 1].strip().split("!")[0])
        react_rate_out = int(lines[current_line + 2].strip().split("!")[0])
        fuel_dens_out = int(lines[current_line + 3].strip().split("!")[0])
        QF_wind_out = int(lines[current_line + 4].strip().split("!")[0])
        QU_wind_inst_out = int(lines[current_line + 5].strip().split("!")[0])
        QU_wind_avg_out = int(lines[current_line + 6].strip().split("!")[0])
        # ! Output plume trajectories
        fuel_moist_out = int(lines[current_line + 8].strip().split("!")[0])
        mass_burnt_out = int(lines[current_line + 9].strip().split("!")[0])
        firebrand_out = int(lines[current_line + 10].strip().split("!")[0])
        emissions_out = int(lines[current_line + 11].strip().split("!")[0])
        radiation_out = int(lines[current_line + 12].strip().split("!")[0])
        intensity_out = int(lines[current_line + 13].strip().split("!")[0])
        # ! AUTOKILL
        auto_kill = int(lines[current_line + 15].strip().split("!")[0])

        return cls(fire_flag=fire_flag,
                   random_seed=random_seed,
                   time_now=time_now,
                   sim_time=sim_time,
                   fire_time_step=fire_time_step,
                   quic_time_step=quic_time_step,
                   output_times=output_times,
                   nz=nz,
                   stretch_grid_flag=stretch_grid_flag,
                   dz=dz,
                   dz_array=dz_array,
                   fuel_flag=fuel_flag,
                   fuel_params=fuel_params,
                   ignition_type=ignition_type,
                   ignitions_per_cell=ignitions_per_cell,
                   firebrand_flag=firebrand_flag,
                   eng_to_atm_out=eng_to_atm_out,
                   react_rate_out=react_rate_out,
                   fuel_dens_out=fuel_dens_out,
                   QF_wind_out=QF_wind_out,
                   QU_wind_inst_out=QU_wind_inst_out,
                   QU_wind_avg_out=QU_wind_avg_out,
                   fuel_moist_out=fuel_moist_out,
                   mass_burnt_out=mass_burnt_out,
                   firebrand_out=firebrand_out,
                   emissions_out=emissions_out,
                   radiation_out=radiation_out,
                   intensity_out=intensity_out,
                   auto_kill=auto_kill)
