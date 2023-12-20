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
from typing import Literal, Union

# External Imports
import numpy as np
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    SerializeAsAny,
)

# Internal imports
from quicfire_tools.ignitions import (
    IgnitionSources,
    CircularRingIgnition,
    IgnitionType,
    RectangleIgnition,
    SquareRingIgnition,
    default_line_ignition,
)
from quicfire_tools.topography import (
    TopoSources,
    CanyonTopo,
    CosHillTopo,
    GaussianHillTopo,
    HalfCircleTopo,
    HillPassTopo,
    SinusoidTopo,
    SlopeMesaTopo,
    TopoType,
)
from quicfire_tools.utils import compute_parabolic_stretched_grid


DOCS_PATH = (
    importlib.resources.files("quicfire_tools")
    .joinpath("data")
    .joinpath("documentation")
)
TEMPLATES_PATH = (
    importlib.resources.files("quicfire_tools").joinpath("data").joinpath("templates")
)


class SimulationInputs:
    """
    Class representing a QUIC-Fire input file deck.

    This is the fundamental class in the quicfire_tools.data module. It is
    used to create, modify, and write QUIC-Fire input file decks. It is also
    used to read in existing QUIC-Fire input file decks.

    Attributes
    ----------
    rasterorigin: RasterOrigin
        Object representing the rasterorigin.txt file.
    qu_buildings: QU_Buildings
        Object representing the QU_buildings.inp file.
    qu_fileoptions: QU_Fileoptions
        Object representing the QU_fileoptions.inp file.
    qfire_advanced_user_inputs: QFire_Advanced_User_Inputs
        Object representing the qfire_advanced_user_inputs.inp file.
    qfire_bldg_advanced_user_inputs: QFire_Bldg_Advanced_User_Inputs
        Object representing the qfire_bldg_advanced_user_inputs.inp file.
    qfire_plume_advanced_user_inputs: QFire_Plume_Advanced_User_Inputs
        Object representing the qfire_plume_advanced_user_inputs.inp file.
    runtime_advanced_user_inputs: RuntimeAdvancedUserInputs
        Object representing the runtime_advanced_user_inputs.inp file.
    qu_movingcoords: QU_movingcoords
        Object representing the QU_movingcoords.inp file.
    qp_buildout: QP_buildout
        Object representing the qp_buildout.inp file.
    qu_metparams: QU_metparams
        Object representing the QU_metparams.inp file.
    quic_fire: QUIC_fire
        Object representing the QUIC_fire.inp file.
    gridlist: Gridlist
        Object representing the gridlist.txt file.
    sensor1: Sensor1
        Object representing the sensor1.inp file.
    qu_topoinputs: QU_TopoInputs
        Object representing the QU_topoinputs.inp file.
    qu_simparams: QU_Simparams
        Object representing the QU_simparams.inp file.
    """

    def __init__(
        self,
        rasterorigin: RasterOrigin,
        qu_buildings: QU_Buildings,
        qu_fileoptions: QU_Fileoptions,
        qfire_advanced_user_inputs: QFire_Advanced_User_Inputs,
        qfire_bldg_advanced_user_inputs: QFire_Bldg_Advanced_User_Inputs,
        qfire_plume_advanced_user_inputs: QFire_Plume_Advanced_User_Inputs,
        runtime_advanced_user_inputs: RuntimeAdvancedUserInputs,
        qu_movingcoords: QU_movingcoords,
        qp_buildout: QP_buildout,
        qu_metparams: QU_metparams,
        quic_fire: QUIC_fire,
        gridlist: Gridlist,
        sensor1: Sensor1,
        qu_topoinputs: QU_TopoInputs,
        qu_simparams: QU_Simparams,
    ):
        # Store the input files as attributes
        self.rasterorigin = rasterorigin
        self.qu_buildings = qu_buildings
        self.qu_fileoptions = qu_fileoptions
        self.qfire_advanced_user_inputs = qfire_advanced_user_inputs
        self.qfire_bldg_advanced_user_inputs = qfire_bldg_advanced_user_inputs
        self.qfire_plume_advanced_user_inputs = qfire_plume_advanced_user_inputs
        self.runtime_advanced_user_inputs = runtime_advanced_user_inputs
        self.qu_movingcoords = qu_movingcoords
        self.qp_buildout = qp_buildout
        self.qu_metparams = qu_metparams
        self.quic_fire = quic_fire
        self.gridlist = gridlist
        self.sensor1 = sensor1
        self.qu_topoinputs = qu_topoinputs
        self.qu_simparams = qu_simparams

        # Create a dictionary from the local variables
        self._input_files_dict = {
            "rasterorigin": rasterorigin,
            "qu_buildings": qu_buildings,
            "qu_fileoptions": qu_fileoptions,
            "qfire_advanced_user_inputs": qfire_advanced_user_inputs,
            "qfire_bldg_advanced_user_inputs": qfire_bldg_advanced_user_inputs,
            "qfire_plume_advanced_user_inputs": qfire_plume_advanced_user_inputs,
            "runtime_advanced_user_inputs": runtime_advanced_user_inputs,
            "qu_movingcoords": qu_movingcoords,
            "qp_buildout": qp_buildout,
            "qu_metparams": qu_metparams,
            "quic_fire": quic_fire,
            "gridlist": gridlist,
            "sensor1": sensor1,
            "qu_topoinputs": qu_topoinputs,
            "qu_simparams": qu_simparams,
        }

    @classmethod
    def create_simulation(
        cls,
        nx: int,
        ny: int,
        fire_nz: int,
        wind_speed: float,
        wind_direction: int,
        simulation_time: int,
    ):
        """
        Creates a SimulationInputs object to build a QUIC-Fire input file deck
        and run a simulation.

        Parameters
        ----------
        nx: int
            Number of cells in the x-direction [-]
        ny: int
            Number of cells in the y-direction [-]
        fire_nz: int
            Number of cells in the z-direction for the fire grid [-]
        wind_speed: float
            Wind speed [m/s]
        wind_direction: float
            Wind direction [deg]. 0 deg is north, 90 deg is east, etc. Must
            be in range [0, 360).
        simulation_time: int
            Number of seconds to run the simulation for [s]

        Returns
        -------
        SimulationInputs
            Class containing the data to build a QUIC-Fire
            input file deck and run a simulation using default parameters.
        """
        # Initialize default input files
        rasterorigin = RasterOrigin()
        qu_buildings = QU_Buildings()
        qu_fileoptions = QU_Fileoptions()
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        qfire_bldg_advanced_user_inputs = QFire_Bldg_Advanced_User_Inputs()
        qfire_plume_advanced_user_inputs = QFire_Plume_Advanced_User_Inputs()
        runtime_advanced_user_inputs = RuntimeAdvancedUserInputs()
        qu_movingcoords = QU_movingcoords()
        qp_buildout = QP_buildout()
        qu_metparams = QU_metparams()

        # Initialize input files with required parameters
        start_time = int(time.time())
        ignition_type = default_line_ignition(nx, ny, wind_direction)
        quic_fire = QUIC_fire(
            nz=fire_nz,
            time_now=start_time,
            sim_time=simulation_time,
            ignition_type=ignition_type,
        )
        gridlist = Gridlist(n=nx, m=ny, l=fire_nz)
        sensor1 = Sensor1(
            time_now=start_time, wind_speed=wind_speed, wind_direction=wind_direction
        )
        qu_topoinputs = QU_TopoInputs()
        qu_simparams = QU_Simparams(nx=nx, ny=ny, wind_times=[start_time])

        # Create the SimulationInputs object
        return cls(
            rasterorigin=rasterorigin,
            qu_buildings=qu_buildings,
            qu_fileoptions=qu_fileoptions,
            qfire_advanced_user_inputs=qfire_advanced_user_inputs,
            qfire_bldg_advanced_user_inputs=qfire_bldg_advanced_user_inputs,
            qfire_plume_advanced_user_inputs=qfire_plume_advanced_user_inputs,
            runtime_advanced_user_inputs=runtime_advanced_user_inputs,
            qu_movingcoords=qu_movingcoords,
            qp_buildout=qp_buildout,
            qu_metparams=qu_metparams,
            quic_fire=quic_fire,
            gridlist=gridlist,
            sensor1=sensor1,
            qu_topoinputs=qu_topoinputs,
            qu_simparams=qu_simparams,
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a directory containing a
        QUIC-Fire input file deck.

        Parameters
        ----------
        directory: str | Path
            Directory containing a QUIC-Fire input file deck.

        Returns
        -------
        SimulationInputs
            Class containing the input files in the QUIC-Fire input file deck.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        return cls(
            rasterorigin=RasterOrigin.from_file(directory),
            qu_buildings=QU_Buildings.from_file(directory),
            qu_fileoptions=QU_Fileoptions.from_file(directory),
            qfire_advanced_user_inputs=QFire_Advanced_User_Inputs.from_file(directory),
            qfire_bldg_advanced_user_inputs=QFire_Bldg_Advanced_User_Inputs.from_file(
                directory
            ),
            qfire_plume_advanced_user_inputs=QFire_Plume_Advanced_User_Inputs.from_file(
                directory
            ),
            runtime_advanced_user_inputs=RuntimeAdvancedUserInputs.from_file(directory),
            qu_movingcoords=QU_movingcoords.from_file(directory),
            qp_buildout=QP_buildout.from_file(directory),
            qu_metparams=QU_metparams.from_file(directory),
            quic_fire=QUIC_fire.from_file(directory),
            gridlist=Gridlist.from_file(directory),
            sensor1=Sensor1.from_file(directory),
            qu_topoinputs=QU_TopoInputs.from_file(directory),
            qu_simparams=QU_Simparams.from_file(directory),
        )

    @classmethod
    def from_dict(cls, data: dict) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a dictionary containing
        input file data.

        Parameters
        ----------
        data: dict
            Dictionary containing input file data.
        """
        return cls(
            rasterorigin=RasterOrigin.from_dict(data["rasterorigin"]),
            qu_buildings=QU_Buildings.from_dict(data["qu_buildings"]),
            qu_fileoptions=QU_Fileoptions.from_dict(data["qu_fileoptions"]),
            qfire_advanced_user_inputs=QFire_Advanced_User_Inputs.from_dict(
                data["qfire_advanced_user_inputs"]
            ),
            qfire_bldg_advanced_user_inputs=QFire_Bldg_Advanced_User_Inputs.from_dict(
                data["qfire_bldg_advanced_user_inputs"]
            ),
            qfire_plume_advanced_user_inputs=QFire_Plume_Advanced_User_Inputs.from_dict(
                data["qfire_plume_advanced_user_inputs"]
            ),
            runtime_advanced_user_inputs=RuntimeAdvancedUserInputs.from_dict(
                data["runtime_advanced_user_inputs"]
            ),
            qu_movingcoords=QU_movingcoords.from_dict(data["qu_movingcoords"]),
            qp_buildout=QP_buildout.from_dict(data["qp_buildout"]),
            qu_metparams=QU_metparams.from_dict(data["qu_metparams"]),
            quic_fire=QUIC_fire.from_dict(data["quic_fire"]),
            gridlist=Gridlist.from_dict(data["gridlist"]),
            sensor1=Sensor1.from_dict(data["sensor1"]),
            qu_topoinputs=QU_TopoInputs.from_dict(data["qu_topoinputs"]),
            qu_simparams=QU_Simparams.from_dict(data["qu_simparams"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a JSON file.

        Parameters
        ----------
        path: str | Path
            Path to the JSON file.
        """
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def write_inputs(self, directory: str | Path, version: str = "latest") -> None:
        """
        Write all input files in the SimulationInputs object to a specified
        directory.

        This method is the core method of the SimulationInputs class. It
        is the principle way to translate a SimulationInputs object into a
        QUIC-Fire input file deck.

        Parameters
        ----------
        directory: str | Path
            Directory to write the input files to.
        version: str
            Version of the input files to write. Default is "latest".
        """
        if isinstance(directory, str):
            directory = Path(directory)

        if not directory.exists():
            raise NotADirectoryError(f"{directory} does not exist")

        self._update_shared_attributes()

        # Skip writing gridlist and rasterorigin if fuel_flag == 1
        skip_inputs = []
        if self.quic_fire.fuel_flag == 1:
            skip_inputs.extend(["gridlist", "rasterorigin"])

        # Write each input file to the output directory
        for input_file in self._input_files_dict.values():
            input_file.to_file(directory, version=version)

        # Copy QU_landuse from the template directory to the output directory
        template_file_path = TEMPLATES_PATH / version / "QU_landuse.inp"
        output_file_path = directory / "QU_landuse.inp"
        with open(template_file_path, "rb") as ftemp:
            with open(output_file_path, "wb") as fout:
                fout.write(ftemp.read())

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation. The SimulationInputs
        object is represented as a nest dictionary, with the name of each
        input file as a key to that input file's dictionary representation.

        Returns:
        --------
        dict
            Dictionary representation of the object.
        """
        return {key: value.to_dict() for key, value in self._input_files_dict.items()}

    def to_json(self, path: str | Path):
        """
        Write the object to a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to write the JSON file to.
        """
        if isinstance(path, str):
            path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def set_custom_simulation(
        self,
        fuel: bool = True,
        ignition: bool = True,
        topo: bool = True,
    ):
        if fuel:
            self.quic_fire.fuel_flag = 3
            self.quic_fire.fuel_density = None
            self.quic_fire.fuel_moisture = None
            self.quic_fire.fuel_height = None
        if ignition:
            self.quic_fire.ignition_type = IgnitionType(
                ignition_flag=IgnitionSources(6)
            )
        if topo:
            self.qu_topoinputs.topo_type = TopoType(topo_flag=TopoSources(5))

    def set_uniform_fuels(
        self,
        fuel_density: float,
        fuel_moisture: float,
        fuel_height: float,
    ):
        self.quic_fire.fuel_flag = 1
        self.quic_fire.fuel_density = fuel_density
        self.quic_fire.fuel_moisture = fuel_moisture
        self.quic_fire.fuel_height = fuel_height

    def set_rectangle_ignition(
        self, x_min: int, y_min: int, x_length: int, y_length: int
    ):
        ignition = RectangleIgnition(
            x_min=x_min, y_min=y_min, x_length=x_length, y_length=y_length
        )
        self.quic_fire.ignition_type = ignition

    def set_output_files(
        self,
        eng_to_atm: bool = False,
        react_rate: bool = False,
        fuel_dens: bool = False,
        qf_wind: bool = False,
        qu_wind_inst: bool = False,
        qu_wind_avg: bool = False,
        fuel_moist: bool = False,
        mass_burnt: bool = False,
        emissions: bool = False,
        radiation: bool = False,
        intensity: bool = False,
    ):
        self.quic_fire.eng_to_atm_out = int(eng_to_atm)
        self.quic_fire.react_rate_out = int(react_rate)
        self.quic_fire.fuel_dens_out = int(fuel_dens)
        self.quic_fire.qf_wind_out = int(qf_wind)
        self.quic_fire.qu_wind_inst_out = int(qu_wind_inst)
        self.quic_fire.qu_wind_avg_out = int(qu_wind_avg)
        self.quic_fire.fuel_moist_out = int(fuel_moist)
        self.quic_fire.mass_burnt_out = int(mass_burnt)
        self.quic_fire.radiation_out = int(radiation)
        self.quic_fire.intensity_out = int(intensity)
        self.quic_fire.emissions_out = 2 if emissions else 0

    def _update_shared_attributes(self):
        self.gridlist.n = self.qu_simparams.nx
        self.gridlist.m = self.qu_simparams.ny
        self.gridlist.l = self.quic_fire.nz
        self.gridlist.dx = self.qu_simparams.dx
        self.gridlist.dy = self.qu_simparams.dy
        if (
            not self.sensor1.time_now
            == self.quic_fire.time_now
            == self.qu_simparams.wind_times[0]
        ):
            # TODO: How to handle conflicts
            print(
                f"WARNING: fire start time must be the same for all input files.\n"
                f"Times: \n"
                f"\tQUIC_fire.inp: {self.quic_fire.time_now}\n"
                f"\tQU_simparams.inp: {self.qu_simparams.wind_times[0]}\n"
                f"\tsensor1.inp: {self.sensor1.time_now}\n"
                f"Setting all values to {self.quic_fire.time_now}"
            )
            self.sensor1.time_now = self.quic_fire.time_now
            self.qu_simparams.wind_times[0] = self.quic_fire.time_now


class InputFile(BaseModel, validate_assignment=True):
    """
    Base class representing an input file.

    This base class provides a common interface for all input files in order to
    accomplish two main goals:
    1) Return documentation for each parameter in the input file.
    2) Provide a method to write the input file to a specified directory.
    """

    name: str
    _extension: str
    _param_info: dict = None

    @property
    def _filename(self):
        return f"{self.name}{self._extension}"

    @property
    def param_info(self):
        """
        Return a dictionary of parameter information for the input file.
        """
        if self._param_info is None:  # open the file if it hasn't been read in
            with open(DOCS_PATH / f"{self._filename}.json", "r") as f:
                self._param_info = json.load(f)
        return self._param_info

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

    def to_dict(self, include_private: bool = False):
        """
        Convert the object to a dictionary, excluding attributes that start
        with an underscore.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        all_fields = self.model_dump(
            exclude={"name", "_extension", "_filename", "param_info"}
        )
        if include_private:
            return all_fields
        return {
            key: value for key, value in all_fields.items() if not key.startswith("_")
        }

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

        template_file_path = TEMPLATES_PATH / version / f"{self._filename}"
        with open(template_file_path, "r") as ftemp:
            src = Template(ftemp.read())

        result = src.substitute(self.to_dict(include_private=True))

        output_file_path = directory / self._filename
        with open(output_file_path, "w") as fout:
            fout.write(result)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# TODO: Unify class naming


class Gridlist(InputFile):
    """
    Class representing the gridlist.txt file. This file contains the grid
    information for the QUIC-Fire simulation when canopies are present.

    Attributes
    ----------
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

    name: str = Field("gridlist", frozen=True)
    _extension: str = ""
    n: PositiveInt
    m: PositiveInt
    l: PositiveInt
    dx: PositiveFloat = 2
    dy: PositiveFloat = 2
    dz: PositiveFloat = 1
    aa1: PositiveFloat = 1.0

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a Gridlist object from a directory containing a
        gridlist.txt file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "gridlist", "r") as f:
            lines = f.read()

        return cls(
            n=int(lines.split("n=")[1].split()[0]),
            m=int(lines.split("m=")[1].split()[0]),
            l=int(lines.split("l=")[1].split()[0]),
            dx=float(lines.split("dx=")[1].split()[0]),
            dy=float(lines.split("dy=")[1].split()[0]),
            dz=float(lines.split("dz=")[1].split()[0]),
            aa1=float(lines.split("aa1=")[1].split()[0]),
        )


class RasterOrigin(InputFile):
    """
    Class representing the rasterorigin.txt file. This file contains the
    coordinates of the south-west corner of the domain in UTM coordinates.

    Attributes
    ----------
    utm_x : float
        UTM-x coordinates of the south-west corner of domain [m]
    utm_y : float
        UTM-y coordinates of the south-west corner of domain [m]
    """

    name: str = Field("rasterorigin", frozen=True)
    _extension: str = ".txt"
    utm_x: NonNegativeFloat = 0.0
    utm_y: NonNegativeFloat = 0.0

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
        return cls(utm_x=float(lines[0].split()[0]), utm_y=float(lines[1].split()[0]))


class QU_Buildings(InputFile):
    """
    Class representing the QU_buildings.inp file. This file contains the
    building-related data for the QUIC-Fire simulation. This class is not
    currently used in QUIC-Fire.

    Attributes
    ----------
    wall_roughness_length : float
        Wall roughness length [m]. Must be greater than 0. Default is 0.1.
    number_of_buildings : int
        Number of buildings [-]. Default is 0. Not currently used in QUIC-Fire.
    number_of_polygon_nodes : int
        Number of polygon building nodes [-]. Default is 0. Not currently used
        in QUIC-Fire.
    """

    name: str = Field("QU_buildings", frozen=True)
    _extension: str = ".inp"
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
        return cls(
            wall_roughness_length=float(lines[1].split()[0]),
            number_of_buildings=int(lines[2].split()[0]),
            number_of_polygon_nodes=int(lines[3].split()[0]),
        )


class QU_Fileoptions(InputFile):
    """
    Class representing the QU_fileoptions.inp file. This file contains
    file output-related options for the QUIC-Fire simulation.

    Attributes
    ----------
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

    name: str = Field("QU_fileoptions", frozen=True)
    _extension: str = ".inp"
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
        return cls(
            output_data_file_format_flag=int(lines[1].split()[0]),
            non_mass_conserved_initial_field_flag=int(lines[2].split()[0]),
            initial_sensor_velocity_field_flag=int(lines[3].split()[0]),
            qu_staggered_velocity_file_flag=int(lines[4].split()[0]),
            generate_wind_startup_files_flag=int(lines[5].split()[0]),
        )


class QU_Simparams(InputFile):
    """
    Class representing the QU_simparams.inp file. This file contains the
    simulation parameters for the QUIC-Fire simulation.

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
    quic_domain_height : float
        QUIC domain height [m]. Recommended value: 300 m
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

    name: str = Field("QU_simparams", frozen=True)
    _extension: str = ".inp"
    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt = 22
    dx: PositiveFloat = 2
    dy: PositiveFloat = 2
    quic_domain_height: PositiveFloat = 300
    wind_times: list[int]
    surface_vertical_cell_size: PositiveFloat = 1.0
    number_surface_cells: PositiveInt = 5
    stretch_grid_flag: Literal[0, 1, 3] = 3
    custom_dz_array: list[PositiveFloat] = []
    utc_offset: int = 0
    sor_iter_max: PositiveInt = 10
    sor_residual_reduction: PositiveInt = 3
    use_diffusion_flag: Literal[0, 1] = 0
    number_diffusion_iterations: PositiveInt = 10
    domain_rotation: float = 0.0
    utm_x: float = 0.0
    utm_y: float = 0.0
    utm_zone_number: PositiveInt = 1
    utm_zone_letter: PositiveInt = 1
    quic_cfd_flag: Literal[0, 1] = 0
    explosive_bldg_flag: Literal[0, 1] = 0
    bldg_array_flag: Literal[0, 1] = 0
    _from_file: bool = False
    _from_file_dz_array: list[PositiveFloat] = []

    @computed_field
    @property
    def _dz_array(self) -> list[float]:
        if self._from_file:
            return self._from_file_dz_array
        elif self.stretch_grid_flag == 0:
            return [self.surface_vertical_cell_size] * self.nz
        elif self.stretch_grid_flag == 1:
            return self.custom_dz_array
        elif self.stretch_grid_flag == 3:
            return compute_parabolic_stretched_grid(
                self.surface_vertical_cell_size,
                self.number_surface_cells,
                self.nz,
                self.quic_domain_height,
            ).tolist()

    @computed_field
    @property
    def _vertical_grid_lines(self) -> str:
        """
        Parses the vertical grid stretching flag and dz_array to generate the
        vertical grid as a string for the QU_simparams.inp file.

        Also modifies dz_array if stretch_grid_flag is not 1.
        """
        stretch_grid_func_map = {
            0: self._stretch_grid_flag_0,
            1: self._stretch_grid_flag_1,
            3: self._stretch_grid_flag_3,
        }
        return stretch_grid_func_map[self.stretch_grid_flag]()

    @computed_field
    @property
    def _wind_time_lines(self) -> str:
        return self._generate_wind_time_lines()

    def _stretch_grid_flag_0(self):
        """
        Generates a uniform vertical grid as a string for the QU_simparams.inp
        file. Adds the uniform grid to dz_array.
        """
        # Create the lines for the uniform grid
        surface_dz_line = (
            f"{float(self.surface_vertical_cell_size)}\t" f"! Surface DZ [m]"
        )
        number_surface_cells_line = (
            f"{self.number_surface_cells}\t" f"! Number of uniform surface cells"
        )

        return f"{surface_dz_line}\n{number_surface_cells_line}"

    def _stretch_grid_flag_1(self):
        """
        Generates a custom vertical grid as a string for the QU_simparams.inp
        file.
        """
        # Verify that dz_array is not empty
        if not self._dz_array:
            raise ValueError(
                "dz_array must not be empty if stretch_grid_flag "
                "is 1. Please provide a custom_dz_array with nz "
                "elements or use a different stretch_grid_flag."
            )

        # Verify that nz is equal to the length of dz_array
        if self.nz != len(self._dz_array):
            raise ValueError(
                f"nz must be equal to the length of dz_array. "
                f"{self.nz} != {len(self._dz_array)}"
            )

        # Verify that the first number_surface_cells_line elements of dz_array
        # are equal to the surface_vertical_cell_size
        for dz in self._dz_array[: self.number_surface_cells]:
            if dz != self.surface_vertical_cell_size:
                raise ValueError(
                    "The first number_surface_cells_line "
                    "elements of dz_array must be equal to "
                    "surface_vertical_cell_size"
                )

        # Write surface vertical cell size line
        surface_dz_line = (
            f"{float(self.surface_vertical_cell_size)}\t! " f"Surface DZ [m]"
        )

        # Write header line
        header_line = "! DZ array [m]"

        # Write dz_array lines
        dz_array_lines_list = []
        for dz in self._dz_array:
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
        surface_dz_line = (
            f"{float(self.surface_vertical_cell_size)}\t! " f"Surface DZ [m]"
        )

        # Write number of surface cells line
        number_surface_cells_line = (
            f"{self.number_surface_cells}\t! " f"Number of uniform surface cells"
        )

        # Write header line
        header_line = "! DZ array [m]"

        # Write dz_array lines
        dz_lines = "\n".join([f"{float(dz)}" for dz in self._dz_array])

        return (
            f"{surface_dz_line}\n{number_surface_cells_line}\n{header_line}"
            f"\n{dz_lines}"
        )

    def _generate_wind_time_lines(self):
        """
        Parses the utc_offset and wind_step_times to generate the wind times
        as a string for the QU_simparams.inp file.
        """
        # Verify that wind_step_times is not empty
        if not self.wind_times:
            raise ValueError(
                "wind_step_times must not be empty. Please "
                "provide a wind_step_times with num_wind_steps "
                "elements or use a different num_wind_steps."
            )

        # Write number of time increments line
        number_time_increments_line = (
            f"{len(self.wind_times)}\t" f"! Number of time increments"
        )

        # Write utc_offset line
        utc_offset_line = f"{self.utc_offset}\t! UTC offset [hours]"

        # Write header line
        header_line = "! Wind step times [s]"

        # Write wind_step_times lines
        wind_step_times_lines_list = []
        for wind_time in self.wind_times:
            wind_step_times_lines_list.append(f"{wind_time}")
        wind_step_times_lines = "\n".join(wind_step_times_lines_list)

        return "\n".join(
            [
                number_time_increments_line,
                utc_offset_line,
                header_line,
                wind_step_times_lines,
            ]
        )

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
            quic_domain_height = surface_vertical_cell_size * number_surface_cells
            current_line = 9
        elif stretch_grid_flag == 1:
            surface_vertical_cell_size = float(lines[7].strip().split("!")[0])
            number_surface_cells = 5
            for i in range(9, 9 + nz):
                custom_dz_array.append(float(lines[i].strip().split("!")[0]))
            quic_domain_height = round(sum(custom_dz_array), 2)
            current_line = 9 + nz
        elif stretch_grid_flag == 3:
            surface_vertical_cell_size = float(lines[7].strip().split("!")[0])
            number_surface_cells = int(lines[8].strip().split("!")[0])
            _ = lines[9].strip().split("!")[0]
            for i in range(10, 10 + nz):
                _from_file_dz_array.append(float(lines[i].strip().split("!")[0]))
            quic_domain_height = round(sum(_from_file_dz_array), 2)
            current_line = 10 + nz
        else:
            raise ValueError("stretch_grid_flag must be 0, 1, or 3.")

        # Read QU wind parameters
        number_wind_steps = int(lines[current_line].strip().split("!")[0])
        utc_offset = int(lines[current_line + 1].strip().split("!")[0])
        _ = lines[current_line + 2].strip().split("!")[0]
        wind_times = []
        for i in range(current_line + 3, current_line + 3 + number_wind_steps):
            wind_times.append(int(lines[i].strip()))
        current_line = current_line + 3 + number_wind_steps

        # Skip not used parameters
        current_line += 9

        # Read remaining QU parameters
        sor_iter_max = int(lines[current_line].strip().split("!")[0])
        sor_residual_reduction = int(lines[current_line + 1].strip().split("!")[0])
        use_diffusion_flag = int(lines[current_line + 2].strip().split("!")[0])
        number_diffusion_iterations = int(lines[current_line + 3].strip().split("!")[0])
        domain_rotation = float(lines[current_line + 4].strip().split("!")[0])
        utm_x = float(lines[current_line + 5].strip().split("!")[0])
        utm_y = float(lines[current_line + 6].strip().split("!")[0])
        utm_zone_number = int(lines[current_line + 7].strip().split("!")[0])
        utm_zone_letter = int(lines[current_line + 8].strip().split("!")[0])
        quic_cfd_flag = int(lines[current_line + 9].strip().split("!")[0])
        explosive_bldg_flag = int(lines[current_line + 10].strip().split("!")[0])
        bldg_array_flag = int(lines[current_line + 11].strip().split("!")[0])

        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            quic_domain_height=quic_domain_height,
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
            _from_file_dz_array=_from_file_dz_array,
        )


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

    name: str = Field("QFire_Advanced_User_Inputs", frozen=True)
    _extension: str = ".inp"
    fraction_cells_launch_firebrands: PositiveFloat = Field(0.05, ge=0, lt=1)
    firebrand_radius_scale_factor: PositiveFloat = Field(40.0, ge=1)
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
        return cls(
            fraction_cells_launch_firebrands=float(lines[0].split()[0]),
            firebrand_radius_scale_factor=float(lines[1].split()[0]),
            firebrand_trajectory_time_step=int(lines[2].split()[0]),
            firebrand_launch_interval=int(lines[3].split()[0]),
            firebrands_per_deposition=int(lines[4].split()[0]),
            firebrand_area_ratio=float(lines[5].split()[0]),
            minimum_burn_rate_coefficient=float(lines[6].split()[0]),
            max_firebrand_thickness_fraction=float(lines[7].split()[0]),
            firebrand_germination_delay=int(lines[8].split()[0]),
            vertical_velocity_scale_factor=float(lines[9].split()[0]),
            minimum_firebrand_ignitions=int(lines[10].split()[0]),
            maximum_firebrand_ignitions=int(lines[11].split()[0]),
            minimum_landing_angle=float(lines[12].split()[0]),
            maximum_firebrand_thickness=float(lines[13].split()[0]),
        )


class QUIC_fire(InputFile):
    """
    Class representing the QUIC_fire.inp input file. This file
    contains the parameters relating to the fire simulation and
    outputs.

    Parameters
    ----------
    fire_flag : Literal[0, 1]
        Fire flag, 1 = run fire; 0 = no fire
    random_seed : int
        Random number generator, -1: use time and date, any other integer > 0
    time_now : PositiveInt
        When the fire is ignited in Unix Epoch time (integer seconds since
        1970/1/1 00:00:00). Must be greater or equal to the time of the first
        wind
    sim_time : PositiveInt
        Total simulation time for the fire [s]
    fire_time_step : PositiveInt
        Time step for the fire simulation [s]
    quic_time_step : PositiveInt
        Number of fire time steps done before updating the quic wind field
        (integer, >= 1)
    out_time_fire : PositiveInt
        After how many fire time steps to print out fire-related files (excluding emissions and radiation)
    out_time_wind : PositiveInt
        After how many quic updates to print out wind-related files
    out_time_emis_rad : PositiveInt
        After how many fire time steps to average emissions and radiation
    out_time_wind_avg : PositiveInt
        After how many quic updates to print out averaged wind-related files
    nz : PositiveInt
        Number of fire grid cells in the z-direction.
    stretch_grid_flag : Literal[0, 1]
        Vertical stretching flag: 0 = uniform dz, 1 = custom
    dz : PositiveInt
        Cell size in the z-direction [m] of the fire grid. Recommended value: 1m
    dz_array : List[PositiveFloat]
        Custom dz, one dz per line must be specified, from the ground to the
        top of the domain
    fuel_flag : Literal[1, 2, 3, 4]
        Flag for fuel data:
            - density
            - moisture
            - height
        1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files
        for quic grid, 4 = Firetech files for different grid
        (need interpolation)
    fuel_density : PositiveFloat
        Fuel density (kg/m3)
    fuel_moisture : PositiveFloat
        Fuel moisture = mass of water/mass of dry fuel
    fuel_height : PositiveFloat
        Fuel height of surface layer (m)
    ignition_type: IgnitionType
        Ignition type specified as an IgnitionsType class from ignitions.py
        1 = rectangle
        2 = square ring
        3 = circular ring
        6 = ignite.dat (Firetec file)
    ignitions_per_cell : PositiveInt
        Number of ignition per cell of the fire model. Recommended max value
        of 100
    firebrand_flag : Literal[0, 1]
        Firebrand flag, 0 = off; 1 = on. Recommended value = 0; firebrands
        are untested for small scale problems
    auto_kill : Literal[0, 1]
        Kill if the fire is out and there are no more ignitions or firebrands
        (0 = no, 1 = yes)
    eng_to_atm_out : Literal[0, 1]
        Output flag [0, 1]: gridded energy-to-atmosphere
        (3D fire grid + extra layers)
    react_rate_out : Literal[0, 1]
        Output flag [0, 1]: compressed array reaction rate (fire grid)
    fuel_dens_out : Literal[0, 1]
        Output flag [0, 1]: compressed array fuel density (fire grid)
    qf_wind_out : Literal[0, 1]
        Output flag [0, 1]: gridded wind (u,v,w,sigma) (3D fire grid)
    qu_wind_inst_out : Literal[0, 1]
        Output flag [0, 1]: gridded QU winds with fire effects, instantaneous
        (QUIC-URB grid)
    qu_wind_avg_out : Literal[0, 1]
        Output flag [0, 1]: gridded QU winds with fire effects, averaged
        (QUIC-URB grid)
    fuel_moist_out : Literal[0, 1]
        Output flag [0, 1]: compressed array fuel moisture (fire grid)
    mass_burnt_out : Literal[0, 1]
        Output flag [0, 1]: vertically-integrated % mass burnt (fire grid)
    firebrand_out : Literal[0, 1]
        Output flag [0, 1]: firebrand trajectories. Must be 0 when firebrand
        flag is 0
    emissions_out : Literal[0, 1, 2, 3, 4, 5]
        Output flag [0, 5]: compressed array emissions (fire grid):
            0 = do not output any emission related variables
            1 = output emissions files and simulate CO in QUIC-SMOKE
            2 = output emissions files and simulate PM2.5 in QUIC- SMOKE
            3 = output emissions files and simulate both CO and PM2.5 in
                QUIC-SMOKE
            4 = output emissions files but use library approach in QUIC-SMOKE
            5 = output emissions files and simulate both water in QUIC-SMOKE
    radiation_out : Literal[0, 1]
        Output flag [0, 1]: gridded thermal radiation (fire grid)
    intensity_out : Literal[0, 1]
        Output flag [0, 1]: surface fire intensity at every fire time step
    """

    name: str = "QUIC_fire"
    _extension: str = ".inp"
    fire_flag: Literal[0, 1] = 1
    random_seed: int = Field(ge=-1, default=-1)
    time_now: PositiveInt
    sim_time: PositiveInt
    fire_time_step: PositiveInt = 1
    quic_time_step: PositiveInt = 1
    out_time_fire: PositiveInt = 30
    out_time_wind: PositiveInt = 30
    out_time_emis_rad: PositiveInt = 30
    out_time_wind_avg: PositiveInt = 30
    nz: PositiveInt
    stretch_grid_flag: Literal[0, 1] = 0
    dz: PositiveInt = 1
    dz_array: list[PositiveFloat] = []
    fuel_flag: Literal[1, 2, 3, 4] = 1
    fuel_density: Union[PositiveFloat, None] = 0.5
    fuel_moisture: Union[PositiveFloat, None] = 0.1
    fuel_height: Union[PositiveFloat, None] = 1.0
    ignition_type: Union[
        RectangleIgnition, SquareRingIgnition, CircularRingIgnition, IgnitionType
    ]
    ignitions_per_cell: PositiveInt = 2
    firebrand_flag: Literal[0, 1] = 0
    auto_kill: Literal[0, 1] = 1
    eng_to_atm_out: Literal[0, 1] = 0
    react_rate_out: Literal[0, 1] = 0
    fuel_dens_out: Literal[0, 1] = 1
    qf_wind_out: Literal[0, 1] = 1
    qu_wind_inst_out: Literal[0, 1] = 0
    qu_wind_avg_out: Literal[0, 1] = 0
    fuel_moist_out: Literal[0, 1] = 0
    mass_burnt_out: Literal[0, 1] = 0
    firebrand_out: Literal[0, 1] = 0
    emissions_out: Literal[0, 1, 2, 3, 4, 5] = 0
    radiation_out: Literal[0, 1] = 0
    intensity_out: Literal[0, 1] = 0

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, v: int) -> int:
        if v == 0:
            raise ValueError("QUIC_fire.inp: random_seed must be not be 0")
        return v

    @computed_field
    @property
    def _stretch_grid_input(self) -> str:
        """
        Writes a custom stretch grid to QUIC_fire.inp, if provided.
        """
        if self.stretch_grid_flag == 1:
            # Verify that dz_array is not empty
            if not self.dz_array:
                raise ValueError(
                    "dz_array must not be empty if stretch_grid_flag "
                    "is 1. Please provide a dz_array with nz elements"
                    " or use a different stretch_grid_flag."
                )

            # Verify that nz is equal to the length of dz_array
            if self.nz != len(self.dz_array):
                raise ValueError(
                    f"nz must be equal to the length of dz_array. "
                    f"{self.nz} != {len(self.dz_array)}"
                )

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
    def _ignition_lines(self) -> str:
        return str(self.ignition_type)

    @computed_field
    @property
    def _fuel_lines(self) -> str:
        flag_line = (
            " 1 = uniform; "
            "2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
            " files for quic grid, 4 = Firetech files for "
            "different grid (need interpolation)"
        )
        fuel_density_flag_line = f"{self.fuel_flag}\t! fuel density flag: " + flag_line
        fuel_moist_flag_line = f"\n{self.fuel_flag}\t! fuel moisture flag: " + flag_line
        fuel_height_flag_line = f"\n{self.fuel_flag}\t! fuel height flag: " + flag_line
        if self.fuel_flag == 1:
            try:
                assert self.fuel_density is not None
                assert self.fuel_moisture is not None
                assert self.fuel_height is not None
            except AssertionError:
                raise ValueError(
                    "fuel_params: FuelInputs class must have values for fuel_density, fuel_moisture, and fuel_height"
                )
            fuel_dens_line = f"\n{self.fuel_density}"
            fuel_moist_line = f"\n{self.fuel_moisture}"
            fuel_height_line = f"\n{self.fuel_height}"
            return (
                fuel_density_flag_line
                + fuel_dens_line
                + fuel_moist_flag_line
                + fuel_moist_line
                + fuel_height_flag_line
                + fuel_height_line
            )
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

        # Read fire grid parameters
        nz = int(lines[12].strip().split("!")[0])
        stretch_grid_flag = int(lines[13].strip().split("!")[0])
        dz_array = []
        if stretch_grid_flag == 0:
            dz = int(lines[14].strip().split("!")[0])
            current_line = 15
        else:
            for i in range(14, 14 + len(dz_array)):
                try:
                    float(lines[i].strip())
                except ValueError:
                    print(
                        "QUIC_fire.inp: dz input value is not a float. Does the number of dz data match nz?"
                    )
                dz_array.append(float(lines[i].strip()))
            current_line = 15 + len(dz_array)

        current_line += 4  # skip unused lines

        # Read fuel data
        # current_line = ! FUEL
        current_line += 1  # header
        fuel_flag = int(lines[current_line].strip().split("!")[0])
        if fuel_flag == 1:
            fuel_density = float(lines[current_line + 1].strip())
            moisture_flag = int(lines[current_line + 2].strip().split("!")[0])
            fuel_moisture = float(lines[current_line + 3].strip())
            height_flag = int(lines[current_line + 4].strip().split("!")[0])
            fuel_height = float(lines[current_line + 5].strip())
            if moisture_flag != fuel_flag or height_flag != fuel_flag:
                raise ValueError(
                    "QUIC_fire.inp: Fuel moisture and fue height flags must match fuel density flag"
                )
            current_line += 6
        else:
            fuel_density = None
            fuel_moisture = None
            fuel_height = None
            current_line += 2

            # Read ignition data
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
            ignition_type = RectangleIgnition(
                x_min=x_min, y_min=y_min, x_length=x_length, y_length=y_length
            )
        elif ignition_flag == 2:
            x_min, y_min, x_length, y_length, x_width, y_width = ignition_params
            ignition_type = SquareRingIgnition(
                x_min=x_min,
                y_min=y_min,
                x_length=x_length,
                y_length=y_length,
                x_width=x_width,
                y_width=y_width,
            )
        elif ignition_flag == 3:
            x_min, y_min, x_length, y_length, ring_width = ignition_params
            ignition_type = CircularRingIgnition(
                x_min=x_min,
                y_min=y_min,
                x_length=x_length,
                y_length=y_length,
                ring_width=ring_width,
            )
        elif ignition_flag == 6:
            ignition_type = IgnitionType(ignition_flag=6)
        else:
            ignition_type = IgnitionType(ignition_flag=ignition_flag)

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
        qf_wind_out = int(lines[current_line + 4].strip().split("!")[0])
        qu_wind_inst_out = int(lines[current_line + 5].strip().split("!")[0])
        qu_wind_avg_out = int(lines[current_line + 6].strip().split("!")[0])
        # ! Output plume trajectories
        fuel_moist_out = int(lines[current_line + 8].strip().split("!")[0])
        mass_burnt_out = int(lines[current_line + 9].strip().split("!")[0])
        firebrand_out = int(lines[current_line + 10].strip().split("!")[0])
        emissions_out = int(lines[current_line + 11].strip().split("!")[0])
        radiation_out = int(lines[current_line + 12].strip().split("!")[0])
        intensity_out = int(lines[current_line + 13].strip().split("!")[0])
        # ! AUTOKILL
        auto_kill = int(lines[current_line + 15].strip().split("!")[0])

        return cls(
            fire_flag=fire_flag,
            random_seed=random_seed,
            time_now=time_now,
            sim_time=sim_time,
            fire_time_step=fire_time_step,
            quic_time_step=quic_time_step,
            out_time_fire=out_time_fire,
            out_time_wind=out_time_wind,
            out_time_emis_rad=out_time_emis_rad,
            out_time_wind_avg=out_time_wind_avg,
            nz=nz,
            stretch_grid_flag=stretch_grid_flag,
            dz=dz,
            dz_array=dz_array,
            fuel_flag=fuel_flag,
            fuel_density=fuel_density,
            fuel_moisture=fuel_moisture,
            fuel_height=fuel_height,
            ignition_type=ignition_type,
            ignitions_per_cell=ignitions_per_cell,
            firebrand_flag=firebrand_flag,
            eng_to_atm_out=eng_to_atm_out,
            react_rate_out=react_rate_out,
            fuel_dens_out=fuel_dens_out,
            qf_wind_out=qf_wind_out,
            qu_wind_inst_out=qu_wind_inst_out,
            qu_wind_avg_out=qu_wind_avg_out,
            fuel_moist_out=fuel_moist_out,
            mass_burnt_out=mass_burnt_out,
            firebrand_out=firebrand_out,
            emissions_out=emissions_out,
            radiation_out=radiation_out,
            intensity_out=intensity_out,
            auto_kill=auto_kill,
        )


class QFire_Bldg_Advanced_User_Inputs(InputFile):
    """
    Class representing the QFire_Bldg_Advanced_User_Inputs.inp input file. This
    file contains advanced parameters related to buildings and fuel.

    Attributes
    ----------
    convert_buildings_to_fuel_flag : int
        Flag to convert QUIC-URB buildings to fuel. 0 = do not convert,
        1 = convert. Recommended value: 0.
    building_fuel_density : PositiveFloat
        Thin fuel density within buildings if no fuel is specified and buildings
        are converted to fuel. Higher value = more fuel. Recommended value: 0.5.
        Units: [kg/m^3]
    building_attenuation_coefficient : PositiveFloat
        Attenuation coefficient within buildings if buildings are converted to
        fuel. Higher value = more drag. Recommended value: 2.
    building_surface_roughness : PositiveFloat
        Surface roughness within buildings if buildings are converted to fuel.
        Higher value = lower wind speed. Recommended value: 0.01 m. Units: [m]
    convert_fuel_to_canopy_flag : int
        Flag to convert fuel to canopy for winds. 0 = do not convert,
        1 = convert. Recommended value: 1.
    update_canopy_winds_flag : int
        Flag to update canopy winds when fuel is consumed. 0 = do not update,
        1 = update. Recommended value: 1.
    fuel_attenuation_coefficient : PositiveFloat
        Attenuation coefficient within fuel for the wind profile. Higher
        value = more drag. Recommended value: 1.
    fuel_surface_roughness : PositiveFloat
        Surface roughness within fuel. Higher value = lower wind speed.
        Recommended value: 0.1 m. Units: [m]
    """

    name: str = Field("QFire_Bldg_Advanced_User_Inputs", frozen=True)
    _extension: str = ".inp"
    convert_buildings_to_fuel_flag: Literal[0, 1] = 0
    building_fuel_density: PositiveFloat = Field(0.5, ge=0)
    building_attenuation_coefficient: PositiveFloat = Field(2.0, ge=0)
    building_surface_roughness: PositiveFloat = Field(0.01, ge=0)
    convert_fuel_to_canopy_flag: Literal[0, 1] = 1
    update_canopy_winds_flag: Literal[0, 1] = 1
    fuel_attenuation_coefficient: PositiveFloat = Field(1.0, ge=0)
    fuel_surface_roughness: PositiveFloat = Field(0.1, ge=0)

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a QFire_Bldg_Advanced_User_Inputs object from a directory
        containing a QFire_Bldg_Advanced_User_Inputs.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QFire_Bldg_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()
        return cls(
            convert_buildings_to_fuel_flag=int(lines[0].split()[0]),
            building_fuel_density=float(lines[1].split()[0]),
            building_attenuation_coefficient=float(lines[2].split()[0]),
            building_surface_roughness=float(lines[3].split()[0]),
            convert_fuel_to_canopy_flag=int(lines[4].split()[0]),
            update_canopy_winds_flag=int(lines[5].split()[0]),
            fuel_attenuation_coefficient=float(lines[6].split()[0]),
            fuel_surface_roughness=float(lines[7].split()[0]),
        )


class QFire_Plume_Advanced_User_Inputs(InputFile):
    """
    Class representing the QFire_Plume_Advanced_User_Inputs.inp input file.
    This file contains advanced parameters related to modeling buoyant plumes.

    Attributes
    ----------
    max_plumes_per_timestep : PositiveInt
        Maximum number of plumes allowed at each time step. Higher values slow
        down the simulation. Default value: 150,000. Recommended range:
        50,000 - 500,000.
    min_plume_updraft_velocity : PositiveFloat
        Minimum plume updraft velocity [m/s]. If plume velocity drops below this
        value, the plume is removed. Higher values reduce number of plumes.
        Default value: 0.1 m/s.
    max_plume_updraft_velocity : PositiveFloat
        Maximum allowed plume updraft velocity [m/s]. Default value: 100 m/s.
    min_velocity_ratio : PositiveFloat
        Minimum ratio between plume updraft velocity and wind speed. If ratio
        drops below this value, plume is removed. Higher values reduce plumes.
        Default value: 0.1.
    brunt_vaisala_freq_squared : NonNegativeFloat
        Inverse of the Brunt-Vaisala frequency squared [1/s^2], a measure of
        atmospheric stability. Default value: 0 1/s^2.
    creeping_flag : Literal[0, 1]
        Flag to enable (1) or disable (0) fire spread by creeping.
        Default value: 1.
    adaptive_timestep_flag : Literal[0, 1]
        Enable (1) or disable (0) adaptive time stepping. Adaptive time stepping
        improves accuracy but increases simulation time. Default value: 0.
    plume_timestep : PositiveFloat
        Time step [s] used to compute buoyant plume trajectories. Higher values
        reduce accuracy. Default value: 1s.
    sor_option_flag : Literal[0, 1]
        SOR solver option. 0 = standard SOR, 1 = memory SOR. Default value: 1.
    sor_alpha_plume_center : PositiveFloat
        SOR alpha value at plume centerline. Higher values reduce influence of
        plumes on winds. Default value: 10.
    sor_alpha_plume_edge : PositiveFloat
        SOR alpha value at plume edge. Higher values reduce influence of plumes
        on winds. Default value: 1.
        max_plume_merging_angle : PositiveFloat
        Maximum angle [degrees] between plumes to determine merging eligibility.
        Higher values increase plume merging. Default value: 30 degrees.
    max_plume_overlap_fraction : PositiveFloat
        Maximum fraction of smaller plume trajectory overlapped by larger plume
        to be considered for merging. Higher values increase merging.
    plume_to_grid_updrafts_flag : Literal[0, 1]
        Method to map plume updrafts to grid. 0 = new method, 1 = old method.
        New method improves accuracy. Default value: 1. New method takes longer,
        but is needed if smoke is simulated afterwards.
    max_points_along_plume_edge : PositiveInt
        Maximum points to sample along grid cell edge for new plume-to-grid
        method. Default value: 10.
    plume_to_grid_intersection_flag : Literal[0, 1]
        Scheme to sum plume-to-grid updrafts when multiple plumes intersect a
        grid cell. 0 = cube method, 1 = max value method. Default value: 1.
    """

    name: str = Field("QFire_Plume_Advanced_User_Inputs", frozen=True)
    _extension: str = ".inp"
    max_plumes_per_timestep: PositiveInt = Field(150000, gt=0)
    min_plume_updraft_velocity: PositiveFloat = Field(0.1, gt=0)
    max_plume_updraft_velocity: PositiveFloat = Field(100.0, gt=0)
    min_velocity_ratio: PositiveFloat = Field(0.1, gt=0)
    brunt_vaisala_freq_squared: NonNegativeFloat = Field(0.0, ge=0)
    creeping_flag: Literal[0, 1] = 1
    adaptive_timestep_flag: Literal[0, 1] = 0
    plume_timestep: PositiveFloat = Field(1.0, gt=0)
    sor_option_flag: Literal[0, 1] = 1
    sor_alpha_plume_center: PositiveFloat = Field(10.0, gt=0)
    sor_alpha_plume_edge: PositiveFloat = Field(1.0, gt=0)
    max_plume_merging_angle: PositiveFloat = Field(30.0, gt=0, le=180)
    max_plume_overlap_fraction: PositiveFloat = Field(0.7, gt=0, le=1)
    plume_to_grid_updrafts_flag: Literal[0, 1] = 1
    max_points_along_plume_edge: PositiveInt = Field(10, ge=1, le=100)
    plume_to_grid_intersection_flag: Literal[0, 1] = 1

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QFire_Plume_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()

        return cls(
            max_plumes_per_timestep=int(lines[0].split()[0]),
            min_plume_updraft_velocity=float(lines[1].split()[0]),
            max_plume_updraft_velocity=float(lines[2].split()[0]),
            min_velocity_ratio=float(lines[3].split()[0]),
            brunt_vaisala_freq_squared=float(lines[4].split()[0]),
            creeping_flag=int(lines[5].split()[0]),
            adaptive_timestep_flag=int(lines[6].split()[0]),
            plume_timestep=float(lines[7].split()[0]),
            sor_option_flag=int(lines[8].split()[0]),
            sor_alpha_plume_center=float(lines[9].split()[0]),
            sor_alpha_plume_edge=float(lines[10].split()[0]),
            max_plume_merging_angle=float(lines[11].split()[0]),
            max_plume_overlap_fraction=float(lines[12].split()[0]),
            plume_to_grid_updrafts_flag=int(lines[13].split()[0]),
            max_points_along_plume_edge=int(lines[14].split()[0]),
            plume_to_grid_intersection_flag=int(lines[15].split()[0]),
        )


class QU_TopoInputs(InputFile):
    """
    Class representing the QU_TopoInputs.inp input file. This file
    contains advanced data pertaining to topography.

    filename : str
        Path to the custom topo file (only used with option 5). Cannot be .bin. Use .dat or .inp
    topo_type : TopoType
        Topography type specified as a TopoType class from topography.py
        0 = no terrain file provided, QUIC-Fire is run with flat terrain
        1 = Gaussian hill
        2 = hill pass
        3 = slope mesa
        4 = canyon
        5 = custom
        6 = half circle
        7 = sinusoid
        8 = cos hill
        9 = terrain is provided via QP_elevation.bin (see Section 2.7)
        10 = terrain is provided via terrainOutput.txt
        11 = terrain.dat (firetec)
    smoothing_method : int
        0 = none (default for idealized topo)
        1 = Blur
        2 = David Robinson’s method based on second derivative
    smoothing_passes : int
        Number of smoothing passes. Real terrain MUST be smoothed
    sor_iterations : int
        Number of SOR iteration to define background winds before starting the fire
    sor_cycles : int
        Number of times the SOR solver initial fields is reset to define
        background winds before starting the fire
    sor_relax : float
        SOR overrelaxation coefficient. Only used if there is topo.
    """

    name: str = "QU_TopoInputs"
    _extension: str = ".inp"
    filename: str = "topo.dat"
    topo_type: SerializeAsAny[TopoType] = TopoType(topo_flag=TopoSources(0))
    smoothing_method: Literal[0, 1, 2] = 2
    smoothing_passes: PositiveInt = Field(le=500, default=500)
    sor_iterations: PositiveInt = Field(le=500, default=200)
    sor_cycles: Literal[0, 1, 2, 3, 4] = 4
    sor_relax: PositiveFloat = Field(le=2, default=0.9)

    @computed_field
    @property
    def _topo_lines(self) -> str:
        return str(self.topo_type)

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_TopoInputs.inp", "r") as f:
            lines = f.readlines()

        # Line 0 is Header
        filename = str(lines[1].strip())
        # Get topo lines
        topo_flag = int(lines[2].strip().split("!")[0])
        add_dict = {
            0: 0,
            1: 4,
            2: 2,
            3: 3,
            4: 5,
            5: 0,
            6: 3,
            7: 2,
            8: 2,
            9: 0,
            10: 0,
            11: 0,
        }
        add = add_dict.get(topo_flag)
        topo_params = []
        for i in range(3, 3 + add):
            topo_params.append(float(lines[i].strip().split("!")[0]))
        if topo_flag == 1:
            x_hilltop, y_hilltop, elevation_max, elevation_std = topo_params
            topo_type = GaussianHillTopo(
                x_hilltop=int(x_hilltop),
                y_hilltop=int(y_hilltop),
                elevation_max=int(elevation_max),
                elevation_std=elevation_std,
            )
        elif topo_flag == 2:
            max_height, location_param = topo_params
            topo_type = HillPassTopo(
                max_height=int(max_height), location_param=location_param
            )
        elif topo_flag == 3:
            slope_axis, slope_value, flat_fraction = topo_params
            topo_type = SlopeMesaTopo(
                slope_axis=int(slope_axis),
                slope_value=slope_value,
                flat_fraction=flat_fraction,
            )
        elif topo_flag == 4:
            x_start, y_center, slope_value, canyon_std, vertical_offset = topo_params
            topo_type = CanyonTopo(
                x_start=int(x_start),
                y_center=int(y_center),
                sloe_value=slope_value,
                canyon_std=canyon_std,
                vertical_offset=vertical_offset,
            )
        elif topo_flag == 6:
            x_location, y_location, radius = topo_params
            topo_type = HalfCircleTopo(
                x_location=int(x_location), y_location=int(y_location), radius=radius
            )
        elif topo_flag == 7:
            period, amplitude = topo_params
            topo_type = SinusoidTopo(period=period, amplitude=amplitude)
        elif topo_flag == 8:
            aspect, height = topo_params
            topo_type = CosHillTopo(aspect=aspect, height=height)
        else:
            topo_type = TopoType(topo_flag=topo_flag)
        current_line = 3 + add
        # Smoothing and SOR
        smoothing_method = int(lines[current_line].strip().split("!")[0])
        smoothing_passes = int(lines[current_line + 1].strip().split("!")[0])
        sor_iterations = int(lines[current_line + 2].strip().split("!")[0])
        sor_cycles = int(lines[current_line + 3].strip().split("!")[0])
        sor_relax = float(lines[current_line + 4].strip().split("!")[0])

        return cls(
            filename=filename,
            topo_type=topo_type,
            smoothing_method=smoothing_method,
            smoothing_passes=smoothing_passes,
            sor_iterations=sor_iterations,
            sor_cycles=sor_cycles,
            sor_relax=sor_relax,
        )


class RuntimeAdvancedUserInputs(InputFile):
    """
    Class representing the Runtime_Advanced_User_Inputs.inp input file.
    This file contains advanced parameters related to computer memory usage.

    Attributes
    ----------
    num_cpus : PositiveInt
        Maximum number of CPU to use. Do not exceed 8. Use 1 for ensemble
        simulations.
    use_acw : Literal[0,1]
        Use Adaptive Computation Window (0=Disabled 1=Enabled)
    """

    name: str = "Runtime_Advanced_User_Inputs"
    _extension: str = ".inp"
    num_cpus: PositiveInt = Field(le=8, default=8)
    use_acw: Literal[0, 1] = 0

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "Runtime_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()

        return cls(
            num_cpus=int(lines[0].strip().split("!")[0]),
            use_acw=int(lines[1].strip().split("!")[0]),
        )


class QU_movingcoords(InputFile):
    """
    Class representing the QU_movingcoords.inp input file.
    This is a QUIC legacy file that is not modified for QUIC-Fire use.
    """

    name: str = "QU_movingcoords"
    _extension: str = ".inp"

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QU_movingcoords.inp", "r") as f:
            lines = f.readlines()

        if int(lines[1].strip().split("!")[0]) == 1:
            print(
                "WARNING: QU_movingcoords.inp: Moving coordinates flag == 1 not supported."
            )

        return cls()


class QP_buildout(InputFile):
    """
    Class representing the QU_buildout.inp input file.
    This is a QUIC legacy file that is not modified for QUIC-Fire use.
    """

    name: str = "QP_buildout"
    _extension: str = ".inp"

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QP_buildout.inp", "r") as f:
            lines = f.readlines()

        if int(lines[0].strip().split("!")[0]) == 1:
            print("WARNING: QP_buildout.inp: number of buildings will be set to 0.")
        if int(lines[1].strip().split("!")[0]) == 1:
            print(
                "WARNING: QP_buildout.inp: number of vegetative canopies will be set to 0."
            )

        return cls()


class QU_metparams(InputFile):
    """
    Class representing the QU_metparams.inp input file.
    This file contains information about wind profiles

    Attributes
    ----------
    num_sensors : int
        Number of measuring sites. Multiple wind profiles are not yet supported.
    sensor_name : str
        Name of the wind profile. This will correspond to the filename of the wind profile, e.g. sensor1.inp
    """

    name: str = "QU_metparams"
    _extension: str = ".inp"
    num_sensors: PositiveInt = 1
    sensor_name: str = "sensor1"

    @computed_field
    @property
    def _sensor_lines(self) -> str:
        return (
            f"{self.sensor_name} !Site Name\n" f"!File name\n" f"{self.sensor_name}.inp"
        )

    @classmethod
    def from_file(cls, directory):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_metparams.inp", "r") as f:
            lines = f.readlines()
        return cls(
            num_sensors=int(lines[2].strip().split()[0]),
            sensor_name=str(lines[4].strip().split()[0].strip()),
        )


class Sensor1(InputFile):
    """
    Class representing the sensor1.inp input file.
    This file contains information on winds, and serves as the
    primary source for wind speed(s) and direction(s)

    Attributes
    ----------
    time_now : PositiveInt
        Begining of time step in Unix Epoch time (integer seconds since
        1970/1/1 00:00:00). Must match time at beginning of fire
        (QU_Simparams.inp and QUIC_fire.inp)
    sensor_height : PositiveFloat
        Wind measurement height (m). Default is 6.1m (20ft)
    wind_speed : PositiveFloat
        Wind speed (m/s)
    wind_direction : NonNegativeInt < 360
        Wind direction (degrees). Use 0° for North
    """

    name: str = "sensor1"
    _extension: str = ".inp"
    time_now: PositiveInt
    sensor_height: PositiveFloat = 6.1
    wind_speed: PositiveFloat
    wind_direction: NonNegativeInt = Field(lt=360)

    @computed_field
    @property
    def _wind_lines(self) -> str:
        """
        This is meant to support wind shifts in the future.
        This computed field could be altered to reproduce the lines below
        for a series of times, speeds, and directions.
        """
        return (
            f"{self.time_now} !Begining of time step in Unix Epoch time\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"{self.sensor_height} {self.wind_speed} {self.wind_direction}"
        )

    @classmethod
    def from_file(cls, directory: str | Path):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "sensor1.inp", "r") as f:
            lines = f.readlines()
        print("\n".join(lines))
        return cls(
            time_now=int(lines[6].strip().split("!")[0]),
            sensor_height=float(lines[11].split(" ")[0]),
            wind_speed=float(lines[11].split(" ")[1]),
            wind_direction=int(lines[11].split(" ")[2]),
        )
