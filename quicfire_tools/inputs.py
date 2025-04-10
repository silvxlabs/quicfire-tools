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
from typing import Literal, Union, List, Optional

# External Imports
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    SerializeAsAny,
)

# Internal imports
from quicfire_tools.ignitions import (
    IgnitionFlags,
    CircularRingIgnition,
    Ignition,
    RectangleIgnition,
    SquareRingIgnition,
    default_line_ignition,
    serialize_ignition,
)
from quicfire_tools.topography import (
    TopoFlags,
    CanyonTopo,
    CosHillTopo,
    GaussianHillTopo,
    HalfCircleTopo,
    HillPassTopo,
    SinusoidTopo,
    SlopeMesaTopo,
    Topography,
    serialize_topography,
)
from quicfire_tools.utils import compute_parabolic_stretched_grid, list_default_factory


DOCS_PATH = Path(
    str(
        importlib.resources.files("quicfire_tools")
        .joinpath("data")
        .joinpath("documentation")
    )
)
TEMPLATES_PATH = Path(
    str(
        importlib.resources.files("quicfire_tools")
        .joinpath("data")
        .joinpath("templates")
    )
)

RESERVED_FILE_NAMES = [
    "gridlist",
    "QFire_Advanced_User_Inputs.inp",
    "QFire_Bldg_Advanced_User_Inputs.inp",
    "QFire_Plume_Advanced_User_Inputs.inp",
    "QP_buildout.inp",
    "QU_buildings.inp",
    "QU_fileoptions.inp",
    "QU_landuse.inp",
    "QU_metparams.inp",
    "QU_movingcoords.inp",
    "QU_simparams.inp",
    "QU_topoinputs.inp",
    "QUIC_fire.inp",
    "RasterOrigin.inp",
    "RuntimeAdvancedUserInputs.inp",
]

LATEST_VERSION = "v6"


class SimulationInputs:
    """
    Class representing a QUIC-Fire input file deck. This class is the primary
    interface for building a QUIC-Fire input file deck and saving the input
    files to a directory for running a simulation.

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
    quic_fire: QUIC_fire
        Object representing the QUIC_fire.inp file.
    wind_sensors: dict[str, WindSensor]
        Object representing the all wind sensor input files, e.g. sensor1.inp.
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
        quic_fire: QUIC_fire,
        wind_sensors: WindSensorArray,
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
        self.quic_fire = quic_fire
        self.wind_sensors = wind_sensors
        self.qu_topoinputs = qu_topoinputs
        self.qu_simparams = qu_simparams

        # Create a dictionary from the local variables
        self._input_files_dict = {
            "qu_buildings": qu_buildings,
            "qu_fileoptions": qu_fileoptions,
            "qfire_advanced_user_inputs": qfire_advanced_user_inputs,
            "qfire_bldg_advanced_user_inputs": qfire_bldg_advanced_user_inputs,
            "qfire_plume_advanced_user_inputs": qfire_plume_advanced_user_inputs,
            "runtime_advanced_user_inputs": runtime_advanced_user_inputs,
            "qu_movingcoords": qu_movingcoords,
            "qp_buildout": qp_buildout,
            "quic_fire": quic_fire,
            "qu_topoinputs": qu_topoinputs,
            "qu_simparams": qu_simparams,
            "wind_sensors": wind_sensors,
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
    ) -> SimulationInputs:
        """
        Creates a SimulationInputs object by taking in the mimum required
        information to build a QUIC-Fire input file deck. Returns a
        SimulationInputs object representing the complete state of the
        QUIC-Fire simulation.

        Parameters
        ----------
        nx: int
            Number of cells in the x-direction [-]. Default cell size is 2m.
        ny: int
            Number of cells in the y-direction [-]. Default cell size is 2m.
        fire_nz: int
            Number of cells in the z-direction for the fire grid [-]. Default
            cell size is 1m.
        wind_speed: float
            Wind speed [m/s].
        wind_direction: float
            Wind direction [deg]. 0 deg is north, 90 deg is east, etc. Must
            be in range [0, 360).
        simulation_time: int
            Number of seconds to run the simulation for [s].

        Returns
        -------
        SimulationInputs
            Class containing the data to build a QUIC-Fire input file deck and
            run a simulation using default parameters.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
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
        qu_topoinputs = QU_TopoInputs()

        # Initialize input files with required parameters
        start_time = int(time.time())
        ignition = default_line_ignition(nx, ny, wind_direction)
        quic_fire = QUIC_fire(
            nz=fire_nz,
            time_now=start_time,
            sim_time=simulation_time,
            ignition=ignition,
        )
        wind_sensor = WindSensor(
            name="sensor1",
            wind_speeds=[wind_speed],
            wind_directions=[wind_direction],
            wind_times=[start_time],
            sensor_heights=6.1,
            x_location=1,
            y_location=1,
        )
        wind_sensor_array = WindSensorArray(sensor_array=[wind_sensor])
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
            quic_fire=quic_fire,
            wind_sensors=wind_sensor_array,
            qu_topoinputs=qu_topoinputs,
            qu_simparams=qu_simparams,
        )

    @classmethod
    def from_directory(
        cls, directory: str | Path, version: str = "latest"
    ) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a directory containing a
        QUIC-Fire input file deck. The function looks for each input file in the
        QUIC-Fire input file deck, reads in the file to an object, and compiles
        the objects to a SimulationInputs object that represents the complete
        state of the QUIC-Fire simulation.

        Parameters
        ----------
        directory: str | Path
            Directory containing a QUIC-Fire input file deck.
        version: str
            QUIC-Fire version of the input files to read. Currently supported
            versions are "v5", "v6", and "latest". Default is "latest".

        Returns
        -------
        SimulationInputs
            Class containing the input files in the QUIC-Fire input file deck.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> simulation_path = "path/to/simulation/directory"
        >>> sim_inputs = SimulationInputs.from_directory(simulation_path)
        """
        if isinstance(directory, str):
            directory = Path(directory)

        version = _validate_and_return_version(version)

        # Read the required input files
        qu_fileoptions = QU_Fileoptions.from_file(directory)
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs.from_file(directory)
        qfire_bldg_advanced_user_inputs = QFire_Bldg_Advanced_User_Inputs.from_file(
            directory
        )
        qfire_plume_advanced_user_inputs = QFire_Plume_Advanced_User_Inputs.from_file(
            directory
        )
        runtime_advanced_user_inputs = RuntimeAdvancedUserInputs.from_file(directory)
        quic_fire = QUIC_fire.from_file(directory, version=version)
        wind_sensors = WindSensorArray.from_file(directory)
        qu_topoinputs = QU_TopoInputs.from_file(directory)
        qu_simparams = QU_Simparams.from_file(directory)

        # Try and read optional input files. Return defaults if not found.
        try:
            raster_origin = RasterOrigin.from_file(directory)
        except FileNotFoundError:
            raster_origin = RasterOrigin()

        # Instantiate file objects that are all default values
        qu_buildings = QU_Buildings()
        qp_buildout = QP_buildout()
        qu_movingcoords = QU_movingcoords()

        return cls(
            rasterorigin=raster_origin,
            qu_buildings=qu_buildings,
            qu_fileoptions=qu_fileoptions,
            qfire_advanced_user_inputs=qfire_advanced_user_inputs,
            qfire_bldg_advanced_user_inputs=qfire_bldg_advanced_user_inputs,
            qfire_plume_advanced_user_inputs=qfire_plume_advanced_user_inputs,
            runtime_advanced_user_inputs=runtime_advanced_user_inputs,
            qu_movingcoords=qu_movingcoords,
            qp_buildout=qp_buildout,
            quic_fire=quic_fire,
            wind_sensors=wind_sensors,
            qu_topoinputs=qu_topoinputs,
            qu_simparams=qu_simparams,
        )

    @classmethod
    def from_dict(cls, data: dict) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary containing input file data.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> json_path = "path/to/json/object"
        >>> sim_inputs = SimulationInputs.from_json(json_path)
        >>> sim_dict = sim_inputs.to_dict()
        >>> new_sim_inputs = SimulationInputs.from_dict(sim_dict)
        """
        return cls(
            rasterorigin=(
                RasterOrigin.from_dict(data["rasterorigin"])
                if data.get("rasterorigin")
                else RasterOrigin()
            ),
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
            quic_fire=QUIC_fire.from_dict(data["quic_fire"]),
            wind_sensors=WindSensorArray.from_dict(data["wind_sensors"]),
            qu_topoinputs=QU_TopoInputs.from_dict(data["qu_topoinputs"]),
            qu_simparams=QU_Simparams.from_dict(data["qu_simparams"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationInputs:
        """
        Initializes a SimulationInputs object from a JSON file.

        Parameters
        ----------
        path: Path | str
            Path to the JSON file.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> json_path = "path/to/json/object"
        >>> sim_inputs = SimulationInputs.from_json(json_path)
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

        This method is a core method of the SimulationInputs class. It
        is the principle way to translate a SimulationInputs object into a
        QUIC-Fire input file deck.

        Parameters
        ----------
        directory: str | Path
            Directory to write the input files to.
        version: str
            Version of the input files to write. Default is "latest".

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.write_inputs("path/to/simulation/directory")
        """
        if isinstance(directory, str):
            directory = Path(directory)

        if not directory.exists():
            directory.mkdir(parents=True)

        version = _validate_and_return_version(version)

        # The fire cannot start before the first wind field
        # Is there a way to check this at assignment rather than write?
        if self.quic_fire.time_now < self.qu_simparams.wind_times[0]:
            raise ValueError(
                f"The fire cannot start before the first wind field update: "
                f"\n\tFire time:{self.quic_fire.time_now}"
                f"\n\tWind time:{self.qu_simparams.wind_times[0]}"
            )

        # Create QU_metparams dynamically from wind_sensors
        QU_metparams(
            site_names=[sensor.name for sensor in self.wind_sensors.sensor_array],
            file_names=[sensor._filename for sensor in self.wind_sensors.sensor_array],
        ).to_file(directory, version=version)

        # Write remaining input files to the output directory
        for input_file in self._input_files_dict.values():
            input_file.to_file(directory, version=version)

        # Write required files for custom fuels
        if self.quic_fire.is_custom_fuel_model:
            Gridlist(
                n=self.qu_simparams.nx,
                m=self.qu_simparams.ny,
                l=self.quic_fire.nz,
                dx=self.qu_simparams.dx,
                dy=self.qu_simparams.dy,
                dz=self.quic_fire.dz,
            ).to_file(directory, version=version)

            RasterOrigin().to_file(directory, version=version)

        # Copy QU_landuse from the template directory to the output directory
        template_file_path = TEMPLATES_PATH / version / "QU_landuse.inp"
        output_file_path = directory / "QU_landuse.inp"
        with open(template_file_path, "rb") as ftemp:
            with open(output_file_path, "wb") as fout:
                fout.write(ftemp.read())

        return None

    def to_dict(self) -> dict:
        """
        Convert the state of the SimulationInputs object to a dictionary.
        The name of each input file in the SimulationInputs object is a key
        to that input file's dictionary form.

        Returns
        -------
        dict
            Dictionary representation of the object.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_dict = sim_inputs.to_dict()
        """
        return {key: value.to_dict() for key, value in self._input_files_dict.items()}

    def to_json(self, path: str | Path) -> None:
        """
        Write the SimulationInputs object to a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to write the JSON file to.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.to_json("path/to/json/object")
        """
        if isinstance(path, str):
            path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def set_custom_simulation(
        self,
        fuel_density: bool = True,
        fuel_moisture: bool = True,
        fuel_height: bool = True,
        size_scale: bool = False,
        patch_and_gap: bool = False,
        ignition: bool = True,
        topo: bool = True,
        interpolate: bool = False,
    ) -> None:
        """
        Sets the simulation to use custom fuel, ignition, and topography
        settings.

        This function can be useful for setting up simulations that use .dat
        files to define custom fuel, topography, or ignition inputs.

        Parameters
        ----------
        fuel_density: bool, optional
            If True, sets the simulation to use fuel density information from
            a treesrhof.dat file (fuel density flag 3). Default is True.
        fuel_moisture: bool, optional
            If True, sets the simulation to use fuel moisture information from
            a treesmoist.dat file (fuel moisture flag 3). Default is True.
        fuel_height: bool, optional
            If True, sets the simulation to use fuel height information from
            a treesdepth.dat file (fuel height flag 3). Default is True.
        size_scale: bool, optional
            If True, sets the simulation to use size scale information from
            a treesss.dat file (size scale flag 3). Defaults to False as this
            is a new feature.
        patch_and_gap: bool, optional
            If True, sets the simulation to use patch and gap information from
            patch.dat and gap.dat files (patch and gap flag 2). Defaults to
            False as this is a new feature.
        ignition : bool, optional
            If True, sets the simulation to use a custom ignition source
            (ignition flag 6). Default is True.
        topo : bool, optional
            If True, sets the simulation to use custom topography settings
            (topography flag 5). Default is True.
        interpolate: bool, optional
            If True, sets the simulation to interpolate the custom fuel inputs
            to the fire grid (fuel flag 4). Default is False. This is also
            useful as it addresses a bug in versions of QUIC-Fire ≤ v6.0.0 where
            custom fuels don't work without the interpolation flag set.
            Interpolation only applies to fuel density, fuel moisture, fuel
            height, and size scale. Patch and gap, ignition, and topography
            are not interpolated.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.set_custom_simulation(fuel=True, ignition=True, topo=True)
        >>> sim_inputs.quic_fire.fuel_flag
        3
        """
        if fuel_density:
            self.quic_fire.fuel_density_flag = 3 if not interpolate else 4
        if fuel_moisture:
            self.quic_fire.fuel_moisture_flag = 3 if not interpolate else 4
        if fuel_height:
            self.quic_fire.fuel_height_flag = 3 if not interpolate else 4
        if size_scale:
            self.quic_fire.size_scale_flag = 3 if not interpolate else 4
        if patch_and_gap:
            self.quic_fire.patch_and_gap_flag = 2
        if ignition:
            self.quic_fire.ignition = Ignition(ignition_flag=IgnitionFlags(7))
        if topo:
            self.qu_topoinputs.topography = Topography(topo_flag=TopoFlags(5))

    def set_uniform_fuels(
        self,
        fuel_density: float,
        fuel_moisture: float,
        fuel_height: float,
        size_scale: float = 0.0005,
        patch_size: float = 0.0,
        gap_size: float = 0.0,
    ) -> None:
        """
        Sets the simulation to use uniform fuel settings. This function updates
        the fuel flag to 1 and sets the fuel density, fuel moisture, and fuel
        height to the specified values.

        Parameters
        ----------
        fuel_density: float
            Fuel bulk density [kg/m^3]. Note: This is the fuel bulk density, so
            the fuel load should be normalized by the height of the fuel bed.
        fuel_moisture: float
            Fuel moisture content [%].
        fuel_height: float
            Fuel bed height [m].
        size_scale: float, optional
            Size scale [m]. Default is 0.0005.
        patch_size: float, optional
            Patch size [m]. Default is 0.0.
        gap_size: float, optional
            Gap size [m]. Default is 0.0.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.set_uniform_fuels(fuel_density=0.5, fuel_moisture=25, fuel_height=1)
        >>> sim_inputs.quic_fire.fuel_density_flag
        1
        >>> sim_inputs.quic_fire.fuel_density
        0.5
        """
        self.quic_fire.fuel_density_flag = 1
        self.quic_fire.fuel_density = fuel_density
        self.quic_fire.fuel_moisture_flag = 1
        self.quic_fire.fuel_moisture = fuel_moisture
        self.quic_fire.fuel_height_flag = 1
        self.quic_fire.fuel_height = fuel_height
        if size_scale != 0.0005:
            self.quic_fire.size_scale_flag = 1
            self.quic_fire.size_scale = size_scale
        else:
            self.quic_fire.size_scale_flag = 0
        if patch_size != 0.0 or gap_size != 0.0:
            self.quic_fire.patch_and_gap_flag = 1
            self.quic_fire.patch_size = patch_size
            self.quic_fire.gap_size = gap_size
        else:
            self.quic_fire.patch_and_gap_flag = 0

    def set_rectangle_ignition(
        self, x_min: float, y_min: float, x_length: float, y_length: float
    ) -> None:
        """
        Sets the simulation to use a rectangle ignition source. This function
        updates the ignition flag to 1 and sets the ignition source to the
        specified rectangle.

        Parameters
        ----------
        x_min: float
            South-west corner in the x-direction [m]
        y_min: float
            South-west corner in the y-direction [m]
        x_length: float
            Length in the x-direction [m]
        y_length: float
            Length in the y-direction [m]

        Examples
        -------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.set_rectangle_ignition(x_min=0, y_min=0, x_length=10, y_length=10)
        """
        ignition = RectangleIgnition(
            x_min=x_min, y_min=y_min, x_length=x_length, y_length=y_length
        )
        self.quic_fire.ignition = ignition

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
        surf_eng: bool = False,
    ) -> None:
        """
        Sets the simulation to output the specified files. Files set to True
        will be output by the simulation, and files set to False will not be
        output.

        Parameters
        ----------
        eng_to_atm: bool, optional
            If True, output the fire-energy_to_atmos.bin file. Default is False.
        react_rate: bool, optional
            If True, output the fire-reaction_rate.bin file. Default is False.
        fuel_dens: bool, optional
            If True, output the fuels-dens.bin file. Default is False.
        qf_wind: bool, optional
            If True, output the windu, windv, and windw .bin files.
            Default is False.
        qu_wind_inst: bool, optional
            If True, output the quic_wind_inst.bin file. Default is False.
        qu_wind_avg: bool, optional
            If True, output the quic_wind_avg.bin file. Default is False.
        fuel_moist: bool, optional
            If True, output the fuels-moist.bin file. Default is False.
        mass_burnt: bool, optional
            If True, output the mburnt_integ.bin file. Default is False.
        emissions: bool, optional
            If True, output the co-emissions and pm-emissions .bin files.
            Default is False.
        radiation: bool, optional
            If True, output the thermaldose and thermalradiation .bin files.
            Default is False.
        surf_eng: bool, optional
            If True, output the surf_eng.bin file. Default is False.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.set_output_files(fuel_dens=True, mass_burnt=True)
        """

        self.quic_fire.eng_to_atm_out = int(eng_to_atm)
        self.quic_fire.react_rate_out = int(react_rate)
        self.quic_fire.fuel_dens_out = int(fuel_dens)
        self.quic_fire.qf_wind_out = int(qf_wind)
        self.quic_fire.qu_wind_inst_out = int(qu_wind_inst)
        self.quic_fire.qu_wind_avg_out = int(qu_wind_avg)
        self.quic_fire.fuel_moist_out = int(fuel_moist)
        self.quic_fire.mass_burnt_out = int(mass_burnt)
        self.quic_fire.radiation_out = int(radiation)
        self.quic_fire.surf_eng_out = int(surf_eng)
        self.quic_fire.emissions_out = 2 if emissions else 0

    def set_output_interval(self, interval: int):
        """
        Sets the interval, in seconds, at which the simulation will write .bin
        files to disk. This function sets the same interval for all output
        files.

        Parameters
        ----------
        interval: int
            Interval in seconds at which to write .bin files to disk.

        Examples
        --------
        >>> from quicfire_tools import SimulationInputs
        >>> sim_inputs = SimulationInputs.create_simulation(nx=100, ny=100, fire_nz=26, wind_speed=1.8, wind_direction=90, simulation_time=600)
        >>> sim_inputs.set_output_interval(60)
        """
        self.quic_fire.out_time_fire = interval
        self.quic_fire.out_time_wind = interval
        self.quic_fire.out_time_wind_avg = interval
        self.quic_fire.out_time_emis_rad = interval

    def add_wind_sensor(
        self,
        wind_speeds: Union[float, List[float]],
        wind_directions: Union[int, List[int]],
        wind_times: Union[int, List[int]],
        sensor_height: float = 6.1,
        x_location: float = 1.0,
        y_location: float = 1.0,
        sensor_name: str = None,
        wind_update_frequency: int = 300,
    ) -> None:
        """
        Adds a new wind sensor to the simulation with specified wind conditions. This method
        handles the coordination between multiple QUIC-Fire input files that contain wind
        information and ensures they stay synchronized.

        Parameters
        ----------
        wind_speeds : float or List[float]
            Wind speed(s) in meters per second. Can be a single value for constant wind
            or a list of values for varying wind conditions.
        wind_directions : int or List[int]
            Wind direction(s) in degrees, where 0° is North and degrees increase clockwise
            (90° is East). Can be a single value or a list matching wind_speeds.
        wind_times : int or List[int]
            Time(s) in seconds relative to simulation start (t=0) when each wind condition begins.
            For constant wind, use 0. For varying winds, provide a list of times corresponding
            to each speed/direction pair (e.g., [0, 600, 1200] for changes at 0, 10, and 20 minutes).
        sensor_height : float, optional
            Height of the sensor in meters. Defaults to 6.1m (20 feet), which is standard
            weather station height.
        x_location : float, optional
            X-coordinate position of the sensor in meters. Defaults to 1.0.
        y_location : float, optional
            Y-coordinate position of the sensor in meters. Defaults to 1.0.
        sensor_name : str, optional
            Custom name for the sensor. If not provided, will be automatically generated as
            "sensorN" where N is the next available number.
        wind_update_frequency : int, optional
            Minimum time in seconds between wind field updates. Defaults to 300 seconds
            (5 minutes). Smaller values increase computation time but may improve accuracy.

        Examples
        --------
        Adding a sensor with constant wind:

        >>> sim_inputs.add_wind_sensor(
        ...     wind_speeds=5.0,
        ...     wind_directions=90,
        ...     wind_times=0
        ... )

        Adding a sensor with varying wind conditions starting at t=0 and changing every 10 minutes:

        >>> sim_inputs.add_wind_sensor(
        ...     wind_speeds=[5.0, 7.0, 6.0],
        ...     wind_directions=[90, 180, 135],
        ...     wind_times=[0, 600, 1200],  # Changes at 0, 10, and 20 minutes
        ...     sensor_height=10.0,
        ...     x_location=50.0,
        ...     y_location=50.0,
        ...     sensor_name="custom_sensor"
        ... )

        Notes
        -----
        - Wind times must be provided relative to simulation start (t=0), not as absolute times.
        - Wind lists must have equal lengths and correspond to each other (i.e., wind_times[0]
          corresponds to wind_speeds[0] and wind_directions[0]).
        - Wind times must be in ascending order.
        - Wind directions must be in degrees from 0 to 360.
        - Multiple sensors can be added to the same simulation to represent spatial variation
          in wind conditions.
        """

        # Generate sensor name if not provided
        if sensor_name is None:
            existing_names = {sensor.name for sensor in self.wind_sensors.sensor_array}
            i = 1
            while f"sensor{i}" in existing_names:
                i += 1
            sensor_name = f"sensor{i}"

        # Create and add the new sensor
        new_sensor = WindSensor(
            name=sensor_name,
            wind_times=wind_times,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            sensor_heights=sensor_height,
            x_location=x_location,
            y_location=y_location,
        )
        for i in range(len(new_sensor.wind_times)):
            new_sensor.wind_times[i] += self.quic_fire.time_now

        self.wind_sensors.sensor_array.append(new_sensor)
        self._update_shared_wind_times(wind_update_frequency)

    def add_wind_sensor_from_dataframe(
        self,
        df: pd.DataFrame,
        x_location: float,
        y_location: float,
        sensor_height: float,
        time_column: str = "wind_times",
        speed_column: str = "wind_speeds",
        direction_column: str = "wind_directions",
        sensor_name: Optional[str] = None,
        wind_update_frequency: int = 300,
    ) -> None:
        """
        Adds a wind sensor to the simulation using wind data from a pandas DataFrame. This is
        particularly useful when importing wind data from CSV files or other tabular data sources.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing wind data. Must include columns for times, speeds, and
            directions (column names can be specified using the column parameters).
        x_location : float
            X-coordinate position of the sensor in meters.
        y_location : float
            Y-coordinate position of the sensor in meters.
        sensor_height : float
            Height of the sensor in meters (typically 6.1m/20ft for standard weather stations).
        time_column : str, optional
            Name of the DataFrame column containing wind times in seconds relative to
            simulation start (t=0). Defaults to "wind_times".
        speed_column : str, optional
            Name of the DataFrame column containing wind speeds in meters per second.
            Defaults to "wind_speeds".
        direction_column : str, optional
            Name of the DataFrame column containing wind directions in degrees (0° = North,
            90° = East). Defaults to "wind_directions".
        sensor_name : str, optional
            Custom name for the sensor. If not provided, will be automatically generated as
            "sensorN" where N is the next available number.
        wind_update_frequency : int, optional
            Minimum time in seconds between wind field updates. Defaults to 300 seconds
            (5 minutes).

        Examples
        --------
        Using default column names:

        >>> import pandas as pd
        >>> wind_data = pd.DataFrame({
        ...     'wind_times': [0, 600, 1200],      # Times at 0, 10, and 20 minutes
        ...     'wind_speeds': [5.0, 7.0, 6.0],    # Wind speeds in m/s
        ...     'wind_directions': [90, 180, 135]   # Wind directions in degrees
        ... })
        >>> sim_inputs.add_wind_sensor_from_dataframe(
        ...     df=wind_data,
        ...     x_location=50.0,
        ...     y_location=50.0,
        ...     sensor_height=6.1
        ... )

        Using custom column names:

        >>> weather_data = pd.DataFrame({
        ...     'time_s': [0, 300, 600],           # Times in seconds from start
        ...     'speed_ms': [4.0, 5.0, 4.5],       # Speeds in m/s
        ...     'direction_deg': [45, 90, 75]      # Directions in degrees
        ... })
        >>> sim_inputs.add_wind_sensor_from_dataframe(
        ...     df=weather_data,
        ...     x_location=100.0,
        ...     y_location=100.0,
        ...     sensor_height=10.0,
        ...     time_column='time_s',
        ...     speed_column='speed_ms',
        ...     direction_column='direction_deg',
        ...     sensor_name='weather_station_1'
        ... )

        Notes
        -----
        - Wind times in the DataFrame must be relative to simulation start (t=0), not absolute times.
        - Times must be in ascending order.
        - Wind directions must be in degrees from 0 to 360.
        - All columns must contain numeric data in the correct units (seconds, m/s, degrees).
        - Multiple sensors can be added to represent spatial variation in wind conditions.

        See Also
        --------
        add_wind_sensor : Add a wind sensor using direct parameter inputs
        """
        # Validate required columns
        required_columns = [time_column, speed_column, direction_column]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")

        self.add_wind_sensor(
            wind_speeds=df[speed_column],
            wind_times=df[time_column],
            wind_directions=df[direction_column],
            sensor_height=sensor_height,
            x_location=x_location,
            y_location=y_location,
            sensor_name=sensor_name,
            wind_update_frequency=wind_update_frequency,
        )

    def remove_wind_sensor(
        self,
        sensor_name: str,
        wind_update_frequency: int = 300,
    ) -> None:
        """
        Removes a wind sensor from the simulation by its name and updates the simulation's
        wind timing parameters accordingly.

        Parameters
        ----------
        sensor_name : str
            Name of the wind sensor to remove. Must match exactly the name of an existing
            sensor in the simulation.
        wind_update_frequency : int, optional
            Minimum time in seconds between wind field updates after sensor removal.
            Defaults to 300 seconds (5 minutes). This parameter is used to recalculate
            wind update times for remaining sensors.

        Raises
        ------
        ValueError
            If the specified sensor_name is not found in the simulation.

        Examples
        --------
        >>> # Add two wind sensors
        >>> sim_inputs.add_wind_sensor(
        ...     wind_speeds=5.0,
        ...     wind_directions=90,
        ...     wind_times=0,
        ...     sensor_name="sensor1"
        ... )
        >>> sim_inputs.add_wind_sensor(
        ...     wind_speeds=6.0,
        ...     wind_directions=180,
        ...     wind_times=0,
        ...     sensor_name="sensor2"
        ... )
        >>>
        >>> # Remove the first sensor
        >>> sim_inputs.remove_wind_sensor("sensor1")

        Notes
        -----
        - After removing a sensor, the simulation's wind update times are automatically
          recalculated based on the remaining sensors.
        - Make sure at least one wind sensor remains in the simulation for valid results.
        - Sensor names are case-sensitive.

        See Also
        --------
        add_wind_sensor : Add a wind sensor using direct parameter inputs
        add_wind_sensor_from_dataframe : Add a wind sensor using data from a DataFrame
        """
        # Find the sensor to remove
        sensor_index = next(
            (
                i
                for i, s in enumerate(self.wind_sensors.sensor_array)
                if s.name == sensor_name
            ),
            None,
        )
        if sensor_index is None:
            raise ValueError(f"Sensor '{sensor_name}' not found")

        # Remove the sensor
        self.wind_sensors.sensor_array.pop(sensor_index)

        # Update global wind times
        self._update_shared_wind_times(wind_update_frequency)

    def _update_shared_wind_times(self, wind_update_frequency: int):
        all_sensor_times = self.wind_sensors.wind_times

        if len(all_sensor_times) == 0:
            return

        first_time = all_sensor_times[0]
        updated_wind_times = [first_time]
        for wind_time in all_sensor_times[1:]:
            if wind_time - updated_wind_times[-1] >= wind_update_frequency:
                updated_wind_times.append(wind_time)

        self.qu_simparams.wind_times = updated_wind_times


class InputFile(BaseModel, validate_assignment=True):
    """
    Base class representing an input file.

    This base class provides a common interface for all input files in order to
    accomplish two main goals:

    1) Return documentation for each parameter in the input file.

    2) Provide a method to write the input file to a directory.
    """

    name: str
    _extension: str

    @property
    def _filename(self):
        return f"{self.name}{self._extension}"

    @property
    def documentation_dict(self) -> dict:
        # Return the documentation dictionary
        with open(DOCS_PATH / f"{self._filename}.json", "r") as f:
            return json.load(f)

    def list_parameters(self) -> list[str]:
        """
        Get a list of the names of all parameters in the input file.
        """
        return list(self.documentation_dict.keys())

    def get_documentation(self, parameter: str = None) -> dict:
        """
        Retrieve documentation for a parameter. If no parameter is specified,
        return documentation for all parameters.
        """
        if parameter:
            return self.documentation_dict.get(parameter, {})
        else:
            return self.documentation_dict

    def print_documentation_table(self, parameter: str = None) -> None:
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

    def to_dict(self, include_private: bool = False) -> dict:
        """
        Convert the object to a dictionary, excluding attributes that start
        with an underscore.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        if isinstance(self, WindSensor):
            exclude_fields = {"_extension", "_filename", "documentation_dict"}
        else:
            exclude_fields = {"name", "_extension", "_filename", "documentation_dict"}
        all_fields = self.model_dump(exclude=exclude_fields)
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

        version = _validate_and_return_version(version)

        if isinstance(self, WindSensor):
            template_file_path = TEMPLATES_PATH / version / "sensor.inp"
        else:
            template_file_path = TEMPLATES_PATH / version / f"{self._filename}"

        with open(template_file_path, "r") as ftemp:
            src = Template(ftemp.read())

        model_dict = self.to_dict(include_private=True)
        result = src.substitute(model_dict)

        output_file_path = directory / self._filename
        with open(output_file_path, "w") as fout:
            fout.write(result)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


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
    dx: PositiveFloat = 2.0
    dy: PositiveFloat = 2.0
    dz: PositiveFloat = 1.0
    aa1: PositiveFloat = 1.0

    @classmethod
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a Gridlist object from a directory containing a
        gridlist.txt file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "gridlist", "r") as f:
            lines = f.read()

        try:
            return cls(
                n=int(lines.split("n=")[1].split()[0]),
                m=int(lines.split("m=")[1].split()[0]),
                l=int(lines.split("l=")[1].split()[0]),
                dx=float(lines.split("dx=")[1].split()[0]),
                dy=float(lines.split("dy=")[1].split()[0]),
                dz=float(lines.split("dz=")[1].split()[0]),
                aa1=float(lines.split("aa1=")[1].split()[0]),
            )
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: gridlist"
                f"\nPlease check the file for correctness."
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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a RasterOrigin object from a directory containing a
        rasterorigin.txt file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "rasterorigin.txt", "r") as f:
            lines = f.readlines()
        try:
            return cls(
                utm_x=float(lines[0].split()[0]), utm_y=float(lines[1].split()[0])
            )
        except IndexError:
            # Optional file, return default values if parsing fails
            return cls()


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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QU_Buildings object from a directory containing a
        QU_buildings.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_buildings.inp", "r") as f:
            lines = f.readlines()
        try:
            return cls(
                wall_roughness_length=float(lines[1].split()[0]),
                number_of_buildings=int(lines[2].split()[0]),
                number_of_polygon_nodes=int(lines[3].split()[0]),
            )
        except IndexError:
            # Optional file, return default values if parsing fails
            return cls()


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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QU_Fileoptions object from a directory containing a
        QU_fileoptions.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_fileoptions.inp", "r") as f:
            lines = f.readlines()
        try:
            return cls(
                output_data_file_format_flag=int(lines[1].split()[0]),
                non_mass_conserved_initial_field_flag=int(lines[2].split()[0]),
                initial_sensor_velocity_field_flag=int(lines[3].split()[0]),
                qu_staggered_velocity_file_flag=int(lines[4].split()[0]),
                generate_wind_startup_files_flag=int(lines[5].split()[0]),
            )
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QU_fileoptions.inp"
                f"\nPlease check the file for correctness."
            )


class QU_Simparams(InputFile):
    """
    Class representing the QU_simparams.inp file. This file contains the
    simulation parameters for the QUIC-Fire simulation.

    Attributes
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
    dx: PositiveFloat = 2.0
    dy: PositiveFloat = 2.0
    quic_domain_height: PositiveFloat = 300.0
    wind_times: list[int]
    surface_vertical_cell_size: PositiveFloat = 1.0
    number_surface_cells: PositiveInt = 5
    stretch_grid_flag: Literal[0, 1, 3] = 3
    custom_dz_array: list[PositiveFloat] = Field(default_factory=list_default_factory)
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
    _from_file_dz_array: Optional[list[PositiveFloat]] = None

    @computed_field
    @property
    def _dz_array(self) -> list[float]:
        if self._from_file_dz_array:
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
        Parses the utc_offset and wind_times to generate the wind times
        as a string for the QU_simparams.inp file.
        """
        # Verify that wind_step_times is not empty
        if not self.wind_times:
            raise ValueError(
                "wind_times must not be empty. Please "
                "provide a wind_times with num_wind_steps "
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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QU_Simparams object from a directory containing a
        QU_simparams.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QU_simparams.inp", "r") as f:
            lines = f.readlines()

        try:
            # Read QU grid parameters
            nx = int(lines[1].strip().split()[0])
            ny = int(lines[2].strip().split()[0])
            nz = int(lines[3].strip().split()[0])
            dx = float(lines[4].strip().split()[0])
            dy = float(lines[5].strip().split()[0])

            # Read stretch grid flag
            stretch_grid_flag = int(lines[6].strip().split()[0])

            # Read vertical grid lines as function of stretch grid flag
            from_file_dz_array = []
            custom_dz_array = []
            if stretch_grid_flag == 0:
                surface_vertical_cell_size = float(lines[7].strip().split()[0])
                number_surface_cells = int(lines[8].strip().split()[0])
                quic_domain_height = surface_vertical_cell_size * number_surface_cells
                current_line = 9
            elif stretch_grid_flag == 1:
                surface_vertical_cell_size = float(lines[7].strip().split()[0])
                number_surface_cells = 5
                for i in range(9, 9 + nz):
                    custom_dz_array.append(float(lines[i].strip().split()[0]))
                quic_domain_height = round(sum(custom_dz_array), 2)
                current_line = 9 + nz
            elif stretch_grid_flag == 3:
                surface_vertical_cell_size = float(lines[7].strip().split()[0])
                number_surface_cells = int(lines[8].strip().split()[0])
                for i in range(10, 10 + nz):
                    from_file_dz_array.append(float(lines[i].strip().split()[0]))
                quic_domain_height = round(sum(from_file_dz_array), 2)
                current_line = 10 + nz
            else:
                raise ValueError("stretch_grid_flag must be 0, 1, or 3.")

            # Read QU wind parameters
            number_wind_steps = int(lines[current_line].strip().split()[0])
            utc_offset = int(lines[current_line + 1].strip().split()[0])
            _ = lines[current_line + 2].strip().split()[0]
            wind_times = []
            for i in range(current_line + 3, current_line + 3 + number_wind_steps):
                wind_times.append(int(lines[i].strip()))
            current_line = current_line + 3 + number_wind_steps

            # Skip not used parameters
            current_line += 9

            # Read remaining QU parameters
            sor_iter_max = int(lines[current_line].strip().split()[0])
            sor_residual_reduction = int(lines[current_line + 1].strip().split()[0])
            use_diffusion_flag = int(lines[current_line + 2].strip().split()[0])
            number_diffusion_iterations = int(
                lines[current_line + 3].strip().split()[0]
            )
            domain_rotation = float(lines[current_line + 4].strip().split()[0])
            utm_x = float(lines[current_line + 5].strip().split()[0])
            utm_y = float(lines[current_line + 6].strip().split()[0])
            utm_zone_number = int(lines[current_line + 7].strip().split()[0])
            utm_zone_letter = int(lines[current_line + 8].strip().split()[0])
            quic_cfd_flag = int(lines[current_line + 9].strip().split()[0])
            explosive_bldg_flag = int(lines[current_line + 10].strip().split()[0])
            bldg_array_flag = int(lines[current_line + 11].strip().split()[0])

        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QU_simparams.inp"
                f"\nPlease check the file for correctness."
            )

        qu_simparams = cls(
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
        )
        qu_simparams._from_file_dz_array = from_file_dz_array
        return qu_simparams


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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QFire_Advanced_User_Inputs object from a directory
        containing a QFire_Advanced_User_Inputs.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QFire_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()

        try:
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
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QFire_Advanced_User_Inputs.inp"
                f"\nPlease check the file for correctness."
            )


class QUIC_fire(InputFile):
    """
    Class representing the QUIC_fire.inp input file. This file
    contains the parameters relating to the fire simulation and
    outputs.

    Attributes
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
    dz : PositiveFloat
        Cell size in the z-direction [m] of the fire grid. Recommended value: 1m
    dz_array : List[PositiveFloat]
        Custom dz, one dz per line must be specified, from the ground to the
        top of the domain
    fuel_density_flag: Literal[1, 2, 3, 4, 5]
        Fuel density flag (defaults to 1):
        1 = density is uniform over the domain,
        2 = density is provided through QF_FuelDensity.inp,
        3 = density is provided through Firetec file (treesrhof.dat) matching QUIC-Fire grid,
        4 = density is provided through Firetec files for an arbitrary grid,
        5 = FastFuels input (assuming uniform dz of 1m)
    fuel_density : PositiveFloat
        Fuel density (kg/m3)
    fuel_moisture_flag : Literal[1, 2, 3, 4, 5]
        Fuel moisture flag (defaults to 1):
        1 = moisture is uniform over the domain,
        2 = moisture is provided through QF_FuelMoisture.inp,
        3 = moisture is provided through Firetec file (treesmoist.dat) matching QUIC-Fire grid,
        4 = moisture is provided through Firetec files for an arbitrary grid,
        5 = FastFuels input (assuming uniform dz of 1m)
    fuel_moisture : PositiveFloat
        Fuel moisture = mass of water/mass of dry fuel (kg/kg). Must be between
        0 and 1.
    fuel_height_flag : Literal[1, 2, 3, 4]
        Fuel height flag (defaults to 1):
        1 = height is uniform over the domain,
        2 = height is provided through QF_FuelHeight.inp,
        3 = height is provided through Firetec file (treesheight.dat) matching QUIC-Fire grid,
        4 = height is provided through Firetec files for an arbitrary grid,
        5 = FastFuels input (assuming uniform dz of 1m)
    fuel_height : PositiveFloat
        Fuel height of surface layer (m)
    size_scale_flag : Literal[0, 1, 2, 3, 4, 5]
        Size scale flag (defaults to 0):
        0 = Default value (0.0005) over entire domain,
        1 = custom uniform value over the domain,
        2 = custom value provided through QF_SizeScale.inp,
        3 = custom value provided through Firetec file (treesss.dat) matching QUIC-Fire grid,
        4 = custom value provided through Firetec files for an arbitrary grid,
        5 = FastFuels input (assuming uniform dz of 1m)
    size_scale : PositiveFloat
        Size scale (m). Defaults to 0.0005.
    patch_and_gap_flag : Literal[0, 1, 2]
        Patch and gap flag (defaults to 0):
        0 = Default values (0, 0) over entire domain,
        1 = custom uniform values over the domain,
        2 = custom values provided by patch.dat and gap.dat
    ignition: Ignition
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
    surf_eng_out : Literal[0, 1]
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
    dz: PositiveFloat = 1.0
    dz_array: list[PositiveFloat] = Field(default_factory=list_default_factory)
    fuel_density_flag: Literal[1, 2, 3, 4, 5] = 1
    fuel_density: Union[PositiveFloat, None] = 0.5
    fuel_moisture_flag: Literal[1, 2, 3, 4, 5] = 1
    fuel_moisture: Union[PositiveFloat, None] = 0.1
    fuel_height_flag: Literal[1, 2, 3, 4] = 1
    fuel_height: Union[PositiveFloat, None] = 1.0
    size_scale_flag: Literal[0, 1, 2, 3, 4, 5] = 0
    size_scale: PositiveFloat = 0.0005
    patch_and_gap_flag: Literal[0, 1, 2] = 0
    patch_size: NonNegativeFloat = 0.0
    gap_size: NonNegativeFloat = 0.0
    ignition: Union[
        RectangleIgnition, SquareRingIgnition, CircularRingIgnition, Ignition
    ]
    ignitions_per_cell: PositiveInt = 2
    firebrand_flag: Literal[0, 1] = 0
    auto_kill: Literal[0, 1] = 1
    eng_to_atm_out: Literal[0, 1] = 0
    react_rate_out: Literal[0, 1] = 0
    fuel_dens_out: Literal[0, 1] = 1
    qf_wind_out: Literal[0, 1] = 0
    qu_wind_inst_out: Literal[0, 1] = 1
    qu_wind_avg_out: Literal[0, 1] = 0
    fuel_moist_out: Literal[0, 1] = 0
    mass_burnt_out: Literal[0, 1] = 0
    firebrand_out: Literal[0, 1] = 0
    emissions_out: Literal[0, 1, 2, 3, 4, 5] = 0
    radiation_out: Literal[0, 1] = 0
    surf_eng_out: Literal[0, 1] = 0

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

            return f"{dz_array_lines}"
        else:
            return str(self.dz)

    @computed_field
    @property
    def _ignition_lines(self) -> str:
        return str(self.ignition)

    @computed_field
    @property
    def _fuel_density_lines(self) -> str:
        fuel_density_flag_line = f"{self.fuel_density_flag}\t! fuel density flag"
        if self.fuel_density_flag == 1:
            return fuel_density_flag_line + f"\n{self.fuel_density}"
        return fuel_density_flag_line

    @computed_field
    @property
    def _fuel_moisture_lines(self) -> str:
        fuel_moist_flag_line = f"\n{self.fuel_moisture_flag}\t! fuel moisture flag"
        if self.fuel_moisture_flag == 1:
            return fuel_moist_flag_line + f"\n{self.fuel_moisture}"
        return fuel_moist_flag_line

    @computed_field
    @property
    def _fuel_height_lines(self) -> str:
        if self.fuel_density_flag == 1:
            fuel_height_flag_line = f"\n{self.fuel_height_flag}\t! fuel height flag"
            if self.fuel_height_flag == 1:
                return fuel_height_flag_line + f"\n{self.fuel_height}"
            return fuel_height_flag_line
        return ""

    @computed_field
    @property
    def _size_scale_lines(self) -> str:
        size_scale_flag_line = f"\n{self.size_scale_flag}\t! size scale flag"
        if self.size_scale_flag == 1:
            return size_scale_flag_line + f"\n{self.size_scale}"
        return size_scale_flag_line

    @computed_field
    @property
    def _patch_and_gap_lines(self) -> str:
        patch_and_gap_flag_line = f"\n{self.patch_and_gap_flag}\t! patch and gap flag"
        if self.patch_and_gap_flag == 1:
            return patch_and_gap_flag_line + f"\n{self.patch_size}\n{self.gap_size}"
        return patch_and_gap_flag_line

    @computed_field
    @property
    def is_custom_fuel_model(self) -> bool:
        return (
            self.fuel_density_flag != 1
            or self.fuel_moisture_flag != 1
            or self.fuel_height_flag != 1
        )

    @classmethod
    def from_dict(cls, data: dict):
        if "ignition" in data:
            data["ignition"] = serialize_ignition(data["ignition"])
        return cls(**data)

    @classmethod
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QUIC_fire object from a directory containing a
        QUIC_Fire.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)

        version = _validate_and_return_version(kwargs.get("version", ""))

        with open(directory / "QUIC_fire.inp", "r") as f:
            lines = f.readlines()

        try:
            # Read fire flag and random seed
            fire_flag = int(lines[0].strip().split()[0])
            random_seed = int(lines[1].strip().split()[0])

            # Read fire times
            time_now = int(lines[3].strip().split()[0])
            sim_time = int(lines[4].strip().split()[0])
            fire_time_step = int(lines[5].strip().split()[0])
            quic_time_step = int(lines[6].strip().split()[0])
            out_time_fire = int(lines[7].strip().split()[0])
            out_time_wind = int(lines[8].strip().split()[0])
            out_time_emis_rad = int(lines[9].strip().split()[0])
            out_time_wind_avg = int(lines[10].strip().split()[0])

            # Read fire grid parameters
            dz = 1.0
            nz = int(lines[12].strip().split()[0])
            stretch_grid_flag = int(lines[13].strip().split()[0])
            dz_array = []
            if stretch_grid_flag == 0:
                dz = float(lines[14].strip().split()[0])
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
            current_line += 1  # skip !FUEL line

            # Read fuel density
            fuel_density_flag = int(lines[current_line].strip().split()[0])
            fuel_density = None
            if fuel_density_flag == 1:
                current_line += 1
                fuel_density = float(lines[current_line].strip().split()[0])

            # Read fuel moisture
            current_line += 1
            fuel_moisture_flag = int(lines[current_line].strip().split()[0])
            fuel_moisture = None
            if fuel_moisture_flag == 1:
                current_line += 1
                fuel_moisture = float(lines[current_line].strip().split()[0])

            # Read fuel height
            fuel_height_flag = 1
            if fuel_density_flag == 1:
                current_line += 1
                fuel_height_flag = int(lines[current_line].strip().split()[0])
            fuel_height = None
            if fuel_density_flag == 1 and fuel_moisture_flag == 1:
                current_line += 1
                fuel_height = float(lines[current_line].strip().split()[0])

            # Read size scale and patch/gap (Supported for v6 and above)
            if version in ("v6"):
                # Check if the next line is the ignition header
                next_line = lines[current_line + 1].strip()
                if next_line.startswith("! IGNITION LOCATIONS"):
                    # Trying to read a v5 file with v6 reader so throw error
                    raise ValueError(
                        "Invalid file version. Selected reader for QUIC-Fire v6, but file is v5."
                    )

                # Read size scale
                current_line += 1
                size_scale_flag = int(lines[current_line].strip().split()[0])
                size_scale = 0.0005
                if size_scale_flag == 1:
                    current_line += 1
                    size_scale = float(lines[current_line].strip().split()[0])

                # Read patch and gap
                current_line += 1
                patch_and_gap_flag = int(lines[current_line].strip().split()[0])
                patch_size = 0.0
                gap_size = 0.0
                if patch_and_gap_flag == 1:
                    current_line += 1
                    patch_size = float(lines[current_line].strip().split()[0])
                    current_line += 1
                    gap_size = float(lines[current_line].strip().split()[0])
            elif version == "v5":
                # The next line should be the ignition header. If it is not, then
                # the file is not a v5 file, and we should throw an error.
                next_line = lines[current_line + 1].strip()
                if not next_line.startswith("! IGNITION LOCATIONS"):
                    # Trying to read a v6 file with v5 reader so throw error
                    raise ValueError(
                        "Invalid file version. Selected reader for QUIC-Fire v5, but file is v6."
                    )

                # Set size scale and patch and gap to defaults
                size_scale_flag = 0
                size_scale = 0.0005
                patch_and_gap_flag = 0
                patch_size = 0.0
                gap_size = 0.0
            else:
                raise ValueError(f"Unsupported version: {version}")

            # Read ignition data
            current_line += 2  # skip ! IGNITION LOCATIONS header
            ignition_flag = int(lines[current_line].strip().split()[0])
            add_lines = {1: 4, 2: 6, 3: 5, 4: 0, 5: 0, 6: 0, 7: 0}
            add = add_lines.get(ignition_flag)
            ignition_params = []
            current_line += 1
            for i in range(current_line, current_line + add):
                ignition_line = float(lines[i].split()[0].strip())
                ignition_params.append(ignition_line)
            if ignition_flag == 1:
                x_min, y_min, x_length, y_length = ignition_params
                ignition = RectangleIgnition(
                    x_min=x_min, y_min=y_min, x_length=x_length, y_length=y_length
                )
            elif ignition_flag == 2:
                x_min, y_min, x_length, y_length, x_width, y_width = ignition_params
                ignition = SquareRingIgnition(
                    x_min=x_min,
                    y_min=y_min,
                    x_length=x_length,
                    y_length=y_length,
                    x_width=x_width,
                    y_width=y_width,
                )
            elif ignition_flag == 3:
                x_min, y_min, x_length, y_length, ring_width = ignition_params
                ignition = CircularRingIgnition(
                    x_min=x_min,
                    y_min=y_min,
                    x_length=x_length,
                    y_length=y_length,
                    ring_width=ring_width,
                )
            else:
                ignition = Ignition(ignition_flag=ignition_flag)

            current_line += add
            ignitions_per_cell = int(lines[current_line].strip().split()[0])
            current_line += 1

            # Read firebrands
            # current_line = ! FIREBRANDS
            current_line += 1  # header
            firebrand_flag = int(lines[current_line].strip().split()[0])
            current_line += 1

            # Read output flags
            # current_line = !OUTPUT_FILES
            eng_to_atm_out = int(lines[current_line + 1].strip().split()[0])
            react_rate_out = int(lines[current_line + 2].strip().split()[0])
            fuel_dens_out = int(lines[current_line + 3].strip().split()[0])
            qf_wind_out = int(lines[current_line + 4].strip().split()[0])
            qu_wind_inst_out = int(lines[current_line + 5].strip().split()[0])
            qu_wind_avg_out = int(lines[current_line + 6].strip().split()[0])
            # ! Output plume trajectories
            fuel_moist_out = int(lines[current_line + 8].strip().split()[0])
            mass_burnt_out = int(lines[current_line + 9].strip().split()[0])
            firebrand_out = int(lines[current_line + 10].strip().split()[0])
            emissions_out = int(lines[current_line + 11].strip().split()[0])
            radiation_out = int(lines[current_line + 12].strip().split()[0])
            surf_eng_out = int(lines[current_line + 13].strip().split()[0])
            # ! AUTOKILL
            auto_kill = int(lines[current_line + 15].strip().split()[0])

        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QUIC_fire.inp"
                f"\nPlease check the file for correctness."
            )

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
            fuel_density_flag=fuel_density_flag,
            fuel_density=fuel_density,
            fuel_moisture_flag=fuel_moisture_flag,
            fuel_moisture=fuel_moisture,
            fuel_height_flag=fuel_height_flag,
            fuel_height=fuel_height,
            size_scale_flag=size_scale_flag,
            size_scale=size_scale,
            patch_and_gap_flag=patch_and_gap_flag,
            patch_size=patch_size,
            gap_size=gap_size,
            ignition=ignition,
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
            surf_eng_out=surf_eng_out,
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
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QFire_Bldg_Advanced_User_Inputs object from a directory
        containing a QFire_Bldg_Advanced_User_Inputs.inp file.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QFire_Bldg_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()
        try:
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
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QFire_Bldg_Advanced_User_Inputs.inp"
                f"\nPlease check the file for correctness."
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
    def from_file(cls, directory: str | Path, **kwargs):
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QFire_Plume_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()
        try:
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
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QFire_Plume_Advanced_User_Inputs.inp"
                f"\nPlease check the file for correctness."
            )


class QU_TopoInputs(InputFile):
    """
    Class representing the QU_TopoInputs.inp input file. This file
    contains advanced data pertaining to topography.

    Attributes
    ----------
    filename : str
        Path to the custom topo file (only used with option 5). Cannot be .bin. Use .dat or .inp
    topography : Topography
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
    smoothing_method : Literal[0, 1, 2]
        0 = none (default for idealized topo)
        1 = Blur
        2 = David Robinson’s method based on second derivative
    smoothing_passes : NonNegativeInt
        Number of smoothing passes. Real terrain MUST be smoothed
    sor_iterations : PositiveInt
        Number of SOR iteration to define background winds before starting the fire
    sor_cycles : Literal[0, 1, 2, 3, 4]
        Number of times the SOR solver initial fields is reset to define
        background winds before starting the fire
    sor_relax : PositiveFloat
        SOR overrelaxation coefficient. Only used if there is topo.
    """

    name: str = "QU_TopoInputs"
    _extension: str = ".inp"
    filename: str = "topo.dat"
    topography: SerializeAsAny[Topography] = Topography(topo_flag=TopoFlags(0))
    smoothing_method: Literal[0, 1, 2] = 2
    smoothing_passes: NonNegativeInt = Field(le=500, default=500)
    sor_iterations: PositiveInt = Field(le=500, default=200)
    sor_cycles: Literal[0, 1, 2, 3, 4] = 4
    sor_relax: PositiveFloat = Field(le=2, default=0.9)

    @computed_field
    @property
    def _topo_lines(self) -> str:
        return str(self.topography)

    @classmethod
    def from_dict(cls, data: dict):
        if "topography" in data:
            data["topography"] = serialize_topography(data["topography"])
        return cls(**data)

    @classmethod
    def from_file(cls, directory: str | Path, **kwargs):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "QU_TopoInputs.inp", "r") as f:
            lines = f.readlines()

        try:
            # Line 0 is Header
            filename = str(lines[1].strip())
            # Get topo lines
            topo_flag = int(lines[2].strip().split()[0])
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
                topo_params.append(float(lines[i].strip().split()[0]))
            if topo_flag == 1:
                x_hilltop, y_hilltop, elevation_max, elevation_std = topo_params
                topography = GaussianHillTopo(
                    x_hilltop=int(x_hilltop),
                    y_hilltop=int(y_hilltop),
                    elevation_max=int(elevation_max),
                    elevation_std=elevation_std,
                )
            elif topo_flag == 2:
                max_height, location_param = topo_params
                topography = HillPassTopo(
                    max_height=int(max_height), location_param=location_param
                )
            elif topo_flag == 3:
                slope_axis, slope_value, flat_fraction = topo_params
                topography = SlopeMesaTopo(
                    slope_axis=int(slope_axis),
                    slope_value=slope_value,
                    flat_fraction=flat_fraction,
                )
            elif topo_flag == 4:
                (
                    x_start,
                    y_center,
                    slope_value,
                    canyon_std,
                    vertical_offset,
                ) = topo_params
                topography = CanyonTopo(
                    x_location=int(x_start),
                    y_location=int(y_center),
                    slope_value=slope_value,
                    canyon_std=canyon_std,
                    vertical_offset=vertical_offset,
                )
            elif topo_flag == 6:
                x_location, y_location, radius = topo_params
                topography = HalfCircleTopo(
                    x_location=int(x_location),
                    y_location=int(y_location),
                    radius=radius,
                )
            elif topo_flag == 7:
                period, amplitude = topo_params
                topography = SinusoidTopo(period=period, amplitude=amplitude)
            elif topo_flag == 8:
                aspect, height = topo_params
                topography = CosHillTopo(aspect=aspect, height=height)
            else:
                topography = Topography(topo_flag=topo_flag)
            current_line = 3 + add
            # Smoothing and SOR
            smoothing_method = int(lines[current_line].strip().split()[0])
            smoothing_passes = int(lines[current_line + 1].strip().split()[0])
            sor_iterations = int(lines[current_line + 2].strip().split()[0])
            sor_cycles = int(lines[current_line + 3].strip().split()[0])
            sor_relax = float(lines[current_line + 4].strip().split()[0])
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QU_TopoInputs.inp"
                f"\nPlease check the file for correctness."
            )

        return cls(
            filename=filename,
            topography=topography,
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
        simulations. Defaults to 1.
    use_acw : Literal[0,1]
        Use Adaptive Computation Window (0=Disabled 1=Enabled). Defaults to 0.
    """

    name: str = "Runtime_Advanced_User_Inputs"
    _extension: str = ".inp"
    num_cpus: PositiveInt = Field(le=8, default=1)
    use_acw: Literal[0, 1] = 0

    @classmethod
    def from_file(cls, directory: str | Path, **kwargs):
        if isinstance(directory, str):
            directory = Path(directory)
        with open(directory / "Runtime_Advanced_User_Inputs.inp", "r") as f:
            lines = f.readlines()
        try:
            return cls(
                num_cpus=int(lines[0].strip().split()[0]),
                use_acw=int(lines[1].strip().split()[0]),
            )
        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: Runtime_Advanced_User_Inputs.inp"
                f"\nPlease check the file for correctness."
            )


class QU_movingcoords(InputFile):
    """
    Class representing the QU_movingcoords.inp input file.
    This is a QUIC legacy file that is not modified for QUIC-Fire use.
    """

    name: str = "QU_movingcoords"
    _extension: str = ".inp"


class QP_buildout(InputFile):
    """
    Class representing the QU_buildout.inp input file.
    This is a QUIC legacy file that is not modified for QUIC-Fire use.
    """

    name: str = "QP_buildout"
    _extension: str = ".inp"


class QU_metparams(InputFile):
    """
    Class representing the QU_metparams.inp input file.
    This file contains information about wind profiles.

    Attributes
    ----------
    site_names : list[str]
        List of site names. Must be the same length as file_names.
    file_names : list[str]
        List of file names. Must be the same length as site_names.
    """

    name: str = "QU_metparams"
    _extension: str = ".inp"
    site_names: list[str] = Field(min_length=1)
    file_names: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_site_and_field_name_lengths(self):
        if len(self.site_names) != len(self.file_names):
            raise ValueError("site_names and file_names must be the same length")
        return self

    @computed_field
    @property
    def num_sensors(self) -> PositiveInt:
        return len(self.site_names)

    @computed_field
    @property
    def _sensor_lines(self) -> str:
        sensor_lines = []
        for i in range(self.num_sensors):
            sensor_name = self.site_names[i]
            file_name = self.file_names[i]
            line = f"{sensor_name} !Site Name\n!File name\n{file_name}"
            sensor_lines.append(line)

        return "\n".join(sensor_lines)

    @classmethod
    def from_file(cls, directory: str | Path, **kwargs):
        """
        Initializes a QU_metparams object from a directory containing a
        QU_metparams.inp file.

        Parameters
        ----------
        directory : str | Path
            Directory containing the QU_metparams.inp file

        Returns
        -------
        QU_metparams
            Initialized QU_metparams object

        Raises
        ------
        ValueError
            If the file cannot be parsed correctly
        """
        if isinstance(directory, str):
            directory = Path(directory)

        with open(directory / "QU_metparams.inp", "r") as f:
            lines = f.readlines()

        try:
            # Get number of sensors from file
            num_sensors = int(lines[2].strip().split("!")[0])

            site_names = []
            file_names = []

            # Parse sensor information
            current_line = 4  # Start at first sensor entry
            for _ in range(num_sensors):
                # Get site name
                site_name = lines[current_line].strip().split("!")[0].strip()
                site_names.append(site_name)

                # Skip "!File name" line
                current_line += 1

                # Get file name
                file_name = lines[current_line + 1].strip()
                file_names.append(file_name)

                current_line += 2  # Move to next sensor block

            return cls(site_names=site_names, file_names=file_names)

        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: QU_metparams.inp\n"
                f"Please check the file for correctness.\n"
                f"Original error: {str(e)}"
            ) from e


class WindSensor(InputFile):
    """
    Class representing a sensor*.inp input file.
    This file contains information on winds, and serves as the
    primary source for wind speed(s) and direction(s).
    Multiple sensor*.inp files may be created.

    Attributes
    ----------
    wind_times : NonNegativeFloat | list(NonNegativeFloat)
        Time in seconds since the start of the fire for each wind shift.
    wind_speeds : PositiveFloat | list(PositiveFloat)
        Wind speed or list of wind speeds (m/s)
    wind_directions : NonNegativeInt < 360 | list(NonNegativeInt < 360)
        Wind direction or list of directions (degrees). Use 0° for North
    sensor_heights : PositiveFloat | list(PositiveFloat)
        Wind measurement height (m). Default is 6.1m (20ft). If a scalar is
        provided, it will be used for all wind_times.
    x_location : PositiveInt
        Location of the sensor in the x-direction
    y_location : PositiveInt
        Location of the sensor in the y-direction
    """

    name: str = Field(..., min_length=1)
    _extension: str = ".inp"
    wind_times: Union[NonNegativeInt, List[NonNegativeInt]]
    wind_speeds: Union[PositiveFloat, List[PositiveFloat]]
    wind_directions: Union[NonNegativeInt, List[NonNegativeInt]]
    sensor_heights: Union[PositiveFloat, List[PositiveFloat]] = 6.1
    x_location: NonNegativeFloat = 1.0
    y_location: NonNegativeFloat = 1.0

    @model_validator(mode="after")
    def validate_reserved_file_name(self):
        if self._filename in RESERVED_FILE_NAMES:
            raise ValueError(
                f"File name '{self._filename}' is reserved and cannot be used for a WindSensor."
            )
        return self

    @model_validator(mode="after")
    def validate_wind_inputs(self):
        """
        Validate wind inputs:
        1. Ensure wind_times, wind_speeds, and wind_directions are lists.
        2. Ensure all wind-related lists have equal lengths.
        """
        # Ensure all wind inputs are lists
        if isinstance(self.wind_times, (float, int)):
            self.wind_times = [self.wind_times]
        if isinstance(self.wind_speeds, (float, int)):
            self.wind_speeds = [self.wind_speeds]
        if isinstance(self.wind_directions, (float, int)):
            self.wind_directions = [self.wind_directions]

        # Validate that all lists have equal lengths
        if not (
            len(self.wind_times) == len(self.wind_speeds) == len(self.wind_directions)
        ):
            raise ValueError(
                "WindSensor: 'wind_times', 'wind_speeds', and 'wind_directions' must have the same length."
            )

        # Validate that wind_times is sorted
        for i in range(len(self.wind_times) - 1):
            if self.wind_times[i] > self.wind_times[i + 1]:
                raise ValueError("Wind times values must be sorted in ascending order")

        return self

    @model_validator(mode="after")
    def validate_sensor_heights(self):
        # if sensor height is a scalar make it a list of len(wind_times)
        if isinstance(self.sensor_heights, (float, int)):
            self.sensor_heights = [self.sensor_heights] * len(self.wind_times)

        # Validate that sensor_heights is a list of the same length as wind_times
        if len(self.sensor_heights) != len(self.wind_times):
            raise ValueError(
                "WindSensor: 'sensor_heights' must have the same length as 'wind_times'."
            )

        return self

    @computed_field
    @property
    def _wind_lines(self) -> str:
        location_lines = (
            f"{self.x_location} !X coordinate (meters)\n"
            f"{self.y_location} !Y coordinate (meters)"
        )
        windshifts = []
        for i, wind_time in enumerate(self.wind_times):
            sensor_height = self.sensor_heights[i]
            wind_speed = self.wind_speeds[i]
            wind_direction = self.wind_directions[i]
            shift = (
                f"\n{wind_time} !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
                f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
                f"0.1 !site zo\n"
                f"0. ! 1/L (default = 0)\n"
                f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
                f"{sensor_height} {wind_speed} {wind_direction}"
            )
            windshifts.append(shift)
        wind_lines = "".join(windshifts)

        return location_lines + wind_lines

    @classmethod
    def from_file(cls, directory: str | Path, sensor_name: str):
        """
        Initializes a WindSensor object from a directory containing a
        sensor .inp file.

        Parameters
        ----------
        directory: str | Path
            Directory containing the sensor .inp file.
        sensor_name: str
            Name of the sensor to read. A ".inp" string is appended to the name
            to get the file name.

        Returns
        -------
        WindSensor
            Initialized WindSensor object.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        sensor_file = sensor_name + ".inp"
        with open(directory / sensor_file, "r") as f:
            lines = f.readlines()

        wind_times = []
        sensor_heights = []
        wind_speeds = []
        wind_directions = []

        try:
            sensor_name = str(lines[0].strip().split("!")[0].strip())
            x_location = float(lines[4].strip().split("!")[0].strip())
            y_location = float(lines[5].strip().split("!")[0].strip())

            # while there are another 6 lines to read
            for i in range(6, len(lines), 6):
                if lines[i].strip() == "":
                    break
                wind_times.append(float(lines[i].split(" ")[0]))
                triplet_line = lines[i + 5].replace("\t", " ").strip()
                sensor_heights.append(float(triplet_line.split(" ")[0]))
                wind_speeds.append(float(triplet_line.split(" ")[1]))
                wind_directions.append(int(triplet_line.split(" ")[2]))

        except IndexError:
            raise ValueError(
                f"Error parsing {cls.__name__} from file: {sensor_file}"
                f"\nPlease check the file for correctness."
            )

        return cls(
            name=sensor_name,
            sensor_heights=sensor_heights,
            wind_times=wind_times,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            x_location=x_location,
            y_location=y_location,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        x_location: float,
        y_location: float,
        sensor_height: float,
        time_column_name: str = "wind_times",
        speed_column_name: str = "wind_speeds",
        direction_column_name: str = "wind_directions",
    ) -> "WindSensor":
        """
        Creates a WindSensor object from a pandas DataFrame containing wind data.

        Parameters
        ----------
        df : DataFrame
            Pandas DataFrame containing wind data with columns for times,
            speeds, and directions.
        name : str, optional
            Name to assign to the sensor.
        x_location : float
            Location of the wind sensor in the x-direction in meters
        y_location : float
            Location of the wind sensor in the y-direction in meters
        sensor_height : float
            Height of the wind sensor in meters
        time_column_name : str, optional
            Name of column containing wind times. Defaults to "wind_times"
        speed_column_name : str, optional
            Name of column containing wind speeds. Defaults to "wind_speeds"
        direction_column_name : str, optional
            Name of column containing wind directions. Defaults to "wind_directions"

        Returns
        -------
        WindSensor
            Initialized WindSensor object with data from the DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> winds_df = pd.read_csv("winds.csv")
        >>> sensor = WindSensor.from_dataframe(
        ...     df, sensor_name="sensor_csv", x_location=50, y_location=50, sensor_height=6.1
        ... )
        """
        # Extract values from DataFrame
        wind_times = df[time_column_name].tolist()
        wind_speeds = df[speed_column_name].tolist()
        wind_directions = df[direction_column_name].tolist()

        return cls(
            name=name,
            wind_times=wind_times,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            sensor_heights=sensor_height,
            x_location=x_location,
            y_location=y_location,
        )


class WindSensorArray(BaseModel):
    """
    Class containing all WindSensor input files and shared attributes.

    Attributes
    ----------
    sensor_array : list(WindSensor)
        List of all WindSensor input files managed by the WindSensorArray.
    """

    sensor_array: List[WindSensor] = Field(default_factory=list_default_factory)

    @model_validator(mode="after")
    def validate_unique_sensor_names(self):
        """
        Validate that all sensor names are unique.
        """
        sensor_names = [sensor.name for sensor in self.sensor_array]
        if len(sensor_names) != len(set(sensor_names)):
            raise ValueError("All sensor names must be unique.")
        return self

    @computed_field
    @property
    def wind_times(self) -> list:
        """
        Creates a global wind times list by combining the wind times lists of each sensor.
        """
        list_times = []
        for sensor in self.sensor_array:
            list_times.extend(sensor.wind_times)
        return sorted(set(list_times))

    def __len__(self):
        return len(self.sensor_array)

    def __iter__(self):
        return iter(self.sensor_array)

    def __getitem__(self, item):
        return self.sensor_array[item]

    @classmethod
    def from_file(cls, directory: str | Path):
        """
        Initializes a WindSensorArray object from a directory by reading sensor names
        and files from QU_metparams.inp.

        Parameters
        ----------
        directory : str | Path
            Directory containing the QU_metparams.inp and sensor files

        Returns
        -------
        WindSensorArray
            Initialized WindSensorArray object containing all wind sensors

        Raises
        ------
        FileNotFoundError
            If QU_metparams.inp or any sensor file is not found
        ValueError
            If there is an error parsing the sensor files
        """
        if isinstance(directory, str):
            directory = Path(directory)

        # Read QU_metparams to get sensor names and files
        try:
            qu_metparams = QU_metparams.from_file(directory)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find QU_metparams.inp in directory: {directory}"
            )
        except ValueError as e:
            raise ValueError(f"Error parsing QU_metparams.inp: {str(e)}")

        # Initialize empty sensor array
        sensor_array = []

        # Read each sensor file specified in QU_metparams
        for site_name, file_name in zip(
            qu_metparams.site_names, qu_metparams.file_names
        ):
            # Extract sensor name without .inp extension
            sensor_name = Path(file_name).stem

            try:
                sensor = WindSensor.from_file(directory, sensor_name)
                sensor_array.append(sensor)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find sensor file {file_name} in directory: {directory}"
                )
            except ValueError as e:
                raise ValueError(f"Error parsing sensor file {file_name}: {str(e)}")

        return cls(sensor_array=sensor_array)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        """
        Convert the object to a dictionary, excluding attributes that start
        with an underscore.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return {"sensor_array": [sensor.to_dict() for sensor in self.sensor_array]}

    def to_file(self, directory: Path | str, version: str = "latest"):
        if isinstance(directory, str):
            directory = Path(directory)

        for sensor in self.sensor_array:
            sensor.to_file(directory, version)


def _validate_and_return_version(version: str) -> str:
    if version not in ("latest", "v5", "v6"):
        raise ValueError(
            f"Version {version} is not supported. Supported versions: 'latest', 'v5', 'v6'"
        )
    return LATEST_VERSION if version == "latest" else version
