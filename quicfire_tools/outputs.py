"""
Module for converting QUIC-Fire output files to duck array data formats.
"""

from __future__ import annotations

# Core imports
import re
from pathlib import Path

# Internal imports
from quicfire_tools.inputs import SimulationInputs

# External imports
import zarr
import numpy as np
import dask.array as da
from numpy import ndarray
from shutil import rmtree
from netCDF4 import Dataset

FUELS_OUTPUTS = {
    "fire-energy_to_atmos": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "en2atmos",
        "output_times": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Energy released to the atmosphere that generates "
        "buoyant plumes",
        "units": "kW",
    },
    "fire-reaction_rate": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Rate of reaction of fuel",
        "units": "kg/m^3/s",
    },
    "fuels-dens": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel density",
        "units": "kg/m^3",
    },
    "fuels-moist": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel moisture content",
        "units": "g water / g air",
    },
    "groundfuelheight": {
        "file_format": "gridded",
        "dimensions": ["y", "x"],
        "grid": "fire",
        "output_times": None,
        "delimiter": None,
        "extension": ".bin",
        "description": "2D array with initial fuel height in the ground layer",
        "units": "m",
    },
    "mburnt_integ": {
        "file_format": "gridded",
        "dimensions": ["y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "2D file containing the percentage of mass burnt for each (i,j) "
        "location on the fire grid (vertically integrated)",
        "units": "%",
    },
    "surfEnergy": {
        "file_format": "gridded",
        "dimensions": ["y", "x"],
        "grid": "fire",
        "output_times": "surf_eng",
        "delimiter": "",
        "extension": ".bin",
        "description": "Array contains total energy output by each surface cell "
        "on a per second interval.",
        "units": "kW/m^2",
    },
}
THERMAL_RADIATION_OUTPUTS = {
    "thermaldose": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
        "collocated with fuel (for health effects).",
        "units": "(kW/m^2)^(4/3)s",
    },
    "thermalradiation": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
        "collocated with fuel (for health effects).",
        "units": "kW/m^2",
    },
}
QF_WIND_OUTPUTS = {
    "windu": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire u-components, cell centered ",
        "units": "m/s",
    },
    "windv": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire v-components, cell centered ",
        "units": "m/s",
    },
    "windw": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire w-components, cell centered ",
        "units": "m/s",
    },
}
QU_WIND_OUTPUTS = {
    "qu_windu": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind",
        "delimiter": "",
        "extension": ".bin",
        "description": "Instantaneous wind component in the x-direction, cell centered ",
        "units": "m/s",
    },
    "qu_windv": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind",
        "delimiter": "",
        "extension": ".bin",
        "description": "Instantaneous wind component in the y-direction, cell centered ",
        "units": "m/s",
    },
    "qu_windw": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind",
        "delimiter": "",
        "extension": ".bin",
        "description": "Instantaneous wind component in the x-direction, cell centered ",
        "units": "m/s",
    },
    "qu_windu_ave": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind_ave",
        "delimiter": "",
        "extension": ".bin",
        "description": "Averaged wind component in the x-direction, cell centered ",
        "units": "m/s",
    },
    "qu_windv_ave": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind_ave",
        "delimiter": "",
        "extension": ".bin",
        "description": "Averaged wind component in the y-direction, cell centered ",
        "units": "m/s",
    },
    "qu_windw_ave": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind_ave",
        "delimiter": "",
        "extension": ".bin",
        "description": "Averaged wind component in the z-direction, cell centered ",
        "units": "m/s",
    },
    "qu_wplume": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "quic",
        "output_times": "wind",
        "delimiter": "",
        "extension": ".bin",
        "description": "Instantaneous wind component in the z-direction generated "
        "by the fire, cell centered ",
        "units": "m/s",
    },
}
EMISSIONS_OUTPUTS = {
    "co_emissions": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of CO emitted between two emission file output"
        "times in grams",
        "units": "g",
    },
    "pm_emissions": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of PM2.5 emitted between two emission file output"
        "times in grams",
        "units": "g",
    },
    "water_emissions": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of water emitted between two emission file output"
        "times (fuel moisture + water from combustion) in grams",
        "units": "g",
    },
    "emissions_distribution": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "output_times": "emis_rad",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Size of PM2.5 emitted between two emission file output"
        "times in microns",
        "units": "g",
    },
}
OUTPUTS_MAP = {
    **FUELS_OUTPUTS,
    **THERMAL_RADIATION_OUTPUTS,
    **QF_WIND_OUTPUTS,
    **QU_WIND_OUTPUTS,
    **EMISSIONS_OUTPUTS,
}


class OutputFile:
    """
    A class representing a single output file. This class provides a common
    interface for processing and retrieving data from output files.

    Attributes
    ----------
    name: str
        The name of the output file this object represents.
    file_format: str
        The format of the output file. Valid options are "gridded" and
        "compressed".
    dimensions: list[str]
        The dimensions of the output file as ["z", "y", "x"] or ["y", "x"].
    shape: tuple
        The shape of the output file array data as (time, nz, ny, nx).
    grid: str
        The grid type of the output file. Valid options are "fire" and "quic".
    delimiter: str
        The delimiter used in the output file name.
    extension: str
        The file extension of the output file.
    description: str
        A description of the output file.
    units: str
        The units of the output file.
    filepaths: list[Path]
        A list of file paths for each timestep.
    times: list[int]
        A list of times corresponding to the timesteps.
    """

    def __init__(
        self,
        name: str,
        file_format: str,
        dimensions: list[str],
        shape: tuple,
        grid: str,
        output_times: str,
        delimiter: str,
        extension: str,
        description: str,
        units: str,
        times: list[int],
        filepaths: list[Path],
        index_map=None,
    ):
        self.name = name
        self.file_format = file_format
        self.dimensions = dimensions
        self.shape = shape
        self.grid = grid
        self.output_times = output_times
        self.delimiter = delimiter
        self.extension = extension
        self.description = description
        self.units = units
        self.times = times
        self.filepaths = filepaths
        function_mappings = {
            "gridded": _process_gridded_bin,
            "compressed": _process_compressed_bin,
        }
        self._output_function = function_mappings.get(self.file_format)
        if self._output_function is None:
            raise ValueError(f"Unknown output format: {self.file_format}")
        self._compressed_index_map = (
            index_map if (self.file_format == "compressed") else None
        )

    def __eq__(self, other):
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()
        for key in self_dict.keys():
            try:
                if self_dict[key] != other_dict[key]:
                    return False
            except KeyError:
                return False
            except ValueError:
                if np.any(self_dict[key] != other_dict[key]):
                    return False
        return True

    def to_numpy(self, timestep: int | list[int] | range = None) -> np.ndarray:
        """
        Return a numpy array for the given output and timestep(s) with shape
        (time, nz, ny, nx). If timestep is None, then all timesteps are
        returned.

        Parameters
        ----------
        timestep: int | list[int]
            The timestep(s) to return. If None, then all timesteps are returned.

        Returns
        -------
        np.ndarray
            A 4D numpy array with shape (time, nz, ny, nx) containing the
            output data.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs", 50, 100, 100)
        >>> fire_energy = outputs.get_output("fire-energy_to_atmos")
        >>> # Get all timesteps for the fire-energy_to_atmos output
        >>> fire_energy_all = fire_energy.to_numpy()
        >>> # Get the first timestep for the fire-energy_to_atmos output
        >>> fire_energy_slice = fire_energy.to_numpy(timestep=0)
        """
        selected_files = self._select_files_based_on_timestep(timestep)
        return self._get_multiple_timesteps(selected_files)

    def to_netcdf(self, directory: str | Path, timestep: int | list[int] = None):
        """
        Write a netCDF file for the given output and timestep(s) with dimensions
        (time, nz, ny, nx). If timestep is None, then all timesteps are
        returned.

        Parameters
        ----------
        directory: str | Path
            The path to the folder where netCDF file will be written.
        timestep: int | list[int] | None
            The timestep(s) to write. If None, then all timesteps are written.

        Builds
        -------
        netCDF file to disk

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs", 50, 100, 100)
        >>> fire_energy = outputs.get_output("fire-energy_to_atmos")
        >>> out_dir = Path(path/to/output/dir)
        >>> # Get all timesteps for the fire-energy_to_atmos output
        >>> fire_energy_all = fire_energy.to_netcdf(directory = out_dir)
        >>> # Get the first timestep for the fire-energy_to_atmos output
        >>> fire_energy_slice = fire_energy.to_netcdf(directory = out_dir, timestep=0)
        """
        if isinstance(directory, str):
            directory = Path(directory)

        if not directory.exists():
            directory.mkdir(parents=True)

        dataset = Dataset(directory / f"{self.name}.nc", "w", format="NETCDF4")
        dataset.title = self.name
        dataset.subtitle = self.description

        dataset.createDimension("nz", self.shape[1])
        dataset.createDimension("ny", self.shape[2])
        dataset.createDimension("nx", self.shape[3])
        dataset.createDimension("time", self.shape[0])

        dataset_nz = dataset.createVariable("z", np.int64, ("nz",))
        dataset_ny = dataset.createVariable("y", np.int64, ("ny",))
        dataset_nx = dataset.createVariable("x", np.int64, ("nx",))
        dataset_time = dataset.createVariable("timestep", np.int64, ("time",))

        dataset_time[:] = np.array(self.times)
        dataset_nz[:] = range(self.shape[0])
        # julia scales the horizontal resolution to meters using dx and dy
        dataset_ny[:] = range(self.shape[1])
        dataset_nx[:] = range(self.shape[2])

        output = dataset.createVariable(
            self.name, np.float32, ("timestep", "nz", "ny", "nx")
        )
        output.units = self.units

        selected_files = self._select_files_based_on_timestep(timestep)
        output[:, :, :, :] = self._get_multiple_timesteps(selected_files)
        dataset.close()
        return

    def _select_files_based_on_timestep(
        self, timestep: int | list[int] | range | None
    ) -> list[Path]:
        """Return files selected based on timestep."""
        if timestep is None:
            return self.filepaths
        if isinstance(timestep, int):
            timestep = [timestep]
        try:
            return [self.filepaths[ts] for ts in timestep]
        except IndexError:
            raise ValueError(f"Invalid timestep: {timestep}")

    def _get_single_timestep(self, output_file: Path) -> np.ndarray:
        """Return a numpy array for the given output file."""
        return self._output_function(
            output_file, self.shape, self._compressed_index_map
        )

    def _get_multiple_timesteps(self, output_files: list[Path]) -> np.ndarray:
        """Return a numpy array for the given output files."""
        arrays = [self._get_single_timestep(of) for of in output_files]
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)


class SimulationOutputs:
    """
    A class responsible for managing and processing simulation outputs,
    including validation, extraction, and organization of data from output
    files. This class facilitates the retrieval of data in various formats
    and can return a numpy array for a specific output, or write the data to a
    zarr file.

    Attributes
    ----------
    output_directory: Path
        The path to the directory containing the simulation outputs. QUIC-Fire
        defaults this to the "Output" directory in the same directory as the
        simulation input files, but that does not have to be the case for
        quicfire-tools. The directory must contain the "fire_indexes.bin" and
        "grid.bin" files.
    fire_nz: int
        The number of vertical cells in the fire grid.
    ny: int
        The number of cells in the y-direction.
    nx: int
        The number of cells in the x-direction.
    dy: float
        The grid spacing in the y-direction (m).
    dx: float
        The grid spacing in the x-direction (m).

    """

    def __init__(
        self,
        output_directory: Path | str,
        fire_nz: int,
        ny: int,
        nx: int,
        dy: float,
        dx: float,
    ):
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)
        self._validate_output_dir(output_directory)

        # Assign attributes
        self.output_directory = output_directory
        self.ny = ny
        self.nx = nx
        self.dy = dy
        self.dx = dx
        self.fire_nz = fire_nz

        # Get grid information from grid.bin and fire_indexes.bin
        self._fire_indexes = _process_fire_indexes(
            output_directory / "fire_indexes.bin"
        )
        self.quic_nz, self._quic_grid, self.en2atmos_nz, self._en2atmos_grid = (
            _process_grid_info(output_directory / "grid.bin", ny, nx)
        )
        self.quic_dz = _get_resolution_from_coords(self._quic_grid)
        self.fire_dz = _get_resolution_from_coords(self._en2atmos_grid)

        # Build a list of present output files and their times
        self.outputs = {}
        self._build_output_files_map()
        self._output_names = list(self.outputs.keys())

    def __iter__(self):
        """Return an iterator for the object."""
        self._index = 0
        return self

    def __next__(self):
        """Return the next item in the iterator."""
        if self._index < len(self._output_names):
            output = self.outputs[self._output_names[self._index]]
            self._index += 1
            return output
        else:
            raise StopIteration

    def __eq__(self, other):
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()
        self_dict.pop("outputs")
        other_dict.pop("outputs")
        for key in self_dict.keys():
            self_value = self_dict[key]
            other_value = other_dict[key]
            try:
                if self_value != other_value:
                    return False
            except KeyError:
                return False
            except ValueError:
                if np.any(self_value != other_value):
                    return False
        for key in self.outputs.keys():
            try:
                if self.outputs[key] != other.outputs[key]:
                    return False
            except KeyError:
                return False
        return True

    @classmethod
    def from_simulation_inputs(
        cls, output_directory: Path | str, simulation_inputs: SimulationInputs
    ) -> "SimulationOutputs":
        """
        Create a SimulationOutputs object from a path to a QUIC-Fire "Output"
        directory and a SimulationInputs object.

        Parameters
        ----------
        output_directory: Path | str
            The path to the directory containing the simulation outputs. The
            directory must contain the "fire_indexes.bin" and "grid.bin" files.
            This is typically the "Output" directory in the same directory as
            the simulation input files, but that does not have to be the case
            for quicfire-tools.
        simulation_inputs: SimulationInputs
            The SimulationInputs object containing the simulation input data.

        Returns
        -------
        SimulationOutputs
            A SimulationOutputs object.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> inputs = qft.SimulationInputs.from_json("path/to/sim_inputs.json")
        >>> outputs = qft.SimulationOutputs.from_simulation_inputs("path/to/outputs", inputs)
        """
        return cls(
            output_directory,
            simulation_inputs.quic_fire.nz,
            simulation_inputs.qu_simparams.ny,
            simulation_inputs.qu_simparams.nx,
            simulation_inputs.qu_simparams.dy,
            simulation_inputs.qu_simparams.dx,
        )

    @staticmethod
    def _validate_output_dir(outputs_directory):
        """
        Validate the outputs directory by checking for the existence of the
        Output directory and the required files.
        """
        # Check if the outputs directory exists and is a directory
        if not outputs_directory.exists():
            raise FileNotFoundError(
                f"The directory {outputs_directory} does not exist."
            )
        elif not outputs_directory.is_dir():
            raise NotADirectoryError(
                f"The path {outputs_directory} is not a directory."
            )

        # Check for required files
        required_files = ["fire_indexes.bin", "grid.bin"]
        for f in required_files:
            if not (outputs_directory / f).exists():
                raise FileNotFoundError(f"Required file {f} not in outputs directory")

    def _build_output_files_map(self):
        """
        Build a key-value map of output files, where the key is the name of
        the output file and the value is an instantiated OutputFile object.
        """
        for key, attributes in OUTPUTS_MAP.items():
            output_files_list = self._get_list_output_paths(
                key, attributes["extension"]
            )

            # Check if any output files of the given type exist in the directory
            if output_files_list:
                shape = self._get_output_shape(attributes)
                times = []
                for file in output_files_list:
                    time = self._get_output_file_time(key, file)
                    times.append(time)

                self.outputs[key] = OutputFile(
                    name=key,
                    file_format=attributes["file_format"],
                    dimensions=attributes["dimensions"],
                    shape=shape,
                    grid=attributes["grid"],
                    delimiter=attributes["delimiter"],
                    extension=attributes["extension"],
                    description=attributes["description"],
                    units=attributes["units"],
                    times=times,
                    filepaths=output_files_list,
                    index_map=self._fire_indexes,
                )

    def _get_list_output_paths(self, name, ext) -> list[Path]:
        """
        Get a sorted list of output files in the Output/ directory for the
        given output name.
        """
        paths = list(self.output_directory.glob(f"{name}*{ext}"))
        paths.sort()
        return paths

    def _get_output_shape(self, attrs) -> tuple:
        grid = attrs["grid"]
        number_dimensions = len(attrs["dimensions"])

        if number_dimensions == 2:
            return 1, self.ny, self.nx
        elif number_dimensions == 3 and attrs["grid"] == "fire":
            return self.fire_nz, self.ny, self.nx
        elif number_dimensions == 3 and attrs["grid"] == "en2atmos":
            return self.en2atmos_nz, self.ny, self.nx
        elif number_dimensions == 3 and attrs["grid"] == "quic":
            return self.quic_nz, self.ny, self.nx
        else:
            raise ValueError(
                f"Invalid number of dimensions ({number_dimensions}) for {grid} grid"
            )

    @staticmethod
    def _get_output_file_time(output, fpath) -> int:
        """Get the time of the output file."""
        delimiter = OUTPUTS_MAP[output]["delimiter"]
        extension = OUTPUTS_MAP[output]["extension"]

        try:
            if delimiter == "":
                # Split on first numeric component
                name_parts = re.split(r"(\d+)", fpath.name)
                file_time = name_parts[1]
            else:
                file_time = fpath.name.split(delimiter)[-1].split(extension)[0]

            return int(file_time)

        except ValueError:  # No numeric component
            return 0

    def _validate_output(self, output: str) -> OutputFile:
        """Validate output."""
        try:
            output = self.outputs[output]
        except KeyError:
            raise ValueError(
                f"{output} is not a valid output. Valid "
                f"outputs are: {self.list_outputs()}"
            )
        return output

    def list_outputs(self) -> list[str]:
        """
        Return a list of available output names.

        Returns
        -------
        list[str]
            A list of available output names.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs")
        >>> outputs.list_outputs()
        ['fire-energy_to_atmos', 'fuels-dens', 'groundfuelheight', 'mburnt_integ']
        """
        return self._output_names

    def get_output(self, key) -> OutputFile:
        """
        Return the OutputFile object for the given output name.

        Parameters
        ----------
        key: str
            The name of the output to return.

        Returns
        -------
        OutputFile
            The OutputFile object for the given output name.

        Raises
        ------
        ValueError
            If the output name is not valid.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs")
        >>> fire_energy = outputs.get_output("fire-energy_to_atmos")
        >>> fire_energy
        <quicfire_tools.outputs.OutputFile object at 0x7f8b1c2b6d90>
        """
        return self._validate_output(key)

    def to_numpy(
        self, key: str, timestep: None | int | list[int] | range = None
    ) -> np.ndarray:
        """
        Returns a 4D numpy array for the given output and timestep(s) with shape
        (time, nz, ny, nx). If timestep is None, then all timesteps are
        returned.

        Parameters
        ----------
        key: str
            The name of the output to return.
        timestep: int | list[int] | None
            The timestep(s) to return. If None, then all timesteps are returned.

        Returns
        -------
        np.ndarray
            A 4D numpy array with shape (time, nz, ny, nx) containing the
            output data.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs", 50, 100, 100)
        >>> output_name = "fire-energy_to_atmos"
        >>>
        >>> # Get all timesteps for the fire-energy_to_atmos output
        >>> fire_energy_all = outputs.to_numpy(output_name)
        >>>
        >>> # Get the first timestep for the fire-energy_to_atmos output
        >>> fire_energy_first_time_step = outputs.to_numpy(output_name, timestep=0)
        """
        output = self._validate_output(key)
        return output.to_numpy(timestep)

    def to_dask(self, key: str) -> da.Array:
        """
        Returns a dask array for the given output with shape (time, nz, ny, nx).

        Parameters
        ----------
        key: str
            The name of the output to return.

        Returns
        -------
        da.Array
            A dask array with shape (time, nz, ny, nx) containing the output
            data with chunks of size (1, nz, ny, nx). The dask array is
            lazily evaluated, so users must call `.compute()` to retrieve the
            data.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs", 50, 100, 100)
        >>> fire_energy_da = outputs.to_dask("fire-energy_to_atmos")
        >>> fire_energy_np = fire_energy_da.compute()  # Retrieve the data
        """
        output = self._validate_output(key)

        # Create a dask array for the output file
        shape = (len(output.times), *output.shape)
        chunks = [1 if i == 0 else shape[i] for i in range(len(shape))]
        dask_array = da.zeros(shape, dtype=float, chunks=chunks)

        # Write each timestep to the dask array
        for time_step in range(len(output.times)):
            data = self.to_numpy(output.name, time_step)
            dask_array[time_step, ...] = data

        return dask_array

    def to_netcdf(
        self,
        directory: str | Path,
        outputs: str | list[str] = None,
    ):
        """
        Write QUIC-Fire output(s) to a netCDF file

        Parameters
        ----------
        directory: str | Path
            The path to the folder where netCDF file will be written.
        outputs: str
            The name or list of names of the output(s) to write to the netCDF file.
            If None, all output files present will be included.


        Builds
        -------
        netCDF file to disk

        TODO: create option for multiple output files?
        """
        if isinstance(directory, str):
            directory = Path(directory)

        if not directory.exists():
            directory.mkdir(parents=True)

        if isinstance(outputs, str):
            outputs = [outputs]
        selected_outputs = self._output_names if outputs is None else outputs

        # Initiate dataset
        dataset = Dataset(directory / f"simulation_outputs.nc", "w", format="NETCDF4")
        dataset.title = "QUIC-Fire Simulation Outputs"
        dataset.subtitle = f"Contains the following outputs: {selected_outputs}"

        # Create nx and ny dimensions and variables in root group
        dataset.createDimension("ny", self.ny)
        dataset.createDimension("nx", self.nx)
        root_ny = dataset.createVariable("y", np.int64, ("ny",))
        root_nx = dataset.createVariable("x", np.int64, ("nx",))
        root_ny[:] = range(self.ny)
        root_nx[:] = range(self.nx)

        # Find which grids and output times are needed
        output_info = self._get_output_info(selected_outputs)

        # Make groups for the different grids present in the data
        # There will always be at least one grid group present, ie the nz grid will never be in the root group
        if "fire" in output_info["grids"]:
            fire_grid = dataset.createGroup("fire_grid")
            fire_grid.createDimension("fire_nz", self.fire_nz)
            fire_grid_nz = fire_grid.createVariable("z", np.int64, ("fire_nz"))

        if "wind" in output_info["grids"]:
            quic_grid = dataset.createGroup("quic_grid")
            quic_grid.createDimension("quic_nz", self.quic_nz)
            quic_grid_nz = quic_grid.createVariable("z", np.int64, ("quic_nz"))

        if "eng2atm" in output_info["grids"]:
            eng2atm_grid = dataset.createGroup("eng2atm_grid")
            eng2atm_grid.createDimension("eng2atm_nz", self.en2atmos_nz)
            eng2atm_grid_nz = eng2atm_grid.createVariable("z", np.int64, ("eng2atm_nz"))

        # Check if the output times in the output files are the same
        identical_times = self._check_identical_times(selected_outputs)
        if identical_times is not None:
            dataset.createDimension("time", len(identical_times))
            root_times = dataset.createVariable("times", np.int64, ("time",))
            root_times[:] = identical_times
        else:
            """Set up different output time groups"""
            # And then by output time interval group
            if "emis_rad" in output_info["output_times"]:
                out_emis_rad = fire_grid.createGroup("emissions_and_radiation")
                out_emis_rad.createDimension("time_emis_rad", self.shape[0])
                out_emis_rad_time = out_emis_rad.createVariable(
                    "timestep_emis_rad", np.int64, ("time_emis_rad",)
                )

            if "surf_eng" in output_info["output_times"]:
                out_surf_eng = fire_grid.createGroup("surface_energy")
                out_surf_eng.createDimension("time_surf_eng", self.shape[0])
                out_surf_eng_time = out_surf_eng.createVariable(
                    "timestep_surf_eng", np.int64, ("time_surf_eng",)
                )

            if "wind" in output_info["output_times"]:
                out_wind = quic_grid.createGroup("instantaneous_wind")
                out_wind.createDimension("time_wind", self.shape[0])
                out_wind_time = out_wind.createVariable(
                    "timestep_wind", np.int64, ("time_wind",)
                )
            if "wind_ave" in output_info["output_times"]:
                out_wind_ave = quic_grid.createGroup("averaged_wind")
                out_wind_ave.createDimension("time_wind_ave", self.shape[0])
                out_wind_ave_time = out_wind_ave.createVariable(
                    "timestep_wind_ave", np.int64, ("time_wind_ave",)
                )

            if "fire" in output_info["output_times"]:
                if "fire" in output_info["grids"]:
                    out_fire = fire_grid.createGroup("fire_related")
                    out_fire.createDimension("time_fire", self.shape[0])
                    out_fire_time = out_fire.createVariable(
                        "timestep_fire", np.int64, ("time_fire",)
                    )
                if "eng2atm" in output_info["grids"]:
                    out_eng2atm = eng2atm_grid.createGroup("eng2atm")
                    out_eng2atm.createDimension("time_eng2atm", self.shape[0])
                    out_fire_time = out_eng2atm.createVariable(
                        "timestep_eng2atm", np.int64, ("time_eng2atm",)
                    )

        for out in selected_outputs:
            output = self._validate_output(out)
            new_var = dataset.createVariable(
                output.name, np.float32, ("timestep", "nz", "ny", "nx")
            )
            new_var.units = output.units

        selected_files = self._select_files_based_on_timestep(timestep)
        output[:, :, :, :] = self._get_multiple_timesteps(selected_files)
        dataset.close()

        return

    # ZCC Changes
    def to_zarr(
        self,
        fpath: str | Path,
        outputs: str | list[str] = None,
        over_write: bool = False,
    ):
        """
        Write the outputs to a zarr file.

        Parameters
        ----------
        fpath: str | Path
            The path to the folder where zarr files are written to be written.
        outputs: str | list[str]
            The name of the output(s) to write to the zarr file. If None, then
            all outputs are written to the zarr file.
        over_write: bool
            Whether or not to over_write zarr if it already exists

        Builds
        -------
        zarr files to disk
        """
        # Hard coding dimentions. Will want to add dimensions to one of the classes.
        dz = 1
        dy = 2
        dx = 2

        if isinstance(fpath, str):
            fpath = Path(fpath)

        if not fpath.exists():
            fpath.mkdir(parents=True)

        self.zarr_folder_path = fpath  # store folder location of zarrs to object

        # Get a list of outputs to write to the zarr file
        if outputs is None:
            outputs = self.list_outputs()
        elif isinstance(outputs, str):
            outputs = [outputs]

        # Write each output to the zarr file
        for output_name in outputs:
            # Get the output object and verify it exists
            output = self.get_output(output_name)

            # Build zarr path and store to output object
            zarr_path = fpath / (output_name + ".zarr")
            output.zarr_path = zarr_path  # store zarr_path to output object

            write_zarr_to_disk = True
            if zarr_path.exists():
                if not over_write:
                    write_zarr_to_disk = False
                    print(
                        "A dataset for {} already exists in the zarr file.\n".format(
                            output_name
                        ),
                        "Set over_write flag to True if you would like to rebuild the zarr.",
                    )
                else:
                    rmtree(zarr_path)

            if write_zarr_to_disk:
                # Create the zarr file
                zarr_file = zarr.open(str(zarr_path), mode="w")

                # Create a zarr dataset for the output
                shape = (len(output.times), *output.shape)
                chunks = [1 if i == 0 else shape[i] for i in range(len(shape))]

                # Output to zarr dataset
                DATA_NAME = "data"
                zarr_dataset = zarr_file.create_dataset(
                    DATA_NAME, shape=shape, chunks=chunks, dtype=float
                )

                # Metadata
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
                zarr_dataset.attrs["long_name"] = output_name
                zarr_dataset.attrs["units"] = output.units
                # Write each timestep to the output's zarr dataset
                for time_step in range(len(output.times)):
                    data = self.to_numpy(output_name, time_step)
                    zarr_dataset[time_step, ...] = data[0, ...]

                # Build datasets for coordinates
                # Build time coordinate
                zarr_dataset = zarr_file.create_dataset(
                    "time", shape=len(output.times), dtype=int
                )
                # Metadata
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["time"]
                zarr_dataset.attrs["long_name"] = "Time since start of simulation"
                zarr_dataset.attrs["units"] = "s"
                # Store Values
                zarr_dataset[...] = np.array(output.times)

                # Build z coordinate
                zarr_dataset = zarr_file.create_dataset(
                    "z", shape=output.shape[0], dtype=int
                )
                # Metadata
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["z"]
                zarr_dataset.attrs["long_name"] = (
                    "cell height: bottom of cell from ground"
                )
                zarr_dataset.attrs["units"] = "m"
                # Store Values
                zarr_dataset[...] = np.array(range(output.shape[0])) * dz

                # Build y coordinate
                zarr_dataset = zarr_file.create_dataset(
                    "y", shape=output.shape[1], dtype=int
                )
                # Metadata
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["y"]
                zarr_dataset.attrs["long_name"] = (
                    "Latitude: cell dist north from southern edge of domain"
                )
                zarr_dataset.attrs["units"] = "m"
                # Store Values
                zarr_dataset[...] = np.array(range(output.shape[1])) * dy

                # Build x coordinate
                zarr_dataset = zarr_file.create_dataset(
                    "x", shape=output.shape[2], dtype=int
                )
                # Metadata
                zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["x"]
                zarr_dataset.attrs["long_name"] = (
                    "Longitude: cell dist east from western edge of domain"
                )
                zarr_dataset.attrs["units"] = "m"
                # Store Values
                zarr_dataset[...] = np.array(range(0, output.shape[2])) * dx

                zarr.convenience.consolidate_metadata(zarr_path)
        return

    # #Anthony's method
    # def to_zarr(
    #     self, fpath: str | Path, outputs: str | list[str] = None
    # ) -> zarr.hierarchy.Group:
    #     """
    #     Write the outputs to a zarr file.

    #     Parameters
    #     ----------
    #     fpath: str | Path
    #         The path to the zarr file to be written.
    #     outputs: str | list[str]
    #         The name of the output(s) to write to the zarr file. If None, then
    #         all outputs are written to the zarr file.

    #     Returns
    #     -------
    #     zarr.hierarchy.Group
    #         The zarr file object.
    #     """
    #     if isinstance(fpath, str):
    #         fpath = Path(fpath)

    #     # Create the zarr file
    #     zarr_file = zarr.open(str(fpath), mode="w")

    #     # Get a list of outputs to write to the zarr file
    #     if outputs is None:
    #         outputs = self.list_available_outputs()
    #     elif isinstance(outputs, str):
    #         outputs = [outputs]

    #     # Write each output to the zarr file
    #     for output_name in outputs:
    #         # Get the output object and verify it exists
    #         output = self.get_output(output_name)

    #         # Create a zarr dataset for the output
    #         shape = (len(output.times), *output.shape)
    #         chunks = [1 if i == 0 else shape[i] for i in range(len(shape))]
    #         zarr_dataset = zarr_file.create_dataset(
    #             output_name, shape=shape, chunks=chunks, dtype=float
    #         )
    #         zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]

    #         # Write each timestep to the output's zarr dataset
    #         for time_step in range(len(output.times)):
    #             data = self.to_numpy(output_name, time_step)
    #             zarr_dataset[time_step, ...] = data[0, ...]

    #     return zarr_file

    def _check_identical_times(self, selected_outputs) -> list | None:
        identical_times = None
        all_the_same = True
        first_output = self.get_output(selected_outputs[0])
        if len(selected_outputs) > 1:
            for i in range(1, len(selected_outputs)):
                next_output = self.get_output(selected_outputs[i])
                if next_output.times == first_output.times:
                    all_the_same = False
        if all_the_same:
            identical_times = first_output.times
        return identical_times

    def _get_output_info(self, selected_outputs) -> dict:
        grids = []
        out_times = []
        for output in selected_outputs:
            info_dict = OUTPUTS_MAP.get(output)
            grids.append(info_dict["grid"])
            out_times.append(info_dict["output_times"])
            grids = list(set(grids))
            out_times = list(set(out_times))
        out_dict = {"grids": grids, "output_times": out_times}
        return out_dict


def _get_resolution_from_coords(coords: list[float]) -> float | list[float]:
    """
    Get the resolution from a list of coordinates. If the resolution is
    constant, then a single float is returned. If the resolution is not
    constant, then a list of floats is returned.
    """
    resolution = np.diff(coords)
    if np.all(resolution == resolution[0]):
        return resolution[0]
    else:
        return resolution


def _process_grid_info(path_to_grid_bin: str | Path, ny: int, nx: int):
    """
    Reads grid information from the grid.bin file and populates attributes
    for fire and quic grid sizes, as well as dx, dy, and dz.
    """
    if isinstance(path_to_grid_bin, str):
        path_to_grid_bin = Path(path_to_grid_bin)

    with open(path_to_grid_bin, "rb") as fid:
        # Read the number of bytes in the vertical QUIC grid
        num_bytes_quic_grid = np.fromfile(fid, dtype=np.int32, count=1)[0]

        # Get the number of cells in the vertical direction
        # num_cells * 32 bits = num_bytes * (8 bits / byte)
        quic_nz = int(num_bytes_quic_grid / 4) - 2

        # QUIC-URB grid bottom values
        quic_grid_bottom = np.fromfile(fid, dtype=np.float32, count=quic_nz + 2)

        np.fromfile(fid, dtype=np.int32, count=2)  # Header
        np.fromfile(fid, dtype=np.float32, count=quic_nz + 2)  # quic grid mid values
        np.fromfile(fid, dtype=np.int32, count=1)  # Header

        # Try and read a topo or a fire header
        test_header = np.fromfile(fid, dtype=np.int32, count=1)
        if len(test_header) == 0:
            raise ValueError("No fire or topo grids found in grid.bin file")

        # Check if topo grid information is present
        if test_header[0] == num_bytes_quic_grid:
            # Read sigma_bottom (same as quic_grid_bottom)
            np.fromfile(fid, dtype=np.float32, count=quic_nz + 2)

            np.fromfile(fid, dtype=np.int32, count=2)  # Header

            # Read sigma_mid (same as quic_grid_mid)
            np.fromfile(fid, dtype=np.float32, count=quic_nz + 2)

            np.fromfile(fid, dtype=np.int32, count=2)  # Header

            quic_grid_bottom_terrain_following = np.fromfile(
                fid, dtype=np.float32, count=ny * nx * (quic_nz + 2)
            )

            np.fromfile(fid, dtype=np.int32, count=2)  # Header

            quic_grid_mid_terrain_following = np.fromfile(
                fid, dtype=np.float32, count=ny * nx * (quic_nz + 2)
            )

            np.fromfile(fid, dtype=np.int32, count=2)  # Header

            quic_grid_volume_correction = np.fromfile(
                fid, dtype=np.float32, count=ny * nx
            )

            np.fromfile(fid, dtype=np.int32, count=1)  # Header

            fire_header = np.fromfile(fid, dtype=np.int32, count=1)[0]
        else:
            fire_header = test_header[0]

        num_eng_to_atmos_cells = np.fromfile(fid, dtype=np.int32, count=1)[0]
        np.fromfile(fid, dtype=np.int32, count=2)  # Header
        eng_to_amos_grid_bottom = np.fromfile(
            fid, dtype=np.float32, count=num_eng_to_atmos_cells + 1
        )

        ending_header = np.fromfile(fid, dtype=np.int32, count=1)

        _ = np.fromfile(fid, dtype=np.int32, count=4)  # Header

        return (
            quic_nz,
            quic_grid_bottom[:],
            num_eng_to_atmos_cells,
            eng_to_amos_grid_bottom[:],
        )


def _process_fire_indexes(path_to_fire_indexes_bin: str | Path) -> ndarray:
    """
    Process the `fire_indexes.bin` file to extract the necessary
    information to rebuild the 3D fields of fire and fuel related variables.
    Function returns a 2D ndarray of shape (num_cells, 3) representing the
    i, j, k indexes of each cell that contains data in the sparse (compressed)
    format.
    """
    if isinstance(path_to_fire_indexes_bin, str):
        path_to_fire_indexes_bin = Path(path_to_fire_indexes_bin)

    with open(path_to_fire_indexes_bin, "rb") as fid:
        np.fromfile(fid, dtype=np.int32, count=1)  # Header
        num_cells = np.fromfile(fid, dtype=np.int32, count=1)[0]
        np.fromfile(fid, dtype=np.int32, count=2)  # Header
        np.fromfile(fid, dtype=np.int32, count=1)  # max k index
        np.fromfile(fid, dtype=np.int32, count=2)  # Header
        np.fromfile(fid, dtype=np.int32, count=num_cells)  # unique cell IDs
        np.fromfile(fid, dtype=np.int32, count=2)  # Header

        # Initialize an array to hold the indices
        ijk = np.zeros((num_cells, 3), dtype=np.int32)

        # Loop over each dimension's indices (Like a sparse meshgrid)
        for i in range(0, 3):
            ijk[:, i] = np.fromfile(fid, dtype=np.int32, count=num_cells)

        # Convert to indices (numbered from 0)
        ijk -= 1

        return ijk


def _process_compressed_bin(filename, dim_zyx, *args) -> ndarray:
    """
    Converts the contents of a sparse .bin file to a dense array with shape
    (1, nz, ny, nx).

    The mapping from sparse to dense is guided by an index map provided as an
    argument. The index map must match the sparse data in the .bin file,
    defining the locations of nonzero values within the dense array.

    Parameters
    ----------
    filename : str | Path
        The name or path of the sparse .bin file to be processed.
    dim_zyx : tuple of int
        The number of z, y, x cells to store in the 3D array, given as a tuple
        of three integers.
    *args : tuple
        Additional arguments, specifically expecting the index map for mapping
        sparse indices to dense. The index map should be a 2D array with shape
        (N, 3), where N is the number of nonzero elements in the sparse data.

    Returns
    -------
    ndarray
        A dense 4D array of shape (1, nz, ny, nx) containing the sparse data.

    Raises
    ------
    IndexError
        If the index map (fire_indexes) is not provided in the arguments.
    ValueError
        If the index map (fire_indexes) is None.
    """
    try:
        fire_indexes = args[0]
    except IndexError:
        raise ValueError(
            "fire_indexes must be provided when processing " "compressed .bin files."
        )
    if fire_indexes is None:
        raise ValueError(
            "fire_indexes must be provided when processing " "compressed .bin files."
        )

    # Initialize the 3D array
    full_array = np.zeros([1, *dim_zyx], dtype=np.float32)
    array_3d = full_array[0, :, :, :]

    with open(filename, "rb") as fid:
        # Read header
        np.fromfile(fid, dtype=np.int32, count=1)

        # Read in the sparse values
        sparse_values = np.fromfile(fid, dtype=np.float32, count=fire_indexes.shape[0])

        # Map indices of the sparse data to the indices of the dense array
        indices = (
            fire_indexes[:, 2],  # k indices (z)
            fire_indexes[:, 1],  # j indices (y)
            fire_indexes[:, 0],  # i indices (x)
        )

        # Update the full array with the sparse indices
        array_3d[indices] = sparse_values

    full_array[0, ...] = array_3d
    return full_array


def _process_gridded_bin(filename, dims_zyx, *args) -> ndarray:
    """
    Converts the data stored in a gridded .bin file to an np array. The
    function takes in a 3D array of dimensions (nz, ny, nx) and returns an
    array of shape (1, nz, ny, nx).

    Parameters
    ----------
    filename : str | Path
        File path to the .bin file that contains the gridded data.
    dims_zyx : tuple
        A tuple of length 3 containing the number of z, y, and x cells in the
        3D array.
    *args : list
        Additional arguments that might be passed to the underlying slice
        processing function (_process_gridded_bin_slice).

    Returns
    -------
    ndarray
        A 4D array of shape (1, nz, ny, nx) containing the gridded data.
    """
    # Initialize the array
    array = np.zeros([1, *dims_zyx], dtype=np.float32)

    nz, ny, nx = dims_zyx
    with open(filename, "rb") as f:
        # Read header
        _ = np.fromfile(f, dtype=np.float32, count=1)

        # Read in gridded data
        for k in range(nz):
            array[0, k, :, :] = _process_gridded_bin_slice(f, ny, nx)
    return array.astype(float)


def _process_gridded_bin_slice(fid, ny, nx) -> ndarray:
    """
    Process a 2D slice from a gridded .bin file.

    This method reads a 2D slice of size (ny, nx) from the provided file
    object, and returns it as a np.float32 array.
    """
    # Read in gridded data
    plane = np.fromfile(fid, dtype=np.float32, count=ny * nx)
    return np.reshape(plane, (ny, nx))
