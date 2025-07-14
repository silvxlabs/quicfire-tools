"""
Module for converting QUIC-Fire output files to duck array data formats.
"""

from __future__ import annotations

# Core imports
import re
import json
from pathlib import Path
import importlib.resources

# Internal imports
from quicfire_tools.inputs import SimulationInputs

# External imports
import zarr
import numpy as np
import pyproj as proj
import dask.array as da
from numpy import ndarray
from netCDF4 import Dataset


OUTPUTS_DIR_PATH = (
    importlib.resources.files("quicfire_tools").joinpath("data").joinpath("outputs")
)
with open(OUTPUTS_DIR_PATH.joinpath("outputs.json")) as f:
    OUTPUTS_MAP = json.load(f)


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
        The shape of the output file array data as (nz, ny, nx).
    grid: str
        The grid type of the output file. Valid options are "fire" and "quic".
    x_coords: list[float]
        The x coordinates of the output file. Coordinates are in the UTM
        coordinate system with the origin at the southwest corner of the
        domain.
    y_coords: list[float]
        The y coordinates of the output file. Coordinates are in the UTM
        coordinate system with the origin at the southwest corner of the
        domain.
    z_coords: list[float]
        The z coordinates of the output file. Coordinates are in meters above
        the ground level.
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
        grid: str,
        x_coords: list[float],
        y_coords: list[float],
        z_coords: list[float],
        delimiter: str,
        extension: str,
        description: str,
        units: str,
        times: list[int],
        filepaths: list[Path],
        index_map=None,
        crs: str = None,
    ):
        self.name = name
        self.file_format = file_format
        self.dimensions = dimensions
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z_coords = z_coords
        self.crs = crs
        self.shape = (len(z_coords), len(y_coords), len(x_coords))
        self.grid = grid
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

        if timestep is None:
            times = self.times
        elif isinstance(timestep, int):
            times = [self.times[timestep]]
        else:
            times = [self.times[i] for i in timestep]

        dataset = Dataset(directory / f"{self.name}.nc", "w", format="NETCDF4")
        dataset.title = self.name
        dataset.subtitle = self.description

        # Time dimension
        dataset.createDimension("time", None)
        dataset_time = dataset.createVariable("time", np.int64, ("time",))
        dataset_time.long_name = "Time since start of simulation"
        dataset_time.units = "s"
        dataset_time[:] = np.array(times)

        # Z dimension
        dataset.createDimension("z", None)
        dataset_z = dataset.createVariable("z", np.int64, ("z",))
        dataset_z.long_name = "Cell height: bottom of cell from ground"
        dataset_z.units = "m"
        dataset_z[:] = self.z_coords

        # Y dimension
        dataset.createDimension("y", None)
        dataset_y = dataset.createVariable("y", np.int64, ("y",))
        dataset_y.long_name = "Northing: y-coordinate of bottom edge of the cell"
        dataset_y.units = "m"
        dataset_y[:] = self.y_coords

        # X dimension
        dataset.createDimension("x", None)
        dataset_x = dataset.createVariable("x", np.int64, ("x",))
        dataset_x.long_name = "Easting: x-coordinate of left edge of the cell"
        dataset_x.units = "m"
        dataset_x[:] = self.x_coords

        # Output variable
        output = dataset.createVariable(self.name, np.float32, ("time", "z", "y", "x"))
        output.long_name = self.description
        output.units = self.units

        selected_files = self._select_files_based_on_timestep(timestep)
        output[:, :, :, :] = self._get_multiple_timesteps(selected_files)

        dataset.close()
        return

    def to_zarr(
        self,
        fpath: str | Path,
        chunk_size: dict[str, int] = None,
        **kwargs,
    ) -> zarr.Group:
        """
        Write the outputs to a zarr file.

        Parameters
        ----------
        fpath: str | Path
            The path to the folder where zarr files are written to be written.
        chunk_size: dict[str, int]
            The chunk size for the zarr array. The dictionary may contain
            keys "time", "z", "y", and "x" with integer values for the chunk
            size in each dimension. The default chunk size is (1, nz, ny, nx).
            If a key is not present, then the chunk size defaults to the size
            of the dimension.
        **kwargs
            Additional keyword arguments to pass to the zarr array creation
            function.

        Returns
        -------
        zarr.Group
            The zarr group containing the output data.

        Examples
        --------
        >>> import quicfire_tools as qft
        >>> outputs = qft.SimulationOutputs("path/to/outputs", 50, 100, 100)
        >>> fire_energy = outputs.get_output("fire-energy_to_atmos")
        >>> # Write fire-energy_to_atmos output to zarr
        >>> zarr_group = fire_energy.to_zarr("path/to/zarr")
        >>> zarr_group["fuels-dens"]
        <zarr.core.Array '/fuels-dens' (2, 1, 200, 200) float64 read-only>
        >>> # Write multiple timesteps to each zarr chunk to improve I/O
        >>> zarr_group = fire_energy.to_zarr("path/to/zarr", chunk_size={"time": 100})
        >>> zarr_group["fuels-dens"].chunks
        (100, 1, 200, 200)
        """
        if isinstance(fpath, str):
            fpath = Path(fpath)

        if not fpath.exists():
            fpath.mkdir(parents=True)

        # Build the zarr file with directory storage
        zarr_path = fpath / (self.name + ".zarr")
        zroot = zarr.open(str(zarr_path), mode="w")
        zroot.attrs["crs"] = self.crs

        # Build the output data array. By default, the array has shape
        # (time, nz, ny, nx) and chunks of size (1, nz, ny, nx).
        shape = (len(self.times), *self.shape)
        if chunk_size is None:
            chunk_size = {}
        chunks = [
            chunk_size.get("time", 1),
            chunk_size.get("z", self.shape[0]),
            chunk_size.get("y", self.shape[1]),
            chunk_size.get("x", self.shape[2]),
        ]
        output_array = zroot.create_dataset(
            self.name, shape=shape, chunks=chunks, dtype=float, **kwargs
        )
        output_array.attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
        output_array.attrs["long_name"] = self.description
        output_array.attrs["units"] = self.units

        # Write each timestep to the output array
        for time_step in range(len(self.times)):
            data = self.to_numpy(time_step)
            output_array[time_step, ...] = data[0, ...]

        # Time dimension
        time_array = zroot.create_dataset("time", shape=len(self.times), dtype=int)
        time_array.attrs["long_name"] = "Time since start of simulation"
        time_array.attrs["units"] = "s"
        time_array[...] = self.times

        # Z dimension
        z_array = zroot.create_dataset("z", shape=self.shape[0], dtype=float)
        z_array.attrs["long_name"] = "Cell height: bottom of cell from ground"
        z_array.attrs["units"] = "m"
        z_array[...] = self.z_coords

        # Y dimension
        y_array = zroot.create_dataset("y", shape=self.shape[1], dtype=float)
        y_array.attrs[
            "long_name"
        ] = "Northing: cell distance north from southern edge of domain"
        y_array.attrs["units"] = "m"
        y_array[...] = self.y_coords

        # X dimension
        x_array = zroot.create_dataset("x", shape=self.shape[2], dtype=float)
        x_array.attrs[
            "long_name"
        ] = "Easting: cell distance east from western edge of domain"
        x_array.attrs["units"] = "m"
        x_array[...] = self.x_coords

        zarr.consolidate_metadata(zarr_path)

        return zroot


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
        The grid spacing in the y-direction (m). Defaults to 2m.
    dx: float
        The grid spacing in the x-direction (m). Defaults to 2m.
    utm_x: float
        UTM-x coordinates of the south-west corner of domain [m]. Default is 0.
    utm_y: float
        UTM-y coordinates of the south-west corner of domain [m]. Default is 0.
    utm_zone: int
        UTM zone of the domain. Default is None. If None, the coordinates are
        considered not to be georeferenced.
    crs: str
        EPSG code for the coordinate reference system. If the UTM zone is
        provided, the crs is set to the UTM zone. If the UTM zone is not
        provided, the crs is set to None.
    """

    def __init__(
        self,
        output_directory: Path | str,
        fire_nz: int,
        ny: int,
        nx: int,
        dy: float = 2,
        dx: float = 2,
        utm_x: float = 0,
        utm_y: float = 0,
        utm_zone: int = 1,
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
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.utm_zone = utm_zone

        # TODO: fire_nz can be optional
        # TODO: Throw warning if fire_nz not provided and qf_wind outputs are present
        # TODO: raise error if fire_nz is not provided and qf_qind is output

        # Compute the x and y coordinate grids
        self.x_coords = np.linspace(utm_x, utm_x + dx * (nx - 1), nx)
        self.y_coords = np.linspace(utm_y, utm_y + dy * (ny - 1), ny)

        # Set the crs if the UTM zone is provided. The if statement here is
        # looking for the default QUIC-Fire values of 1, 0, 0. If the UTM
        # zone is not 1 or the x and y coordinates are not 0, then the crs
        # is set to the UTM zone.
        if utm_zone != 1 or (utm_x != 0 and utm_y != 0):
            crs = proj.CRS.from_proj4(f"+proj=utm +zone={utm_zone} +ellps=WGS84")
            self.crs = f"EPSG:{crs.to_epsg()}"
        else:
            self.crs = None

        # Get grid information from grid.bin and fire_indexes.bin
        self._fire_indexes = _process_fire_indexes(
            output_directory / "fire_indexes.bin"
        )
        (
            self.quic_nz,
            self.quic_z_coords,
            self.en2atmos_nz,
            self.en2atmos_z_coords,
        ) = _process_grid_info(output_directory / "grid.bin", ny, nx)
        self.quic_z_coords = self.quic_z_coords[: self.quic_nz]
        self.en2atmos_z_coords = self.en2atmos_z_coords[: self.en2atmos_nz]
        self.fire_z_coords = self.en2atmos_z_coords[:fire_nz]
        self.quic_dz = _get_resolution_from_coords(self.quic_z_coords)
        self.fire_dz = _get_resolution_from_coords(self.en2atmos_z_coords)

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
    ) -> SimulationOutputs:
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
        >>> inputs = qft.SimulationInputs.from_directory("path/to/inputs")
        >>> outputs = qft.SimulationOutputs.from_simulation_inputs("path/to/outputs", inputs)
        """
        return cls(
            output_directory,
            simulation_inputs.quic_fire.nz,
            simulation_inputs.qu_simparams.ny,
            simulation_inputs.qu_simparams.nx,
            simulation_inputs.qu_simparams.dy,
            simulation_inputs.qu_simparams.dx,
            simulation_inputs.qu_simparams.utm_x,
            simulation_inputs.qu_simparams.utm_y,
            simulation_inputs.qu_simparams.utm_zone_number,
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
                z_coords = self._get_z_coords(attributes)
                times = []
                for file in output_files_list:
                    time = self._get_output_file_time(key, file)
                    times.append(time)

                self.outputs[key] = OutputFile(
                    name=key,
                    file_format=attributes["file_format"],
                    dimensions=attributes["dimensions"],
                    grid=attributes["grid"],
                    x_coords=self.x_coords,
                    y_coords=self.y_coords,
                    z_coords=z_coords,
                    delimiter=attributes["delimiter"],
                    extension=attributes["extension"],
                    description=attributes["description"],
                    units=attributes["units"],
                    times=times,
                    filepaths=output_files_list,
                    index_map=self._fire_indexes,
                    crs=self.crs,
                )

    def _get_list_output_paths(self, name, ext) -> list[Path]:
        """
        Get a sorted list of output files in the Output/ directory for the
        given output name.
        """
        paths = list(self.output_directory.glob(f"{name}*{ext}"))
        paths.sort()
        return paths

    def _get_z_coords(self, attrs) -> tuple:
        number_dimensions = len(attrs["dimensions"])
        grid = attrs["grid"]

        if number_dimensions == 2:
            return [0.0]
        elif number_dimensions == 3 and grid == "fire":
            return self.fire_z_coords
        elif number_dimensions == 3 and grid == "en2atmos":
            return self.en2atmos_z_coords
        elif number_dimensions == 3 and grid == "quic":
            return self.quic_z_coords
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
            np.fromfile(
                fid, dtype=np.float32, count=quic_nz + 2
            )  # Read sigma_bottom (same as quic_grid_bottom)
            np.fromfile(fid, dtype=np.int32, count=2)  # Header
            np.fromfile(
                fid, dtype=np.float32, count=quic_nz + 2
            )  # Read sigma_mid (same as quic_grid_mid)
            np.fromfile(fid, dtype=np.int32, count=2)  # Header
            np.fromfile(
                fid, dtype=np.float32, count=ny * nx * (quic_nz + 2)
            )  # quic grid bottom terrain following
            np.fromfile(fid, dtype=np.int32, count=2)  # Header
            np.fromfile(
                fid, dtype=np.float32, count=ny * nx * (quic_nz + 2)
            )  # quic grid mid terrain following
            np.fromfile(fid, dtype=np.int32, count=2)  # Header
            np.fromfile(
                fid, dtype=np.float32, count=ny * nx
            )  # quic grid volume correction
            np.fromfile(fid, dtype=np.int32, count=1)  # Header
            _ = np.fromfile(fid, dtype=np.int32, count=1)[0]  # fire header
        else:
            _ = test_header[0]  # fire header

        num_eng_to_atmos_cells = np.fromfile(fid, dtype=np.int32, count=1)[0]
        np.fromfile(fid, dtype=np.int32, count=2)  # Header
        eng_to_amos_grid_bottom = np.fromfile(
            fid, dtype=np.float32, count=num_eng_to_atmos_cells + 1
        )

        np.fromfile(fid, dtype=np.int32, count=1)  # ending header
        np.fromfile(fid, dtype=np.int32, count=4)  # Header

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
