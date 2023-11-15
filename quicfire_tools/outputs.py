"""
Module for converting QUIC-Fire output files to duck array data formats.
"""
from __future__ import annotations

# Core imports
import re
from pathlib import Path

# External imports
import zarr
import numpy as np
import dask.array as da
from numpy import ndarray

FUELS_OUTPUTS = {
    "fire-energy_to_atmos": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
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
        "delimiter": "-",
        "extension": ".bin",
        "description": "Rate of reaction of fuel",
        "units": "kg/m^3/s",
    },
    "fuels-dens": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel density",
        "units": "kg/m^3",
    },
    "fuels-moist": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel moisture content",
        "units": "g water / g air",
    },
    "groundfuelheight": {
        "file_format": "gridded",
        "dimensions": ["y", "x"],
        "grid": "fire",
        "delimiter": None,
        "extension": ".bin",
        "description": "2D array with initial fuel height in the ground layer",
        "units": "m",
    },
    "mburnt_integ": {
        "file_format": "gridded",
        "dimensions": ["y", "x"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "2D file containing the percentage of mass burnt for each (i,j) "
        "location on the fire grid (vertically integrated)",
        "units": "%",
    },
}
THERMAL_RADIATION_OUTPUTS = {
    "thermaldose": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
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
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
        "collocated with fuel (for health effects).",
        "units": "kW/m^2",
    },
}
WIND_OUTPUTS = {
    "windu": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire u-components, cell centered ",
        "units": "m/s",
    },
    "windv": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire v-components, cell centered ",
        "units": "m/s",
    },
    "windw": {
        "file_format": "gridded",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire w-components, cell centered ",
        "units": "m/s",
    },
}
EMISSIONS_OUTPUTS = {
    "co-emissions": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of CO emitted between two emission file output"
        "times in grams",
        "units": "g",
    },
    "pm-emissions": {
        "file_format": "compressed",
        "dimensions": ["z", "y", "x"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of PM2.5 emitted between two emission file output"
        "times in grams",
        "units": "g",
    },
}
OUTPUTS_MAP = {
    **FUELS_OUTPUTS,
    **THERMAL_RADIATION_OUTPUTS,
    **WIND_OUTPUTS,
    **EMISSIONS_OUTPUTS,
}


class OutputFile:
    def __init__(
        self,
        name: str,
        file_format: str,
        dimensions: list[str],
        shape: tuple,
        grid: str,
        delimiter: str,
        extension: str,
        description: str,
        units: str,
        index_map=None,
    ):
        self.name = name
        self.file_format = file_format
        self.dimensions = dimensions
        self.shape = shape
        self.grid = grid
        self.delimiter = delimiter
        self.extension = extension
        self.description = description
        self.units = units
        self.times = []  # List of times corresponding to the timesteps
        self.filepaths = []  # List of file paths for each timestep
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

    def to_numpy(self, timestep: int | list[int] = None) -> np.ndarray:
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
        self._validate_timestep(timestep)
        selected_files = self._select_files_based_on_timestep(timestep)
        return self._get_multiple_timesteps(selected_files)

    def _validate_timestep(self, timestep: int | list[int] | None) -> None:
        """Validate the timestep input."""
        if timestep is None:
            return
        if isinstance(timestep, int):
            if timestep not in range(len(self.times)):
                raise ValueError(f"Invalid timestep: {timestep}")
        elif all(isinstance(ts, int) for ts in timestep):
            if any(ts not in range(len(self.times)) for ts in timestep):
                raise ValueError(f"Invalid timestep: {timestep}")
        else:
            raise TypeError(f"Invalid timestep type: {type(timestep)}")

    def _select_files_based_on_timestep(
        self, timestep: int | list[int] | None
    ) -> list[Path]:
        """Return files selected based on timestep."""
        if timestep is None:
            return self.filepaths
        if isinstance(timestep, int):
            return [self.filepaths[timestep]]
        return [self.filepaths[ts] for ts in timestep]

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
    """

    def __init__(self, output_directory: Path | str, nz: int, ny: int, nx: int):
        # Convert to Path and resolve
        output_directory = Path(output_directory).resolve()

        # Validate outputs directory
        self._validate_output_dir(output_directory)

        # Assign attributes
        self.output_directory = output_directory
        self.nz = nz
        self.ny = ny
        self.nx = nx

        # Get indexing information from the fire grid
        self._fire_indexes = self._process_fire_indexes()

        # Build a list of present output files and their times
        self.outputs = {}
        self._build_output_files_map()

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
        required_files = ["fire_indexes.bin"]
        for f in required_files:
            if not (outputs_directory / f).exists():
                raise FileNotFoundError(f"Required file {f} not in outputs directory")

    def _build_output_files_map(self):
        """
        Build a key-value map of output files, where the key is the name of
        the output file and the value is an instantiated OutputFile object.
        """
        for key, attributes in OUTPUTS_MAP.items():
            output_files_list = self._get_list_output_paths(key)

            # Check if any output files of the given type exist in the directory
            if output_files_list:
                shape = self._get_output_shape(key, attributes)
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
                    index_map=self._fire_indexes,
                )

                # Populate the output file with timesteps and filepaths
                for filepath in output_files_list:
                    time = self._get_output_file_time(key, filepath)
                    self.outputs[key].times.append(time)
                    self.outputs[key].filepaths.append(filepath)

    def _get_list_output_paths(self, name) -> list[Path]:
        """
        Get a sorted list of output files in the Output/ directory for the
        given output name.
        """
        paths = list(self.output_directory.glob(f"{name}*"))
        paths.sort()
        return paths

    def _get_output_shape(self, name, attrs) -> tuple:
        number_dimensions = len(attrs["dimensions"])
        if number_dimensions == 2:
            return 1, self.ny, self.nx
        elif number_dimensions == 3 and attrs["grid"] == "fire":
            return self.nz + 1, self.ny, self.nx
        else:
            return self.nz, self.ny, self.nx

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

    def _process_fire_indexes(self) -> ndarray:
        """
        Process the `fire_indexes.bin` file to extract the necessary
        information to rebuild the 3D fields of fire and fuel related variables.

        - Data 1 (INT): Header
        - Data 2 (INT): Number of cells with fuel (firegrid%num_fuel_cells = nfuel below)
        - Data 3 (INT): Max index of the cells with fuel in the vertical direction
        - Data 4-(nfuel+3) (INT): Unique cell identifiers (firegrid%num_fuel_cells)
        - Data (nfuel+4)-(2*nfuel+4) (INT): i,j,k cell indexes
        - Data (INT): Header

        Function returns a 2D ndarray of shape (num_cells, 3) representing the
        i, j, k indexes of each cell.
        """
        with open(self.output_directory / "fire_indexes.bin", "rb") as fid:
            # Read in the number of cells and skip the header
            num_cells = np.fromfile(fid, dtype=np.int32, offset=4, count=1)[0]

            # Skip the max index of the cells with fuel in the vertical
            # direction and unique cell identifiers
            np.fromfile(fid, dtype=np.int32, count=7 + num_cells)

            # Initialize an array to hold the indices
            ijk = np.zeros((num_cells, 3), dtype=np.int32)

            # Loop over each dimension's indices (Like a sparse meshgrid)
            for i in range(0, 3):
                ijk[:, i] = np.fromfile(fid, dtype=np.int32, count=num_cells)

            # Convert to indices (numbered from 0)
            ijk -= 1

            return ijk

    def _validate_output(self, output: str) -> OutputFile:
        """Validate output."""
        try:
            output = self.outputs[output]
        except KeyError:
            raise ValueError(
                f"{output} is not a valid output. Valid "
                f"outputs are: {self.list_available_outputs()}"
            )
        return output

    def list_available_outputs(self) -> list[str]:
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
        >>> outputs.list_available_outputs()
        ['fire-energy_to_atmos', 'fuels-dens', 'groundfuelheight', 'mburnt_integ']
        """
        return list(self.outputs.keys())

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

    def to_numpy(self, key: str, timestep: None | int | list[int] = None) -> np.ndarray:
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
        >>> # Get all timesteps for the fire-energy_to_atmos output
        >>> fire_energy_all = outputs.to_numpy(output_name)
        >>> # Get the first timestep for the fire-energy_to_atmos output
        >>> fire_energy_slice = outputs.to_numpy(output_name, timestep=0)
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
        >>> fire_energy = outputs.to_dask("fire-energy_to_atmos")
        >>> fire_energy.compute()
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

    def to_zarr(
        self, fpath: str | Path, outputs: str | list[str] = None
    ) -> zarr.hierarchy.Group:
        """
        Write the outputs to a zarr file.

        Parameters
        ----------
        fpath: str | Path
            The path to the zarr file to be written.
        outputs: str | list[str]
            The name of the output(s) to write to the zarr file. If None, then
            all outputs are written to the zarr file.

        Returns
        -------
        zarr.hierarchy.Group
            The zarr file object.
        """
        if isinstance(fpath, str):
            fpath = Path(fpath)

        # Create the zarr file
        zarr_file = zarr.open(str(fpath), mode="w")

        # Get a list of outputs to write to the zarr file
        if outputs is None:
            outputs = self.list_available_outputs()
        elif isinstance(outputs, str):
            outputs = [outputs]

        # Write each output to the zarr file
        for output_name in outputs:
            # Get the output object and verify it exists
            output = self.get_output(output_name)

            # Create a zarr dataset for the output
            shape = (len(output.times), *output.shape)
            chunks = [1 if i == 0 else shape[i] for i in range(len(shape))]
            zarr_dataset = zarr_file.create_dataset(
                output_name, shape=shape, chunks=chunks, dtype=float
            )
            zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]

            # Write each timestep to the output's zarr dataset
            for time_step in range(len(output.times)):
                data = self.to_numpy(output_name, time_step)
                zarr_dataset[time_step, ...] = data[0, ...]

        return zarr_file


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
    return full_array.astype(float)


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
