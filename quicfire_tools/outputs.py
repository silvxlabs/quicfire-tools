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

# Internal imports
from quicfire_tools.parameters import SimulationParameters

FUELS_OUTPUTS = {
    "fire-energy_to_atmos": {
        "file_format": "compressed",
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Energy released to the atmosphere that generates "
        "buoyant plumes",
        "units": "kW",
    },
    "fire-reaction_rate": {
        "file_format": "compressed",
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Rate of reaction of fuel",
        "units": "kg/m^3/s",
    },
    "fuels-dens": {
        "file_format": "compressed",
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel density",
        "units": "kg/m^3",
    },
    "fuels-moist": {
        "file_format": "compressed",
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel moisture content",
        "units": "g water / g air",
    },
    "groundfuelheight": {
        "file_format": "gridded",
        "number_dimensions": 2,
        "grid": "fire",
        "delimiter": None,
        "extension": ".bin",
        "description": "2D array with initial fuel height in the ground layer",
        "units": "m",
    },
    "mburnt_integ": {
        "file_format": "gridded",
        "number_dimensions": 2,
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
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
        "collocated with fuel (for health effects).",
        "units": "(kW/m^2)^(4/3)s",
    },
    "thermalradiation": {
        "file_format": "compressed",
        "number_dimensions": 3,
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
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire u-components, cell centered ",
        "units": "m/s",
    },
    "windv": {
        "file_format": "gridded",
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire v-components, cell centered ",
        "units": "m/s",
    },
    "windw": {
        "file_format": "gridded",
        "number_dimensions": 3,
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
        "number_dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of CO emitted between two emission file output"
        "times in grams",
        "units": "g",
    },
    "pm-emissions": {
        "file_format": "compressed",
        "number_dimensions": 3,
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

    def add_timestep(self, time: float, filepath: str) -> None:
        """Add a new timestep to the OutputFile."""
        self.times.append(time)
        self.filepaths.append(filepath)

    def to_numpy(self, timestep: int | list[int] | None = None) -> np.ndarray:
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
        return arrays[0] if len(arrays) == 1 else np.stack(arrays)


class SimulationOutputs:
    """
    A class responsible for managing and processing simulation outputs,
    including validation, extraction, and organization of data from output
    files. This class facilitates the retrieval of data in various formats
    and can return a numpy array for a specific output, or write the data to a
    zarr file.
    """

    def __init__(
        self, output_directory: Path | str, params: SimulationParameters
    ) -> None:
        # Convert to Path and resolve
        output_directory = Path(output_directory).resolve()

        # Validate outputs directory
        self._validate_output_dir(output_directory)

        # Assign attributes
        self.output_directory = output_directory
        self.params = params

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
                raise FileNotFoundError(
                    f"Required file {f} not found in outputs directory"
                )

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
                    self.outputs[key].add_timestep(time, filepath)

    def _get_list_output_paths(self, name) -> list[Path]:
        """
        Get a sorted list of output files in the Output/ directory for the
        given output name.
        """
        paths = list(self.output_directory.glob(f"{name}*"))
        paths.sort()
        return paths

    def _get_output_shape(self, name, attrs) -> tuple:
        if attrs["number_dimensions"] == 2:
            return self.params.ny, self.params.nx, 1
        # elif attrs["number_dimensions"] == 3 and name == "fire-energy_to_atmos":
        #     _process_vertical_grid(self.output_directory / "grid.bin", self.params.nx, self.params.ny)
        elif attrs["number_dimensions"] == 3 and attrs["grid"] == "fire":
            return self.params.ny, self.params.nx, self.params.nz + 1
        else:
            return self.params.ny, self.params.nx, self.params.nz

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

    def list_available_outputs(self):
        """Return a list of keys representing available outputs."""
        return list(self.outputs.keys())

    def get_output(self, key):
        """Return a list of times for a given output key."""
        return self.outputs[key]

    def to_numpy(
        self, key: str | OutputFile, timestep: None | int | list[int] = None
    ) -> np.ndarray:
        """Return a numpy array for the given output and timestep(s)."""
        output = self._validate_output(key)
        return output.to_numpy(timestep)

    def _validate_output(self, output: str | OutputFile) -> OutputFile:
        """Validate output."""
        if isinstance(output, str):
            try:
                output = self.outputs[output]
            except KeyError:
                raise ValueError(
                    f"{output} is not a valid output. Valid "
                    f"outputs are: {self.list_available_outputs()}"
                )
        elif isinstance(output, OutputFile):
            if output not in self.outputs.values():
                raise ValueError(
                    f"{output} is not a valid output. Valid "
                    f"outputs are: {self.list_available_outputs()}"
                )
        else:
            raise TypeError(
                f"output must be a string or OutputFile object. " f"Got {type(output)}"
            )
        return output

    def to_dask(
        self, key: str | OutputFile, timestep: None | int | list[int] = None
    ) -> np.ndarray:
        """Return a dask array for the given output and timestep(s)."""
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

    def to_zarr(self, fpath: Path, outputs: str | list[str] = None):
        """Write the data to a zarr file."""
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
            zarr_dataset.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "z"]

            # Write each timestep to the output's zarr dataset
            for time_step in range(len(output.times)):
                data = self.to_numpy(output_name, time_step)
                zarr_file[output_name][time_step, ...] = data

        return zarr_file


def _process_compressed_bin(filename, dim_yxz, *args) -> ndarray:
    """
    Converts the contents of a sparse .bin file to a dense NumPy array.

    This function reads a .bin file containing sparse data and converts it to
    a dense 3D NumPy array. The mapping from sparse to dense is guided by an
    index map provided as an argument. The index map must match the sparse data
    in the .bin file, defining the locations of nonzero values within the dense
    array.

    Parameters
    ----------
    filename : str | Path
        The name or path of the sparse .bin file to be processed.
    dim_yxz : tuple of int
        The number of y, x, z cells to store in the 3D array, given as a tuple
        of three integers.
    *args : tuple
        Additional arguments, specifically expecting the index map for mapping
        sparse indices to dense. The index map should be a 2D array with shape
        (N, 3), where N is the number of nonzero elements in the sparse data.

    Returns
    -------
    ndarray
        A dense 3D NumPy array with shape defined by `dim_yxz`, and data type
        np.float32. The array represents the dense form of the sparse data
        from the .bin file.

    Raises
    ------
    IndexError
        If the index map (fire_indexes) is not provided in the arguments.
    ValueError
        If the index map (fire_indexes) is None.
    """
    # TODO: Rename fire_indexes to something more generic
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
    full_3d_array = np.zeros(dim_yxz, dtype=np.float32)

    with open(filename, "rb") as fid:
        # Read header
        np.fromfile(fid, dtype=np.int32, count=1)

        # Read in the sparse values
        sparse_values = np.fromfile(fid, dtype=np.float32, count=fire_indexes.shape[0])

        # Map indices of the sparse data to the indices of the dense array
        indices = (
            fire_indexes[:, 1],
            fire_indexes[:, 0],
            fire_indexes[:, 2],
        )

        # Update the full array with the sparse indices
        full_3d_array[indices] = sparse_values

    return full_3d_array


def _process_gridded_bin(filename, dims, *args) -> ndarray:
    """
    Converts the data stored in a gridded .bin file to an np array. The
    function handles both 2D and 3D data based on the dimensions provided.

    Parameters
    ----------
    filename : str | Path
        File path to the .bin file that contains the gridded data.
    dims : tuple
        Dimensions of the data. If a 2-element tuple (ny, nx) is provided,
        the function processes 2D data. If a 3-element tuple (ny, nx, nz)
        is provided, the function processes 3D data.
    *args : list
        Additional arguments that might be passed to the underlying slice
        processing function (_process_gridded_bin_slice).

    Returns
    -------
    ndarray
        A NumPy array containing the data extracted from the .bin file. The
        array's shape corresponds to the provided dimensions, and its data type
        is np.float32.
    """
    # Initialize the array
    array = np.zeros(dims, dtype=np.float32)

    with open(filename, "rb") as f:
        # Read header
        _ = np.fromfile(f, dtype=np.float32, count=1)

        # Read in gridded data
        if len(dims) == 2:  # 2D data
            array[:, :] = _process_gridded_bin_slice(f, *dims)
        else:  # 3D data
            for k in range(dims[2]):
                array[:, :, k] = _process_gridded_bin_slice(f, *dims[:2])
    return array


def _process_gridded_bin_slice(fid, ny, nx) -> ndarray:
    """
    Process a 2D slice from a gridded .bin file.

    This method reads a 2D slice of size (ny, nx) from the provided file
    object, and returns it as a np.float32 array.
    """
    # Read in gridded data
    plane = np.fromfile(fid, dtype=np.float32, count=ny * nx)
    return np.reshape(plane, (ny, nx))
