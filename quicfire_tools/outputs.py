"""
Module for converting QUIC-Fire output files to duck array data formats.
"""
from __future__ import annotations

# Core imports
from pathlib import Path

# Internal imports
from quicfire_tools.parameters import SimulationParameters

# External imports
import zarr
import numpy as np
from numpy import ndarray

FUELS_OUTPUTS = {
    "fire-energy_to_atmos": {
        "format": "gridded",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Energy released to the atmosphere that generates "
                       "buoyant plumes",
        "units": "kW",
    },
    "fire-reaction_rate": {
        "format": "compressed",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Rate of reaction of fuel",
        "units": "kg/m^3/s",
    },
    "fuels-dens": {
        "format": "compressed",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel density",
        "units": "kg/m^3",
    },
    "fuels-moist": {
        "format": "compressed",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel moisture content",
        "units": "g water / g air",
    },
    "groundfuelheight": {
        "format": "gridded",
        "dimensions": 2,
        "grid": "fire",
        "delimiter": None,
        "extension": ".bin",
        "description": "2D array with initial fuel height in the ground layer",
        "units": "m",
    },
    "mburnt_integ": {
        "format": "gridded",
        "dimensions": 2,
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
        "format": "compressed",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
                       "collocated with fuel (for health effects).",
        "units": "(kW/m^2)^(4/3)s",
    },
    "thermalradiation": {
        "format": "compressed",
        "dimensions": 3,
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
        "format": "gridded",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire u-components, cell centered ",
        "units": "m/s",
    },
    "windv": {
        "format": "gridded",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire v-components, cell centered ",
        "units": "m/s",
    },
    "windw": {
        "format": "gridded",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire w-components, cell centered ",
        "units": "m/s",
    },
}
EMISSIONS_OUTPUTS = {
    "co-emissions": {
        "format": "compressed",
        "dimensions": 3,
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of CO emitted between two emission file output"
                       "times in grams",
        "units": "g",
    },
    "pm-emissions": {
        "format": "compressed",
        "dimensions": 3,
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


class SimulationOutputs:

    def __init__(self, outputs_directory: Path | str,
                 params: SimulationParameters) -> None:
        # Convert to Path and resolve
        outputs_directory = Path(outputs_directory).resolve()

        # Validate outputs directory
        self._validate_outputs_dir(outputs_directory)

        # Assign attributes
        self._outputs_directory = outputs_directory
        self._params = params

        # Build a list of present output files and their times
        self.simulation_output_files = self._build_output_files_map()

        # Get indexing information from the fire grid
        self._fire_indexes = self._process_fire_indexes()

        # Build a map of output formats to functions that process them
        self._format_to_func = {
            'compressed': self._process_compressed_bin,
            'gridded': self._process_gridded_bin
        }

    def get_available_outputs(self) -> list[str]:
        """Get a list of available output variables."""
        return list(self.simulation_output_files.keys())

    def get_output_times(self, output):
        """Get list of output times for a given output."""

        # Validate output
        if output not in self.simulation_output_files:
            raise ValueError(f"{output} is not a valid output")

        # Return saved times for that output
        return self.simulation_output_files[output]

    @staticmethod
    def _validate_outputs_dir(outputs_directory):
        """Validate the outputs directory."""
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

    def _build_output_files_map(self) -> dict[str, list[int]]:
        """Build a dictionary of present output files and their times."""
        outputs_dict = {}
        for output in OUTPUTS_MAP.keys():
            output_paths = self._get_list_output_paths(output)

            # Skip if no output files are found
            if not output_paths:
                continue

            # Check if the output is a single file
            if OUTPUTS_MAP[output]["delimiter"] is None:
                # Outputs with no delimiter are single files w/o time steps
                outputs_dict[output] = [0]
                continue

            # Store the output times for the output
            outputs_dict[output] = self._get_output_times(output, output_paths)
        return outputs_dict

    def _get_list_output_paths(self, output) -> list[Path]:
        """Get a sorted list of output files for the given output."""
        paths = list(self._outputs_directory.glob(f"{output}*"))
        paths.sort()
        return paths

    def _get_output_times(self, output, output_paths) -> list[int]:
        """Get a sorted list of output times for the given output."""
        output_times = []
        for output_path in output_paths:
            output_time = self._get_output_file_time(output, output_path)
            output_times.append(output_time)
        output_times.sort()
        return output_times

    @staticmethod
    def _get_output_file_time(output, fpath) -> int:
        """Get the time of the output file."""
        delimiter = OUTPUTS_MAP[output]['delimiter']
        extension = OUTPUTS_MAP[output]['extension']
        file_time = fpath.name.split(delimiter)[-1].split(extension)[0]
        return int(file_time)

    def get_output_file_path(self, output_name, time):
        padded_time = str(time).zfill(5)
        file_name = f"{output_name}{OUTPUTS_MAP[output_name]['delimiter']}" \
                    f"{padded_time}{OUTPUTS_MAP[output_name]['extension']}"
        return self._outputs_directory / file_name

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
        with open(self._outputs_directory / "fire_indexes.bin", "rb") as fid:
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

    def _process_compressed_bin(self, filename, dim_yxz) -> ndarray:
        """
        Writes the contents of a sparse .bin file to a dense zarr array.

        Args:
            filename (str): name of the sparse .bin file
            dim_yxz (tuple): Number of y, x, z cells to store in the 3D array

        Returns:
            ndarray: Dense np.float32 representation of the sparse .bin data
        """
        # Initialize the 3D array
        full_3d_array = np.zeros(dim_yxz, dtype=np.float32)

        with open(filename, "rb") as fid:
            # Read in the sparse values
            sparse_values = np.fromfile(
                fid, dtype=np.float32, offset=4,
                count=self._fire_indexes.shape[0]
            )

            # Map indices of the sparse data to the indices of the dense array
            indices = (
                self._fire_indexes[:, 1],
                self._fire_indexes[:, 0],
                self._fire_indexes[:, 2],
            )

            # Update the full array with the sparse indices
            full_3d_array[indices] = sparse_values

        return full_3d_array

    def _process_gridded_bin(self, filename, dims) -> ndarray:
        """
        Converts the data stored in a gridded .bin file to an np array. The
        function handles both 2D and 3D data based on the dimensions provided.

        This method expects the dimensions as a tuple. If a 2-element tuple
        (ny, nx) is provided, the function processes 2D data. If a 3-element
        tuple (ny, nx, nz) is provided, the function processes 3D data.
        """
        # Initialize the array
        array = np.zeros(dims, dtype=np.float32)

        with open(filename, "rb") as f:
            # Read header
            _ = np.fromfile(f, dtype=np.float32, count=1)

            # Read in gridded data
            if len(dims) == 2:  # 2D data
                array[:, :] = self._process_gridded_bin_slice(f, *dims)
            else:  # 3D data
                for k in range(dims[2]):
                    array[:, :, k] = self._process_gridded_bin_slice(f,
                                                                     *dims[:2])
        return array

    @staticmethod
    def _process_gridded_bin_slice(fid, ny, nx) -> ndarray:
        """
        Process a 2D slice from a gridded .bin file.

        This method reads a 2D slice of size (ny, nx) from the provided file
        object, and returns it as a np.float32 array.
        """
        # Read in gridded data
        plane = np.fromfile(fid, dtype=np.float32, count=ny * nx)
        return np.reshape(plane, (ny, nx), order="F")

    def to_zarr(self, fpath: Path):
        """Write the data to a zarr file."""
        # Create the zarr file
        zarr_file = zarr.open(str(fpath), mode="w")

        # Write the data to the zarr file
        for output_name, output_times in self.simulation_output_files.items():
            num_time_steps = len(output_times)
            num_dimensions = OUTPUTS_MAP[output_name]['dimensions']
            if num_dimensions == 2:
                dims = (num_time_steps, self._params.ny, self._params.nx)
            elif num_dimensions == 3:
                dims = (num_time_steps, self._params.ny, self._params.nx,
                        self._params.nz)
            else:
                raise ValueError(f"Unknown number of dimensions: "
                                 f"{num_dimensions}")
            chunk_size = [1 if i == 0 else dims[i] for i in range(len(dims))]
            zarr_file.create_dataset(
                output_name,
                shape=dims,
                chunks=chunk_size,
                dtype=float)
            for time_step in range(len(output_times)):
                data = self.to_numpy(output_name, time_step)
                zarr_file[output_name][time_step, ...] = data

        return zarr_file

    def to_numpy(self, output: str,
                 timestep: None | int | list[int] = None) -> np.ndarray:
        """Return a numpy array for the given output and timestep(s)."""
        # Validate inputs
        self._validate_output(output)
        self._validate_timesteps(timestep, output)

        # Get processing function for the output format
        output_format = OUTPUTS_MAP[output]['format']
        function = self._format_to_func.get(output_format)
        if function is None:
            raise ValueError(f"Unknown output format: {output_format}")

        # Select files to process based on timestep input
        output_files = self._get_list_output_paths(output)
        if timestep is None:
            selected_files = output_files
        elif isinstance(timestep, int):
            selected_files = [output_files[timestep]]
        else:
            selected_files = [output_files[ts] for ts in timestep]

        # Process the data for the selected files
        data = self._get_multiple_timesteps(output, selected_files, function)

        return data

    def _validate_output(self, output: str):
        """Validate output."""
        if output not in self.get_available_outputs():
            raise ValueError(f"{output} is not a valid output. Valid outputs "
                             f"are:  {self.get_available_outputs()}")

    def _validate_timesteps(self, timestep: int | list[int] | None,
                            output: str):
        """Validate timestep(s)."""
        num_timesteps = len(self.get_output_times(output))
        if timestep is not None:
            if isinstance(timestep, int):
                timestep = [timestep]  # Convert to list
            invalid_timesteps = [ts for ts in timestep if
                                 not 0 <= ts < num_timesteps]
            if invalid_timesteps:
                raise ValueError(
                    f"The following timestep(s) are not valid for this output: "
                    f"{invalid_timesteps}. Valid timesteps are in the range "
                    f"[0, {num_timesteps - 1}].")

    def _get_single_timestep(self,
                             output_name: str,
                             output_file: Path,
                             func: callable) -> np.ndarray:
        """Return a numpy array for the given output file."""
        dimensions = OUTPUTS_MAP[output_name]['dimensions']
        if dimensions == 2:
            dims = (self._params.ny, self._params.nx)
        elif dimensions == 3:
            dims = (self._params.ny, self._params.nx, self._params.nz)
        else:
            raise ValueError(f"Unknown dimensions: {dimensions}")
        return func(output_file, dims)

    def _get_multiple_timesteps(self, output_name: str,
                                output_files: list[Path],
                                func: callable) -> np.ndarray:
        """Return a numpy array for the given output files."""
        arrays = []
        for output_file in output_files:
            data = self._get_single_timestep(output_name, output_file, func)
            arrays.append(data)
        if len(arrays) == 1:
            return arrays[0]
        return np.stack(arrays)
