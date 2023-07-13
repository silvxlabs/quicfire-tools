"""
Module for converting QUIC-Fire output files to duck array data formats.
"""
from __future__ import annotations

# Core imports
from pathlib import Path

# External imports
import numpy as np
from numpy import ndarray

FUELS_OUTPUTS = {
    "fire-energy_to_atmos": {
        "format": "gridded",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Energy released to the atmosphere that generates "
                       "buoyant plumes",
        "units": "kW",
    },
    "fire-reaction_rate": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Rate of reaction of fuel",
        "units": "kg/m^3/s",
    },
    "fuels-dens": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel density",
        "units": "kg/m^3",
    },
    "fuels-moist": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Fuel moisture content",
        "units": "g water / g air",
    },
    "groundfuelheight": {
        "format": "gridded",
        "dimensions": ["x", "y"],
        "grid": "fire",
        "delimiter": None,
        "extension": ".bin",
        "description": "2D array with initial fuel height in the ground layer.",
        "units": "m"
    },
    "mburnt_integ": {
        "format": "gridded",
        "dimensions": ["x", "y"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "2D file containing the % of mass burnt for each (i,j) "
                       "location on the fire grid (vertically integrated)",
        "units": "%",
    }
}
THERMAL_RADIATION_OUTPUTS = {
    "thermaldose": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "The output shows the thermal flux to skin of a person "
                       "collocated with fuel (for health effects).",
        "units": "(kW/m^2)^(4/3)s",
    },
    "thermalradiation": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
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
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire u-components, cell centered ",
        "units": "m/s",
    },
    "windv": {
        "format": "gridded",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "",
        "extension": ".bin",
        "description": "Fire v-components, cell centered ",
        "units": "m/s",
    },
    "windw": {
        "format": "gridded",
        "dimensions": ["x", "y", "z"],
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
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of CO emitted between two emission file output "
                       "times in grams",
        "units": "g",
    },
    "pm-emissions": {
        "format": "compressed",
        "dimensions": ["x", "y", "z"],
        "grid": "fire",
        "delimiter": "-",
        "extension": ".bin",
        "description": "Mass of PM2.5 emitted between two emission file output "
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

    def __init__(self, outputs_directory: Path | str) -> None:
        if isinstance(outputs_directory, str):
            path = Path(outputs_directory)
            outputs_directory = path.resolve()

        # Validate the outputs directory
        if not outputs_directory.exists():
            raise FileNotFoundError(
                f"The directory {outputs_directory} does not exist.")
        elif not outputs_directory.is_dir():
            raise NotADirectoryError(
                f"The path {outputs_directory} is not a directory.")
        self._outputs_directory = outputs_directory

        # Build a list of present output files and their times
        self.simulation_output_files = self._build_output_files_map()

        # Get indexing information from the fire grid
        self._fire_indexes = self._process_fire_indexes()

    def _build_output_files_map(self):
        """Build a dictionary of present output files and their times."""
        outputs_dict = {}
        for output in OUTPUTS_MAP.keys():
            output_paths = self._get_output_paths(output)

            # Skip if no output files are found
            if not output_paths:
                continue

            # Check if the output is a single file
            if OUTPUTS_MAP[output]['delimiter'] is None:
                # Output files with no delimiter are single files w/o time steps
                outputs_dict[output] = [0]
                continue

            # Store the output times for the output
            outputs_dict[output] = self._get_output_times(output, output_paths)
        return outputs_dict

    def _get_output_paths(self, output):
        """Get a list of output files for the given output."""
        return list(self._outputs_directory.glob(f"{output}*"))

    def _get_output_times(self, output, output_paths):
        """Get a list of output times for the given output."""
        output_times = []
        for output_path in output_paths:
            output_time = self._get_output_file_time(output, output_path)
            output_times.append(output_time)
        output_times.sort()
        return output_times

    @staticmethod
    def _get_output_file_time(output, fpath):
        """Get the time of the output file."""
        delimiter = OUTPUTS_MAP[output]['delimiter']
        extension = OUTPUTS_MAP[output]['extension']
        file_time = fpath.name.split(delimiter)[-1].split(extension)[0]
        return int(file_time)

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
        with open(self._outputs_directory / 'fire_indexes.bin', 'rb') as fid:
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

        with open(filename, 'rb') as fid:
            # Read in the sparse values
            sparse_values = np.fromfile(fid, dtype=np.float32, offset=4,
                                        count=self._fire_indexes.shape[0])

            # Map indices of the sparse data to the indices of the dense array
            indices = (self._fire_indexes[:, 1],
                       self._fire_indexes[:, 0],
                       self._fire_indexes[:, 2])

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

        with open(filename, 'rb') as f:
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
        return np.reshape(plane, (ny, nx), order='F')
