from __future__ import annotations

import zarr
import pytest
import numpy as np
import xarray as xr
import dask.array as da

# from scipy.io import FortranFile
from pathlib import Path
import os
from quicfire_tools import outputs


TEST_DIR = Path(os.path.abspath(__file__)).parent
DATA_PATH = TEST_DIR / "data" / "test-output-zarr" / "linefire_example"
OUTPUT_PATH = DATA_PATH / "Output"


def main():
    tso = TestSimulationOutputs()
    tso.test_zarr_single_output()
    tso.test_zarr_multiple_outputs()
    tso.test_zarr_all_outputs()
    ds = tso.test_zarr_xarray_connection_single_output()


class TestSimulationOutputs:
    nz = 1
    ny = 200
    nx = 200
    sut = outputs.SimulationOutputs(OUTPUT_PATH, nz, ny, nx)

    def test_zarr_single_output(self, single_output_name="fuels-dens"):
        """
        Run a test to ensure that the zarr output contains a single output
        """
        self.sut.to_zarr(
            OUTPUT_PATH / "zarr_outputs", outputs=single_output_name, over_write=True
        )

        assert self.sut.zarr_folder_path.exists()
        for output_name in self.sut.list_available_outputs():
            if output_name == single_output_name:
                output = self.sut.get_output(output_name)
                assert output.zarr_path.exists()
                zarr_file = zarr.open(output.zarr_path, mode="r")
                assert isinstance(zarr_file, zarr.hierarchy.Group)
                assert isinstance(zarr_file["data"], zarr.core.Array)
                output_data = output.to_numpy()
                assert zarr_file["data"].shape == output_data.shape

    def test_zarr_multiple_outputs(
        self, multiple_output_names=["mburnt_integ", "fuels-dens"]
    ):
        """
        Run a test to ensure that the zarr output contains multiple outputs
        """
        self.sut.to_zarr(
            OUTPUT_PATH / "zarr_outputs", outputs=multiple_output_names, over_write=True
        )
        assert self.sut.zarr_folder_path.exists()
        for output_name in self.sut.list_available_outputs():
            if output_name in multiple_output_names:
                output = self.sut.get_output(output_name)
                assert output.zarr_path.exists()
                zarr_file = zarr.open(output.zarr_path, mode="r")
                assert isinstance(zarr_file, zarr.hierarchy.Group)
                assert isinstance(zarr_file["data"], zarr.core.Array)
                output_data = output.to_numpy()
                assert zarr_file["data"].shape == output_data.shape

    def test_zarr_all_outputs(self):
        """
        Run a test to ensure that the zarr output contains all outputs
        """
        self.sut.to_zarr(OUTPUT_PATH / "zarr_outputs", over_write=False)
        assert self.sut.zarr_folder_path.exists()
        for output_name in self.sut.list_available_outputs():
            output = self.sut.get_output(output_name)
            assert output.zarr_path.exists()
            zarr_file = zarr.open(output.zarr_path, mode="r")
            assert isinstance(zarr_file, zarr.hierarchy.Group)
            assert isinstance(zarr_file["data"], zarr.core.Array)
            output_data = output.to_numpy()
            assert zarr_file["data"].shape == output_data.shape

    def test_zarr_xarray_connection_single_output(
        self, single_output_name="fuels-dens"
    ):
        output = self.sut.get_output(single_output_name)
        ds = xr.open_zarr(
            output.zarr_path, decode_times=False
        )  # zarr path will only exist if you run .to_zarr()
        assert ds.data.long_name == single_output_name
        ds.info()
        return ds

    def test_zarr_rechunker(self):
        pass


if __name__ == "__main__":
    main()
