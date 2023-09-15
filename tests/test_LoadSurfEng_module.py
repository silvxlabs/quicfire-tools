import sys
import os

#sys.path.append("../quicfire_tools")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from quicfire_tools import outputs
from quicfire_tools.parameters import SimulationParameters

import zarr
import pytest
import numpy as np
from scipy.io import FortranFile
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

DATA_PATH = PurePath("/mnt/c/Users/zacha/Documents/0_Projects")
SIMULATION_PATH = DATA_PATH.joinpath("0016_FtStewart", "F6_4", "1_Runs", "01_FastFuelsAerialIg531")
OUTPUT_PATH = SIMULATION_PATH.joinpath("Output")
DRAWFIRE_PATH = OUTPUT_PATH.joinpath("drawfire")
ZARR_PATH = OUTPUT_PATH.joinpath("outputs.zarr")

# Create simulation parameters object
SIM_PARAMS = SimulationParameters(
    nx=968,
    ny=1978,
    nz=40,
    dx=2,
    dy=2,
    dz=1,
    wind_speed=6.5,
    wind_direction=270,
    sim_time=4067,
    auto_kill=0,
    num_cpus=8,
    fuel_flag=5,
    ignition_flag=7,
    output_time=100,
    topo_flag=0,
)

# DATA_PATH = Path("data")
# SIMULATION_PATH = DATA_PATH / "crazy-canyon-simulation"
# OUTPUT_PATH = SIMULATION_PATH / "Output"
# DRAWFIRE_PATH = OUTPUT_PATH / "drawfire"

# # Create simulation parameters object
# SIM_PARAMS = SimulationParameters(
#     nx=100,
#     ny=100,
#     nz=56,
#     dx=1,
#     dy=1,
#     dz=1,
#     wind_speed=6.5,
#     wind_direction=270,
#     sim_time=600,
#     auto_kill=0,
#     num_cpus=4,
#     fuel_flag=4,
#     ignition_flag=1,
#     output_time=100,
#     topo_flag=0,
# )

def main():
    ###
    ###This is crashing. Need to set-up to run with zarr and dask
    ###
    SMOLDER_THRESHOLD = 25
    #Use library to load and calculate surfEnergy outputs
    simulation_outputs = outputs.SimulationOutputs(OUTPUT_PATH, SIM_PARAMS)

    # output = simulation_outputs.get_output("surfEnergy")
    # arr = simulation_outputs.to_numpy(output)

    # # Create a DataArray object from the numpy array
    # da = xr.DataArray(arr, dims=["time", "y", "x"])

    # # Create a Dataset object from the DataArray object
    # ds = da.to_dataset(name="data")
    if not os.path.exists(ZARR_PATH):
        zarr_file = simulation_outputs.to_zarr(ZARR_PATH)
    zarr.convenience.consolidate_metadata(ZARR_PATH)
    ds = xr.open_zarr(ZARR_PATH)

    #Calc percent burned & time for max power
    ds = ds.fillna(0) #Convert nan to 0 for dask
    xarr_max_power_time = ds.surfEnergy.argmax('time')
    xarr_max_power = ds.surfEnergy[xarr_max_power_time.compute()]   
    xarr_max_power_time = xr.where(xarr_max_power_time==0,np.nan,xarr_max_power_time)

    ###Calc Times: arrival, stop, residence
    ##Removed forloop to improve speed
    #https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    #https://stackoverflow.com/questions/66305130/index-of-last-occurence-of-true-in-every-row
    burned_binary = (ds>SMOLDER_THRESHOLD)
    #Arrival time
    xarr_arrival_time = burned_binary.surfEnergy.argmax('time')
    xarr_arrival_time = xr.where(xarr_arrival_time==0,np.nan,xarr_arrival_time) #0 to nan
    #xarr_arrival_time = xarr_arrival_time.compute()

    #Fire stop time
    xarr_fire_stop_time = burned_binary.dims['time'] - burned_binary.surfEnergy[::-1,:,:].argmax('time') - 1
    xarr_fire_stop_time = xr.where((burned_binary.surfEnergy[-1,:,:]==0) & (xarr_fire_stop_time==xarr_fire_stop_time.max()),np.nan,xarr_fire_stop_time) #non-burning cells to nan
    del burned_binary

    xarr_residence_time = xarr_fire_stop_time - xarr_arrival_time        
    
    def scale_for_figs_x_and_y(arr, dx=2, dy=2):
        arr = np.array(arr)
        arr = np.repeat(np.repeat(arr, dy, axis=0), dx, axis=1)
        plt.imshow(arr, cmap='YlOrRd', origin="lower")
    
    save_dir = DRAWFIRE_PATH

    #Plot Spatial metrics
    scale_for_figs_x_and_y(xarr_arrival_time)
    plt.colorbar()
    plt.title("Arrival Time (s)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(os.path.join(save_dir,"arrival_time.png"))
    plt.close()
    
    #Spatial Figures of Metrics
    scale_for_figs_x_and_y(xarr_fire_stop_time)
    plt.colorbar()
    plt.title("Burn Completion Time (s)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(os.path.join(save_dir,"stop_time.png"))
    plt.close()
    
    scale_for_figs_x_and_y(xarr_residence_time)
    plt.colorbar()
    plt.title("Residence Time (s)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(os.path.join(save_dir,"residence_time.png"))
    plt.close()

    scale_for_figs_x_and_y(xarr_max_power)
    plt.colorbar()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Max Power (kW/m^2)")
    plt.savefig(os.path.join(save_dir,"max_power.png"))
    plt.close()


"""
class TestOutputFile:
    simulation_outputs = outputs.SimulationOutputs(OUTPUT_PATH, SIM_PARAMS)

    def test_output_single_timestep(self):
        # Test 1: 2D gridded data
        output = self.simulation_outputs.get_output("groundfuelheight")
        output_path = output.filepaths[0]
        data = output._get_single_timestep(output_path)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx)

        # # Test 2: 3D gridded data
        # output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        # output_path = output.filepaths[0]
        # data = output._get_single_timestep(output_path)
        # assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # Test 3: 3D Compressed data
        output = self.simulation_outputs.get_output("fuels-dens")
        output_path = output.filepaths[0]
        data = output._get_single_timestep(output_path)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

    def test_get_multiple_timesteps(self):
        # Test 1: 2D gridded data (single timestep)
        output = self.simulation_outputs.get_output("groundfuelheight")
        output_path = output.filepaths
        data = output._get_multiple_timesteps(output_path)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx)

        # # Test 2: 3D gridded data (all timesteps)
        # output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        # output_path = output.filepaths
        # data = output._get_multiple_timesteps(output_path)
        # assert data.shape == (
        #     2, SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # Test 3: 3D Compressed data (all timesteps)
        output = self.simulation_outputs.get_output("fuels-dens")
        output_path = output.filepaths
        data = output._get_multiple_timesteps(output_path)
        assert data.shape == (
            2, SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1) #(time, ny, nx, nz) Time is 41 in my example

    def test_to_numpy(self):
        # Test 1: 2D gridded data (single timestep)
        output = self.simulation_outputs.get_output("groundfuelheight")
        data = self.simulation_outputs.to_numpy(output, 0)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx)

        # Test 2: 2D gridded data (all timesteps)
        output = self.simulation_outputs.get_output("groundfuelheight")
        data = self.simulation_outputs.to_numpy(output)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx)

        # # Test 3: 3D gridded data (single timestep)
        # output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        # data = self.simulation_outputs.to_numpy(output, 0)
        # assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # # Test 4: 3D gridded data (all timesteps)
        # output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        # data = self.simulation_outputs.to_numpy(output)
        # assert data.shape == (
        #     2, SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # Test 5: 3D gridded data (list timesteps)
        output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        data = self.simulation_outputs.to_numpy(output, [0, 1])
        assert data.shape == (
            2, SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # Test 6: 3D gridded data (invalid timesteps)
        output = self.simulation_outputs.get_output("fire-energy_to_atmos")
        with pytest.raises(ValueError):
            self.simulation_outputs.to_numpy(output, [0, 1, 2])

        # Test 7: 3D compressed data (single timestep)
        output = self.simulation_outputs.get_output("fuels-dens")
        data = self.simulation_outputs.to_numpy(output, 0)
        assert data.shape == (SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)

        # Test 8: 3D compressed data (all timesteps)
        output = self.simulation_outputs.get_output("fuels-dens")
        data = self.simulation_outputs.to_numpy(output)
        assert data.shape == (
            2, SIM_PARAMS.ny, SIM_PARAMS.nx, SIM_PARAMS.nz + 1)


class TestProcessGriddedBinSlice:
    nx = 100
    ny = 100
    nz = 56

    @staticmethod
    def _process_gridded_bin_slice(f, ny, nx):
        return outputs._process_gridded_bin_slice(f, ny, nx)

    def _load_drawfire_data(self, fpath):
        data = np.load(fpath)
        if data.shape[-1] == 1:
            return data.squeeze(axis=-1)
        return data

    def _load_and_process_input(self, input_fpath):
        input_fortran_file = FortranFile(input_fpath, "r")
        input_data = input_fortran_file.read_reals(dtype=np.float32)
        input_data = input_data.reshape((self.nz, self.ny, self.nx))
        input_data = np.moveaxis(input_data, 0, 2).astype(np.float32)
        input_data = input_data[..., 0]
        input_fortran_file.close()
        return input_data

    def _assert_data_equality(self, data1, data2):
        assert data1.shape == data2.shape
        assert np.allclose(data1, data2)

    def test_groundfuelheight(self):
        input_data = self._load_and_process_input(
            SIMULATION_PATH / "treesfueldepth.dat")

        output_fpath = OUTPUT_PATH / "groundfuelheight.bin"
        with open(output_fpath, "rb") as f:
            np.fromfile(f, dtype=np.float32, count=1)  # Read header
            output_data = self._process_gridded_bin_slice(f, self.ny, self.nx)

        input_data[0, :] = 0
        input_data[-1, :] = 0
        input_data[:, 0] = 0
        input_data[:, -1] = 0

        self._assert_data_equality(output_data, input_data)

    def _generic_drawfire_slice_test(self, test_name, dims):
        for t in (0, 300):
            # Load drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"{test_name}_{t}.npy"
            drawfire_data = self._load_drawfire_data(drawfire_fpath)

            # Load output data
            output_fpath = OUTPUT_PATH / f"{test_name}00{t:03}.bin"
            f = open(output_fpath, "rb")
            np.fromfile(f, dtype=np.float32, count=1)  # Read header

            # Read in each slice of the output data and compare to the
            # equivalent slice of the drawfire data
            for k in range(dims[2]):
                output_data = self._process_gridded_bin_slice(f, dims[0],
                                                              dims[1])
                self._assert_data_equality(drawfire_data[..., k], output_data)

            f.close()

    def test_fire_energy_to_atmos(self):
        self._generic_drawfire_slice_test("fire-energy_to_atmos-",
                                          (self.ny, self.nx, self.nz))

    def test_windu(self):
        self._generic_drawfire_slice_test("windu", (self.ny, self.nx, self.nz))


class TestProcessGriddedBin:
    nx = 100
    ny = 100
    nz = 56

    @staticmethod
    def _process_gridded_bin(f, dims):
        return outputs._process_gridded_bin(f, dims)

    def _load_fastfuels_data(self):
        zarray = zarr.open(SIMULATION_PATH / "crazy_canyon_100m", mode="r")
        return zarray["surface"]["fuel-depth"][...]

    def _load_and_process_input(self, input_fpath):
        input_fortran_file = FortranFile(input_fpath, "r")
        input_data = input_fortran_file.read_reals(dtype=np.float32)
        input_data = input_data.reshape((self.nz, self.ny, self.nx))
        input_data = np.moveaxis(input_data, 0, 2).astype(np.float32)
        input_data = input_data[..., 0]
        input_fortran_file.close()
        return input_data

    def _load_drawfire_data(self, fpath):
        data = np.load(fpath)
        if data.shape[-1] == 1:
            return data.squeeze(axis=-1)
        return data

    def _assert_data_equality(self, data1, data2):
        assert data1.shape == data2.shape
        assert np.allclose(data1, data2)

    def test_groundfuelheight(self):
        fuel_depth = self._load_fastfuels_data()
        input_data = self._load_and_process_input(
            SIMULATION_PATH / "treesfueldepth.dat")

        self._assert_data_equality(fuel_depth, input_data)

        output_data = self._process_gridded_bin(
            OUTPUT_PATH / "groundfuelheight.bin", (self.ny, self.nx))

        fuel_depth[0, :] = 0
        fuel_depth[-1, :] = 0
        fuel_depth[:, 0] = 0
        fuel_depth[:, -1] = 0

        self._assert_data_equality(output_data, fuel_depth)

    def test_total_initial_fuel_height(self):
        fuel_depth = self._load_fastfuels_data()
        input_data = self._load_and_process_input(
            SIMULATION_PATH / "treesfueldepth.dat")

        self._assert_data_equality(fuel_depth, input_data)

        output_data = self._process_gridded_bin(
            OUTPUT_PATH / "totalinitialfuelheight.bin", (self.ny, self.nx))

        fuel_depth[0, :] = 0
        fuel_depth[-1, :] = 0
        fuel_depth[:, 0] = 0
        fuel_depth[:, -1] = 0

        self._assert_data_equality(output_data, fuel_depth)

    def _generic_drawfire_test(self, test_name, dims, *args):

        for t in (0, 300):
            drawfire_data = self._load_drawfire_data(
                DRAWFIRE_PATH / f"{test_name}_{t}.npy")
            if drawfire_data.shape[-1] != dims[-1]:
                drawfire_data = drawfire_data[..., :dims[2]]
            output_data = self._process_gridded_bin(
                OUTPUT_PATH / f"{test_name}00{t:03}.bin", dims)

            self._assert_data_equality(drawfire_data, output_data)

    def test_mburnt_integ(self):
        self._generic_drawfire_test("mburnt_integ-", (self.ny, self.nx))

    def test_fire_energy_to_atmos(self):
        self._generic_drawfire_test("fire-energy_to_atmos-",
                                    (self.ny, self.nx, self.nz))

    def test_windu(self):
        self._generic_drawfire_test("windu", (self.ny, self.nx, self.nz + 1))


class TestProcessCompressedBin:
    nx = 100
    ny = 100
    nz = 56
    dims = (ny, nx, nz)
    sut = outputs.SimulationOutputs(OUTPUT_PATH, SIM_PARAMS)
    fire_indexes = sut._fire_indexes

    def _process_compressed_bin(self, f):
        return outputs._process_compressed_bin(
            f, self.dims, self.fire_indexes)

    def test_fuels_dens(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"fuels-dens-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"fuels-dens-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)

            # Check that the input and output data are the same in the first
            # time step
            if t == 0:
                # Load input data
                input_fpath = SIMULATION_PATH / "treesrhof.dat"
                input_fortran_file = FortranFile(input_fpath, "r")
                input_data = input_fortran_file.read_reals(dtype=np.float32)
                input_fortran_file.close()
                input_data = input_data.reshape((self.nz, self.ny, self.nx))
                input_data = np.moveaxis(input_data, 0, 2).astype(np.float32)

                # Remove edge cells from the input data
                input_data[0, :] = 0
                input_data[-1, :] = 0
                input_data[:, 0] = 0
                input_data[:, -1] = 0

                # Check that the input and output data are the same
                assert input_data.shape == output_data.shape
                assert np.allclose(input_data, output_data, atol=1e-3)

    def test_fuels_moist(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"fuels-moist-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"fuels-moist-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)

            # Check that the input and output data are the same in the first
            # time step
            if t == 0:
                # Load input data
                input_fpath = SIMULATION_PATH / "treesmoist.dat"
                input_fortran_file = FortranFile(input_fpath, "r")
                input_data = input_fortran_file.read_reals(dtype=np.float32)
                input_fortran_file.close()
                input_data = input_data.reshape((self.nz, self.ny, self.nx))
                input_data = np.moveaxis(input_data, 0, 2).astype(np.float32)

                # Remove edge cells from the input data
                input_data[0, :] = 0
                input_data[-1, :] = 0
                input_data[:, 0] = 0
                input_data[:, -1] = 0

                # Check that the input and output data are the same
                assert input_data.shape == output_data.shape
                # assert np.allclose(input_data, output_data, atol=1e-3)

    def test_array_reaction_rate(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"fire-reaction_rate-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"fire-reaction_rate-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)

    def test_co_emissions(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"co_emissions-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"co_emissions-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)

    def test_thermaldose(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"thermaldose-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"thermaldose-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)

    def test_thermalradiation(self):
        for t in (0, 300):
            # Load the drawfire data
            drawfire_fpath = DRAWFIRE_PATH / f"thermalradiation-_{t}.npy"
            drawfire_data = np.load(drawfire_fpath)

            # Load the output data
            output_fpath = OUTPUT_PATH / f"thermalradiation-00{t:03}.bin"
            output_data = self._process_compressed_bin(output_fpath)

            # Check that the drawfire and output data are the same
            assert drawfire_data.shape == output_data.shape
            assert np.allclose(drawfire_data, output_data)
"""
            
if __name__ == '__main__':
    main()
