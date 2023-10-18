import sys
import os

#sys.path.append("../quicfire_tools")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from quicfire_tools import outputs
from quicfire_tools.parameters import SimulationParameters

import zarr
import pytest
import numpy as np
import pandas as pd
from scipy.io import FortranFile
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import random

from pathlib import Path, PurePath

# DATA_PATH = PurePath("/mnt/c/Users/zacha/Documents/0_Code/quicfire-tools/tests/data")
# SIMULATION_PATH = DATA_PATH.joinpath("test_run_eng")
# OUTPUT_PATH = SIMULATION_PATH.joinpath("Output")
# DRAWFIRE_PATH = SIMULATION_PATH.joinpath("drawfire")
# ZARR_PATH = OUTPUT_PATH.joinpath("outputs.zarr")

# # Create simulation parameters object
# SIM_PARAMS = SimulationParameters(
#     nx=400,
#     ny=200,
#     nz=1,
#     dx=2,
#     dy=2,
#     dz=1,
#     wind_speed=6,
#     wind_direction=270,
#     sim_time=600,
#     auto_kill=0,
#     num_cpus=8,
#     fuel_flag=4,
#     ignition_flag=1,
#     output_time=100,
#     topo_flag=0,
# )

DATA_PATH = PurePath("/mnt/c/Users/zacha/Documents/0_Projects")
SIMULATION_PATH = DATA_PATH.joinpath("0016_FtStewart", "F6_4", "1_Runs", "01_FastFuelsAerialIg531")
OUTPUT_PATH = SIMULATION_PATH.joinpath("Output")
DRAWFIRE_PATH = SIMULATION_PATH.joinpath("drawfire")
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

def main():
    #Setup drawfire folder:
    if not os.path.exists(DRAWFIRE_PATH):
        os.makedirs(DRAWFIRE_PATH)
    save_dir = DRAWFIRE_PATH
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

    #Calc time for max power
    ds = ds.fillna(0) #Convert nan to 0 for dask
    xarr_max_power_time = ds.surfEnergy.argmax('time')
    xarr_max_power = ds.surfEnergy[xarr_max_power_time.compute()]   

    #Calc Total Energy
    xarr_total_energy = ds.surfEnergy.sum(dim='time')

    ###Calc Times: arrival, stop, residence
    ##Removed forloop to improve speed
    #https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    #https://stackoverflow.com/questions/66305130/index-of-last-occurence-of-true-in-every-row
    burned_binary = (ds>SMOLDER_THRESHOLD)
    #Arrival time
    xarr_arrival_time = burned_binary.surfEnergy.argmax('time')
    xarr_arrival_time = xr.where(xarr_arrival_time==0,np.nan,xarr_arrival_time) #0 to nan
    xarr_arrival_time = xarr_arrival_time.compute()

    #Fire stop time
    xarr_fire_stop_time = burned_binary.dims['time'] - burned_binary.surfEnergy[::-1,:,:].argmax('time') - 1
    xarr_fire_stop_time = xr.where((burned_binary.surfEnergy[-1,:,:]==0) & (xarr_fire_stop_time==xarr_fire_stop_time.max()),np.nan,xarr_fire_stop_time) #non-burning cells to nan
    xarr_fire_stop_time = xarr_fire_stop_time.compute()
    del burned_binary

    xarr_residence_time = xarr_fire_stop_time - xarr_arrival_time

    #NEW code to run for avg power and stdDev:
    burn_indexes = np.where(xarr_residence_time>0)
    num_burn_cells = len(burn_indexes[0])
    max_res_time = xarr_residence_time.max()
    power_burn_cells = np.empty((num_burn_cells,max_res_time))
    for i in range(num_burn_cells):
        ty = burn_indexes[0][i] #temp y
        tx = burn_indexes[1][i]
        start_t = int(xarr_arrival_time[y_cell, x_cell])
        stop_t = int(xarr_fire_stop_time[y_cell, x_cell])
        cell_power = ds.surfEnergy[start_t:stop_t,y_cell,x_cell]
        power_burn_cells[i,:(stop_t-start_t)] = cell_power.to_numpy()
    
    power_mean_overtime = np.nanmean(power_burn_cells, axis=1)
    power_median_overtime = np.nanmedian(power_burn_cells, axis=1)
    power_stdev_overtime = np.nanstd(power_burn_cells, axis=1)

    #Raw data for Joe:
    df = pd.DataFrame({"power_mean_overtime":power_mean_overtime,"power_median_overtime":power_median_overtime,
                       "power_stdev_overtime":power_stdev_overtime})
    df.to_csv(os.path.join(save_dir,"FtStewart_PowerOvertime.csv"), index=False)
    del df

    plt.plot(max_res_time, power_mean_overtime)
    plt.fill_between(power_mean_overtime,power_mean_overtime-power_stdev_overtime,power_mean_overtime+power_stdev_overtime)
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW/m^2)')
    plt.title('Average Power of Surface Cells After Ignition')
    plt.savefig(os.path.join(save_dir, 'SufaceCell_MeanPowerAfterIg.png'))
    plt.close()  

    plt.plot(max_res_time, power_median_overtime)
    plt.fill_between(power_median_overtime,power_median_overtime-power_stdev_overtime,power_median_overtime+power_stdev_overtime)
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW/m^2)')
    plt.title('Median Power of Surface Cells After Ignition')
    plt.savefig(os.path.join(save_dir, 'SufaceCell_MedianPowerAfterIg.png'))
    plt.close()   
    
    ###This method is dumb. I should be able to use np.where to index the burned cells:
    #Sample burning cells
    def find_cells_that_burned(xarr_residence_time, SIM_PARAMS, n=1, time_len=15):
        """
        xarr_residence_time: residence times
        SIM_PARAMS: class of simulation parameters
        n: # of cells to sample
        time_len: length of time to consider cell burned for sample
        """
        PICKLE_PATH = os.path.join(save_dir, 'cell_that_burned.pkl')
        if not os.path.exists(PICKLE_PATH):
            nx = SIM_PARAMS.nx
            ny = SIM_PARAMS.ny
            burned_cells = []
            print('Starting while loop')
            while len(burned_cells) < n:
                temp_x = int(nx*random.random())
                temp_y = int(ny*random.random())
                temp_tup = (temp_x, temp_y)
                if xarr_residence_time[temp_y, temp_x]>0:
                    if temp_tup not in burned_cells:
                        burned_cells.append(temp_tup)
            with open(PICKLE_PATH, 'wb') as f:
                pickle.dump(burned_cells,f)
            print('While loop complete.')
        else: #reload previous list
            with open(PICKLE_PATH, 'rb') as f:
                burned_cells = pickle.load(f)
        return burned_cells 
    
    #Graph power overtime
    def build_power_graph(power, x_cell, y_cell, roll_avg, save_dir=save_dir):
        x_cell_m = x_cell * 2
        y_cell_m = y_cell * 2
        plt.plot(range(len(power)), power)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (kW/m^2)')
        plt.title('Power From Surface Cell\nx={}m, y={}m, Rolling Avg={}'.format(x_cell_m,y_cell_m,roll_avg))
        plt.savefig(os.path.join(save_dir, 'SufaceCellx-{}_y-{}_RollingAvg={}.png'.format(x_cell_m,y_cell_m,roll_avg)))
        plt.close()

    CF_PATH = os.path.join(save_dir, "Cell_Figures")
    if not os.path.exists(CF_PATH):
        os.makedirs(CF_PATH)
    # burned_cells = find_cells_that_burned(xarr_residence_time, SIM_PARAMS, n=100)
    # power_metrics = {'max_power':[],'total_eng':[]}
    # import time
    # strt_time = time.time()
    # for i, bc in enumerate(burned_cells):
    #     print(i)
    #     print(time.time()-strt_time)
    #     x_cell, y_cell = bc
    #     start_t = int(xarr_arrival_time[y_cell, x_cell])
    #     stop_t = int(xarr_fire_stop_time[y_cell, x_cell])
    #     cell_power = ds.surfEnergy[start_t:stop_t,y_cell,x_cell]
    #     cell_power = cell_power.to_numpy()
    #     df = pd.DataFrame({"cell_power_1":cell_power})
    #     roll_vals = [1, 5, 10, 15, 20]
    #     roll_names = ["cell_power_1", "cell_power_5", "cell_power_10", "cell_power_15", "cell_power_20"]
    #     for i in range(len(roll_vals)):
    #         rn = roll_names[i]
    #         rv = roll_vals[i]
    #         if i == 0:
    #             cp = cell_power
    #         else:
    #             df[rn]=df[roll_names[0]].rolling(rv, min_periods=1).mean()
    #             cp = np.array(df[rn])
    #         temp_sav_dir = os.path.join(CF_PATH, rn)
    #         if not os.path.exists(temp_sav_dir):
    #             os.makedirs(temp_sav_dir)
    #         build_power_graph(cp, x_cell, y_cell, rv, temp_sav_dir)
    #     power_metrics["max_power"].append(float(xarr_max_power[y_cell, x_cell]))
    #     power_metrics["total_eng"].append(float(cell_power.sum()))
    #     np.savetxt(os.path.join(temp_sav_dir,'SufaceCellx-{}_y-{}.csv'.format(x_cell*2,y_cell*2)), cell_power, delimiter=",")
    # df = pd.DataFrame(power_metrics)
    # df.to_csv(os.path.join(save_dir,'power_metrics.csv'))

    # plt.hist(df['max_power'])
    # plt.title('Maximum Power for Selected Cells')
    # plt.xlabel('Max Power kW/m^2')
    # plt.ylabel('Frequency')
    # plt.savefig(os.path.join(CF_PATH,'max_power_hist_select_cells.png'))
    # plt.close()

    # plt.hist(df['total_eng'])
    # plt.title('Total Energy for Selected Cells')
    # plt.xlabel('Total Energy kJ/m^2')
    # plt.ylabel('Frequency')
    # plt.savefig(os.path.join(CF_PATH,'total_eng_hist_select_cells.png'))
    # plt.close()

    #Build Figures
    def scale_for_figs_x_and_y(arr, dx=2, dy=2):
        arr = np.array(arr)
        arr = np.repeat(np.repeat(arr, dy, axis=0), dx, axis=1)
        plt.imshow(arr, cmap='YlOrRd', origin="lower")

    def build_hist_remove_zeros(arr):
        arr = np.array(arr)
        arr = arr[arr>0]
        plt.hist(arr)

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
    build_hist_remove_zeros(xarr_residence_time)
    plt.title('Residence Time Histogram')
    plt.xlabel('Residence Time (s)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(CF_PATH,'residence_time_hist.png'))
    plt.close()
    np.savetxt(os.path.join(save_dir,'ResidenceTimes.csv'), xarr_residence_time, delimiter=",")

    scale_for_figs_x_and_y(xarr_total_energy)
    plt.colorbar()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Total Energy (kJ/m^2)")
    plt.savefig(os.path.join(save_dir,"total_energy.png"))
    plt.close()
    build_hist_remove_zeros(xarr_total_energy)
    plt.title('Total Energy Histogram')
    plt.xlabel('Total Energy (kJ/m^2)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(CF_PATH,'total_energy_hist.png'))
    plt.close()
    np.savetxt(os.path.join(save_dir,'MaxPower.csv'), xarr_max_power, delimiter=",")

    scale_for_figs_x_and_y(xarr_max_power)
    plt.colorbar()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Max Power (kW/m^2)")
    plt.savefig(os.path.join(save_dir,"max_power.png"))
    plt.close()
    build_hist_remove_zeros(xarr_max_power)
    plt.title('Max Power Histogram')
    plt.xlabel('Max Power (kW/m^2)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(CF_PATH,'max_power_hist.png'))
    plt.close()
    np.savetxt(os.path.join(save_dir,'MaxPower.csv'), xarr_max_power, delimiter=",")

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
