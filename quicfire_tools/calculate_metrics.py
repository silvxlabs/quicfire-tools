"""
Module for running standard spatial metric calulations.
"""
# Internal imports
from quicfire_tools.outputs import SimulationOutputs

# External imports
import os
import matplotlib.pyplot as plt
import pickle
import random
import xarray as xr
import numpy as np
import pandas as pd

def _build_save_dir(simulation_outputs: SimulationOutputs, OUTPUT_NAME):
    save_dir = os.path.join(simulation_outputs.plots_directory, OUTPUT_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def surfeng_metrics(simulation_outputs: SimulationOutputs, SMOLDER_THRESHOLD = 25):
    OUTPUT_NAME = 'surfEnergy'
    save_dir = _build_save_dir(simulation_outputs, OUTPUT_NAME)
    ds = xr.open_zarr(simulation_outputs.get_output(OUTPUT_NAME).zarr_path, consolidated=True)

    #Calc percent burned & time for max power
    ds = ds.fillna(0) #Convert nan to 0 for dask
    xarr_max_power_time = ds.data.argmax('time')
    xarr_max_power = ds.data[xarr_max_power_time.compute()]   
    xarr_max_power_time = xr.where(xarr_max_power_time==0,np.nan,xarr_max_power_time)

    ###Calc Times: arrival, stop, residence
    ##Removed forloop to improve speed
    #https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    #https://stackoverflow.com/questions/66305130/index-of-last-occurence-of-true-in-every-row
    burned_binary = (ds>SMOLDER_THRESHOLD)
    #Arrival time
    xarr_arrival_time = burned_binary.data.argmax('time')
    xarr_arrival_time = xr.where(xarr_arrival_time==0,np.nan,xarr_arrival_time) #0 to nan
    xarr_arrival_time = xarr_arrival_time.compute()

    #Fire stop time
    xarr_fire_stop_time = burned_binary.dims['time'] - burned_binary.data[::-1,:,:].argmax('time') - 1
    xarr_fire_stop_time = xr.where((burned_binary.data[-1,:,:]==0) & (xarr_fire_stop_time==xarr_fire_stop_time.max()),np.nan,xarr_fire_stop_time) #non-burning cells to nan
    xarr_fire_stop_time = xarr_fire_stop_time.compute()
    del burned_binary

    xarr_residence_time = xarr_fire_stop_time - xarr_arrival_time        
    
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
    def build_power_graph(power, x_cell, y_cell, save_dir=save_dir):
        x_cell_m = x_cell * 2
        y_cell_m = y_cell * 2
        plt.plot(range(len(power)), power)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (kW/m^2)')
        plt.title('Power From Surface Cell x={}m, y={}m'.format(x_cell_m,y_cell_m))
        plt.savefig(os.path.join(save_dir, "Cell_Figures", 'SufaceCellx-{}_y-{}.png'.format(x_cell_m,y_cell_m)))
        plt.close()

    CF_PATH = os.path.join(save_dir, "Cell_Figures")
    if not os.path.exists(CF_PATH):
        os.makedirs(CF_PATH)
    burned_cells = find_cells_that_burned(xarr_residence_time, simulation_outputs.params, n=10)
    power_metrics = {'max_power':[],'total_eng':[]}
    import time
    strt_time = time.time()
    for i, bc in enumerate(burned_cells):
        print(i)
        print(time.time()-strt_time)
        x_cell, y_cell = bc
        start_t = int(xarr_arrival_time[y_cell, x_cell])
        stop_t = int(xarr_fire_stop_time[y_cell, x_cell])
        cell_power = ds.data[start_t:stop_t,y_cell,x_cell]
        build_power_graph(cell_power, x_cell, y_cell)
        power_metrics["max_power"].append(float(xarr_max_power[y_cell, x_cell]))
        power_metrics["total_eng"].append(float(cell_power.sum()))
        np.savetxt(os.path.join(save_dir, "Cell_Figures",'SufaceCellx-{}_y-{}.csv'.format(x_cell*2,y_cell*2)), cell_power, delimiter=",")
    
    df = pd.DataFrame(power_metrics)
    df.to_csv(os.path.join(save_dir,'power_metrics.csv'))

    plt.hist(df['max_power'])
    plt.title('Maximum Power for Selected Cells')
    plt.xlabel('Max Power kW/m^2')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir,'max_power_hist.png'))
    plt.close()

    plt.hist(df['total_eng'])
    plt.title('Total Energy for Selected Cells')
    plt.xlabel('Total Energy kJ/m^2')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir,'total_eng_hist.png'))
    plt.close()

    #Build Figures
    def scale_for_figs_x_and_y(arr, dx=2, dy=2):
        arr = np.array(arr)
        arr = np.repeat(np.repeat(arr, dy, axis=0), dx, axis=1)
        plt.imshow(arr, cmap='YlOrRd', origin="lower")

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
    np.savetxt(os.path.join(save_dir,'ResidenceTimes.csv'), xarr_residence_time, delimiter=",")

    scale_for_figs_x_and_y(xarr_max_power)
    plt.colorbar()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Max Power (kW/m^2)")
    plt.savefig(os.path.join(save_dir,"max_power.png"))
    plt.close()
    np.savetxt(os.path.join(save_dir,'MaxPower.csv'), xarr_max_power, delimiter=",")
