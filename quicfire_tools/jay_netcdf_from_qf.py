from quicfire_tools.outputs import SimulationOutputs
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import sys

#CHANGE ME TO YOUR PROJECT FILE!!!!!!
base_pf = '/Users/joliveto/Desktop/Projects/SOPHIE_GUI/CleanGUI/GUI/projects/Konza_S8/' #REPLACE WITH YOUR OWN BASE PROJECT PATH
#CHANGE ME TO YOUR PROJECT FILE!!!!!!


#================================================================
#================================================================
#================================================================
# NOTHING BELOW SHOULD BE CHANGED
# NOTHING BELOW SHOULD BE CHANGED
#================================================================
#================================================================
#================================================================

#FUCTION
def createDataSet(fname,v,dx):
    if os.path.isfile(fname):
        os.remove(fname)
    netfile = nc.Dataset(fname, "w")

    var = v.to_numpy()
    timerange = v.times
    time,nz,ny,nx = np.shape(var)
    #save the match of days
    time_dim = netfile.createDimension('time', time)
    nz_dim   = netfile.createDimension('nz', nz)
    ny_dim   = netfile.createDimension('ny', ny)
    nx_dim   = netfile.createDimension('nx', nx)
    time_new = netfile.createVariable('time', np.int64, ('time',))
    nz_new   = netfile.createVariable('nz', np.int64, ('nz',))
    ny_new   = netfile.createVariable('ny', np.int64, ('ny',))
    nx_new   = netfile.createVariable('nx', np.int64, ('nx',))

    nx_new[:] = range(0,nx*dx,dx)
    ny_new[:] = range(0,ny*dx,dx)
    nz_new[:] = range(0,nz)
    time_new[:] = np.array(timerange)
    return netfile

def saveDataset(netfile,var,varname):
    avg_var = netfile.createVariable(varname,np.float32,('time','nz','ny','nx'), fill_value=-9999) # note: unlimited dimension is leftmost
    avg_var[:,:,:,:] = var
    return 0

def read_input_info(pf):
    nx = 0.0
    ny = 0.0
    nz = 0.0
    dx = 0.0   
    wind_nz = 0.0
    if os.path.exists(os.path.join(pf,'gridlist')):
        with open(os.path.join(pf,'gridlist'), 'r') as fire:
            l = fire.readlines()
            nx = int(l[0].split('=')[1])
            ny = int(l[1].split('=')[1])
            nz = int(l[2].split('=')[1])
            dx = int(l[5].split('=')[1])
    else: 
        print('WARNING, NO GRIDLIST FOUND!')
        sys.exit()
    
    if os.path.exists(os.path.join(pf,'QU_simparams.inp')):
        with open(os.path.join(pf,'QU_simparams.inp'), 'r') as wind:
            w = wind.readlines()
            wind_nz = int(w[3].split('!')[0])
    else: 
        print('WARNING, NO QUSIMPARAMS FOUND!')
        sys.exit()

    return nx,ny,nz,dx,wind_nz
#END OF FUNCTION
    
projectname = os.path.normpath(base_pf).split(os.path.sep)[-1]
print(projectname)
nx,ny,nz,dx,wind_nz = read_input_info(base_pf)
output_directory = os.path.join(base_pf,"Output")

#for fire grid outputs
simulation_outputs = SimulationOutputs(output_directory, nz, ny, nx)
sim_time = simulation_outputs.get_output(simulation_outputs.list_available_outputs()[0]).times

fname = os.path.join(base_pf,projectname+'_fireoutputs.nc')
netfile = createDataSet(fname,
                        simulation_outputs.get_output(simulation_outputs.list_available_outputs()[0]),
                        dx)
for output in simulation_outputs.list_available_outputs():
    print('WRITING ',output)
    output_file = simulation_outputs.get_output(output)
    if len(sim_time) != len(output_file.times):
        print('ERROR IN SIMULATION TIME LENGTH!!, NOT WRITING ',output)
    else:
        output_data = output_file.to_numpy()
        saveDataset(netfile,output_data,output)

netfile.close()


#test = nc.Dataset(fname, 'r')
#print(test)
#test.close()

