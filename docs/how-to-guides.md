## Inputs

QUIC-Fire is controlled by a deck of input files that specify fuel parameters, wind conditions, ignitions, topography, etc. 
The inputs module provides a simple interface to programatically create and modify new input file decks or read existing decks.
The following guide provides step-by-step instructions for working with simple QUIC-Fire input decks. 
Please see [inputs](reference.md%quicfire_tools.inputs) for full documentation.

### Create a QUIC-Fire simulation

QUIC-Fire input decks are created and modified using the [`SimulationInputs`](reference.md#quicfire_tools.inputs.SimulationInputs) class.

```python
from quicfire_tools.inputs import SimulationInputs
```

To create a simulation, use the [`create_simulation`](reference.md#quicfire_tools.inputs.SimulationInputs.create_simulation) method, and
input basic information about the burn domain, initial weather conditions, and simulation parameters. In the following example, we are
setting up a simulation that is 200x200 grid cells, with one vertical layer of fuels. The wind is blowing from the east at 1.7 m/s, and 
the simulation will run for 600 seconds of simulation time.

```python
simulation = SimulationInputs.create_simulation(
    nx=200,
    ny=200,
    fire_nz=1,
    wind_speed=1.7,
    wind_direction=90,
    simulation_time=600
)
```

- **nx** and **ny** define the number of cells in the x and y axis of the simulation grid.
- **fire_nz** determines the number of cells in the z-axis for the fire grid.
- **wind_speed** and **wind_direction** set the initial wind conditions.
- **simulation_time** specifies how long the simulation will run in seconds.

This creates a basic input deck with default values for fuels, ignitions, and topography. This simulation can be run as-is, or modified using 
methods described below.

### Use set_* methods for common simulation parameters

Once a simulation is created, it can be modified directly through methods in the `SimulationInputs` class. For common modifications, convenience methods starting with `set_*` are available. Guides for all `set_*` methods are below.

#### Set uniform fuel conditions

To set and modify fuel parameters for uniform fuels, use the [`set_uniform_fuels`](reference.md#quicfire_tools.inputs.SimulationInputs.set_uniform_fuels) method.
In the following example, we are setting surface fuel density to 0.7 kg/m^3, fuel moisture to 5% of its dry weight, and fuel height to 1 meter.

```python
simulation.set_uniform_fuels(
    fuel_density=0.7,
    fuel_moisture=0.10,
    fuel_height=1.0
)
```

- **fuel_density** sets the surface fuel density in kg/m^3.
- **fuel_moisture** sets the surface fuel moisture content as a fraction of the fuel's dry weight.
- **fuel_height** sets the surface fuel height in meters.

#### Set rectangle ignition pattern

By default, ignitions are set up perpendicular to the wind direction specified in `create_simulation`, spanning 80% of the domain
edge length, 10% from either side. A different igntion line can be created using the [`set_rectangle_ignition`](reference.md#quicfire_tools.inputs.SimulationInputs.set_rectangle_ignition) method.

```python
simulation.set_rectangle_ignition(
    x_min=150,
    y_min=100,
    x_length=10,
    y_length=100
)
```

- **x_min** and **y_min** set the coordinates of the bottom left corner of the ignition zone. These coordinates are specified in meters, not grid cells.
- **x_length** and **y_length** set the length of the ignition zone in the x and y directions in meters.

Ignition patterns other than rectangular can be specified using the [ignitions module](how-to-guides.md#define-ignitions-using-ignitionspy).
Please see [ignitions](reference.md#quicfire_tools.ignitions) for a full list of available ignition patterns.

#### Specify output files

Depending on the desired analyses, different files (.bin) may need to be output. To specify with files should be written, use the [`set_output_files`](reference.md#quicfire_tools.inputs.SimulationInputs.set_output_files) method and set the desired files to true. In the following example, we are specifying outputs for fuel density, various emissions, and wind grid components.

```python
simulation.set_output_files(
    fuel_dens=True,
    emissions=True,
    qu_wind_inst=True
)
```

- **fuel_dens** specifies a compressed array fuel density output.
- **emissions** specifies CO, PM2.5, and water emissions outputs.
- **qu_wind_inst** specifies gridded wind components (u, v, w) for the QUIC (wind) grid.

By default, an unmodified simulation will output fuel density and QUIC winds. After using `set_output_files`, all output files not specified as `True` will be set to false.

Please see the [`QUIC_fire`](reference.md#quicfire_tools.inputs.QUIC_fire) class for a full list of available output files.

#### Set custom fuels, ignitions, and topography from .dat files

For more advanced simulations, it may be necessary to specify custom fuel parameters, ignition patterns, or topography using .dat files. These files must be created and provided by the user, but the [`set_cunstom_simulation`](reference.md#quicfire_tools.inputs.SimulationInputs.set_custom_simualtion) method can be used to specify which custom .dat files should be used in the simulation.

```python
simulation.set_custom_simulation(
    fuel = True,
    ignition = True,
    topo = True
)
```

- **fuel** specifies that fuel density, moisture, and height are provided by treesrhof.dat, treesmoist.dat, and treesfueldepth.dat, respectively.
- **ignition** specifies that ignitions are provided by ignite.dat
- **topo** specifies that topography is provided by topo.dat

Any parameter not set to `True` will not be specified by a custom .dat file.

### Directly modify input files

Every QUIC-Fire input file is represented in an `InputFile` class in the [inputs](reference.md#quicfire_tools.inputs) module. These classes can be accessed as attributes of the `SimulationInputs` class, where their parameters can be modified. Some of the more commonly modified input files are below:

- **QUIC_fire** contains parameters relating to the fire simulation, including the fire grid, fuels, ignitions, and output files.
- **QU_TopoInputs** contains parameters relating to the underlying topography of the simulation, including topography type and smoothing parameters.
- **Sensor1** contains parameters relating to the wind conditions throughout the simulation.
- **Runtime_Adananced_User_Inputs** contains parameters relating to the internal processing of the simulation.

#### QUIC_fire.inp

The [`QUIC_fire`](reference.md#quicfire_tools.inputs.QUIC_fire) input file class contains many parameters relating to the fire simulation. Once a simulation is created, these parameters can be accessed and modified through [`SimulationInputs.quic_fire`](reference.md#quicfire_tools.inputs.SimulationInputs.quic_fire). In the following example, some parameters not accessed by the `set_*` methods are modified.

```python
simulation.quic_fire.random_seed = 47
simulation.quic_fire.out_time_fire = 60
simulation.quic_fire.ignitions_per_cell = 5
simulation.quic_fire.auto_kill = 1
```

Please see [`QUIC_fire`](reference.md#quicfire_tools.inputs.QUIC_fire) for a full list of parameters associated with the QUIC_fire.inp input file.

#### QU_TopoInputs.inp

The [`QU_TopoInputs`](reference.md#quicfire_tools.inputs.QU_TopoInputs) input file class contains parameters relating to the underlying topography of the simulation. Once a simulation is created, these parameters can be accessed and modified throught [`QU_TopoInputs`](reference.md#quicfire_tools.inputs.SimulationInputs.qu_topoinputs). For information on setting custom topography using built-in methods in the [topography](reference.md#quicfire_tools.topography) module, see [Set custom topography](how-to-guides.md#define-topography-using-topographypy). In the following example, some parameters not accessed by the `set_*` methods or the `topography` module are modified.

```python
simulation.qu_topoinputs.smoothing_passes = 500
simulation.qu_topoinputs.sor_iteration = 300
```

Please see [`QU_TopoInputs`](reference.md#quicfire_tools.inputs.QU_TopoInputs) for a full list of parameters associated with the QUIC_fire.inp input file.

#### sensor1.inp
The ['Sensor1](reference.md#quicfire_tools.inputs.Sensor1) input file class contains parameters defining wind conditions throughout the simulation. Once a simulation is created, these parameters can be accessed and modified through [`SimulationInputs.sensor1`](reference.md#quicfire_tools.inputs.SimulationInputs.sensor1).

```python
simulation.sensor1.sensor_height = 10
```

Please see [`Sensor1`](reference.md#quicfire_tools.inputs.Senor1) for a full list of parameters associated with the sensor1.inp input file.

#### Runtime_Advanced_User_Inputs.inp

The [`Runtime_Advanced_User_Inputs`](reference.md#quicfire_tools.inputs.Runtime_Advanced_User_Inputs) input file class condtains two parameters relating to the internal processing of the simulation. Once a simulation is created, these parameters can be accessed and modified throught [`Runtime_Advanced_User_Inputs`](reference.md#quicfire_tools.inputs.SimulationInputs.runtime_advanced_user_inputs). In the following example, the number of CPUs/threads is specifed, along with whether or not to use the adaptive computational window.

```python
simulation.runtime_advanced_user_inputs.num_cpus = 1
simulation.runtime_advanced_user_inputs.use_acw = 1
```

### Define topography using topography.py

In addition to flat topography (the default) and [custom topography](how-to-guides.md#set-custom-fuels-ignitions-and-topography-from-dat-files), there are various built-in topography types, all of which can be set using classes in the [topopgraphy](reference.md#quicfire_tools.topography) module. The following example creates and sets Gaussian hill topography.

```python
from quicfire_tools.topography import GaussianHillTopo

topo = GaussianHillTopo(
    x_hilltop = 100,
    y_hilltop = 150,
    elevation_max = 50,
    elevation_std = 15
)

simulation.qu_topoinputs.topo_type = topo
```

Please see [topopgraphy](reference.md#quicfire_tools.topography) for a full list of topography types.

### Define ignitions using ignitions.py

In addition to [rectangle ignitions](how-to-guides.md#set-rectangle-ignition-patterns) (the default) and [custom ignitions](how-to-guides.md#set-custom-fuels-ignitions-and-topography-from-dat-files), there are various build-in ignition patterns, all of which can be set using class in the [ignnitions](reference.md#quicfire_tools.ignitions) module. The following example creates and sets a circular ring ignition.

```python
from quicfire_tools.ignitions import CurcularRingIgnition

ignition = CircularRingIgnition(
    x_min = 50,
    y_min = 50,
    x_length = 20,
    y_length = 20,
    ring_width = 10
)
```

Please see [igntions](reference.md#quicfire_tools.ignitions) for a full list of ignition patterns.

### Set weather conditions

What goes here?

### Read and write input file decks

#### Write a SimulationInputs object to an input deck

#### Read in an existing input deck

#### Write a simulation to JSON

#### Load a simulation from a JSON file

## Outputs

### Load a numpy array from a QUIC-Fire output file

### Load a time slice from a QUIC-Fire output file

### Load a dask array from a QUIC-Fire output file
