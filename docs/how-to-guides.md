## Inputs

QUIC-Fire is controlled by a deck of input files that specify fuel parameters, wind conditions, ignitions, topography, etc.
The inputs module provides a simple interface to programatically create and modify new input file decks or read existing decks.
The following guides provide step-by-step instructions for working with simple QUIC-Fire input decks.
Please see [inputs](reference.md#quicfire_tools.inputs) for full documentation.

### How to create a basic QUIC-Fire simulation

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

### How to use set_* methods for common simulation parameters

Once a simulation is created, it can be modified directly through methods in the `SimulationInputs` class.
For common modifications, convenience methods starting with `set_*` are available. Guides for all `set_*` methods are below.

#### How to set uniform fuel conditions

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

#### How to set a rectangle ignition pattern

By default, ignitions are set up perpendicular to the wind direction specified in `create_simulation`, spanning 80% of the domain
edge length, 10% from either side. A different igntion line can be created using the [`set_rectangle_ignition`](reference.md#quicfire_tools.inputs.SimulationInputs.set_rectangle_ignition) method.
In the following example, ignitions start 150m from the western edge of the domain (x=0) and 100m from the southern edge of the domain (y=0). The rectangle extends 10 meters east in the x-direction,
and 100m north in the y-direction.

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

#### How to specify which files to output

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

#### How to set custom fuels, ignitions, and topography from .dat files

For more advanced simulations, it may be necessary to specify custom fuel parameters, ignition patterns, or topography using .dat files. These files must be created and provided by the user, but the [`set_custom_simulation`](reference.md#quicfire_tools.inputs.SimulationInputs.set_custom_simulation) method can be used to specify which custom .dat files should be used in the simulation.

```python
simulation.set_custom_simulation(
    fuel_density=True,    # Use treesrhof.dat for fuel density
    fuel_moisture=True,   # Use treesmoist.dat for fuel moisture
    fuel_height=True,     # Use treesfueldepth.dat for fuel height
    size_scale=False,     # Use default size scale
    patch_and_gap=False,  # Use default patch and gap
    ignition=True,        # Use ignite.dat for ignition pattern
    topo=True,            # Use topo.dat for topography
    interpolate=False     # Don't interpolate custom fuel inputs (default)
)
```

The parameters control which aspects of the simulation will use custom .dat files:

- **fuel_density**: Use treesrhof.dat for fuel density data
- **fuel_moisture**: Use treesmoist.dat for fuel moisture data
- **fuel_height**: Use treesfueldepth.dat for fuel height data
- **size_scale**: Use treesss.dat for size scale data
- **patch_and_gap**: Use patch.dat and gap.dat for patch and gap data
- **ignition**: Use ignite.dat for ignition pattern
- **topo**: Use topo.dat for topography
- **interpolate**: Control whether custom fuel inputs are interpolated

Any parameter not set to `True` will not be specified by a custom .dat file.

##### Understanding the interpolate parameter

The `interpolate` parameter controls how QUIC-Fire handles the grid spacing of custom fuel inputs. This is particularly important when working with data from sources like FastFuels where the fuel grid spacing may not match the QUIC-Fire grid spacing.

When `interpolate=False` (default):
- Sets fuel flags to 3 (for enabled fuel parameters)
- Assumes fuel grid spacing matches QUIC-Fire grid spacing
- No interpolation is performed
- Good for when you know your input data matches the QUIC-Fire grid exactly

When `interpolate=True`:
- Sets fuel flags to 4 (for enabled fuel parameters)
- Allows fuel grid spacing to differ from QUIC-Fire grid
- Interpolates fuel data to match QUIC-Fire grid
- May be required for proper functionality in some older versions of QUIC-Fire

For example, to properly handle FastFuels data which typically has different grid spacing:

```python
simulation.set_custom_simulation(
    fuel_density=True,
    fuel_moisture=True, 
    fuel_height=True,
    interpolate=True
)
```

Note: For versions of QUIC-Fire ≤ v6.0.0, setting `interpolate=True` may be required for custom fuels to work properly, regardless of grid spacing.

### How to directly modify input files

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

### How to define topography using topography.py

In addition to flat topography (the default) and [custom topography](how-to-guides.md#set-custom-fuels-ignitions-and-topography-from-dat-files), there are various built-in topography types, all of which can be set using classes in the [topopgraphy](reference.md#quicfire_tools.topography) module. The following example creates and sets Gaussian hill topography.

```python
from quicfire_tools.topography import GaussianHillTopo

# First, create an TopoType class
topo = GaussianHillTopo(
    x_hilltop=100,
    y_hilltop=150,
    elevation_max=50,
    elevation_std=15
)

# Next, assign it to the topography attribute of the qu_topoinputs InputFile
simulation.qu_topoinputs.topography = topo
```

Please see [topopgraphy](reference.md#quicfire_tools.topography) for a full list of topography types.

### How to define ignitions using ignitions.py

In addition to [rectangle ignitions](how-to-guides.md#set-rectangle-ignition-patterns) (the default) and [custom ignitions](how-to-guides.md#set-custom-fuels-ignitions-and-topography-from-dat-files), there are various build-in ignition patterns, all of which can be set using class in the [ignnitions](reference.md#quicfire_tools.ignitions) module. The following example creates and sets a circular ring ignition.

```python
from quicfire_tools.ignitions import CurcularRingIgnition

# First, create an IgnitionType class
ignition = CircularRingIgnition(
    x_min = 50,
    y_min = 50,
    x_length = 20,
    y_length = 20,
    ring_width = 10
)

# Next, assign it to the ignition attribute of the quic_fire InputFile
simulation.quic_fire.ignition
```

Please see [igntions](reference.md#quicfire_tools.ignitions) for a full list of ignition patterns.

### How to manage wind conditions

Wind conditions in QUIC-Fire are managed through wind sensors, which specify wind speeds and directions at specific locations and times. Multiple wind sensors can be used to represent spatial variation in wind conditions across the simulation domain.

#### Adding wind sensors

The simplest way to add a wind sensor is using the `add_wind_sensor` method. In the following example, we're creating a sensor with a constant wind speed of 5 m/s blowing from the east:

```python
simulation.add_wind_sensor(
    wind_speeds=5.0,
    wind_directions=90,
    wind_times=0
)
```

For varying wind conditions, provide lists of values that change over time. Times are specified in seconds relative to the simulation start (t=0):

```python
simulation.add_wind_sensor(
    wind_speeds=[5.0, 7.0, 6.0],       # Wind speeds in m/s
    wind_directions=[90, 180, 135],     # Wind directions in degrees
    wind_times=[0, 600, 1200],         # Changes at 0, 10, and 20 minutes
    sensor_height=10.0,                 # Sensor height in meters
    x_location=50.0,                    # X-coordinate in meters
    y_location=50.0,                    # Y-coordinate in meters
    sensor_name="station_1"             # Optional custom name
)
```

- **wind_speeds** specifies wind speeds in meters per second.
- **wind_directions** specifies wind directions in degrees (0° = North, 90° = East).
- **wind_times** specifies when each wind condition begins, in seconds from simulation start.
- **sensor_height** sets the height of the sensor in meters (defaults to 6.1m/20ft).
- **x_location** and **y_location** set the sensor position in meters.
- **sensor_name** provides a custom identifier for the sensor.

#### Adding wind sensors from data files

For wind data stored in CSV files or pandas DataFrames, use the `add_wind_sensor_from_dataframe` method:

```python
import pandas as pd

# Read wind data from CSV
wind_data = pd.read_csv("weather_station_data.csv")

simulation.add_wind_sensor_from_dataframe(
    df=wind_data,
    x_location=100.0,
    y_location=100.0,
    sensor_height=6.1,
    time_column="time_seconds",         # Column containing times
    speed_column="windspeed_ms",        # Column containing wind speeds
    direction_column="direction_deg",    # Column containing wind directions
    sensor_name="weather_station_2"
)
```

The DataFrame must contain columns for:
- Times in seconds relative to simulation start
- Wind speeds in meters per second
- Wind directions in degrees

#### Removing wind sensors

To remove a wind sensor from the simulation, use the `remove_wind_sensor` method with the sensor's name:

```python
simulation.remove_wind_sensor("station_1")
```

#### Notes on wind sensors

- Multiple wind sensors can be used to represent spatial variation in wind conditions.
- Wind times must be in ascending order and relative to simulation start (t=0).
- Wind directions must be in degrees from 0° to 360°.
- At least one wind sensor must remain in the simulation.
- The simulation automatically manages wind field update times based on all active sensors.
### How to read and write input file decks

#### How to write a SimulationInputs object to an input deck

Once a `SimulationInputs` object has been created and modified, it can be used to write all the necessary input files to a directory containing the QUIC-Fire executable.
This is done using the [`write_inputs`](reference.md#quicfire_tools.inputs.SimulationInputs.write_inputs) method.

```python
simulation.write_inputs("path/to/directory")
```

#### How to load an existing input deck

Input decks that already exist may be read in as a `SimulationInputs` object. The [`from_directory`](reference.md#quicfire_tools.inputs.SimulationInputs.from_directory) method
is used for this purpose.

```python
simulation = SimulationInputs.from_directory("path/to/directory")
```

#### How to write a simulation to JSON

All the information in a `SimulationInputs` object may be saved in JSON format using the [`to_json`](reference.md#quicire_tools.inputs.SimulationInputs.to_json) method.

```python
simulation.to_json("path/to/directory")
```

#### How to load a simulation from a JSON file

Input decks saved in JSON format by quicfire-tools can be loaded into a `SimulationInputs` oject using the [`from_json`](reference.md#quicfire_tools.inputs.SimulationInputs.from_json) method.

```python
simulation = SimulationInputs.from_json("path/to/directory")
```

## Outputs

### How to create a SimulationOutputs object from a directory containing QUIC-Fire output files

To read and process QUIC-Fire output files, use the [`SimulationOutputs`](reference.md#quicfire_tools.outputs.SimulationOutputs) class.
You need to specify the path to the directory containing the output files, as well as the number of cells in the z, y, and x directions of the simulation grid.

```python
from quicfire_tools.outputs import SimulationOutputs

output_directory = "/path/to/output/directory"
nz = 56  # number of z cells
ny = 100  # number of y cells
nx = 100  # number of x cells

simulation_outputs = SimulationOutputs(output_directory, nz, ny, nx)
```

### How to get an OutputFile object from a SimulationOutputs object

Once you have a `SimulationOutputs` object, you can use it to get an [`OutputFile`](reference.md#quicfire_tools.outputs.OutputFile) object for a specific output file.
To do this, use the [`get_output_file`](reference.md#quicfire_tools.outputs.SimulationOutputs.get_output_file) method of the `SimulationOutputs` class.

```python
from quicfire_tools.outputs import SimulationOutputs
simulation_outputs = SimulationOutputs("/path/to/output/directory", 56, 100, 100)

output_name = "fire-energy_to_atmos"  # replace with the name of the output you are interested in
output_file = simulation_outputs.get_output(output_name)
```

### How to get a numpy array from an OutputFile object at a specific timestep

Once you have an `OutputFile` object, you can use it to get a numpy array for the output data using the
[`to_numpy`](reference.md#quicfire_tools.outputs.OutputFile.to_numpy) method of the `OutputFile` instance. You can
specify the timestep(s) you are interested in. If you don't provide a timestep, all timesteps will be returned:

```python
from quicfire_tools.outputs import SimulationOutputs
simulation_outputs = SimulationOutputs("/path/to/output/directory", 56, 100, 100)
output_file = simulation_outputs.get_output("fire-energy_to_atmos")

timestep = 0  # replace with the timestep you are interested in
output_data = output_file.to_numpy(timestep)
```

#### How to get the data across all timesteps

By not specifying a timestep, you can get the data across all timesteps. This will return a 4D numpy array with the
shape (nt, nz, ny, nx), where nt is the number of timesteps.

_caution:_ please be aware of the memory requirements your data. This approach may not be appropriate for large simulations
or for computers with limited memory.

```python
from quicfire_tools.outputs import SimulationOutputs
simulation_outputs = SimulationOutputs("/path/to/output/directory", 56, 100, 100)
output_file = simulation_outputs.get_output("fire-energy_to_atmos")

output_data = output_file.to_numpy()
```

### How to get a dask array from a QUIC-Fire output file

You can use the [`to_dask`](reference.md#quicfire_tools.outputs.SimulationOutputs.to_dask) method of the `SimulationOutputs` class to
get a dask array for the desired output file. This method returns a
[`dask.array`](https://docs.dask.org/en/latest/array.html) object, which can be used to read and process data in parallel,
perform lazy computations, and work with data that is too large to fit in memory.

Note that this method returns a dask array, not a numpy array. To get a numpy array, you can use the
[`compute`](https://docs.dask.org/en/latest/array-api.html#dask.array.Array.compute) method of the dask array to perform
the computations and load data into memory.

The dask array returned by this method has the same format as the numpy array returned by the
[`to_numpy`](reference.md#quicfire_tools.outputs.OutputFile.to_numpy) method, a 4D array with the shape
(nt, nz, ny, nx), where nt is the number of timesteps.

```python
from quicfire_tools.outputs import SimulationOutputs
simulation_outputs = SimulationOutputs("/path/to/output/directory", 56, 100, 100)
dask_array = simulation_outputs.to_dask("fire-energy_to_atmos")
```
