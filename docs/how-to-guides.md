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

#### Set custom ignition patterns

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

Ignition patterns other than rectangular can be specified using the `ignitions` module.
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

### Set custom topography

### Set weather conditions

### Modify simulation outputs

### Write a simulation to JSON

### Load a simulation from a JSON file

## Outputs

### Load a numpy array from a QUIC-Fire output file

### Load a time slice from a QUIC-Fire output file

### Load a dask array from a QUIC-Fire output file
