## Creating a QUIC-Fire Simulation with quicfire-tools.

Welcome to the tutorial on how to create a QUIC-Fire simulation using quicfire-tools, a Python package designed to
streamline the process of managing QUIC-Fire input file decks and processing output files. This tutorial will take you
through the necessary steps to set up a basic QUIC-Fire simulation, focusing on using the SimulationInputs class from
the inputs module.

### Prerequisites

Before starting this tutorial, ensure that you have:

- Python 3.8 or higher installed on your system.
- The quicfire-tools package installed on your system. If you do not have quicfire-tools installed, please see the
  [installation instructions](index.md#installation) in the documentation.

### Step 1: Import the SimulationInputs class

Start by importing the SimulationInputs class from the quicfire_tools.inputs module:

```python
from quicfire_tools.inputs import SimulationInputs
```

### Step 2: Creating a Uniform Line Fire Simulation

In this step, we will create a function to set up a basic QUIC-Fire simulation. This involves initializing the
simulation, defining fuel characteristics, setting the ignition area, and specifying output files.

#### Step 2.1: Initialize the simulation

First, we initialize the simulation with the grid size, simulation time, and wind conditions. The
[`create_simulation`](reference.md#quicfire_tools.inputs.SimulationInputs.create_simulation) method of
[`SimulationInputs`](reference.md#quicfire_tools.inputs.SimulationInputs) is used for this purpose.

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

#### Step 2.2: Define Uniform Fuel Characteristics

Next, we define the characteristics of the fuel. The [`set_uniform_fuels`](reference.md#quicfire_tools.inputs.SimulationInputs.set_uniform_fuels)
method sets the fuel density, moisture, and height uniformly across the simulation grid.

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

#### Step 2.3: Set the Ignition Area

Now, we specify the area where the fire will start. The [`set_rectangle_ignition`](reference.md#quicfire_tools.inputs.SimulationInputs.set_rectangle_ignition)
method is used to create a rectangular ignition zone simulating a line fire ignition pattern.

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

#### Step 2.4: Specify Output Files

Finally, we specify the types of output files the simulation will generate.
The [`set_output_files`](reference.md#quicfire_tools.inputs.SimulationInputs.set_output_files)
method allows you to choose which data to output.

```python
simulation.set_output_files(
    fuel_dens=True,
    emissions=True,
    qu_wind_inst=True
)
```

In this example:
- **fuel_dens** specifies a compressed array fuel density output.
- **emissions** specifies CO, PM2.5, and water emissions outputs.
- **qu_wind_inst** specifies gridded wind components (u, v, w) for the QUIC (wind) grid.

### Step 3: Write the Simulation to a Directory

Now that we have created the simulation, we can write it to a directory containing a QUIC-Fire executable to run the
fire model.
The [`write_simulation`](reference.md#quicfire_tools.inputs.SimulationInputs.write_simulation)
method of is used for this purpose.

```python
simulation.write_inputs("path/to/directory")
```

### Conclusion

In this tutorial, we have learned how to create a basic QUIC-Fire simulation using the SimulationInputs class from the
inputs module. We have also learned how to write the simulation to a directory containing a QUIC-Fire executable to run
the fire model.

quicfire-tools is designed to set up simulations in a quick, easy, and repeatable manner. The scope of the inputs module
ends at writing the simulation to a directory. Running a simulation and understanding the relevant inputs are the
responsibility of the user.

For more information about setting up a QUIC-Fire simulation, please see the [How-to-Guide](how-to-guides.md)
for more examples of working with the inputs module,
and the [Reference](reference.md#quicfire_tools.inputs) page for a complete list of available methods and attributes.
