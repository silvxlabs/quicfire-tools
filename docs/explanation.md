## Inputs module

The [`inputs`](reference.md#quicfire_tools.inputs) module in `quicfire_tools` is a central part of the package that handles the creation, modification,
and reading of QUIC-Fire input file decks. It provides a programmatic interface to interact with the various input
files required for a QUIC-Fire simulation.

### SimulationInputs Class

The [`SimulationInputs`](reference.md#quicfire_tools.inputs.SimulationInputs) class is the primary class in the
`inputs` module.
It represents a QUIC-Fire input file deck with attributes that represent the state of each input file in the deck.
These attributes are instances of various [`InputFile`] subclasses, each corresponding to a specific input file.

The `SimulationInputs` class provides several methods for creating and modifying simulations. For example, the
`create_simulation` method is used to create a new simulation with default values for fuels, ignitions, and topography.
To modify these default values, `SimulationInputs` offers `set_*` methods, such as `set_uniform_fuels` and
`set_rectangle_ignition`, to modify common simulation parameters and maintain consistency across input files.

The `write_inputs` method is also used to write all input files in the `SimulationInputs` object to a
specified directory. This method translates the state stored in the `SimulationInputs` object into the
text format used by QUIC-Fire input files. In addition, simulation state can be saved to a JSON file for sharing,
version control, or modification by using the `to_json` method.

The `SimulationInputs` class also provides methods for reading existing input file decks and converting
`SimulationInputs` objects to and from JSON format. The `from_directory` method is used to initialize a
`SimulationInputs` object from a directory containing a QUIC-Fire input file deck. The `to_json` and `from_json`
methods are used to save and load `SimulationInputs` objects in JSON format.

### InputFile Class

Each input file in a QUIC-Fire input file deck has a unique [Pydantic](https://docs.pydantic.dev/latest/)
Base Model class that represents the state of that input file. These classes are subclasses of the [`InputFile`]
class, which provides common functionality for all input files.

This common functionality includes methods for reading and writing input files to disk, as well as methods for
converting input files to and from JSON format. The `InputFile` class also provides methods to output the documentation
for a QUIC-Fire parameter in a Python dictionary format.

#### InputFiles objects and Pydantic

The `InputFile` class is a subclass of the Pydantic `BaseModel` class. Pydantic is a Python library that provides
data validation and serialization for Python data structures. Pydantic is used in `quicfire_tools` to validate the
state of input files and to serialize input files to and from JSON format.

Valid data types, ranges, and defaults are provided for each input file's attributes in the Pydantic `BaseModel`
subclass for that input file. This allows Pydantic to validate the state of input files and raise errors when
invalid values are provided.

## Outputs module

The [`outputs`](reference.md#quicfire_tools.outputs)
module in `quicfire-tools` is responsible for managing and processing simulation outputs produced by QUIC-Fire.
It provides efficient ways to access, extract, and organize data from various output files, enabling further analysis
and visualization of simulation results.

### OutputFile class
The [`OutputFile`](reference.md#quicfire_tools.outputs.OutputFile)
class represents a single output variable from a QUIC-Fire simulation.
It handles loading the data from disk into NumPy arrays for a given timestep.

Key functionality includes:

* Mapping compressed formats to dense NumPy arrays
* Automatically detecting available timesteps
* Extracting data for specific timesteps
* Integration with SimulationOutputs class

### SimulationOutputs class
The [`SimulationOutputs`](reference.md#quicfire_tools.outputs.SimulationOutputs)
class is the main interface for working with QUIC-Fire output data.
It collects available outputs in an Outputs directory and constructs `OutputFile` instances for each output variable.

Key functionality includes:

* Detecting available output variables
* Validating directory structure
* Getting outputs as NumPy arrays
* Writing outputs to Zarr files
* Integration with Dask arrays

## Ignite.dat Files module

The [`ignite.dat files`](reference.md#quicfire_tools.ignite_dat_files) module in `quicfire_tools` is a standalone part of the package for creating and visualizing ignite.dat files. It is not built into the [`inputs`](reference.md#quicfire_tools.inputs.set_rectangle_ignition) module because there are many different ways to create ignition plans for QUIC-Fire and this module only covers a few of them. The ignitions options that are pre-set in the [`inputs`](reference.md#quicfire_tools.inputs) module do not require any additional files from the user. [`Ignite.dat files`](reference.md#quicfire_tools.ignite_dat_files)  module, on the other hand, requires either a pre-made geojson of points or lines (likely made by hand in a GIS) or a csv of timestamped points (for example, a gps track), which are used as the basis for ignitions.

The functions for visualizing ignite.dat files can be run on any ignite.dat file, regardless of if it was created by this module or from another source. These visuals allow users to sanity check their ignition files before running a QUIC-Fire simulation and provide a tool for communicating burn plans or explaining the simulation. The [`animation function`](reference.md#quicfire_tools.ignite_dat_files.animate_geojson_ignition) only runs on geojsons formatted for use in this module. It does not work on ignite.dat files. The animation shows ignition groups (simultaneous ignitions) in different colors and preserves the timing of different groups relative to each other, which is useful for designing and checking complex ignition plans.
