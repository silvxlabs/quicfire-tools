---
title: 'quicfire-tools: A Python package for QUIC-Fire simulation management'
tags:
  - Python
  - wildfire
  - fire modeling
  - simulation
  - fire spread
  - fire behavior
authors:
  - name: Anthony A. Marcozzi
    orcid: 0000-0003-4697-8154
    affiliation: 1
  - name: Niko J. Tutland
    orcid: 0000-0002-3167-6842
    affiliation: 1
  - name: Zachary Cope
    orcid: 0000-0003-1214-5365
    affiliation: 2
affiliations:
 - name: New Mexico Consortium, Center for Applied Fire and Ecosystem Sciences, Los Alamos, NM, USA
   index: 1
 - name: USDA Forest Service Center for Forest Disturbance Science, Athens, GA, 30602, USA
   index: 2
date: 20 March 2025
bibliography: paper.bib
---

# Summary

Fire behavior modeling is a critical tool for understanding and predicting fire spread across landscapes; informing risk assessment, planning prescribed burns, and developing mitigation strategies.
QUIC-Fire is a coupled fire-atmospheric modeling tool designed to rapidly simulate the complex interactions between fire, fuels, and atmospheric that are essential for predicting wildland fire behavior [@Linn2020a].
QUIC-Fire simulations require preparing numerous input files with complex interdependencies and generate large volumes of output data in a variety of formats.

Here we introduce `quicfire-tools`, a Python package that provides a streamlined interface for creating, managing, and analyzing QUIC-Fire simulations.
The package handles two primary aspects of the QUIC-Fire workflow: (1) programmatic creation and management of input file decks with validation and documentation, and (2) processing of simulation outputs into standard data structures compatible with the scientific Python ecosystem.
By simplifying these tasks, `quicfire-tools` enables researchers, fire managers, and modelers to focus on scientific questions rather than the technical details of file format specifications.

# Statement of need

Physics-based fire behavior models, such as QUIC-Fire, produce high-fidelity simulations of wildland fire behavior, but present significant barriers to entry due to their complex input requirements and output formats.
Users of QUIC-Fire must navigate more than 15 interdependent input files, each with dozens of parameters governing aspects like terrain, fuel characteristics, ignition patterns, and weather conditions.
Manipulating these files manually is error-prone and time-consuming, particularly when setting up parameter studies or batch simulations.

Similarly, processing QUIC-Fire outputs poses challenges as the model produces binary files in various formats (compressed sparse arrays and gridded arrays) that must be properly interpreted to extract meaningful data.
QUIC-Fire generates outputs for metrics such as energy release to atmosphere, reaction rate, fuel density, wind components, and fuel moisture, all of which require understanding of file formats, grid specifications, and coordinate transformations to interpret correctly.

Despite these challenges, no standardized tools existed to programmatically manage QUIC-Fire simulations before `quicfire-tools`.
While ad-hoc scripts were previously used, these were neither standardized, tested, nor comprehensively documented, leading to duplicated efforts and potential inconsistencies across research groups.

`quicfire-tools` addresses these needs by providing:

1. A consistent, validated interface for creating and modifying QUIC-Fire input files, with particular attention to the complex interdependencies between parameters
2. Programmatic management of spatially and temporally varying parameters like fuels, winds, and topography
3. Robust output processing capabilities that convert binary data to standard formats (NumPy arrays, Zarr archives, netCDF files)
4. Integration with the scientific Python ecosystem (NumPy, Dask, Pandas, Xarray)
5. Comprehensive documentation with step-by-step tutorials and examples

By standardizing these essential workflow components, `quicfire-tools` enables users to focus on the scientific aspects of fire simulation rather than technical implementation details.
The package facilitates more complex simulation studies, enhances reproducibility, and lowers the barrier to entry for new QUIC-Fire users.

# Key Features

## Input Management

The inputs module provides a comprehensive interface for creating and managing QUIC-Fire input decks, using Pydantic data models for validation and representation [@pydantic2025].
This approach solves two critical problems:

1. **Input file validation:** By leveraging Pydantic's validation capabilities, each parameter is checked against allowable ranges and types, preventing invalid input states before the simulation runs.

2. **Unified representation:** Instead of manually maintaining 15+ separate input files with complex interdependencies, `quicfire-tools` provides a single `SimulationInputs` object that encapsulates the entire simulation state, which can be serialized to JSON or written to individual input files to run the simulation.

The package repository contains detailed examples of QUIC-Fire simulation creation in the [how-to-guides](https://silvxlabs.github.io/quicfire-tools/how-to-guides/) and [tutorials](https://silvxlabs.github.io/quicfire-tools/tutorials/) documentation pages. 
These examples demonstrate typical workflows for creating simulations with uniform or custom fuels, various ignition patterns, and complex terrain features.
The package includes specialized modules for defining ignition patterns, topographic features, and wind conditions, all backed by Pydantic models for validation and consistency. 
These components can be easily configured programmatically as demonstrated in the repository documentation.

## Output Processing

The outputs module simplifies loading binary QUIC-Fire outputs into common Python data structures. 
The `SimulationOutputs` class automatically detects available output files and creates corresponding `OutputFile` instances. 
Each `OutputFile` class is initialized with the necessary attributes to properly load and interpret the output files, including domain dimensionality, binary file format, and metadata.

Examples of reading and interpreting QUIC-Fire output files using the outputs module are included in the [how-to-guides](https://silvxlabs.github.io/quicfire-tools/how-to-guides/) documentation. 
The documentation also showcases how output files can be saved in various data formats (NumPy arrays, netCDF, Zarr) for further analysis.

# Implementation

`quicfire-tools` is implemented in Python using a modular design centered around two main components:

1. The `SimulationInputs` class manages QUIC-Fire input files, each represented by specialized classes that inherit from the `InputFile` base class. These classes use Pydantic for validation, ensuring parameter values are within acceptable ranges. The use of Pydantic provides several key benefits:
   - Automatic validation of input parameters against physical constraints
   - Type checking and conversion of input values
   - Self-documenting models with clear parameter descriptions
   - Serialization/deserialization to/from JSON for storage and sharing
   - Extensibility through inheritance for specialized input types

2. The `SimulationOutputs` class manages QUIC-Fire output files, providing methods to extract data from both compressed and gridded binary formats. It includes functionality to convert outputs to standard formats like NumPy arrays, Dask arrays, netCDF files, and Zarr archives. The output processing system handles:
   - Automatic detection of available output files
   - Mapping between sparse compressed formats and dense array representations
   - Coordinate system transformations and grid alignment
   - Lazy loading for efficient memory management with large datasets

The package includes robust validation to prevent common errors, comprehensive documentation of available parameters, and extensive examples demonstrating typical workflows. Integration with scientific Python libraries ensures compatibility with common data analysis pipelines.

# Conclusion

`quicfire-tools` simplifies the creation, management, and analysis of QUIC-Fire simulations, enabling users to programatically and interact with the inputs and outputs.
By defining a consistent framework for representing input files, the package provides a user-friendly python class structure that facilites integration of QUIC-Fire with fuel modeling platforms [@marcozzi_fastfuels_2025] and large ensemble applications [@ahmed_towards_2024].
Furthermore, its streamlined processing of QUIC-Fire output files aids with data analysis, visualization, and communication of results.
The flexible framework of `quicfire-tools` also allows for the package to keep pace with QUIC-Fire's active development.


# Acknowledgements

We thank David Robinson, Rod Linn, and the rest of the QUIC-Fire development team. We are also grateful for the input and testing from the QUIC-Fire community of practice, including Julia Oliveto, Sophie Bonner, Jay Charney, Leticia Lee, Alex Massarie, Mary Brady, and many others.

# References
