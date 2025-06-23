# quicfire-tools

## Quick-Links

[Documentation](https://silvxlabs.github.io/quicfire-tools/) - [PyPi Package](https://pypi.org/project/quicfire-tools/) [Conda](https://anaconda.org/conda-forge/quicfire-tools)

## What is quicfire-tools?

quicfire-tools is a Python package that provides a convenient interface for programmatically creating and managing
QUIC-Fire input file decks and processing QUIC-Fire output files into standard Python array data structures.

The goals of quicfire-tools are to:

1. Make it easy to write Python code to work with QUIC-Fire input and output files.
2. Unify code, scripts, and workflows across the QUIC-Fire ecosystem into a single package to support the development of
   new QUIC-Fire tools and applications.
3. Provide a platform for collaboration among QUIC-Fire developers and users.

## What is QUIC-Fire?

QUIC-Fire is a fast-running, coupled fire-atmospheric modeling tool developed by Los Alamos National Laboratory for wildland fire behavior prediction and prescribed fire planning.
It combines a 3D wind solver (QUIC-URB) with a physics-based cellular automata fire spread model (Fire-CA) to rapidly simulate the complex interactions between fire, fuels, and atmospheric conditions.

**Important Licensing Note**: QUIC-Fire is a closed-source simulation tool. The maintainers of quicfire-tools are not responsible for QUIC-Fire licensing. Users must obtain QUIC-Fire access through appropriate channels.


## Installation

quicfire-tools can be installed using `pip` or `conda`.

### pip

```bash
pip install quicfire-tools
```

### conda

```bash
conda install conda-forge::quicfire-tools
```

## Issues

If you encounter any issues with the quicfire-tools package, please submit an issue on the quicfire-tools GitHub
repository [issues page](https://github.com/silvxlabs/quicfire-tools/issues).
