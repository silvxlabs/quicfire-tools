import json
import urllib.request
from setuptools import find_packages, setup


def read_file(fname):
    with open(fname, encoding="utf-8") as fd:
        return fd.read()


def get_version():
    """Get the version number."""
    with urllib.request.urlopen(
        "https://api.github.com/repos/silvxlabs/quicfire-tools/releases/latest"
    ) as response:
        data = json.loads(response.read().decode("utf-8"))
    version = data["tag_name"]
    return version[1:]  # Remove the leading "v" from the version number


NAME = "quicfire-tools"
DESCRIPTION = "Input and output management tools for the QUIC-Fire fire model"
LONG_DESCRIPTION = read_file("README.md")
VERSION = get_version()
LICENSE = "MIT"
URL = "https://github.com/silvxlabs/quicfire-tools"
PROJECT_URLS = {"Bug Tracker": f"{URL}/issues"}
REQUIREMENTS = [
    "dask",
    "dask-expr",
    "netCDF4",
    "numpy",
    "pydantic>=2",
    "pyproj",
    "scipy",
    "zarr",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={
        "quicfire_tools": [
            "data/templates/*/*",
            "data/documentation/*",
            "data/outputs/*",
        ]
    },
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires=">=3.8",
)
