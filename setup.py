import os

from setuptools import find_packages, setup


def read_file(fname):
    with open(fname, encoding="utf-8") as fd:
        return fd.read()


def get_version():
    with open(os.path.join("quicfire_tools", "_version.py")) as fd:
        # contents are __version__ = "<vstring>"
        return fd.read().split("=")[1].strip().strip('"')


NAME = "quicfire-tools"
DESCRIPTION = "Input and output management tools for the QUIC-Fire fire model"
LONG_DESCRIPTION = read_file("README.md")
VERSION = get_version()
LICENSE = "MIT"
URL = "https://github.com/silvxlabs/quicfire-tools"
PROJECT_URLS = {
    "Bug Tracker": f"{URL}/issues"
}

print(NAME, VERSION)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=["Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10",
                 "Programming Language :: Python :: 3.11",
                 "Topic :: Scientific/Engineering", ],
    package_dir={"": "."},
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={
        'quicfire_tools': ['input-templates/'],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "pydantic>=2",
        "zarr>=2",
    ],
    python_requires=">=3.8",
)