# Core Imports
import sys

# Internal Imports
sys.path.append("../quicfire_tools")
from quicfire_tools.inputs import *

# External Imports
import pytest


# TEST_PARAMS = {
#     "nx": 100,
#     "ny": 100,
#     "nz": 1,
#     "dx": 1.,
#     "dy": 1.,
#     "dz": 1.,
#     "wind_speed": 4.,
#     "wind_direction": 265,
#     "sim_time": 60,
#     "auto_kill": 1,
#     "num_cpus": 1,
#     "fuel_flag": 3,
#     "ignition_flag": 1,
#     "output_time": 10,
# }


# SUT = SimulationInputs("pytest-simulation")


class TestGridList:
    def test_init(self):
        """Test the initialization of a Gridlist object."""
        # Test the default initialization
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        assert isinstance(gridlist, Gridlist)
        assert gridlist.n == 10
        for i in ["n", "m", "l", "dx", "dy", "dz", "aa1"]:
            assert i in gridlist.list_params()

        # Pass bad parameters: non-real numbers
        with pytest.raises(TypeError):
            Gridlist(n=2.5, m=10, l=10, dx="", dy=1, dz=1, aa1=1)
        with pytest.raises(TypeError):
            Gridlist(n="10", m=10, l=10, dx=1, dy=1, dz=1, aa1=1)

        # Pass bad parameters: zero or negative values
        with pytest.raises(ValueError):
            Gridlist(n=-10, m=10, l=10, dx=0, dy=1, dz=1, aa1=1)
        with pytest.raises(ValueError):
            Gridlist(n=10, m=10, l=10, dx=1, dy=0, dz=1, aa1=1)

    def test_to_dict(self):
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        result_dict = gridlist.to_dict()
        assert result_dict['n'] == 10
        assert result_dict['m'] == 10
        assert result_dict['l'] == 10
        assert result_dict['dx'] == 1
        assert result_dict['dy'] == 1
        assert result_dict['dz'] == 1
        assert '_validate_inputs' not in result_dict

    def test_to_file(self):
        """Test the write_file method of a Gridlist object."""
        gridlist = Gridlist(n=10, m=10, l=10, dx=1., dy=1., dz=1., aa1=1.)
        gridlist.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/gridlist", 'r') as file:
            lines = file.readlines()
            assert lines[0].split("=")[1].strip() == "10"
            assert lines[1].split("=")[1].strip() == "10"
            assert lines[2].split("=")[1].strip() == "10"
            assert lines[3].split("=")[1].strip() == "1.0"
            assert lines[4].split("=")[1].strip() == "1.0"
            assert lines[5].split("=")[1].strip() == "1.0"
            assert lines[6].split("=")[1].strip() == "1.0"

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            gridlist.to_file("/non_existent_path/gridlist.txt")


class TestRasterOrigin:
    def test_init(self):
        """Test the initialization of a RasterOrigin object."""
        # Test the default initialization
        raster_origin = RasterOrigin()
        assert isinstance(raster_origin, RasterOrigin)
        assert raster_origin.utm_x == 0.
        assert raster_origin.utm_y == 0.

        # Test the default initialization
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        assert isinstance(raster_origin, RasterOrigin)
        assert raster_origin.utm_x == 500.0
        assert raster_origin.utm_y == 1000.0

        # Pass bad parameters: non-real numbers
        with pytest.raises(TypeError):
            RasterOrigin(utm_x="500", utm_y=1000.0)
        with pytest.raises(TypeError):
            RasterOrigin(utm_x=500.0, utm_y="1000")

        # Pass bad parameters: zero or negative values
        with pytest.raises(ValueError):
            RasterOrigin(utm_x=-1, utm_y=1000.0)
        with pytest.raises(ValueError):
            RasterOrigin(utm_x=500.0, utm_y=-1000.0)

    def test_to_dict(self):
        """Test the to_dict method of a RasterOrigin object."""
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        result_dict = raster_origin.to_dict()
        assert result_dict['utm_x'] == 500.0
        assert result_dict['utm_y'] == 1000.0
        assert '_validate_inputs' not in result_dict

    def test_to_file(self):
        """Test the to_file method of a RasterOrigin object."""
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        raster_origin.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/rasterorigin.txt", 'r') as file:
            lines = file.readlines()
            assert lines[0].strip() == "500.0"
            assert lines[1].strip() == "1000.0"

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            raster_origin.to_file("/non_existent_path/rasterorigin.txt")


class TestQU_Buildings:
    def test_init(self):
        """Test the initialization of a QU_Buildings object."""
        # Test the default initialization
        qu_buildings = QU_Buildings()
        assert isinstance(qu_buildings, QU_Buildings)
        assert qu_buildings.wall_roughness_length == 0.1
        assert qu_buildings.number_of_buildings == 0
        assert qu_buildings.number_of_polygon_nodes == 0

        # Test custom initialization
        qu_buildings = QU_Buildings(wall_roughness_length=1.0,
                                    number_of_buildings=0,
                                    number_of_polygon_nodes=0)
        assert isinstance(qu_buildings, QU_Buildings)
        assert qu_buildings.wall_roughness_length == 1.0
        assert qu_buildings.number_of_buildings == 0
        assert qu_buildings.number_of_polygon_nodes == 0

        # Pass bad parameters: negative values
        with pytest.raises(ValueError):
            QU_Buildings(wall_roughness_length=-1, number_of_buildings=0,
                         number_of_polygon_nodes=0)
        with pytest.raises(ValueError):
            QU_Buildings(wall_roughness_length=1, number_of_buildings=-1,
                         number_of_polygon_nodes=0)

        # Pass bad parameters: incorrect types
        with pytest.raises(TypeError):
            QU_Buildings(wall_roughness_length=1.0, number_of_buildings=0.,
                         number_of_polygon_nodes=0)
        with pytest.raises(TypeError):
            QU_Buildings(wall_roughness_length="1.0", number_of_buildings=0,
                         number_of_polygon_nodes=0)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_buildings = QU_Buildings(wall_roughness_length=1.0,
                                    number_of_buildings=0,
                                    number_of_polygon_nodes=0)
        result_dict = qu_buildings.to_dict()
        assert result_dict['wall_roughness_length'] == 1.0
        assert result_dict['number_of_buildings'] == 0
        assert result_dict['number_of_polygon_nodes'] == 0
        assert '_validate_inputs' not in result_dict

    def test_to_file(self):
        """Test the to_file method of a QU_Buildings object."""
        qu_buildings = QU_Buildings(wall_roughness_length=0.1,
                                    number_of_buildings=0,
                                    number_of_polygon_nodes=0)
        qu_buildings.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/QU_buildings.inp", 'r') as file:
            lines = file.readlines()
            assert float(lines[1].strip().split("\t")[0]) == 0.1
            assert int(lines[2].strip().split("\t")[0]) == 0
            assert int(lines[3].strip().split("\t")[0]) == 0

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qu_buildings.to_file("/non_existent_path/QU_buildings.inp")
