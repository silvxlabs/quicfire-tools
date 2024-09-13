"""
Test module for the data module of the quicfire_tools package.
"""

from __future__ import annotations

import shutil
import pytest
from pydantic import ValidationError
from pandera.errors import SchemaError
from pathlib import Path
import pandas as pd

from quicfire_tools.inputs import (
    InputFile,
    Gridlist,
    RasterOrigin,
    QU_Buildings,
    QU_Fileoptions,
    QU_Simparams,
    QFire_Advanced_User_Inputs,
    QUIC_fire,
    QFire_Bldg_Advanced_User_Inputs,
    QFire_Plume_Advanced_User_Inputs,
    QU_TopoInputs,
    RuntimeAdvancedUserInputs,
    QU_movingcoords,
    QP_buildout,
    QU_metparams,
    SimulationInputs,
    WindSensorArray,
    WindSensor,
)
from quicfire_tools.ignitions import (
    Ignition,
    IgnitionFlags,
    RectangleIgnition,
    SquareRingIgnition,
)
from quicfire_tools.topography import (
    Topography,
    TopoFlags,
    GaussianHillTopo,
    CanyonTopo,
    CosHillTopo,
    HillPassTopo,
    SinusoidTopo,
    SlopeMesaTopo,
)

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data" / "test-inputs"
SAMPLES_DIR = TEST_DIR / "data" / "samples"
TMP_DIR = TEST_DIR / "tmp"
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(exist_ok=True)


class TestInputFile:
    def test_to_file_v5(self):
        input_file = InputFile(name="QP_buildout")
        input_file._extension = ".inp"
        input_file.to_file(TMP_DIR, version="v5")

    def test_to_file_v6(self):
        input_file = InputFile(name="QP_buildout")
        input_file._extension = ".inp"
        input_file.to_file(TMP_DIR, version="v6")

    def test_to_file_invalid_version(self):
        input_file = InputFile(name="")
        with pytest.raises(ValueError):
            input_file.to_file(TMP_DIR, version="v1")


class TestGridList:
    def test_init(self):
        """Test the initialization of a Gridlist object."""
        # Test the default initialization
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        assert isinstance(gridlist, Gridlist)
        assert gridlist.n == 10
        for i in ["n", "m", "l", "dx", "dy", "dz", "aa1"]:
            assert i in gridlist.list_parameters()

        # Test data type casting
        gridlist = Gridlist(n="10", m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        assert isinstance(gridlist.n, int)
        assert gridlist.n == 10

        # Pass bad parameters: non-real numbers
        with pytest.raises(ValidationError):
            Gridlist(n=2.5, m=10, l=10, dx="", dy=1, dz=1, aa1=1)
        with pytest.raises(ValidationError):
            Gridlist(n="a", m=10, l=10, dx=1, dy=1, dz=1, aa1=1)

        # Pass bad parameters: zero or negative values
        with pytest.raises(ValidationError):
            Gridlist(n=-10, m=10, l=10, dx=0, dy=1, dz=1, aa1=1)
        with pytest.raises(ValidationError):
            Gridlist(n=10, m=10, l=10, dx=1, dy=0, dz=1, aa1=1)

    def test_to_dict(self):
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        result_dict = gridlist.to_dict()
        assert result_dict["n"] == 10
        assert result_dict["m"] == 10
        assert result_dict["l"] == 10
        assert result_dict["dx"] == 1
        assert result_dict["dy"] == 1
        assert result_dict["dz"] == 1
        assert "_validate_inputs" not in result_dict

    def test_to_docs(self):
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        result_dict = gridlist.to_dict()
        result_docs = gridlist.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        """Test the write_file method of a Gridlist object."""
        gridlist = Gridlist(n=10, m=10, l=10, dx=1.0, dy=1.0, dz=1.0, aa1=1.0)
        gridlist.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "gridlist", "r") as file:
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

    def test_from_file(self):
        """Test initializing a class from a gridlist file."""
        gridlist = Gridlist(n=10, m=10, l=10, dx=1.0, dy=1.0, dz=1.0, aa1=1.0)
        gridlist.to_file(TMP_DIR)
        test_object = Gridlist.from_file(TMP_DIR)
        assert isinstance(test_object, Gridlist)
        assert gridlist == test_object


class TestRasterOrigin:
    def test_init(self):
        """Test the initialization of a RasterOrigin object."""
        # Test the default initialization
        raster_origin = RasterOrigin()
        assert isinstance(raster_origin, RasterOrigin)
        assert raster_origin.utm_x == 0.0
        assert raster_origin.utm_y == 0.0

        # Test the default initialization
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        assert isinstance(raster_origin, RasterOrigin)
        assert raster_origin.utm_x == 500.0
        assert raster_origin.utm_y == 1000.0

        # Test data type casting
        raster_origin = RasterOrigin(utm_x="500", utm_y=1000.0)
        assert isinstance(raster_origin.utm_x, float)
        assert raster_origin.utm_x == 500.0

        # Pass bad parameters: non-real numbers
        with pytest.raises(ValidationError):
            RasterOrigin(utm_x="x", utm_y=1000.0)

        # Pass bad parameters: zero or negative values
        with pytest.raises(ValidationError):
            RasterOrigin(utm_x=-1, utm_y=1000.0)
        with pytest.raises(ValidationError):
            RasterOrigin(utm_x=500.0, utm_y=-1000.0)

    def test_to_dict(self):
        """Test the to_dict method of a RasterOrigin object."""
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        result_dict = raster_origin.to_dict()
        assert result_dict["utm_x"] == raster_origin.utm_x
        assert result_dict["utm_y"] == raster_origin.utm_y

    def test_from_dict(self):
        """Test the from_dict method of a RasterOrigin object."""
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        result_dict = raster_origin.to_dict()
        test_object = RasterOrigin.from_dict(result_dict)
        assert isinstance(test_object, RasterOrigin)
        assert raster_origin == test_object

    def test_to_docs(self):
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        result_dict = raster_origin.to_dict()
        result_docs = raster_origin.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        """Test the to_file method of a RasterOrigin object."""
        raster_origin = RasterOrigin(utm_x=500.0, utm_y=1000.0)
        raster_origin.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "rasterorigin.txt", "r") as file:
            lines = file.readlines()
            assert float(lines[0].strip()) == raster_origin.utm_x
            assert float(lines[1].strip()) == raster_origin.utm_y

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            raster_origin.to_file("/non_existent_path/rasterorigin.txt")

    def test_from_file(self):
        """Test initializing a class from a rasterorigin.txt file."""
        raster_origin = RasterOrigin()
        raster_origin.to_file(TMP_DIR)
        test_object = RasterOrigin.from_file(TMP_DIR)
        assert isinstance(test_object, RasterOrigin)
        assert raster_origin == test_object


class TestQU_Buildings:
    def test_init(self):
        """Test the initialization of a QU_Buildings object."""
        # Test the default initialization
        qu_buildings = QU_Buildings()
        assert qu_buildings.wall_roughness_length == 0.1
        assert qu_buildings.number_of_buildings == 0
        assert qu_buildings.number_of_polygon_nodes == 0

        # Test custom initialization
        qu_buildings = QU_Buildings(
            wall_roughness_length=1.0, number_of_buildings=0, number_of_polygon_nodes=0
        )
        assert qu_buildings.wall_roughness_length == 1.0
        assert qu_buildings.number_of_buildings == 0
        assert qu_buildings.number_of_polygon_nodes == 0

        # Test data type casting
        qu_buildings = QU_Buildings(
            wall_roughness_length="1.0", number_of_buildings=1.0
        )
        assert isinstance(qu_buildings.wall_roughness_length, float)
        assert qu_buildings.wall_roughness_length == 1.0
        assert isinstance(qu_buildings.number_of_buildings, int)
        assert qu_buildings.number_of_buildings == 1

        # Pass bad parameters
        with pytest.raises(ValidationError):
            QU_Buildings(
                wall_roughness_length=-1,
                number_of_buildings=0,
                number_of_polygon_nodes=0,
            )
        with pytest.raises(ValidationError):
            QU_Buildings(
                wall_roughness_length=1,
                number_of_buildings=-1,
                number_of_polygon_nodes=0,
            )
        with pytest.raises(ValidationError):
            QU_Buildings(wall_roughness_length=0)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_buildings = QU_Buildings()
        result_dict = qu_buildings.to_dict()
        assert (
            result_dict["wall_roughness_length"] == qu_buildings.wall_roughness_length
        )
        assert result_dict["number_of_buildings"] == qu_buildings.number_of_buildings
        assert (
            result_dict["number_of_polygon_nodes"]
            == qu_buildings.number_of_polygon_nodes
        )

    def test_from_dict(self):
        """Test the from_dict method of a QU_Buildings object."""
        qu_buildings = QU_Buildings()
        result_dict = qu_buildings.to_dict()
        test_object = QU_Buildings.from_dict(result_dict)
        assert isinstance(test_object, QU_Buildings)
        assert qu_buildings == test_object

    def test_to_docs(self):
        qu_buildings = QU_Buildings()
        result_dict = qu_buildings.to_dict()
        result_docs = qu_buildings.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        """Test the to_file method of a QU_Buildings object."""
        qu_buildings = QU_Buildings()
        qu_buildings.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QU_buildings.inp", "r") as file:
            lines = file.readlines()
            assert (
                float(lines[1].strip().split("\t")[0])
                == qu_buildings.wall_roughness_length
            )
            assert (
                int(lines[2].strip().split("\t")[0]) == qu_buildings.number_of_buildings
            )
            assert (
                int(lines[3].strip().split("\t")[0])
                == qu_buildings.number_of_polygon_nodes
            )

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qu_buildings.to_file("/non_existent_path/QU_buildings.inp")

    def test_from_file(self):
        """Test initializing a class from a QU_buildings.inp file."""
        qu_buildings = QU_Buildings()
        qu_buildings.to_file(TMP_DIR)
        test_object = QU_Buildings.from_file(TMP_DIR)
        assert isinstance(test_object, QU_Buildings)
        assert qu_buildings == test_object


class TestQU_Fileoptions:
    def test_init(self):
        # Test default initialization
        qu_fileoptions = QU_Fileoptions()
        assert qu_fileoptions.output_data_file_format_flag == 2
        assert qu_fileoptions.non_mass_conserved_initial_field_flag == 0
        assert qu_fileoptions.initial_sensor_velocity_field_flag == 0
        assert qu_fileoptions.qu_staggered_velocity_file_flag == 0
        assert qu_fileoptions.generate_wind_startup_files_flag == 0

        # Test custom initialization #1
        qu_fileoptions = QU_Fileoptions(output_data_file_format_flag=1)
        assert qu_fileoptions.output_data_file_format_flag == 1

        # Test custom initialization #2
        qu_fileoptions = QU_Fileoptions(non_mass_conserved_initial_field_flag=1)
        assert qu_fileoptions.non_mass_conserved_initial_field_flag == 1

        # Test custom initialization #3
        qu_fileoptions = QU_Fileoptions(generate_wind_startup_files_flag=1)
        assert qu_fileoptions.generate_wind_startup_files_flag == 1

        # Test invalid output_data_file_format_flag flags
        for invalid_flag in [-1, 0, 5, "1", 1.0, 1.5]:
            with pytest.raises(ValidationError):
                QU_Fileoptions(output_data_file_format_flag=invalid_flag)

        # Test invalid non_mass_conserved_initial_field_flag flag
        for invalid_flag in [-1, 0.0, "1", 2]:
            with pytest.raises(ValidationError):
                QU_Fileoptions(non_mass_conserved_initial_field_flag=invalid_flag)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_fileoptions = QU_Fileoptions()
        result_dict = qu_fileoptions.to_dict()
        assert (
            result_dict["output_data_file_format_flag"]
            == qu_fileoptions.output_data_file_format_flag
        )
        assert (
            result_dict["non_mass_conserved_initial_field_flag"]
            == qu_fileoptions.non_mass_conserved_initial_field_flag
        )
        assert (
            result_dict["initial_sensor_velocity_field_flag"]
            == qu_fileoptions.initial_sensor_velocity_field_flag
        )
        assert (
            result_dict["qu_staggered_velocity_file_flag"]
            == qu_fileoptions.qu_staggered_velocity_file_flag
        )
        assert (
            result_dict["generate_wind_startup_files_flag"]
            == qu_fileoptions.generate_wind_startup_files_flag
        )

    def test_from_dict(self):
        """Test the from_dict method of a QU_Buildings object."""
        qu_fileoptions = QU_Fileoptions()
        result_dict = qu_fileoptions.to_dict()
        test_object = QU_Fileoptions.from_dict(result_dict)
        assert isinstance(test_object, QU_Fileoptions)
        assert qu_fileoptions == test_object

    def test_to_docs(self):
        qu_fileoptions = QU_Fileoptions()
        result_dict = qu_fileoptions.to_dict()
        result_docs = qu_fileoptions.get_documentation()
        for key in result_dict:
            assert key in result_docs

    def test_to_file(self):
        """Test the to_file method of a QU_Buildings object."""
        qu_fileoptions = QU_Fileoptions()
        qu_fileoptions.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QU_fileoptions.inp", "r") as file:
            lines = file.readlines()
            assert int(lines[1].strip().split("!")[0]) == 2
            assert int(lines[2].strip().split("!")[0]) == 0
            assert int(lines[3].strip().split("!")[0]) == 0
            assert int(lines[4].strip().split("!")[0]) == 0
            assert int(lines[5].strip().split("!")[0]) == 0

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qu_fileoptions.to_file("/non_existent_path/QU_buildings.inp")

    def test_from_file(self):
        """Test initializing a class from a QU_fileoptions.inp file."""
        qu_fileoptions = QU_Fileoptions()
        qu_fileoptions.to_file(TMP_DIR)
        test_object = QU_Fileoptions.from_file(TMP_DIR)
        assert isinstance(test_object, QU_Fileoptions)
        assert qu_fileoptions == test_object


class TestQU_Simparams:
    @staticmethod
    def get_test_object():
        return QU_Simparams(
            nx=100, ny=100, nz=26, dx=2.0, dy=2, quic_domain_height=250, wind_times=[0]
        )

    def test_init(self):
        # Test default initialization
        qu_simparams = self.get_test_object()
        assert qu_simparams.nx == 100
        assert qu_simparams.ny == 100
        assert qu_simparams.nz == 26
        assert qu_simparams.dx == 2
        assert qu_simparams.dy == 2
        assert qu_simparams.surface_vertical_cell_size == 1
        assert qu_simparams.number_surface_cells == 5
        assert qu_simparams.stretch_grid_flag == 3
        assert len(qu_simparams._dz_array) == 26
        assert len(qu_simparams._vertical_grid_lines.split("\n")) == 29

        # Test changing the default values
        qu_simparams.nx = 150
        assert qu_simparams.nx == 150

        # Test property setters
        qu_simparams.nz = 30
        assert qu_simparams.nz == 30
        assert len(qu_simparams._dz_array) == 30
        assert len(qu_simparams._vertical_grid_lines.split("\n")) == 33

        # Test data type casting
        qu_simparams = QU_Simparams(
            nx="100", ny=100, nz=26, dx=2, dy=2, quic_domain_height=5, wind_times=[0.0]
        )
        assert qu_simparams.nx == 100
        assert qu_simparams.wind_times == [0]

        # Test invalid stretch_grid_flags
        for invalid_flag in [-1, 4, "1", 1.0, 1.5, 2]:
            with pytest.raises(ValidationError):
                QU_Simparams(stretch_grid_flag=invalid_flag)

    def test_dz_array(self):
        # Test with stretch_grid_flag = 0
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        assert (
            qu_simparams._dz_array
            == [qu_simparams.surface_vertical_cell_size] * qu_simparams.nz
        )

        # Test with stretch_grid_flag = 1
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.custom_dz_array = [0.5] * qu_simparams.nz
        assert qu_simparams._dz_array == [0.5] * qu_simparams.nz

        # Test with stretch_grid_flag = 3
        qu_simparams = self.get_test_object()
        assert len(qu_simparams._dz_array) == qu_simparams.nz

    def test_stretch_grid_flag_0(self):
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        vertical_grid_lines = qu_simparams._stretch_grid_flag_0()
        with open(TEST_DIR / "data/test-inputs/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_stretch_grid_flag_1(self):
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 1

        # Test with no _dz_array input
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with 19 custom_dz_array data
        qu_simparams.custom_dz_array = [1] * (qu_simparams.nz - 1)
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with dz data that don't match the surface values
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        qu_simparams.custom_dz_array[0] = 2
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test valid case
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        vertical_grid_lines = qu_simparams._stretch_grid_flag_1()
        with open(TEST_DIR / "data/test-inputs/stretchgrid_1.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_stretch_grid_flag_3(self):
        qu_simparams = self.get_test_object()
        vertical_grid_lines = qu_simparams._stretch_grid_flag_3()
        with open(TEST_DIR / "data/test-inputs/stretchgrid_3.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_generate_vertical_grid(self):
        qu_simparams = self.get_test_object()

        # Test stretch_grid_flag = 0
        qu_simparams.stretch_grid_flag = 0
        with open(TEST_DIR / "data/test-inputs/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams._vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 1
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        with open(TEST_DIR / "data/test-inputs/stretchgrid_1.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams._vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 3
        qu_simparams.stretch_grid_flag = 3
        with open(TEST_DIR / "data/test-inputs/stretchgrid_3.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams._vertical_grid_lines == "".join(expected_lines)

    def test_generate_wind_time_lines(self):
        # Test valid wind_step_times
        qu_simparams = self.get_test_object()
        qu_simparams.wind_times = [0]
        wind_times_lines = qu_simparams._generate_wind_time_lines()
        with open(TEST_DIR / "data/test-inputs/wind_times.txt") as f:
            expected_lines = f.readlines()
        assert wind_times_lines == "".join(expected_lines)

        # Test invalid wind_step_times
        qu_simparams.wind_times = []
        with pytest.raises(ValueError):
            qu_simparams._generate_wind_time_lines()

    def test_to_dict(self):
        """
        Test the to_dict method of a QU_Simparams object.
        """
        qu_simparams = self.get_test_object()
        result_dict = qu_simparams.to_dict()

        # Test the passed parameters
        assert result_dict["nx"] == qu_simparams.nx
        assert result_dict["ny"] == qu_simparams.ny
        assert result_dict["nz"] == qu_simparams.nz
        assert result_dict["dx"] == qu_simparams.dx
        assert result_dict["dy"] == qu_simparams.dy
        assert (
            result_dict["surface_vertical_cell_size"]
            == qu_simparams.surface_vertical_cell_size
        )
        assert result_dict["number_surface_cells"] == qu_simparams.number_surface_cells

        # Test the default parameters
        assert (
            result_dict["surface_vertical_cell_size"]
            == qu_simparams.surface_vertical_cell_size
        )
        assert result_dict["number_surface_cells"] == qu_simparams.number_surface_cells
        assert result_dict["stretch_grid_flag"] == qu_simparams.stretch_grid_flag
        assert result_dict["utc_offset"] == qu_simparams.utc_offset
        assert result_dict["wind_times"] == qu_simparams.wind_times
        assert result_dict["sor_iter_max"] == qu_simparams.sor_iter_max
        assert (
            result_dict["sor_residual_reduction"] == qu_simparams.sor_residual_reduction
        )
        assert result_dict["use_diffusion_flag"] == qu_simparams.use_diffusion_flag
        assert (
            result_dict["number_diffusion_iterations"]
            == qu_simparams.number_diffusion_iterations
        )
        assert result_dict["domain_rotation"] == qu_simparams.domain_rotation
        assert result_dict["utm_x"] == qu_simparams.utm_x
        assert result_dict["utm_y"] == qu_simparams.utm_y
        assert result_dict["utm_zone_number"] == qu_simparams.utm_zone_number
        assert result_dict["utm_zone_letter"] == qu_simparams.utm_zone_letter
        assert result_dict["quic_cfd_flag"] == qu_simparams.quic_cfd_flag
        assert result_dict["explosive_bldg_flag"] == qu_simparams.explosive_bldg_flag
        assert result_dict["bldg_array_flag"] == qu_simparams.bldg_array_flag

    def test_from_dict(self):
        """
        Test the from_dict method of a QU_Simparams object.
        """
        qu_simparams = self.get_test_object()
        result_dict = qu_simparams.to_dict()
        test_object = QU_Simparams.from_dict(result_dict)
        assert isinstance(test_object, QU_Simparams)
        assert qu_simparams == test_object

    def test_to_docs(self):
        qu_simparams = self.get_test_object()
        result_dict = qu_simparams.to_dict()
        result_docs = qu_simparams.get_documentation()
        for key in result_dict:
            if key in [
                "_dz_array",
                "_vertical_grid_lines",
                "_wind_time_lines",
                "custom_dz_array",
                "quic_domain_height",
            ]:
                continue
            assert key in result_docs
        for key in result_docs:
            if key in ["dz_array"]:
                continue
            assert key in result_dict

    def test_to_file(self):
        """
        Test the to_file method of a QU_Simparams object.
        """
        qu_simparams = self.get_test_object()
        qu_simparams.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QU_simparams.inp", "r") as file:
            lines = file.readlines()

        # Check nx, ny, nz, dx, dy
        assert int(lines[1].strip().split("!")[0]) == qu_simparams.nx
        assert int(lines[2].strip().split("!")[0]) == qu_simparams.ny
        assert int(lines[3].strip().split("!")[0]) == qu_simparams.nz
        assert float(lines[4].strip().split("!")[0]) == qu_simparams.dx
        assert float(lines[5].strip().split("!")[0]) == qu_simparams.dy

        # Check stretch_grid_flag, surface_vertical_cell_size,
        # number_surface_cells
        assert int(lines[6].strip().split("!")[0]) == qu_simparams.stretch_grid_flag
        assert (
            float(lines[7].strip().split("!")[0])
            == qu_simparams.surface_vertical_cell_size
        )
        assert int(lines[8].strip().split("!")[0]) == qu_simparams.number_surface_cells

        # Check _dz_array
        assert lines[9] == "! DZ array [m]\n"
        for i in range(qu_simparams.nz):
            index = i + 10
            dz = qu_simparams._dz_array[i]
            assert float(lines[index].strip()) == dz

        # Update lines index
        i_current = 10 + qu_simparams.nz

        # Check number of time increments, utc_offset
        assert int(lines[i_current].strip().split("!")[0]) == len(
            qu_simparams.wind_times
        )
        assert (
            int(lines[i_current + 1].strip().split("!")[0]) == qu_simparams.utc_offset
        )

        # Check wind_step_times
        assert lines[i_current + 2] == "! Wind step times [s]\n"
        for i in range(len(qu_simparams.wind_times)):
            index = i_current + 3 + i
            wind_time = qu_simparams.wind_times[i]
            assert int(lines[index].strip()) == wind_time

        # Update lines index
        i_current = i_current + 3 + len(qu_simparams.wind_times)
        i_current += 9  # Skip not used lines

        # Check sor_iter_max, sor_residual_reduction
        assert int(lines[i_current].strip().split("!")[0]) == qu_simparams.sor_iter_max
        assert (
            int(lines[i_current + 1].strip().split("!")[0])
            == qu_simparams.sor_residual_reduction
        )

        # Check use_diffusion_flag, number_diffusion_iterations, domain_rotation
        # utm_x, utm_y, utm_zone_number, utm_zone_letter, quic_cfd_flag,
        # explosive_bldg_flag, bldg_array_flag
        assert (
            int(lines[i_current + 2].strip().split("!")[0])
            == qu_simparams.use_diffusion_flag
        )
        assert (
            int(lines[i_current + 3].strip().split("!")[0])
            == qu_simparams.number_diffusion_iterations
        )
        assert (
            float(lines[i_current + 4].strip().split("!")[0])
            == qu_simparams.domain_rotation
        )
        assert float(lines[i_current + 5].strip().split("!")[0]) == qu_simparams.utm_x
        assert float(lines[i_current + 6].strip().split("!")[0]) == qu_simparams.utm_y
        assert (
            int(lines[i_current + 7].strip().split("!")[0])
            == qu_simparams.utm_zone_number
        )
        assert (
            int(lines[i_current + 8].strip().split("!")[0])
            == qu_simparams.utm_zone_letter
        )
        assert (
            int(lines[i_current + 9].strip().split("!")[0])
            == qu_simparams.quic_cfd_flag
        )
        assert (
            int(lines[i_current + 10].strip().split("!")[0])
            == qu_simparams.explosive_bldg_flag
        )
        assert (
            int(lines[i_current + 11].strip().split("!")[0])
            == qu_simparams.bldg_array_flag
        )

    def test_from_file(self):
        """
        Test initializing a class from a QU_simparams.inp file.
        """
        # Test stretch grid flag = 3
        qu_simparams = self.get_test_object()
        qu_simparams.to_file(TMP_DIR)
        test_object = QU_Simparams.from_file(TMP_DIR)
        assert isinstance(test_object, QU_Simparams)
        test_object_dict = test_object.to_dict()
        qu_simparams_dict = qu_simparams.to_dict()
        assert test_object_dict == qu_simparams_dict

        # # Test stretch grid flag = 0
        # qu_simparams = self.get_test_object()
        # qu_simparams.stretch_grid_flag = 0
        # qu_simparams.to_file(TMP_DIR)
        # test_object = QU_Simparams.from_file(TMP_DIR)
        # assert isinstance(test_object, QU_Simparams)
        # assert qu_simparams == test_object

        # # Test stretch grid flag = 1
        # qu_simparams = self.get_test_object()
        # qu_simparams.stretch_grid_flag = 1
        # qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        # qu_simparams.to_file(TMP_DIR)
        # test_object = QU_Simparams.from_file(TMP_DIR)
        # assert isinstance(test_object, QU_Simparams)
        # assert qu_simparams == test_object


class TestQFire_Advanced_User_Inputs:
    def test_init(self):
        """Test the initialization of a QFire_Advanced_User_Inputs object."""
        # Test the default initialization
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        assert qfire_advanced_user_inputs.fraction_cells_launch_firebrands == 0.05

        # Test custom initialization
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs(
            fraction_cells_launch_firebrands=0.1
        )
        assert qfire_advanced_user_inputs.fraction_cells_launch_firebrands == 0.1

        # Test data type casting
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs(
            fraction_cells_launch_firebrands="0.1"
        )
        assert isinstance(
            qfire_advanced_user_inputs.fraction_cells_launch_firebrands, float
        )
        assert qfire_advanced_user_inputs.fraction_cells_launch_firebrands == 0.1

        # Pass bad parameters: negative numbers
        with pytest.raises(ValidationError):
            QFire_Advanced_User_Inputs(fraction_cells_launch_firebrands=-1)

        # Pass bad parameters: not a fraction
        with pytest.raises(ValidationError):
            QFire_Advanced_User_Inputs(fraction_cells_launch_firebrands=2)

        # Pass bad parameters: not a valid range for theta
        with pytest.raises(ValidationError):
            QFire_Advanced_User_Inputs(minimum_landing_angle=361)

    def test_to_dict(self):
        """Test the to_dict method of a QFire_Advanced_User_Inputs object."""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        result_dict = qfire_advanced_user_inputs.to_dict()
        assert (
            result_dict["fraction_cells_launch_firebrands"]
            == qfire_advanced_user_inputs.fraction_cells_launch_firebrands
        )
        assert (
            result_dict["firebrand_radius_scale_factor"]
            == qfire_advanced_user_inputs.firebrand_radius_scale_factor
        )
        assert (
            result_dict["firebrand_trajectory_time_step"]
            == qfire_advanced_user_inputs.firebrand_trajectory_time_step
        )
        assert (
            result_dict["firebrand_launch_interval"]
            == qfire_advanced_user_inputs.firebrand_launch_interval
        )
        assert (
            result_dict["firebrands_per_deposition"]
            == qfire_advanced_user_inputs.firebrands_per_deposition
        )
        assert (
            result_dict["firebrand_area_ratio"]
            == qfire_advanced_user_inputs.firebrand_area_ratio
        )
        assert (
            result_dict["minimum_burn_rate_coefficient"]
            == qfire_advanced_user_inputs.minimum_burn_rate_coefficient
        )
        assert (
            result_dict["max_firebrand_thickness_fraction"]
            == qfire_advanced_user_inputs.max_firebrand_thickness_fraction
        )
        assert (
            result_dict["firebrand_germination_delay"]
            == qfire_advanced_user_inputs.firebrand_germination_delay
        )
        assert (
            result_dict["vertical_velocity_scale_factor"]
            == qfire_advanced_user_inputs.vertical_velocity_scale_factor
        )
        assert (
            result_dict["minimum_firebrand_ignitions"]
            == qfire_advanced_user_inputs.minimum_firebrand_ignitions
        )
        assert (
            result_dict["maximum_firebrand_ignitions"]
            == qfire_advanced_user_inputs.maximum_firebrand_ignitions
        )
        assert (
            result_dict["minimum_landing_angle"]
            == qfire_advanced_user_inputs.minimum_landing_angle
        )
        assert (
            result_dict["maximum_firebrand_thickness"]
            == qfire_advanced_user_inputs.maximum_firebrand_thickness
        )

    def test_from_dict(self):
        """Test class initialization from a dictionary object"""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        result_dict = qfire_advanced_user_inputs.to_dict()
        test_obj = QFire_Advanced_User_Inputs.from_dict(result_dict)
        assert test_obj == qfire_advanced_user_inputs

    def test_to_docs(self):
        """Test the to_docs method of a QFire_Advanced_User_Inputs object."""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        result_dict = qfire_advanced_user_inputs.to_dict()
        result_docs = qfire_advanced_user_inputs.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        """Test the to_file method of a QFire_Advanced_User_Inputs object."""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs(
            fraction_cells_launch_firebrands=0.1
        )
        qfire_advanced_user_inputs.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QFire_Advanced_User_Inputs.inp", "r") as file:
            lines = file.readlines()
            assert (
                float(lines[0].strip().split("!")[0])
                == qfire_advanced_user_inputs.fraction_cells_launch_firebrands
            )
            assert (
                float(lines[1].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrand_radius_scale_factor
            )
            assert (
                float(lines[2].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrand_trajectory_time_step
            )
            assert (
                float(lines[3].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrand_launch_interval
            )
            assert (
                float(lines[4].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrands_per_deposition
            )
            assert (
                float(lines[5].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrand_area_ratio
            )
            assert (
                float(lines[6].strip().split("!")[0])
                == qfire_advanced_user_inputs.minimum_burn_rate_coefficient
            )
            assert (
                float(lines[7].strip().split("!")[0])
                == qfire_advanced_user_inputs.max_firebrand_thickness_fraction
            )
            assert (
                float(lines[8].strip().split("!")[0])
                == qfire_advanced_user_inputs.firebrand_germination_delay
            )
            assert (
                float(lines[9].strip().split("!")[0])
                == qfire_advanced_user_inputs.vertical_velocity_scale_factor
            )
            assert (
                float(lines[10].strip().split("!")[0])
                == qfire_advanced_user_inputs.minimum_firebrand_ignitions
            )
            assert (
                float(lines[11].strip().split("!")[0])
                == qfire_advanced_user_inputs.maximum_firebrand_ignitions
            )
            assert (
                float(lines[12].strip().split("!")[0])
                == qfire_advanced_user_inputs.minimum_landing_angle
            )
            assert (
                float(lines[13].strip().split("!")[0])
                == qfire_advanced_user_inputs.maximum_firebrand_thickness
            )

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qfire_advanced_user_inputs.to_file(
                "/non_existent_path/QFIRE_advanced_user_inputs.inp"
            )

    def test_from_file(self):
        """Test initializing a class from a QFIRE_advanced_user_inputs.inp
        file."""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        qfire_advanced_user_inputs.to_file(TMP_DIR)
        test_object = QFire_Advanced_User_Inputs.from_file(TMP_DIR)
        assert isinstance(test_object, QFire_Advanced_User_Inputs)
        assert qfire_advanced_user_inputs == test_object


class TestQUIC_fire:
    @staticmethod
    def get_basic_test_object():
        return QUIC_fire(
            nz=26,
            sim_time=60,
            time_now=1695311421,
            ignition=RectangleIgnition(x_min=20, y_min=20, x_length=10, y_length=160),
        )

    def test_init(self):
        # Test default initialization
        quic_fire = self.get_basic_test_object()
        assert quic_fire.nz == 26
        assert quic_fire.sim_time == 60

        # Test changing the default values
        quic_fire.nz = 27
        assert quic_fire.nz == 27

        # Test data type casting
        quic_fire.nz = "26"
        assert quic_fire.nz == 26

        # Test stretch grid input
        assert quic_fire.stretch_grid_flag == 0
        assert quic_fire._stretch_grid_input == "1"

        # Test invalid random_seed
        with pytest.raises(ValidationError):
            quic_fire.random_seed = 0

    def test_fuel_lines_uniform(self):
        quic_fire = self.get_basic_test_object()

        quic_fire.fuel_density_flag = 1
        quic_fire.fuel_density = 0.5
        quic_fire.fuel_moisture_flag = 1
        quic_fire.fuel_moisture = 1
        quic_fire.fuel_height_flag = 1
        quic_fire.fuel_height = 0.75
        quic_fire.size_scale_flag = 1
        quic_fire.size_scale = 0.001
        quic_fire.patch_and_gap_flag = 1
        quic_fire.patch_size = 0.5
        quic_fire.gap_size = 0.5
        with open(TEST_DIR / "data/test-inputs/fuel_lines_uniform.txt") as f:
            expected_lines = f.readlines()
        fuel_lines = (
            quic_fire._fuel_density_lines
            + quic_fire._fuel_moisture_lines
            + quic_fire._fuel_height_lines
            + quic_fire._size_scale_lines
            + quic_fire._patch_and_gap_lines
        )
        assert "".join(expected_lines) == fuel_lines

        quic_fire.to_file(TMP_DIR, version="v5")

    def test_fuel_lines_custom(self):
        quic_fire = self.get_basic_test_object()

        quic_fire.fuel_density_flag = 3
        quic_fire.fuel_moisture_flag = 3
        quic_fire.size_scale_flag = 0
        quic_fire.patch_and_gap_flag = 0
        with open(TEST_DIR / "data/test-inputs/fuel_lines_custom.txt") as f:
            expected_lines = f.readlines()
        fuel_lines = (
            quic_fire._fuel_density_lines
            + quic_fire._fuel_moisture_lines
            + quic_fire._fuel_height_lines
            + quic_fire._size_scale_lines
            + quic_fire._patch_and_gap_lines
        )
        assert "".join(expected_lines) == fuel_lines

        quic_fire.to_file(TMP_DIR, version="v5")

    def test_to_dict(self):
        """Test the to_dict method of a QUIC_fire object."""
        quic_fire = self.get_basic_test_object()
        result_dict = quic_fire.to_dict()

        assert result_dict["nz"] == quic_fire.nz
        assert result_dict["time_now"] == quic_fire.time_now
        assert result_dict["sim_time"] == quic_fire.sim_time
        assert result_dict["fire_flag"] == quic_fire.fire_flag
        assert result_dict["random_seed"] == quic_fire.random_seed
        assert result_dict["fire_time_step"] == quic_fire.fire_time_step
        assert result_dict["quic_time_step"] == quic_fire.quic_time_step
        assert result_dict["out_time_fire"] == quic_fire.out_time_fire
        assert result_dict["out_time_wind"] == quic_fire.out_time_wind
        assert result_dict["out_time_emis_rad"] == quic_fire.out_time_emis_rad
        assert result_dict["out_time_wind_avg"] == quic_fire.out_time_wind_avg
        assert result_dict["stretch_grid_flag"] == quic_fire.stretch_grid_flag
        assert result_dict["dz"] == quic_fire.dz
        assert result_dict["dz_array"] == quic_fire.dz_array
        assert result_dict["fuel_density_flag"] == quic_fire.fuel_density_flag
        assert result_dict["fuel_density"] == quic_fire.fuel_density
        assert result_dict["fuel_moisture_flag"] == quic_fire.fuel_moisture_flag
        assert result_dict["fuel_moisture"] == quic_fire.fuel_moisture
        assert result_dict["fuel_height_flag"] == quic_fire.fuel_height_flag
        assert result_dict["fuel_height"] == quic_fire.fuel_height
        assert result_dict["size_scale_flag"] == quic_fire.size_scale_flag
        assert result_dict["size_scale"] == quic_fire.size_scale
        assert result_dict["patch_and_gap_flag"] == quic_fire.patch_and_gap_flag
        assert result_dict["patch_size"] == quic_fire.patch_size
        assert result_dict["gap_size"] == quic_fire.gap_size
        assert (
            result_dict["ignition"]["ignition_flag"] == quic_fire.ignition.ignition_flag
        )
        assert result_dict["ignition"]["x_min"] == quic_fire.ignition.x_min
        assert result_dict["ignition"]["y_min"] == quic_fire.ignition.y_min
        assert result_dict["ignition"]["x_length"] == quic_fire.ignition.x_length
        assert result_dict["ignition"]["y_length"] == quic_fire.ignition.y_length
        assert result_dict["ignitions_per_cell"] == quic_fire.ignitions_per_cell
        assert result_dict["firebrand_flag"] == quic_fire.firebrand_flag
        assert result_dict["auto_kill"] == quic_fire.auto_kill
        assert result_dict["eng_to_atm_out"] == quic_fire.eng_to_atm_out
        assert result_dict["react_rate_out"] == quic_fire.react_rate_out
        assert result_dict["fuel_dens_out"] == quic_fire.fuel_dens_out
        assert result_dict["qf_wind_out"] == quic_fire.qf_wind_out
        assert result_dict["qu_wind_inst_out"] == quic_fire.qu_wind_inst_out
        assert result_dict["qu_wind_avg_out"] == quic_fire.qu_wind_avg_out
        assert result_dict["fuel_moist_out"] == quic_fire.fuel_moist_out
        assert result_dict["mass_burnt_out"] == quic_fire.mass_burnt_out
        assert result_dict["firebrand_out"] == quic_fire.firebrand_out
        assert result_dict["emissions_out"] == quic_fire.emissions_out
        assert result_dict["radiation_out"] == quic_fire.radiation_out
        assert result_dict["surf_eng_out"] == quic_fire.surf_eng_out

    def test_from_dict(self):
        quic_fire = self.get_basic_test_object()
        test_dict = quic_fire.to_dict()
        test_object = QUIC_fire.from_dict(test_dict)
        assert test_object == quic_fire

    def test_to_file(self):
        quic_fire = self.get_basic_test_object()
        for version in ["v5", "v6"]:
            quic_fire.to_file(TMP_DIR, version=version)
            with open(TMP_DIR / "QUIC_fire.inp", "r") as file:
                lines = file.readlines()

            assert quic_fire.fire_flag == int(lines[0].strip().split("!")[0])
            assert quic_fire.random_seed == int(lines[1].strip().split("!")[0])
            assert quic_fire.time_now == int(lines[3].strip().split("!")[0])
            assert quic_fire.sim_time == int(lines[4].strip().split("!")[0])
            assert quic_fire.fire_time_step == int(lines[5].strip().split("!")[0])
            assert quic_fire.quic_time_step == int(lines[6].strip().split("!")[0])
            assert quic_fire.out_time_fire == int(lines[7].strip().split("!")[0])
            assert quic_fire.out_time_wind == int(lines[8].strip().split("!")[0])
            assert quic_fire.out_time_emis_rad == int(lines[9].strip().split("!")[0])
            assert quic_fire.out_time_wind_avg == int(lines[10].strip().split("!")[0])
            assert quic_fire.nz == int(lines[12].strip().split("!")[0])
            assert quic_fire.stretch_grid_flag == int(lines[13].strip().split("!")[0])
            dz_array = []
            for i in range(14, 14 + len(quic_fire.dz_array)):
                dz_array.append(float(lines[i].strip()))
            assert quic_fire.dz_array == dz_array
            current_line = 15 + len(quic_fire.dz_array)
            current_line += 4  # skip unused lines
            current_line += 1  # header
            assert quic_fire.fuel_density_flag == int(
                lines[current_line].strip().split("!")[0]
            )
            assert quic_fire.fuel_density == float(lines[current_line + 1].strip())
            assert quic_fire.fuel_moisture_flag == int(
                lines[current_line + 2].strip().split("!")[0]
            )
            assert quic_fire.fuel_moisture == float(lines[current_line + 3].strip())
            assert quic_fire.fuel_height_flag == int(
                lines[current_line + 4].strip().split("!")[0]
            )
            assert quic_fire.fuel_height == float(lines[current_line + 5].strip())
            if version == "v6":
                assert quic_fire.size_scale_flag == int(
                    lines[current_line + 6].strip().split("!")[0]
                )
                assert quic_fire.patch_and_gap_flag == int(
                    lines[current_line + 7].strip().split("!")[0]
                )
                current_line += 8
            else:
                current_line += 6

            current_line += 1  # header
            ignition_flag = int(lines[current_line].strip().split("!")[0])
            ignition_params = []
            current_line += 1
            for i in range(current_line, current_line + 4):
                ignition_params.append(float(lines[i].strip().split("!")[0]))
            x_min, y_min, x_length, y_length = ignition_params
            assert quic_fire.ignition == RectangleIgnition(
                ignition_flag=ignition_flag,
                x_min=x_min,
                y_min=y_min,
                x_length=x_length,
                y_length=y_length,
            )
            current_line += 4
            assert quic_fire.ignitions_per_cell == int(
                lines[current_line].strip().split("!")[0]
            )
            current_line += 1
            current_line += 1  # header
            assert quic_fire.firebrand_flag == int(
                lines[current_line].strip().split("!")[0]
            )
            current_line += 1
            assert quic_fire.eng_to_atm_out == int(
                lines[current_line + 1].strip().split("!")[0]
            )
            assert quic_fire.react_rate_out == int(
                lines[current_line + 2].strip().split("!")[0]
            )
            assert quic_fire.fuel_dens_out == int(
                lines[current_line + 3].strip().split("!")[0]
            )
            assert quic_fire.qf_wind_out == int(
                lines[current_line + 4].strip().split("!")[0]
            )
            assert quic_fire.qu_wind_inst_out == int(
                lines[current_line + 5].strip().split("!")[0]
            )
            assert quic_fire.qu_wind_avg_out == int(
                lines[current_line + 6].strip().split("!")[0]
            )
            assert quic_fire.fuel_moist_out == int(
                lines[current_line + 8].strip().split("!")[0]
            )
            assert quic_fire.mass_burnt_out == int(
                lines[current_line + 9].strip().split("!")[0]
            )
            assert quic_fire.firebrand_out == int(
                lines[current_line + 10].strip().split("!")[0]
            )
            assert quic_fire.emissions_out == int(
                lines[current_line + 11].strip().split("!")[0]
            )
            assert quic_fire.radiation_out == int(
                lines[current_line + 12].strip().split("!")[0]
            )
            assert quic_fire.surf_eng_out == int(
                lines[current_line + 13].strip().split("!")[0]
            )
            assert quic_fire.auto_kill == int(
                lines[current_line + 15].strip().split("!")[0]
            )

    def test_from_file_v5(self):
        """Test initializing a class from a QUIC_fire.inp
        file."""
        quic_fire = self.get_basic_test_object()
        quic_fire.to_file(TMP_DIR, version="v5")
        test_object = QUIC_fire.from_file(TMP_DIR, version="v5")
        assert isinstance(test_object, QUIC_fire)
        assert quic_fire == test_object

    def test_from_file_v6(self):
        """Test initializing a class from a QUIC_fire.inp
        file."""
        quic_fire = self.get_basic_test_object()
        quic_fire.to_file(TMP_DIR, version="v6")
        test_object = QUIC_fire.from_file(TMP_DIR, version="v6")
        assert isinstance(test_object, QUIC_fire)
        assert quic_fire == test_object

    def test_from_file_v6_with_v5_reader(self):
        """
        This test tries to read in a version 6 file using the version="v5"
        argument which should result in an error.
        """
        quic_fire = self.get_basic_test_object()
        quic_fire.to_file(TMP_DIR, version="v6")
        with pytest.raises(ValueError):
            QUIC_fire.from_file(TMP_DIR, version="v5")

    def test_from_file_v5_with_v6_reader(self):
        quic_fire = self.get_basic_test_object()
        quic_fire.to_file(TMP_DIR, version="v5")
        with pytest.raises(ValueError):
            QUIC_fire.from_file(TMP_DIR, version="v6")


class Test_QFire_Bldg_Advanced_User_Inputs:
    def test_default_init(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()

        assert bldg_inputs.convert_buildings_to_fuel_flag == 0
        assert bldg_inputs.building_fuel_density == 0.5
        assert bldg_inputs.building_attenuation_coefficient == 2.0
        assert bldg_inputs.building_surface_roughness == 0.01
        assert bldg_inputs.convert_fuel_to_canopy_flag == 1
        assert bldg_inputs.update_canopy_winds_flag == 1
        assert bldg_inputs.fuel_attenuation_coefficient == 1.0
        assert bldg_inputs.fuel_surface_roughness == 0.1

    def test_custom_init(self):
        # Change a flag
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs(convert_buildings_to_fuel_flag=1)
        assert bldg_inputs.convert_buildings_to_fuel_flag == 1

        # Change a float
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs(building_fuel_density=0.6)
        assert bldg_inputs.building_fuel_density == 0.6

        # Test data type casting
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs(building_fuel_density="0.6")
        assert isinstance(bldg_inputs.building_fuel_density, float)

    def test_init_invalid_values(self):
        # Test invalid convert_buildings_to_fuel_flag
        for invalid_flag in [-1, 2, "1", 1.0, 1.5]:
            with pytest.raises(ValidationError):
                QFire_Bldg_Advanced_User_Inputs(
                    convert_buildings_to_fuel_flag=invalid_flag
                )

        # Test invalid building_fuel_density
        for invalid_density in [-1, ""]:
            with pytest.raises(ValidationError):
                QFire_Bldg_Advanced_User_Inputs(building_fuel_density=invalid_density)

    def test_to_dict(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()
        result_dict = bldg_inputs.to_dict()

        assert (
            result_dict["convert_buildings_to_fuel_flag"]
            == bldg_inputs.convert_buildings_to_fuel_flag
        )
        assert result_dict["building_fuel_density"] == bldg_inputs.building_fuel_density
        assert (
            result_dict["building_attenuation_coefficient"]
            == bldg_inputs.building_attenuation_coefficient
        )
        assert (
            result_dict["building_surface_roughness"]
            == bldg_inputs.building_surface_roughness
        )
        assert (
            result_dict["convert_fuel_to_canopy_flag"]
            == bldg_inputs.convert_fuel_to_canopy_flag
        )
        assert (
            result_dict["update_canopy_winds_flag"]
            == bldg_inputs.update_canopy_winds_flag
        )
        assert (
            result_dict["fuel_attenuation_coefficient"]
            == bldg_inputs.fuel_attenuation_coefficient
        )
        assert (
            result_dict["fuel_surface_roughness"] == bldg_inputs.fuel_surface_roughness
        )

    def test_from_dict(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()
        result_dict = bldg_inputs.to_dict()
        test_obj = QFire_Bldg_Advanced_User_Inputs.from_dict(result_dict)
        assert test_obj == bldg_inputs

    def test_to_docs(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()
        result_dict = bldg_inputs.to_dict()
        result_docs = bldg_inputs.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()
        bldg_inputs.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QFire_Bldg_Advanced_User_Inputs.inp", "r") as file:
            lines = file.readlines()
        assert (
            int(lines[0].strip().split("!")[0])
            == bldg_inputs.convert_buildings_to_fuel_flag
        )
        assert (
            float(lines[1].strip().split("!")[0]) == bldg_inputs.building_fuel_density
        )
        assert (
            float(lines[2].strip().split("!")[0])
            == bldg_inputs.building_attenuation_coefficient
        )
        assert (
            float(lines[3].strip().split("!")[0])
            == bldg_inputs.building_surface_roughness
        )
        assert (
            int(lines[4].strip().split("!")[0])
            == bldg_inputs.convert_fuel_to_canopy_flag
        )
        assert (
            int(lines[5].strip().split("!")[0]) == bldg_inputs.update_canopy_winds_flag
        )
        assert (
            float(lines[6].strip().split("!")[0])
            == bldg_inputs.fuel_attenuation_coefficient
        )
        assert (
            float(lines[7].strip().split("!")[0]) == bldg_inputs.fuel_surface_roughness
        )

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            bldg_inputs.to_file(
                "/non_existent_path/QFire_Bldg_Advanced_User_Inputs.inp"
            )

    def test_from_file(self):
        bldg_inputs = QFire_Bldg_Advanced_User_Inputs()
        bldg_inputs.to_file(TMP_DIR)
        test_object = QFire_Bldg_Advanced_User_Inputs.from_file(TMP_DIR)
        assert isinstance(test_object, QFire_Bldg_Advanced_User_Inputs)
        assert bldg_inputs == test_object


class Test_QFire_Plume_Advanced_User_Inputs:
    def test_default_init(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()

        assert plume_inputs.max_plumes_per_timestep == 150000
        assert plume_inputs.min_plume_updraft_velocity == 0.1
        assert plume_inputs.max_plume_updraft_velocity == 100.0
        assert plume_inputs.min_velocity_ratio == 0.1
        assert plume_inputs.brunt_vaisala_freq_squared == 0.0
        assert plume_inputs.creeping_flag == 1
        assert plume_inputs.adaptive_timestep_flag == 0
        assert plume_inputs.plume_timestep == 1.0
        assert plume_inputs.sor_option_flag == 1
        assert plume_inputs.sor_alpha_plume_center == 10.0
        assert plume_inputs.sor_alpha_plume_edge == 1.0
        assert plume_inputs.max_plume_merging_angle == 30.0
        assert plume_inputs.max_plume_overlap_fraction == 0.7
        assert plume_inputs.plume_to_grid_updrafts_flag == 1
        assert plume_inputs.max_points_along_plume_edge == 10
        assert plume_inputs.plume_to_grid_intersection_flag == 1

    def test_custom_init(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs(
            max_plumes_per_timestep=100000,
            min_plume_updraft_velocity=0.2,
            creeping_flag=0,
            max_plume_updraft_velocity="100.",
        )
        assert plume_inputs.max_plumes_per_timestep == 100000
        assert plume_inputs.min_plume_updraft_velocity == 0.2
        assert plume_inputs.creeping_flag == 0
        assert plume_inputs.max_plume_updraft_velocity == 100.0

    def test_invalid_values(self):
        # Invalid max_plumes_per_timestep (Positive integer)
        for value in [-1, 0, 1.5, "a"]:
            with pytest.raises(ValidationError):
                QFire_Plume_Advanced_User_Inputs(max_plumes_per_timestep=value)

        # Invalid min_plume_updraft_velocity (Positive float)
        for value in [-1, 0, "a"]:
            with pytest.raises(ValidationError):
                QFire_Plume_Advanced_User_Inputs(min_plume_updraft_velocity=value)

        # Invalid brunt_vaisala_freq_squared (Non-negative float)
        for value in [-1, "a"]:
            with pytest.raises(ValidationError):
                QFire_Plume_Advanced_User_Inputs(brunt_vaisala_freq_squared=value)

        # Invalid creeping_flag (Literal 0 or 1)
        for value in [-1, 2, 1.5, "a", "0"]:
            with pytest.raises(ValidationError):
                QFire_Plume_Advanced_User_Inputs(creeping_flag=value)

    def test_to_dict(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()
        result_dict = plume_inputs.to_dict()

        assert (
            result_dict["max_plumes_per_timestep"]
            == plume_inputs.max_plumes_per_timestep
        )
        assert (
            result_dict["min_plume_updraft_velocity"]
            == plume_inputs.min_plume_updraft_velocity
        )
        assert (
            result_dict["max_plume_updraft_velocity"]
            == plume_inputs.max_plume_updraft_velocity
        )
        assert result_dict["min_velocity_ratio"] == plume_inputs.min_velocity_ratio
        assert (
            result_dict["brunt_vaisala_freq_squared"]
            == plume_inputs.brunt_vaisala_freq_squared
        )
        assert result_dict["creeping_flag"] == plume_inputs.creeping_flag
        assert (
            result_dict["adaptive_timestep_flag"] == plume_inputs.adaptive_timestep_flag
        )
        assert result_dict["plume_timestep"] == plume_inputs.plume_timestep
        assert result_dict["sor_option_flag"] == plume_inputs.sor_option_flag
        assert (
            result_dict["sor_alpha_plume_center"] == plume_inputs.sor_alpha_plume_center
        )
        assert result_dict["sor_alpha_plume_edge"] == plume_inputs.sor_alpha_plume_edge
        assert (
            result_dict["max_plume_merging_angle"]
            == plume_inputs.max_plume_merging_angle
        )
        assert (
            result_dict["max_plume_overlap_fraction"]
            == plume_inputs.max_plume_overlap_fraction
        )
        assert (
            result_dict["plume_to_grid_updrafts_flag"]
            == plume_inputs.plume_to_grid_updrafts_flag
        )
        assert (
            result_dict["max_points_along_plume_edge"]
            == plume_inputs.max_points_along_plume_edge
        )
        assert (
            result_dict["plume_to_grid_intersection_flag"]
            == plume_inputs.plume_to_grid_intersection_flag
        )

    def test_from_dict(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()
        result_dict = plume_inputs.to_dict()
        test_obj = QFire_Plume_Advanced_User_Inputs.from_dict(result_dict)
        assert test_obj == plume_inputs

    def test_to_docs(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()
        result_dict = plume_inputs.to_dict()
        result_docs = plume_inputs.get_documentation()
        for key in result_dict:
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()
        plume_inputs.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QFire_Plume_Advanced_User_Inputs.inp", "r") as file:
            lines = file.readlines()
        assert (
            int(lines[0].strip().split("!")[0]) == plume_inputs.max_plumes_per_timestep
        )
        assert (
            float(lines[1].strip().split("!")[0])
            == plume_inputs.min_plume_updraft_velocity
        )
        assert (
            float(lines[2].strip().split("!")[0])
            == plume_inputs.max_plume_updraft_velocity
        )
        assert float(lines[3].strip().split("!")[0]) == plume_inputs.min_velocity_ratio
        assert (
            float(lines[4].strip().split("!")[0])
            == plume_inputs.brunt_vaisala_freq_squared
        )
        assert int(lines[5].strip().split("!")[0]) == plume_inputs.creeping_flag
        assert (
            int(lines[6].strip().split("!")[0]) == plume_inputs.adaptive_timestep_flag
        )
        assert float(lines[7].strip().split("!")[0]) == plume_inputs.plume_timestep
        assert int(lines[8].strip().split("!")[0]) == plume_inputs.sor_option_flag
        assert (
            float(lines[9].strip().split("!")[0]) == plume_inputs.sor_alpha_plume_center
        )
        assert (
            float(lines[10].strip().split("!")[0]) == plume_inputs.sor_alpha_plume_edge
        )
        assert (
            float(lines[11].strip().split("!")[0])
            == plume_inputs.max_plume_merging_angle
        )
        assert (
            float(lines[12].strip().split("!")[0])
            == plume_inputs.max_plume_overlap_fraction
        )
        assert (
            int(lines[13].strip().split("!")[0])
            == plume_inputs.plume_to_grid_updrafts_flag
        )
        assert (
            int(lines[14].strip().split("!")[0])
            == plume_inputs.max_points_along_plume_edge
        )
        assert (
            int(lines[15].strip().split("!")[0])
            == plume_inputs.plume_to_grid_intersection_flag
        )

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            plume_inputs.to_file(
                "/non_existent_path/QFire_Plume_Advanced_User_Inputs.inp"
            )

    def test_from_file(self):
        plume_inputs = QFire_Plume_Advanced_User_Inputs()
        plume_inputs.to_file(TMP_DIR)
        test_object = QFire_Plume_Advanced_User_Inputs.from_file(TMP_DIR)
        assert isinstance(test_object, QFire_Plume_Advanced_User_Inputs)
        assert plume_inputs == test_object


class TestQUTopoInputs:
    def get_default_test_object(self):
        return QU_TopoInputs(topography=Topography(topo_flag=0))

    def get_complex_test_object(self):
        return QU_TopoInputs(
            topography=GaussianHillTopo(
                x_hilltop=100, y_hilltop=150, elevation_max=500, elevation_std=20
            ),
            smoothing_method=1,
            sor_relax=1.78,
        )

    def test_default_inputs(self):
        topoinputs = self.get_default_test_object()
        assert topoinputs.filename == "topo.dat"
        assert isinstance(topoinputs.topography, Topography)
        assert topoinputs.smoothing_passes == 500

    def test_complex_inputs(self):
        topoinputs = self.get_complex_test_object()
        assert topoinputs.topography.topo_flag.value == 1
        assert topoinputs._topo_lines == (
            "1\t\t! N/A, "
            "topo flag: 0 = flat, 1 = Gaussian hill, "
            "2 = hill pass, 3 = slope mesa, 4 = canyon, "
            "5 = custom, 6 = half circle, 7 = sinusoid, "
            "8 = cos hill, 9 = QP_elevation.inp, "
            "10 = terrainOutput.txt (ARA), "
            "11 = terrain.dat (firetec)\n"
            "100.0\t! m, x-center\n"
            "150.0\t! m, y-center\n"
            "500.0\t! m, max height\n"
            "20.0\t! m, std"
        )
        assert topoinputs.smoothing_method == 1
        assert topoinputs.sor_relax == 1.78

    def test_to_dict(self):
        topoinputs = self.get_default_test_object()
        test_dict = topoinputs.to_dict()
        assert test_dict["smoothing_method"] == topoinputs.smoothing_method
        assert test_dict["smoothing_passes"] == topoinputs.smoothing_passes
        assert test_dict["sor_iterations"] == topoinputs.sor_iterations
        assert test_dict["sor_cycles"] == topoinputs.sor_cycles
        assert test_dict["sor_relax"] == topoinputs.sor_relax

        topoinputs = self.get_complex_test_object()
        test_dict = topoinputs.to_dict()
        assert test_dict["topography"]["x_hilltop"] == topoinputs.topography.x_hilltop
        assert test_dict["topography"]["y_hilltop"] == topoinputs.topography.y_hilltop
        assert (
            test_dict["topography"]["elevation_max"]
            == topoinputs.topography.elevation_max
        )
        assert (
            test_dict["topography"]["elevation_std"]
            == topoinputs.topography.elevation_std
        )
        assert test_dict["smoothing_method"] == topoinputs.smoothing_method
        assert test_dict["smoothing_passes"] == topoinputs.smoothing_passes
        assert test_dict["sor_iterations"] == topoinputs.sor_iterations
        assert test_dict["sor_cycles"] == topoinputs.sor_cycles
        assert test_dict["sor_relax"] == topoinputs.sor_relax

    def test_from_dict(self):
        topoinputs = self.get_default_test_object()
        result_dict = topoinputs.to_dict()
        test_obj = QU_TopoInputs.from_dict(result_dict)
        assert test_obj == topoinputs

    def test_to_file(self):
        topoinputs = self.get_default_test_object()
        topoinputs.to_file(TMP_DIR)
        with open(TMP_DIR / "QU_TopoInputs.inp", "r") as file:
            lines = file.readlines()
        topo_flag = int(lines[2].strip().split("!")[0])
        assert topo_flag == topoinputs.topography.topo_flag.value
        add_dict = {
            0: 0,
            1: 4,
            2: 2,
            3: 3,
            4: 5,
            5: 0,
            6: 3,
            7: 2,
            8: 2,
            9: 0,
            10: 0,
            11: 0,
        }
        add = add_dict.get(topo_flag)
        current_line = current_line = 3 + add
        assert (
            int(lines[current_line].strip().split("!")[0])
            == topoinputs.smoothing_method
        )
        assert (
            int(lines[current_line + 1].strip().split("!")[0])
            == topoinputs.smoothing_passes
        )
        assert (
            int(lines[current_line + 2].strip().split("!")[0])
            == topoinputs.sor_iterations
        )
        assert (
            int(lines[current_line + 3].strip().split("!")[0]) == topoinputs.sor_cycles
        )
        assert (
            float(lines[current_line + 4].strip().split("!")[0]) == topoinputs.sor_relax
        )

        topoinputs = self.get_complex_test_object()
        topoinputs.to_file(TMP_DIR)
        with open(TMP_DIR / "QU_TopoInputs.inp", "r") as file:
            lines = file.readlines()
        topo_flag = int(lines[2].strip().split("!")[0])
        assert topo_flag == topoinputs.topography.topo_flag.value
        add_dict = {
            0: 0,
            1: 4,
            2: 2,
            3: 3,
            4: 5,
            5: 0,
            6: 3,
            7: 2,
            8: 2,
            9: 0,
            10: 0,
            11: 0,
        }
        add = add_dict.get(topo_flag)
        assert float(lines[3].strip().split("!")[0]) == topoinputs.topography.x_hilltop
        assert float(lines[4].strip().split("!")[0]) == topoinputs.topography.y_hilltop
        assert (
            float(lines[5].strip().split("!")[0]) == topoinputs.topography.elevation_max
        )
        assert (
            float(lines[6].strip().split("!")[0]) == topoinputs.topography.elevation_std
        )
        current_line = current_line = 3 + add
        assert (
            int(lines[current_line].strip().split("!")[0])
            == topoinputs.smoothing_method
        )
        assert (
            int(lines[current_line + 1].strip().split("!")[0])
            == topoinputs.smoothing_passes
        )
        assert (
            int(lines[current_line + 2].strip().split("!")[0])
            == topoinputs.sor_iterations
        )
        assert (
            int(lines[current_line + 3].strip().split("!")[0]) == topoinputs.sor_cycles
        )
        assert (
            float(lines[current_line + 4].strip().split("!")[0]) == topoinputs.sor_relax
        )

    def test_from_file(self):
        topoinputs = self.get_default_test_object()
        topoinputs.to_file(TMP_DIR)
        test_object = QU_TopoInputs.from_file(TMP_DIR)
        assert isinstance(test_object, QU_TopoInputs)
        assert topoinputs == test_object

        topoinputs = self.get_complex_test_object()
        topoinputs.to_file(TMP_DIR)
        test_object = QU_TopoInputs.from_file(TMP_DIR)
        assert isinstance(test_object, QU_TopoInputs)
        assert topoinputs == test_object


class TestRuntimeAdvancedUserInputs:
    def get_test_object(self):
        return RuntimeAdvancedUserInputs()

    def test_init(self):
        raui = self.get_test_object()
        assert isinstance(raui, RuntimeAdvancedUserInputs)
        assert raui.num_cpus == 1
        assert raui.use_acw == 0

    def test_from_file(self):
        raui = self.get_test_object()
        raui.to_file(TMP_DIR)
        test_object = RuntimeAdvancedUserInputs.from_file(TMP_DIR)
        assert raui == test_object


class TestQUmovingcoords:
    def get_test_object(self):
        return QU_movingcoords()

    def test_init(self):
        qu_moving = self.get_test_object()
        assert isinstance(qu_moving, QU_movingcoords)


class TestQUbuildout:
    def get_test_object(self):
        return QP_buildout()

    def test_init(self):
        qp_buildout = self.get_test_object()
        assert isinstance(qp_buildout, QP_buildout)


class Test_QU_metparams:
    def get_test_object(self):
        return QU_metparams()

    def test_init(self):
        qu_metparams = self.get_test_object()
        assert qu_metparams.num_sensors == 1

    def test_to_dict(self):
        qu_metparams = self.get_test_object()
        result_dict = qu_metparams.to_dict()
        assert result_dict["num_sensors"] == qu_metparams.num_sensors

    def test_from_dict(self):
        qu_metparams = self.get_test_object()
        result_dict = qu_metparams.to_dict()
        test_obj = QU_metparams.from_dict(result_dict)
        assert test_obj == qu_metparams

    def test_to_file(self):
        qu_metparams = self.get_test_object()
        qu_metparams.to_file(TMP_DIR)

        # Read the content of the file and check for correctness
        with open(TMP_DIR / "QU_metparams.inp", "r") as file:
            lines = file.readlines()
        assert int(lines[2].strip().split("!")[0]) == qu_metparams.num_sensors

    def test_from_file(self):
        qu_metparams = self.get_test_object()
        qu_metparams.to_file(TMP_DIR)
        test_object = QU_metparams.from_file(TMP_DIR)
        assert isinstance(test_object, QU_metparams)
        assert qu_metparams == test_object


class TestWindSensorArray:
    def test_add_sensor(self):
        windarray = WindSensorArray(time_now=1)
        sensor1 = windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        assert len(windarray.sensor_array) == 1
        assert windarray.sensor_array[0] == sensor1
        sensor2 = windarray.add_sensor(6, 230, 0, 10, 1, 1)
        assert len(windarray.sensor_array) == 2
        assert windarray.sensor_array[0] == sensor1
        assert windarray.sensor_array[1] == sensor2
        sensor3 = windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        assert len(windarray.sensor_array) == 3
        assert windarray.sensor_array[0] == sensor1
        assert windarray.sensor_array[1] == sensor2
        assert windarray.sensor_array[2] == sensor3

    def test_getattr(self):
        windarray = WindSensorArray()
        # sensor list is empty, so trying to get a sensor throws an error
        with pytest.raises(AttributeError):
            windarray.sensor1
        # We should get an error for any attribute not in WindSensorArray or not called sensor*
        with pytest.raises(AttributeError):
            windarray.nonsense
        # Now add three wind sensors and see if we can get them back
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        sensor1 = windarray.sensor1
        sensor2 = windarray.sensor2
        sensor3 = windarray.sensor3
        assert isinstance(sensor1, WindSensor)
        assert isinstance(sensor2, WindSensor)
        assert isinstance(sensor3, WindSensor)
        assert sensor1 == windarray.sensor_array[0]
        assert sensor2 == windarray.sensor_array[1]
        assert sensor3 == windarray.sensor_array[2]

    def test_update_sensor(self):
        windarray = WindSensorArray(time_now=1)
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        # Make sure a nonexistent sensor can't be updated
        with pytest.raises(AttributeError):
            windarray.update_sensor(sensor_name="sensor2", wind_speeds=3)
        with pytest.raises(AttributeError):
            windarray.update_sensor(sensor_name="nonsense", wind_speeds=3)
        # Make sure the right parameters are updated
        windarray.update_sensor(sensor_name="sensor1", wind_speeds=3)
        assert windarray.sensor1.wind_speeds == [3]
        assert windarray.sensor1.wind_directions == [270]
        # Add another sensor and update everything
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.update_sensor(
            sensor_name="sensor2",
            wind_times=[0, 2, 3],
            wind_speeds=[4, 5, 6],
            wind_directions=[90, 180, 270],
            sensor_height=5.1,
            x_location=2,
            y_location=2,
        )
        assert windarray.sensor2.wind_times == [0, 2, 3]
        assert windarray.sensor2.wind_speeds == [4, 5, 6]
        assert windarray.sensor2.wind_directions == [90, 180, 270]
        assert windarray.sensor2.sensor_height == 5.1
        assert windarray.sensor2.x_location == 2
        assert windarray.sensor2.y_location == 2
        # Add a third sensor and try updating
        windarray.add_sensor(4, 10, 0, 10, 1, 1)
        windarray.update_sensor(sensor_name="sensor3", sensor_height=11)
        assert windarray.sensor3.sensor_height == 11
        assert windarray.sensor3.wind_speeds == [4]

    # TODO: make sure wind sensors don't get put in the same location?
    # def test_validate_sensor_location(self):
    #     # Validation should not allow two sensors to be in the same place at the same height
    #     windarray = WindSensorArray(time_now=1)
    #     windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
    #     with pytest.raises(ValueError):
    #         windarray.add_sensor(5, 270, 0, 6.1, 1, 1)

    def test_to_file(self):
        # Test case with one sensor and one windshift
        windarray = WindSensorArray()
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        windarray.to_file(time_now=1, directory=TMP_DIR)
        assert windarray.wind_times == [0]
        sensor1_path = TMP_DIR / "sensor1.inp"
        assert sensor1_path.exists()
        # Test case with multiple sensors and one windshift
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        windarray.to_file(time_now=1, directory=TMP_DIR)
        sensor2_path = TMP_DIR / "sensor2.inp"
        sensor3_path = TMP_DIR / "sensor3.inp"
        assert sensor2_path.exists()
        assert sensor3_path.exists()
        assert windarray.wind_times == [0]
        # Test case with multiple sensors with different wind shifts
        windarray.update_sensor(
            sensor_name="sensor2",
            wind_times=[0, 100, 300],
            wind_speeds=[6, 4, 5],
            wind_directions=[270, 270, 270],
        )
        windarray.update_sensor(
            sensor_name="sensor3",
            wind_times=[0, 200, 400],
            wind_speeds=[4, 4, 4],
            wind_directions=[10, 30, 45],
        )
        windarray.to_file(time_now=1, directory=TMP_DIR)
        assert windarray.wind_times == [0, 100, 200, 300, 400]

    def test_from_file(self):
        # Test three wind sensors with a single windshift
        windarray = WindSensorArray()
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        windarray.to_file(time_now=1, directory=TMP_DIR)
        new_array = WindSensorArray.from_file(TMP_DIR)
        assert isinstance(new_array, WindSensorArray)
        assert len(new_array.sensor_array) == 3
        assert new_array.sensor1.wind_speeds == [5]
        assert new_array.sensor2.wind_directions == [230]
        # Add a windshift to one sensor
        windarray.update_sensor(
            "sensor1",
            wind_times=[0, 60],
            wind_speeds=[5, 6],
            wind_directions=[270, 270],
        )
        windarray.to_file(time_now=1, directory=TMP_DIR)
        new_array = WindSensorArray.from_file(TMP_DIR)
        assert windarray.wind_times == [0, 60]

    def test_to_dict(self):
        windarray = WindSensorArray(time_now=1)
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        result_dict = windarray.to_dict()
        assert result_dict["wind_times"] == windarray.wind_times
        assert (
            result_dict["sensor_array"][0]["wind_speeds"]
            == windarray.sensor1.wind_speeds
        )
        assert (
            result_dict["sensor_array"][1]["wind_directions"]
            == windarray.sensor2.wind_directions
        )
        assert (
            result_dict["sensor_array"][2]["wind_times"] == windarray.sensor2.wind_times
        )

    def test_from_dict(self):
        windarray = WindSensorArray(time_now=1)
        windarray.add_sensor(5, 270, 0, 6.1, 1, 1)
        windarray.add_sensor(6, 230, 0, 10, 1, 1)
        windarray.add_sensor(4, 10, 0, 6.1, 2, 2)
        result_dict = windarray.to_dict()
        new_windarray = WindSensorArray.from_dict(result_dict)
        assert new_windarray == windarray


class TestWindSensor:
    def test_validation(self):
        # make sure float/int inputs are converted to lists
        sensor = WindSensor(
            name="sensor1",
            time_now=1,
            wind_times=0,
            wind_speeds=4.5,
            wind_directions=270,
        )
        assert sensor.wind_times == [0]
        assert sensor.wind_speeds == [4.5]
        assert sensor.wind_directions == [270]
        # first element of wind_times must be zero
        with pytest.raises(ValueError):
            sensor.wind_times = [1, 2, 3]
        # wind lists have to be the same length
        with pytest.raises(ValueError):
            sensor = WindSensor(
                name="sensor1",
                time_now=1,
                wind_times=0,
                wind_speeds=[1, 2, 3],
                wind_directions=270,
            )
        with pytest.raises(ValueError):
            sensor = WindSensor(
                name="sensor1",
                time_now=1,
                wind_times=[0, 100, 200],
                wind_speeds=[1, 2, 3],
                wind_directions=270,
            )

    def test_get_wind_lines(self):
        # test scenario where there are 3 global wind times, but wind sensor has only 1 shift
        global_times = [0, 100, 200]
        sensor = WindSensor(
            name="sensor1",
            wind_times=0,
            wind_speeds=4.5,
            wind_directions=270,
        )
        wind_lines = sensor.get_wind_lines(global_times, time_now=1)
        assert isinstance(wind_lines, str)
        assert wind_lines == (
            f"1 !X coordinate (meters)\n"
            f"1 !Y coordinate (meters)"
            f"\n1 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 4.5 270"
            f"\n101 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 4.5 270"
            f"\n201 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 4.5 270"
        )

        # test scenario where there are 3 global wind times, and two wind shifts
        global_times = [0, 100, 200]
        sensor = WindSensor(
            name="sensor1",
            wind_times=[0, 200],
            wind_speeds=[4.5, 5.5],
            wind_directions=[270, 330],
        )
        wind_lines = sensor.get_wind_lines(global_times, time_now=1)
        assert isinstance(wind_lines, str)
        assert wind_lines == (
            f"1 !X coordinate (meters)\n"
            f"1 !Y coordinate (meters)"
            f"\n1 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 4.5 270"
            f"\n101 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 4.5 270"
            f"\n201 !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n"
            f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n"
            f"0.1 !site zo\n"
            f"0. ! 1/L (default = 0)\n"
            f"!Height (m), Speed (m/s), Direction (deg relative to true N)\n"
            f"6.1 5.5 330"
        )

    def test_to_file(self):
        global_times = [0, 200]
        sensor = WindSensor(
            name="sensor1",
            wind_times=[0, 200],
            wind_speeds=[4.5, 5.5],
            wind_directions=[270, 330],
        )
        sensor.to_file(global_times, 1, TMP_DIR, "latest")
        with open(TMP_DIR / "sensor1.inp", "r") as file:
            lines = file.readlines()
        assert str(lines[0].strip().split("!")[0].strip()) == sensor.name
        assert float(lines[11].split(" ")[0]) == sensor.sensor_height
        assert float(lines[11].split(" ")[1]) == sensor.wind_speeds[0]
        assert float(lines[11].split(" ")[2]) == sensor.wind_directions[0]

    def test_from_file(self):
        global_times = [0, 200]
        sensor = WindSensor(
            name="sensor1",
            wind_times=[0, 200],
            wind_speeds=[4.5, 5.5],
            wind_directions=[270, 330],
        )
        sensor.to_file(global_times, 1, TMP_DIR, "latest")
        sensor1 = WindSensor.from_file(TMP_DIR, "sensor1")
        assert isinstance(sensor1, WindSensor)
        assert sensor1.name == sensor.name
        assert sensor1.wind_times == sensor.wind_times
        assert sensor1.wind_directions == sensor.wind_directions
        assert sensor1.wind_speeds == sensor.wind_speeds
        assert sensor1.sensor_height == sensor.sensor_height
        assert sensor1.x_location == sensor.x_location
        assert sensor1.y_location == sensor.y_location

    def test_to_dict(self):
        sensor = WindSensor(
            name="sensor1",
            time_now=1,
            wind_times=[0, 200],
            wind_speeds=[4.5, 5.5],
            wind_directions=[270, 330],
        )
        result_dict = sensor.to_dict()
        assert result_dict["name"] == "sensor1"
        assert result_dict["wind_directions"] == [270, 330]

    def test_from_dict(self):
        sensor = WindSensor(
            name="sensor1",
            time_now=1,
            wind_times=[0, 200],
            wind_speeds=[4.5, 5.5],
            wind_directions=[270, 330],
        )
        result_dict = sensor.to_dict()
        new_sensor = WindSensor.from_dict(result_dict)
        assert isinstance(new_sensor, WindSensor)
        assert sensor == new_sensor


class TestSimulationInputs:
    @staticmethod
    def get_test_object():
        return SimulationInputs.create_simulation(
            nx=150,
            ny=150,
            fire_nz=1,
            wind_speed=5.0,
            wind_direction=90,
            simulation_time=65,
        )

    def test_basic_inputs(self):
        sim_inputs = self.get_test_object()
        assert isinstance(sim_inputs, SimulationInputs)

    def test_input_files(self):
        sim_inputs = self.get_test_object()
        assert isinstance(sim_inputs.gridlist, Gridlist)
        assert isinstance(sim_inputs.rasterorigin, RasterOrigin)
        assert isinstance(sim_inputs.qu_buildings, QU_Buildings)
        assert isinstance(sim_inputs.qu_fileoptions, QU_Fileoptions)
        assert isinstance(sim_inputs.qu_simparams, QU_Simparams)
        assert isinstance(
            sim_inputs.qfire_advanced_user_inputs, QFire_Advanced_User_Inputs
        )
        assert isinstance(sim_inputs.quic_fire, QUIC_fire)
        assert isinstance(
            sim_inputs.qfire_bldg_advanced_user_inputs, QFire_Bldg_Advanced_User_Inputs
        )
        assert isinstance(
            sim_inputs.qfire_plume_advanced_user_inputs,
            QFire_Plume_Advanced_User_Inputs,
        )
        assert isinstance(sim_inputs.qu_topoinputs, QU_TopoInputs)
        assert isinstance(
            sim_inputs.runtime_advanced_user_inputs, RuntimeAdvancedUserInputs
        )
        assert isinstance(sim_inputs.qu_movingcoords, QU_movingcoords)
        assert isinstance(sim_inputs.qp_buildout, QP_buildout)
        assert isinstance(sim_inputs.qu_metparams, QU_metparams)
        assert isinstance(sim_inputs.windsensors, WindSensorArray)

        assert sim_inputs.quic_fire.nz == 1
        assert sim_inputs.quic_fire.sim_time == 65
        assert sim_inputs.qu_simparams.nx == 150
        assert sim_inputs.qu_simparams.ny == 150
        assert sim_inputs.qu_simparams.wind_times[0] == sim_inputs.quic_fire.time_now
        assert sim_inputs.windsensors.sensor1.wind_speeds == [5.0]
        assert sim_inputs.windsensors.sensor1.wind_directions == [90]

    def test_set_uniform_fuels(self):
        sim_inputs = self.get_test_object()

        # Test with default size scale and patch/gap
        sim_inputs.set_uniform_fuels(
            fuel_density=0.6, fuel_moisture=0.05, fuel_height=0.9
        )
        assert sim_inputs.quic_fire.fuel_density_flag == 1
        assert sim_inputs.quic_fire.fuel_density == 0.6
        assert sim_inputs.quic_fire.fuel_moisture_flag == 1
        assert sim_inputs.quic_fire.fuel_moisture == 0.05
        assert sim_inputs.quic_fire.fuel_height_flag == 1
        assert sim_inputs.quic_fire.fuel_height == 0.9
        assert sim_inputs.quic_fire.size_scale_flag == 0
        assert sim_inputs.quic_fire.patch_and_gap_flag == 0

        # Test with custom uniform size scale and patch/gap values
        sim_inputs.set_uniform_fuels(
            fuel_density=0.6,
            fuel_moisture=0.05,
            fuel_height=0.9,
            size_scale=0.00025,
            patch_size=0.5,
            gap_size=0.5,
        )
        assert sim_inputs.quic_fire.size_scale_flag == 1
        assert sim_inputs.quic_fire.size_scale == 0.00025
        assert sim_inputs.quic_fire.patch_and_gap_flag == 1
        assert sim_inputs.quic_fire.patch_size == 0.5
        assert sim_inputs.quic_fire.gap_size == 0.5

    def test_set_rectangle_ignition(self):
        sim_inputs = self.get_test_object()
        sim_inputs.set_rectangle_ignition(
            x_min=20,
            y_min=20,
            x_length=10,
            y_length=110,
        )
        assert sim_inputs.quic_fire.ignition == RectangleIgnition(
            x_min=20,
            y_min=20,
            x_length=10,
            y_length=110,
        )

    def test_set_output_files(self):
        sim_inputs = self.get_test_object()
        sim_inputs.set_output_files(mass_burnt=True, emissions=True)
        assert sim_inputs.quic_fire.eng_to_atm_out == 0
        assert sim_inputs.quic_fire.react_rate_out == 0
        assert sim_inputs.quic_fire.fuel_dens_out == 0
        assert sim_inputs.quic_fire.qf_wind_out == 0
        assert sim_inputs.quic_fire.qu_wind_inst_out == 0
        assert sim_inputs.quic_fire.qu_wind_avg_out == 0
        assert sim_inputs.quic_fire.fuel_moist_out == 0
        assert sim_inputs.quic_fire.mass_burnt_out == 1
        assert sim_inputs.quic_fire.radiation_out == 0
        assert sim_inputs.quic_fire.surf_eng_out == 0
        assert sim_inputs.quic_fire.emissions_out == 2

    def test_set_output_times(self):
        sim_inputs = self.get_test_object()
        sim_inputs.set_output_interval(60)
        assert sim_inputs.quic_fire.out_time_fire == 60
        assert sim_inputs.quic_fire.out_time_wind == 60
        assert sim_inputs.quic_fire.out_time_wind_avg == 60
        assert sim_inputs.quic_fire.out_time_emis_rad == 60

        # Test with invalid intervals
        with pytest.raises(ValueError):
            sim_inputs.set_output_interval(-1)
        with pytest.raises(ValueError):
            sim_inputs.set_output_interval(0)
        with pytest.raises(ValueError):
            sim_inputs.set_output_interval("a")

    def test_set_custom_simulation(self):
        # Test default set custom simulation (no arguments)
        sim_inputs = self.get_test_object()
        sim_inputs.set_custom_simulation()
        assert sim_inputs.quic_fire.fuel_density_flag == 3
        assert sim_inputs.quic_fire.fuel_moisture_flag == 3
        assert sim_inputs.quic_fire.fuel_height_flag == 3
        assert sim_inputs.quic_fire.size_scale_flag == 0  # Default
        assert sim_inputs.quic_fire.patch_and_gap_flag == 0  # Default
        assert sim_inputs.quic_fire.ignition == Ignition(ignition_flag=IgnitionFlags(7))
        assert sim_inputs.qu_topoinputs.topography == Topography(topo_flag=TopoFlags(5))

        # Test interpolate argument
        sim_inputs = self.get_test_object()
        sim_inputs.set_custom_simulation(interpolate=True)
        assert sim_inputs.quic_fire.fuel_density_flag == 4
        assert sim_inputs.quic_fire.fuel_moisture_flag == 4
        assert sim_inputs.quic_fire.fuel_height_flag == 4
        assert sim_inputs.quic_fire.ignition == Ignition(ignition_flag=IgnitionFlags(7))
        assert sim_inputs.qu_topoinputs.topography == Topography(topo_flag=TopoFlags(5))

        # Test including size scale and patch/gap from set custom simulation
        sim_inputs = self.get_test_object()
        sim_inputs.set_custom_simulation(size_scale=True, patch_and_gap=True)
        assert sim_inputs.quic_fire.size_scale_flag == 3
        assert sim_inputs.quic_fire.patch_and_gap_flag == 2

        # Test excluding topo flag from set custom simulation
        sim_inputs = self.get_test_object()
        sim_inputs.set_custom_simulation(topo=False)
        assert sim_inputs.qu_topoinputs.topography == Topography(topo_flag=TopoFlags(0))

    def test_write_inputs_v5(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR, version="v5")

    def test_write_inputs_v6(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR, version="v6")

    def test_new_wind_sensor(self):
        sim_inputs = self.get_test_object()
        # Add a new wind sensor
        sim_inputs.new_wind_sensor(
            x_location=2,
            y_location=2,
            wind_speeds=[6, 6],
            wind_directions=[270, 350],
            wind_times=[0, 100],
            sensor_height=6.1,
        )
        # test that a windsensor was added to the sensor array
        assert len(sim_inputs.windsensors.sensor_array) == 2
        # and that windarray wind times were updated
        assert sim_inputs.windsensors.wind_times == [0, 100]
        # but the wind times in qu_simparams should not be reflected until the write stage
        assert len(sim_inputs.qu_simparams.wind_times) == 1
        sim_inputs.write_inputs(TMP_DIR)
        assert sim_inputs.windsensors.wind_times == [0, 100]
        assert sim_inputs.qu_simparams.wind_times == [
            s + sim_inputs.quic_fire.time_now for s in sim_inputs.windsensors.wind_times
        ]
        # Try replacing sensor1
        sim_inputs.new_wind_sensor(
            update="sensor1",
            wind_speeds=[6, 6, 6],
            wind_directions=[270, 350, 270],
            wind_times=[0, 100, 200],
        )
        # test that a windsensor was not added to the sensor array
        assert len(sim_inputs.windsensors.sensor_array) == 2
        # and that windarray wind times were updated
        assert sim_inputs.windsensors.wind_times == [0, 100, 200]
        # but the wind times in qu_simparams should not be reflected until the write stage
        assert len(sim_inputs.qu_simparams.wind_times) == 2
        sim_inputs.write_inputs(TMP_DIR)
        assert sim_inputs.windsensors.wind_times == [0, 100, 200]
        assert sim_inputs.qu_simparams.wind_times == [
            s + sim_inputs.quic_fire.time_now for s in sim_inputs.windsensors.wind_times
        ]
        # Test updating nonexistent sensors
        with pytest.raises(AttributeError):
            sim_inputs.new_wind_sensor(
                update="sensor3", wind_speeds=6, wind_directions=270
            )
        with pytest.raises(AttributeError):
            sim_inputs.new_wind_sensor(
                update="sensor 1", wind_speeds=6, wind_directions=270
            )
        # Test adding a new windsensor without putting in values for all the arguments
        with pytest.raises(TypeError):
            sim_inputs.new_wind_sensor(
                wind_speeds=[6, 6],
                wind_directions=[270, 350],
                wind_times=[0, 100],
                sensor_height=6.1,
            )

    def test_new_wind_sensor_from_csv(self):
        sim_inputs = self.get_test_object()
        # first test updating sensor1
        sim_inputs.new_wind_sensor_from_csv(
            update="sensor1", directory=TEST_DATA_DIR, filename="sample_raws_data.csv"
        )
        # now add a wind sensor with the same data
        sim_inputs.new_wind_sensor_from_csv(
            TEST_DATA_DIR,
            "sample_raws_data.csv",
            sensor_height=6.1,
            x_location=1,
            y_location=1,
        )
        assert (
            sim_inputs.windsensors.sensor1.wind_times
            == sim_inputs.windsensors.sensor2.wind_times
        )
        assert (
            sim_inputs.windsensors.sensor1.wind_speeds
            == sim_inputs.windsensors.sensor2.wind_speeds
        )
        assert (
            sim_inputs.windsensors.sensor1.wind_directions
            == sim_inputs.windsensors.sensor2.wind_directions
        )
        # Test incorrect data frame
        csv_path = TEST_DATA_DIR / "sample_raws_data.csv"
        raws = pd.read_csv(csv_path)
        missing_column_path = TMP_DIR / "raws_missing_column.csv"
        wrong_datatype_path = TMP_DIR / "raws_wrong_datatype.csv"
        north_is_360__path = TMP_DIR / "raws_north_is_360.csv"
        # when a column is missing
        raws.drop(["wind_times"], axis=1).to_csv(missing_column_path)
        # when a column has the wrong data type
        raws.astype({"wind_directions": "float64"}).to_csv(wrong_datatype_path)
        # when 360 deg is used for north
        raws["wind_directions"].apply(lambda x: 360 if x == 0 else x).to_csv(
            north_is_360__path
        )
        with pytest.raises(SchemaError):
            sim_inputs.new_wind_sensor_from_csv(
                TMP_DIR,
                "raws_missing_column.csv",
                sensor_height=6.1,
                x_location=1,
                y_location=1,
            )
        with pytest.raises(SchemaError):
            sim_inputs.new_wind_sensor_from_csv(
                TMP_DIR,
                "raws_wrong_datatype.csv",
                sensor_height=6.1,
                x_location=1,
                y_location=1,
            )
        with pytest.raises(SchemaError):
            sim_inputs.new_wind_sensor_from_csv(
                TMP_DIR,
                "raws_north_is_360.csv",
                sensor_height=6.1,
                x_location=1,
                y_location=1,
            )

    def test_write_inputs_latest(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR, version="latest")

    def test_write_inputs_invalid_version(self):
        sim_inputs = self.get_test_object()
        with pytest.raises(ValueError):
            sim_inputs.write_inputs(TMP_DIR, version="invalid_version")

    def test_update_shared_attributes(self):
        sim_inputs = self.get_test_object()
        # assign some new parameters directly to the input files
        sim_inputs.qu_simparams.nx = 100
        sim_inputs.qu_simparams.ny = 100
        sim_inputs.quic_fire.nz = 2
        sim_inputs.quic_fire.time_now = 12345667
        # first test that these changes are not reflected in the same attributes of other input files
        assert sim_inputs.qu_simparams.nx != sim_inputs.gridlist.n
        assert sim_inputs.qu_simparams.ny != sim_inputs.gridlist.m
        assert sim_inputs.quic_fire.nz != sim_inputs.gridlist.l
        assert sim_inputs.quic_fire.time_now != sim_inputs.qu_simparams.wind_times[0]
        # then write to a file and confirm that they are changed across input files at the write stage
        sim_inputs.write_inputs(TMP_DIR)
        assert sim_inputs.qu_simparams.nx == sim_inputs.gridlist.n
        assert sim_inputs.qu_simparams.ny == sim_inputs.gridlist.m
        assert sim_inputs.quic_fire.nz == sim_inputs.gridlist.l
        assert sim_inputs.quic_fire.time_now == sim_inputs.qu_simparams.wind_times[0]
        # now add a sensor with windshifts and see if those changes get reflected
        sim_inputs.new_wind_sensor(
            x_location=1,
            y_location=1,
            wind_speeds=[6, 6],
            wind_directions=[270, 350],
            wind_times=[0, 100],
            sensor_height=6.1,
        )
        # wind times should be updated in windsensors but not qu_simparams
        assert sim_inputs.windsensors.wind_times == [0, 100]
        assert len(sim_inputs.windsensors.wind_times) != len(
            sim_inputs.qu_simparams.wind_times
        )
        # until the write stage
        sim_inputs.write_inputs(TMP_DIR)
        assert sim_inputs.qu_simparams.wind_times == [
            s + sim_inputs.quic_fire.time_now for s in sim_inputs.windsensors.wind_times
        ]

    def test_from_directory(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR)
        sensor2 = TMP_DIR / "sensor2.inp"
        sensor3 = TMP_DIR / "sensor3.inp"
        for file in [sensor2, sensor3]:
            if file.exists():
                file.unlink()
        test_object = SimulationInputs.from_directory(TMP_DIR)
        assert isinstance(test_object, SimulationInputs)

        # Check that the inputs are the same
        compare_simulation_inputs(sim_inputs, test_object)

    def test_from_directory_optional_files(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR)

        # Remove optional files
        gridlist_path = TMP_DIR / "gridlist"
        gridlist_path.unlink()
        raster_origin_path = TMP_DIR / "rasterorigin.txt"
        raster_origin_path.unlink()

        test_object = SimulationInputs.from_directory(TMP_DIR)
        assert isinstance(test_object, SimulationInputs)
        compare_simulation_inputs(sim_inputs, test_object)

    def test_from_directory_missing_files(self):
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR)
        quic_fire_path = TMP_DIR / "QUIC_fire.inp"
        quic_fire_path.unlink()

        with pytest.raises(FileNotFoundError):
            SimulationInputs.from_directory(TMP_DIR)

    def test_to_dict(self):
        sim_inputs = self.get_test_object()
        sim_inputs.to_dict()

    def test_from_dict(self):
        sim_inputs = self.get_test_object()
        sim_dict = sim_inputs.to_dict()
        test_obj = SimulationInputs.from_dict(sim_dict)
        assert isinstance(test_obj, SimulationInputs)

        # Check that the inputs are the same
        compare_simulation_inputs(sim_inputs, test_obj)

    def test_to_json(self):
        sim_inputs = self.get_test_object()
        sim_inputs.to_json(TMP_DIR / "test.json")

    def test_from_json(self):
        sim_inputs = self.get_test_object()
        sim_inputs.to_json(TMP_DIR / "test.json")
        test_obj = SimulationInputs.from_json(TMP_DIR / "test.json")
        assert isinstance(test_obj, SimulationInputs)

        # Check that the inputs are the same
        compare_simulation_inputs(sim_inputs, test_obj)


class TestSamples:
    def test_canyon(self):
        canyon_sim = SimulationInputs.from_directory(SAMPLES_DIR / "Canyon")

        # Check topography
        assert isinstance(canyon_sim.qu_topoinputs.topography, CanyonTopo)
        assert canyon_sim.qu_topoinputs.topography.topo_flag == 4
        assert canyon_sim.qu_topoinputs.topography.x_location == 300
        assert canyon_sim.qu_topoinputs.topography.y_location == 150
        assert canyon_sim.qu_topoinputs.topography.canyon_std == 100
        assert canyon_sim.qu_topoinputs.topography.vertical_offset == 20

        # Check ignition
        assert isinstance(canyon_sim.quic_fire.ignition, RectangleIgnition)
        assert canyon_sim.quic_fire.ignition.ignition_flag == 1
        assert canyon_sim.quic_fire.ignition.x_min == 250
        assert canyon_sim.quic_fire.ignition.y_min == 225
        assert canyon_sim.quic_fire.ignition.x_length == 10
        assert canyon_sim.quic_fire.ignition.y_length == 100

        # Check I/O
        canyon_sim.write_inputs(TMP_DIR)
        canyon_sim.to_json(TMP_DIR / "sim.json")
        canyon_sim_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(canyon_sim, canyon_sim_from_json)

    def test_cos_hill(self):
        cos_hill = SimulationInputs.from_directory(SAMPLES_DIR / "CosHill")

        # Check topography
        assert isinstance(cos_hill.qu_topoinputs.topography, CosHillTopo)
        assert cos_hill.qu_topoinputs.topography.topo_flag == 8
        assert cos_hill.qu_topoinputs.topography.aspect == 100
        assert cos_hill.qu_topoinputs.topography.height == 10

        # Check I/O
        cos_hill.write_inputs(TMP_DIR)
        cos_hill.to_json(TMP_DIR / "sim.json")
        cos_hill_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(cos_hill, cos_hill_from_json)

    def test_eglin_canopy(self):
        eglin_canopy = SimulationInputs.from_directory(SAMPLES_DIR / "EglinCanopyTest")

        # Check topography
        assert isinstance(eglin_canopy.qu_topoinputs.topography, Topography)
        assert eglin_canopy.qu_topoinputs.topography.topo_flag == 0

        # Check fuel flags
        assert eglin_canopy.quic_fire.fuel_density_flag == 5
        assert eglin_canopy.quic_fire.fuel_moisture_flag == 5

        # Check ignition
        assert isinstance(eglin_canopy.quic_fire.ignition, Ignition)
        assert eglin_canopy.quic_fire.ignition.ignition_flag == 7

        # Check I/O
        eglin_canopy.write_inputs(TMP_DIR)
        eglin_canopy.to_json(TMP_DIR / "sim.json")
        eglin_canopy_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(eglin_canopy, eglin_canopy_from_json)

    def test_gauss_hill(self):
        gauss_hill = SimulationInputs.from_directory(SAMPLES_DIR / "GaussHill")

        # Check topography
        assert isinstance(gauss_hill.qu_topoinputs.topography, GaussianHillTopo)
        assert gauss_hill.qu_topoinputs.topography.topo_flag == 1
        assert gauss_hill.qu_topoinputs.topography.x_hilltop == 400
        assert gauss_hill.qu_topoinputs.topography.y_hilltop == 300
        assert gauss_hill.qu_topoinputs.topography.elevation_max == 100
        assert gauss_hill.qu_topoinputs.topography.elevation_std == 150

        # Check I/O
        gauss_hill.write_inputs(TMP_DIR)
        gauss_hill.to_json(TMP_DIR / "sim.json")
        gauss_hill_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(gauss_hill, gauss_hill_from_json)

    def test_hill_pass(self):
        hill_pass = SimulationInputs.from_directory(SAMPLES_DIR / "HillPass")

        # Check topography
        assert isinstance(hill_pass.qu_topoinputs.topography, HillPassTopo)
        assert hill_pass.qu_topoinputs.topography.topo_flag == 2
        assert hill_pass.qu_topoinputs.topography.max_height == 70
        assert hill_pass.qu_topoinputs.topography.location_param == 5

        # Check I/O
        hill_pass.write_inputs(TMP_DIR)
        hill_pass.to_json(TMP_DIR / "sim.json")
        hill_pass_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(hill_pass, hill_pass_from_json)

    def test_line_fire(self):
        line_fire = SimulationInputs.from_directory(SAMPLES_DIR / "LineFire")

        # Check parabolic quic grid
        assert line_fire.qu_simparams.stretch_grid_flag == 3
        assert line_fire.qu_simparams._dz_array[0] == 1
        assert line_fire.qu_simparams._dz_array[4] == 1
        assert line_fire.qu_simparams._dz_array[5] == 1.078399
        assert line_fire.qu_simparams._dz_array[6] == 1.313596
        assert line_fire.qu_simparams._dz_array[23] == 29.302056
        assert line_fire.qu_simparams._dz_array[24] == 32.354352

        # Check topography
        assert isinstance(line_fire.qu_topoinputs.topography, Topography)
        assert line_fire.qu_topoinputs.topography.topo_flag == 0

        # Check ignition
        assert isinstance(line_fire.quic_fire.ignition, SquareRingIgnition)
        assert line_fire.quic_fire.ignition.ignition_flag == 2
        assert line_fire.quic_fire.ignition.x_min == 50
        assert line_fire.quic_fire.ignition.y_min == 50
        assert line_fire.quic_fire.ignition.x_length == 200
        assert line_fire.quic_fire.ignition.y_length == 200
        assert line_fire.quic_fire.ignition.x_width == 10
        assert line_fire.quic_fire.ignition.y_width == 10

        # Check I/O
        line_fire.write_inputs(TMP_DIR)
        line_fire.to_json(TMP_DIR / "sim.json")
        line_fire_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(line_fire, line_fire_from_json)

    def test_sinusoid(self):
        sinusoid = SimulationInputs.from_directory(SAMPLES_DIR / "Sinusoid")

        # Check topography
        assert isinstance(sinusoid.qu_topoinputs.topography, SinusoidTopo)
        assert sinusoid.qu_topoinputs.topography.topo_flag == 7
        assert sinusoid.qu_topoinputs.topography.period == 20
        assert sinusoid.qu_topoinputs.topography.amplitude == 80

        # Check I/O
        sinusoid.write_inputs(TMP_DIR)
        sinusoid.to_json(TMP_DIR / "sim.json")
        sinusoid_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(sinusoid, sinusoid_from_json)

    def test_slope_mesa(self):
        slope_mesa = SimulationInputs.from_directory(SAMPLES_DIR / "SlopeMesa")

        # Check topography
        assert isinstance(slope_mesa.qu_topoinputs.topography, SlopeMesaTopo)
        assert slope_mesa.qu_topoinputs.topography.topo_flag == 3
        assert slope_mesa.qu_topoinputs.topography.slope_axis == 0
        assert slope_mesa.qu_topoinputs.topography.slope_value == 0.2
        assert slope_mesa.qu_topoinputs.topography.flat_fraction == 0.5

        # Check I/O
        slope_mesa.write_inputs(TMP_DIR)
        slope_mesa.to_json(TMP_DIR / "sim.json")
        slope_mesa_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(slope_mesa, slope_mesa_from_json)

    def test_transient_winds(self):
        transient_winds = SimulationInputs.from_directory(
            SAMPLES_DIR / "TransientWinds"
        )

        # Check wind sensors
        assert transient_winds.qu_simparams.wind_times == [1653321600, 1653321700]
        assert transient_winds.qu_metparams.num_sensors == 1
        assert len(transient_winds.windsensors) == 1
        assert isinstance(transient_winds.windsensors.sensor1, WindSensor)
        assert transient_winds.windsensors.sensor1.wind_times == [0, 100]
        assert transient_winds.windsensors.sensor1.wind_speeds == [6, 6]
        assert transient_winds.windsensors.sensor1.wind_directions == [270, 180]
        assert transient_winds.windsensors.sensor1.sensor_height == 10.0

        # Check I/O
        transient_winds.write_inputs(TMP_DIR)
        transient_winds.to_json(TMP_DIR / "sim.json")
        transient_winds_from_json = SimulationInputs.from_json(TMP_DIR / "sim.json")
        compare_simulation_inputs(transient_winds, transient_winds_from_json)


def compare_simulation_inputs(a: SimulationInputs, b: SimulationInputs):
    for a_input_file, b_input_file in zip(
        a._input_files_dict.values(), b._input_files_dict.values()
    ):
        if isinstance(a_input_file, QU_Simparams) or isinstance(
            b_input_file, QU_Simparams
        ):
            a_input_file._from_file_dz_array = None
            b_input_file._from_file_dz_array = None
        assert a_input_file == b_input_file
