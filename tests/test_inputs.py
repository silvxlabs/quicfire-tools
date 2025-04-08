"""
Test module for the data module of the quicfire_tools package.
"""

# Core imports
from __future__ import annotations
import shutil
from pathlib import Path

# Internal imports
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

# External imports
import pytest
import pandas as pd
from pydantic import ValidationError

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data" / "test-inputs"
SAMPLES_DIR = TEST_DIR / "data" / "samples"
TMP_DIR = TEST_DIR / "tmp"


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

    @pytest.mark.parametrize("flag", [-1, 0, 5, "1", 1.5])
    def test_invalid_output_data_file_format_flag(self, flag):
        with pytest.raises(ValidationError):
            QU_Fileoptions(output_data_file_format_flag=flag)  # noqa

    @pytest.mark.parametrize("flag", [-1, "1", 2])
    def test_invalid_non_mass_conserved_initial_field_flag(self, flag):
        with pytest.raises(ValidationError):
            QU_Fileoptions(non_mass_conserved_initial_field_flag=flag)  # noqa

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
        assert quic_fire._stretch_grid_input == "1.0"

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

    def test_dz_custom_dz_array(self):
        quic_fire = self.get_basic_test_object()
        quic_fire.nz = 3
        quic_fire.stretch_grid_flag = 1
        dz_write = [1.0, 2.0, 3.0]
        quic_fire.dz_array = dz_write
        quic_fire.to_file(TMP_DIR, version="v6")
        with open(TMP_DIR / "QUIC_fire.inp", "r") as file:
            lines = file.readlines()
        dz_array = []
        for i in range(14, 14 + len(quic_fire.dz_array)):
            dz_array.append(float(lines[i].strip().split("!")[0]))
        assert dz_array == dz_write
        assert quic_fire.dz_array == dz_array
        current_line = 14 + len(quic_fire.dz_array)
        assert lines[current_line] == "! FILE PATH\n"

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
            current_line = 14 + len(quic_fire.dz_array) + 1  # + 1 for quic_fire.dz
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

    @pytest.mark.parametrize("flag", [-1, 2, "1", 1.5])
    def test_invalid_convert_buildings_to_fuel_flag(self, flag):
        with pytest.raises(ValidationError):
            QFire_Bldg_Advanced_User_Inputs(convert_buildings_to_fuel_flag=flag)  # noqa

    @pytest.mark.parametrize("flag", [-1, ""])
    def test_invalid_building_fuel_density(self, flag):
        with pytest.raises(ValidationError):
            QFire_Bldg_Advanced_User_Inputs(building_fuel_density=flag)

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
    @pytest.fixture(scope="class")
    def qu_metparams(self):
        """Fixture that returns a basic QU_metparams object with a single sensor"""
        return QU_metparams(site_names=["sensor1"], file_names=["sensor1.inp"])

    def test_init(self, qu_metparams):
        # Test basic initialization
        assert qu_metparams.num_sensors == 1
        assert qu_metparams.site_names == ["sensor1"]
        assert qu_metparams.file_names == ["sensor1.inp"]

    def test_multiple_sensors(self):
        # Test initialization with multiple sensors
        qu_metparams = QU_metparams(
            site_names=["sensor1", "sensor2", "sensor3"],
            file_names=["sensor1.inp", "sensor2.inp", "sensor3.inp"],
        )
        assert qu_metparams.num_sensors == 3
        assert len(qu_metparams.site_names) == 3
        assert len(qu_metparams.file_names) == 3

    def test_validation_error(self):
        # Test that validation catches mismatched lengths
        with pytest.raises(
            ValueError, match="site_names and file_names must be the same length"
        ):
            QU_metparams(site_names=["sensor1", "sensor2"], file_names=["sensor1.inp"])

    def test_min_length_validation(self):
        # Test that empty lists are not allowed
        with pytest.raises(ValidationError):
            QU_metparams(site_names=[], file_names=[])

    def test_to_dict(self, qu_metparams):
        result_dict = qu_metparams.to_dict()
        assert "site_names" in result_dict
        assert "file_names" in result_dict
        assert result_dict["site_names"] == ["sensor1"]
        assert result_dict["file_names"] == ["sensor1.inp"]

    def test_from_dict(self, qu_metparams):
        result_dict = qu_metparams.to_dict()
        test_obj = QU_metparams.from_dict(result_dict)
        assert test_obj == qu_metparams

    def test_to_file(self, qu_metparams, tmp_path):
        qu_metparams.to_file(tmp_path)

        # Read the content of the file and check for correctness
        with open(tmp_path / "QU_metparams.inp", "r") as file:
            lines = file.readlines()

        # Check number of sensors line
        assert int(lines[2].strip().split("!")[0]) == qu_metparams.num_sensors

        # Check sensor name and file name
        assert "sensor1" in lines[4]  # Site name line
        assert "sensor1.inp" in lines[6]  # File name line

    def test_to_file_multiple_sensors(self, tmp_path):
        qu_metparams = QU_metparams(
            site_names=["sensor1", "sensor2", "sensor3"],
            file_names=["sensor1.inp", "sensor2.inp", "sensor3.inp"],
        )
        qu_metparams.to_file(tmp_path)

        with open(tmp_path / "QU_metparams.inp", "r") as file:
            content = file.read()

        # Check that all sensor names and file names are present
        for i in range(3):
            assert f"sensor{i + 1}" in content
            assert f"sensor{i + 1}.inp" in content

    def test_from_file(self, qu_metparams, tmp_path):
        qu_metparams.to_file(tmp_path)
        loaded = QU_metparams.from_file(tmp_path)

        assert isinstance(loaded, QU_metparams)
        assert loaded.site_names == qu_metparams.site_names
        assert loaded.file_names == qu_metparams.file_names
        assert loaded.num_sensors == qu_metparams.num_sensors

    def test_from_file_multiple_sensors(self, tmp_path):
        """Test that from_file correctly reads multiple sensors"""
        qu_metparams = QU_metparams(
            site_names=["sensor1", "sensor2", "sensor3"],
            file_names=["sensor1.inp", "sensor2.inp", "sensor3.inp"],
        )
        qu_metparams.to_file(tmp_path)

        result = QU_metparams.from_file(tmp_path)
        assert result.num_sensors == 3
        assert result.site_names == ["sensor1", "sensor2", "sensor3"]
        assert result.file_names == ["sensor1.inp", "sensor2.inp", "sensor3.inp"]

    def test_from_file_malformed(self, tmp_path):
        """Test that from_file handles malformed files appropriately"""
        # Create a malformed file
        content = """!QUIC 6.26
    0 !Met input flag (0=QUIC,1=WRF,2=ITT MM5,3=HOTMAC)
    1 !Number of measuring sites"""

        with open(tmp_path / "QU_metparams.inp", "w") as f:
            f.write(content)

        with pytest.raises(ValueError, match="Error parsing"):
            QU_metparams.from_file(tmp_path)

    def test_sensor_lines_property(self, qu_metparams):
        expected_lines = "sensor1 !Site Name\n!File name\nsensor1.inp"
        assert qu_metparams._sensor_lines == expected_lines


class TestWindSensorArray:
    @pytest.fixture
    def single_value_sensor(self):
        return WindSensor(
            name="sensor1",
            wind_times=0,
            wind_speeds=4.5,
            wind_directions=270,
        )

    @pytest.fixture
    def multi_value_sensor(self):
        return WindSensor(
            name="sensor1",
            wind_times=[0, 60, 120],
            wind_speeds=[4.5, 5.5, 6.5],
            wind_directions=[270, 330, 45],
        )

    @pytest.fixture
    def sensor_array_one_sensor(self, single_value_sensor):
        return WindSensorArray(sensor_array=[single_value_sensor])

    @pytest.fixture
    def sensor_array_multiple_sensors(self, single_value_sensor, multi_value_sensor):
        sensor1 = single_value_sensor.model_copy()
        sensor2 = single_value_sensor.model_copy()
        sensor2.name = "sensor2"
        sensor2.wind_times = 30
        sensor2.x_location, sensor2.y_location = 10, 10
        sensor3 = multi_value_sensor.model_copy()
        sensor3.name = "sensor3"
        sensor3.x_location, sensor3.y_location = 20, 20
        return WindSensorArray(sensor_array=[sensor1, sensor2, sensor3])

    @pytest.mark.parametrize(
        "num_sensors",
        [0, 1, 2, 3, 10],
        ids=["no_sensors", "1_sensor", "2_sensors", "3_sensors", "10_sensors"],
    )
    def test_init(self, num_sensors, single_value_sensor):
        sensor_list = []
        for i in range(num_sensors):
            sensor = single_value_sensor.model_copy()
            sensor.name = f"sensor{i + 1}"  # Ensure unique names
            sensor_list.append(sensor)
        wind_array = WindSensorArray(sensor_array=sensor_list)
        assert len(wind_array) == num_sensors
        for sensor1, sensor2 in zip(wind_array, sensor_list):
            assert sensor1 == sensor2

    @pytest.mark.parametrize(
        "sensor_array_fixture",
        ["sensor_array_one_sensor", "sensor_array_multiple_sensors"],
    )
    def test_wind_times(self, sensor_array_fixture, request):
        sensor_array = request.getfixturevalue(sensor_array_fixture)

        # Assert that wind_times is sorted
        assert all(
            sensor_array.wind_times[i] <= sensor_array.wind_times[i + 1]
            for i in range(len(sensor_array.wind_times) - 1)
        )

        # Assert that each time belonging to a sensor is in the wind_times array
        for sensor in sensor_array:
            for time in sensor.wind_times:
                assert time in sensor_array.wind_times

    @pytest.mark.parametrize(
        "sensor_array_fixture",
        ["sensor_array_one_sensor", "sensor_array_multiple_sensors"],
    )
    def test_to_dict(self, sensor_array_fixture, request):
        sensor_array = request.getfixturevalue(sensor_array_fixture)
        result_dict = sensor_array.to_dict()

        assert "sensor_array" in result_dict
        assert len(result_dict["sensor_array"]) == len(sensor_array)
        for sensor1, sensor2 in zip(sensor_array, result_dict["sensor_array"]):
            assert sensor1.to_dict() == sensor2

    @pytest.mark.parametrize(
        "sensor_array_fixture",
        ["sensor_array_one_sensor", "sensor_array_multiple_sensors"],
    )
    def test_from_dict(self, sensor_array_fixture, request):
        sensor_array = request.getfixturevalue(sensor_array_fixture)
        result_dict = sensor_array.to_dict()
        test_obj = WindSensorArray.from_dict(result_dict)
        assert test_obj is not sensor_array
        assert test_obj == sensor_array

    @pytest.mark.parametrize(
        "sensor_array_fixture",
        ["sensor_array_one_sensor", "sensor_array_multiple_sensors"],
    )
    def test_to_file(self, sensor_array_fixture, request):
        sensor_array = request.getfixturevalue(sensor_array_fixture)
        sensor_array.to_file(TMP_DIR)

        tmp_dir_file_names = [file.name for file in TMP_DIR.iterdir()]
        for sensor in sensor_array:
            assert sensor._filename in tmp_dir_file_names
            assert WindSensor.from_file(TMP_DIR, sensor.name) == sensor

    @pytest.mark.parametrize(
        "sensor_array_fixture",
        ["sensor_array_one_sensor", "sensor_array_multiple_sensors"],
    )
    def test_from_file(self, sensor_array_fixture, request):
        sensor_array = request.getfixturevalue(sensor_array_fixture)

        # First create a QU_metparams object and write it to a file
        site_names = [sensor.name for sensor in sensor_array]
        file_names = [sensor._filename for sensor in sensor_array]
        QU_metparams(site_names=site_names, file_names=file_names).to_file(TMP_DIR)

        sensor_array.to_file(TMP_DIR)
        test_obj = WindSensorArray.from_file(TMP_DIR)
        assert test_obj == sensor_array
        assert test_obj is not sensor_array


class TestWindSensor:
    @pytest.fixture
    def single_value_sensor(self):
        return WindSensor(
            name="sensor1",
            wind_times=0,
            wind_speeds=4.5,
            wind_directions=270,
        )

    @pytest.fixture
    def single_value_sensor_nonzero_start_time(self):
        return WindSensor(
            name="sensor1",
            wind_times=60,
            wind_speeds=4.5,
            wind_directions=270,
        )

    @pytest.fixture
    def multi_value_sensor(self):
        return WindSensor(
            name="sensor1",
            wind_times=[0, 60, 120],
            wind_speeds=[4.5, 5.5, 6.5],
            wind_directions=[270, 330, 45],
        )

    @pytest.mark.parametrize(
        "wind_times, wind_speeds, wind_directions",
        [
            [0, 4.5, 270],
            [[0], [4.5], [270]],
            [[0, 60, 120], [4.5, 5.5, 6.5], [270, 330, 45]],
        ],
        ids=["single_values", "list_of_values", "multiple_values"],
    )
    def test_init(self, wind_times, wind_speeds, wind_directions):
        sensor = WindSensor(
            name="sensor1",
            wind_times=wind_times,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
        )

        # Test attribute assignment
        assert sensor.name == "sensor1"
        if isinstance(wind_times, int):
            wind_times = [wind_times]
        assert sensor.wind_times == wind_times
        if isinstance(wind_speeds, (int, float)):
            wind_speeds = [wind_speeds]
        assert sensor.wind_speeds == wind_speeds
        if isinstance(wind_directions, int):
            wind_directions = [wind_directions]
        assert sensor.wind_directions == wind_directions
        assert sensor.sensor_heights == [6.1 for _ in range(len(wind_times))]
        assert sensor.x_location == 1
        assert sensor.y_location == 1

        # Assert computed field (property) attributes
        assert sensor._filename == "sensor1.inp"

    @pytest.mark.parametrize(
        "wind_times, wind_speeds, wind_directions",
        [
            [[0, 60], [1.8, 4.5, 2.5], [270, 330, 45]],
            [[0, 60, 270], [1.8, 4.5], [270, 330, 45]],
            [[0, 60, 270], [1.8, 4.5, 2.5], [270, 330]],
        ],
        ids=["mismatched_speeds", "mismatched_directions", "mismatched_times"],
    )
    def test_init_invalid_list_length(self, wind_times, wind_speeds, wind_directions):
        with pytest.raises(ValidationError):
            WindSensor(
                name="sensor1",
                wind_times=wind_times,
                wind_speeds=wind_speeds,
                wind_directions=wind_directions,
            )

    def test_init_invalid_name(self):
        with pytest.raises(ValidationError):
            WindSensor(
                name="",
                wind_times=0,
                wind_speeds=4.5,
                wind_directions=270,
            )

    def test_init_reserved_name(self):
        with pytest.raises(ValidationError):
            WindSensor(
                name="QU_simparams",
                wind_times=0,
                wind_speeds=4.5,
                wind_directions=270,
            )

    @pytest.mark.parametrize(
        "sensor_fixture",
        ["single_value_sensor", "multi_value_sensor"],
    )
    def test_updates(self, sensor_fixture, request):
        sensor = request.getfixturevalue(sensor_fixture)

        wind_times_len_before = len(sensor.wind_times)
        wind_speeds_len_before = len(sensor.wind_speeds)
        wind_directions_len_before = len(sensor.wind_directions)

        sensor.wind_times.append(300)
        sensor.wind_speeds.append(7.5)
        sensor.wind_directions.append(90)

        assert len(sensor.wind_times) == wind_times_len_before + 1
        assert len(sensor.wind_speeds) == wind_speeds_len_before + 1
        assert len(sensor.wind_directions) == wind_directions_len_before + 1

    @pytest.mark.parametrize(
        "sensor_fixture",
        ["single_value_sensor", "multi_value_sensor"],
    )
    def test_update_sensor_heights(self, sensor_fixture, request):
        sensor = request.getfixturevalue(sensor_fixture)
        sensor.sensor_heights = 2.0
        assert sensor.sensor_heights == [2.0 for _ in range(len(sensor.wind_times))]

    @pytest.mark.parametrize(
        "version",
        ["latest", "v6", "v5"],
    )
    @pytest.mark.parametrize(
        "sensor_fixture",
        ["single_value_sensor", "multi_value_sensor"],
    )
    def test_to_file(self, sensor_fixture, version, request):
        sensor = request.getfixturevalue(sensor_fixture)
        sensor.to_file(TMP_DIR, version)

        # Verify the file was created
        assert (TMP_DIR / "sensor1.inp").exists()
        with open(TMP_DIR / "sensor1.inp", "r") as file:
            lines = file.readlines()

        # Verify the contents of the file
        assert str(lines[0].strip().split("!")[0].strip()) == sensor.name
        assert float(lines[4].strip().split("!")[0].strip()) == sensor.x_location
        assert float(lines[5].strip().split("!")[0].strip()) == sensor.y_location

        # Check wind lines across time steps
        for i in range(len(sensor.wind_times)):
            file_index = 6 + i * 6

            # Check beginning time
            assert float(lines[file_index].split("!")[0]) == sensor.wind_times[i]

            # Check sensor measurements
            assert (
                float(lines[file_index + 5].split(" ")[0]) == sensor.sensor_heights[i]
            )
            assert float(lines[file_index + 5].split(" ")[1]) == sensor.wind_speeds[i]
            assert (
                float(lines[file_index + 5].split(" ")[2]) == sensor.wind_directions[i]
            )

    @pytest.mark.parametrize(
        "version",
        ["latest", "v6", "v5"],
    )
    @pytest.mark.parametrize(
        "sensor_fixture",
        [
            "single_value_sensor",
            "single_value_sensor_nonzero_start_time",
            "multi_value_sensor",
        ],
    )
    def test_from_file(self, sensor_fixture, version, request):
        test_sensor = request.getfixturevalue(sensor_fixture)
        test_sensor.to_file(TMP_DIR, version)

        new_sensor = WindSensor.from_file(TMP_DIR, "sensor1")
        assert isinstance(new_sensor, WindSensor)
        assert new_sensor is not test_sensor
        assert new_sensor == test_sensor

    @pytest.mark.parametrize(
        "sensor_fixture",
        [
            "single_value_sensor",
            "single_value_sensor_nonzero_start_time",
            "multi_value_sensor",
        ],
    )
    def test_to_dict(self, sensor_fixture, request):
        test_sensor = request.getfixturevalue(sensor_fixture)
        result_dict = test_sensor.to_dict()

        assert result_dict["name"] == test_sensor.name
        assert result_dict["wind_times"] == test_sensor.wind_times
        assert result_dict["wind_speeds"] == test_sensor.wind_speeds
        assert result_dict["wind_directions"] == test_sensor.wind_directions
        assert result_dict["sensor_heights"] == test_sensor.sensor_heights
        assert result_dict["x_location"] == test_sensor.x_location
        assert result_dict["y_location"] == test_sensor.y_location
        assert result_dict["sensor_heights"] == test_sensor.sensor_heights

    @pytest.mark.parametrize(
        "sensor_fixture",
        [
            "single_value_sensor",
            "single_value_sensor_nonzero_start_time",
            "multi_value_sensor",
        ],
    )
    def test_from_dict(self, sensor_fixture, request):
        test_sensor = request.getfixturevalue(sensor_fixture)
        result_dict = test_sensor.to_dict()

        new_sensor = WindSensor.from_dict(result_dict)
        assert isinstance(new_sensor, WindSensor)
        assert new_sensor is not test_sensor
        assert new_sensor == test_sensor

    def test_from_csv_defaults(self):
        wind_data = pd.read_csv(TEST_DATA_DIR / "sample_raws_data.csv")
        test_sensor = WindSensor.from_dataframe(
            df=wind_data,
            name="sensor_from_csv",
            x_location=50,
            y_location=50,
            sensor_height=6.1,
        )

        # Check that the sensor was created correctly
        assert test_sensor.name == "sensor_from_csv"
        assert test_sensor.x_location == 50
        assert test_sensor.y_location == 50
        assert test_sensor.sensor_heights == [6.1 for _ in range(len(wind_data))]
        assert test_sensor.wind_times == wind_data["wind_times"].tolist()
        assert test_sensor.wind_speeds == wind_data["wind_speeds"].tolist()
        assert test_sensor.wind_directions == wind_data["wind_directions"].tolist()

    def test_from_csv_custom_column_names(self):
        wind_data = pd.read_csv(TEST_DATA_DIR / "sample_raws_data.csv")
        wind_data["WindDir_D1_WVT"] = wind_data["WindDir_D1_WVT"].astype(int)
        test_sensor = WindSensor.from_dataframe(
            df=wind_data,
            name="sensor_from_csv",
            x_location=50,
            y_location=50,
            sensor_height=6.1,
            time_column_name="UNIX_INT",
            speed_column_name="WS_ms_Max",
            direction_column_name="WindDir_D1_WVT",
        )

        # Check that the sensor was created correctly
        assert test_sensor.name == "sensor_from_csv"
        assert test_sensor.x_location == 50
        assert test_sensor.y_location == 50
        assert test_sensor.sensor_heights == [6.1 for _ in range(len(wind_data))]
        assert test_sensor.wind_times == wind_data["UNIX_INT"].tolist()
        assert test_sensor.wind_speeds == wind_data["WS_ms_Max"].tolist()
        assert test_sensor.wind_directions == wind_data["WindDir_D1_WVT"].tolist()


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
        assert isinstance(sim_inputs.wind_sensors, WindSensorArray)

        assert sim_inputs.quic_fire.nz == 1
        assert sim_inputs.quic_fire.sim_time == 65
        assert sim_inputs.qu_simparams.nx == 150
        assert sim_inputs.qu_simparams.ny == 150

        assert sim_inputs.qu_simparams.wind_times[0] == sim_inputs.quic_fire.time_now
        assert sim_inputs.wind_sensors[0].wind_times == [sim_inputs.quic_fire.time_now]
        assert sim_inputs.wind_sensors[0].wind_speeds == [5.0]
        assert sim_inputs.wind_sensors[0].wind_directions == [90]

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

    def test_add_wind_sensor(self):
        """
        This test adds a wind sensor to the simulation with default arguments.
        Want to test that the sensor is correctly added to the wind sensor array
        and that the qu_simparams win_times is updated correctly.

        By default, the times_relative_to_simulation_start and
        enforce_300s_wind_updates arguments are set to False.
        """
        sim_inputs = self.get_test_object()
        sim_time = sim_inputs.quic_fire.time_now
        original_wind_times = sim_inputs.qu_simparams.wind_times
        assert len(sim_inputs.wind_sensors) == 1
        sim_inputs.add_wind_sensor(
            wind_speeds=1,
            wind_directions=2,
            wind_times=100,
            sensor_height=6,
        )

        # Check that the sensor was added correctly
        assert len(sim_inputs.wind_sensors) == 2
        new_sensor = sim_inputs.wind_sensors[1]
        assert new_sensor.wind_speeds == [1]
        assert new_sensor.wind_directions == [2]
        assert new_sensor.wind_times == [sim_time + 100]
        assert new_sensor.sensor_heights == [6]

        # Check that the qu_simparams wind times did not update since update time
        # is 300s by default and we added a sensor that updates after 100s
        assert sim_inputs.qu_simparams.wind_times == original_wind_times

        # Add a new sensor that updates 300s after sim start time
        sim_inputs.add_wind_sensor(
            wind_speeds=1,
            wind_directions=2,
            wind_times=300,
            sensor_height=6,
        )

        # qu_simparams wind times should have updated
        assert sim_inputs.qu_simparams.wind_times == [sim_time, sim_time + 300]

        # Add a wind sensor and set update frequency to 100
        sim_inputs.add_wind_sensor(
            wind_speeds=1,
            wind_directions=2,
            wind_times=400,
            sensor_height=6,
            wind_update_frequency=100,
        )

        # There should now be 4 wind time updates
        assert sim_inputs.qu_simparams.wind_times == [
            sim_time,
            sim_time + 100,
            sim_time + 300,
            sim_time + 400,
        ]

    def test_add_wind_sensor_from_datafrane(self):
        sim_inputs = self.get_test_object()
        start_time = sim_inputs.quic_fire.time_now
        df = pd.read_csv(TEST_DATA_DIR / "sample_raws_data.csv")
        sim_inputs.add_wind_sensor_from_dataframe(
            df, x_location=0, y_location=0, sensor_height=2
        )
        assert len(sim_inputs.qu_simparams.wind_times) == 893
        assert sim_inputs.qu_simparams.wind_times[0] == start_time
        for i in range(len(sim_inputs.qu_simparams.wind_times) - 1):
            assert (
                sim_inputs.qu_simparams.wind_times[i + 1]
                - sim_inputs.qu_simparams.wind_times[i]
                >= 300
            )

    def test_remove_wind_sensor(self):
        sim_inputs = self.get_test_object()
        sim_inputs.remove_wind_sensor("sensor1")
        assert len(sim_inputs.wind_sensors) == 0

    @pytest.mark.parametrize("version", ["v5", "v6", "latest"])
    def test_write_inputs(self, version):
        shutil.rmtree(TMP_DIR)
        sim_inputs = self.get_test_object()
        sim_inputs.write_inputs(TMP_DIR, version=version)

        # Assert that the expected default files are present in the directory
        sim_files = [file_name.name for file_name in Path(TMP_DIR).iterdir()]
        for sim_object in sim_inputs._input_files_dict.values():
            if isinstance(sim_object, WindSensorArray):
                continue
            file_name = sim_object.name + sim_object._extension
            assert file_name in sim_files

        # Assert that the custom fuel files were not written
        assert "gridlist" not in sim_files
        assert "rasterorigin.txt" not in sim_files

    def test_write_inputs_custom_fuels(self):
        sim_inputs = self.get_test_object()
        sim_inputs.set_custom_simulation()
        sim_inputs.write_inputs(TMP_DIR, version="latest")
        sim_files = [file_name.name for file_name in Path(TMP_DIR).iterdir()]
        assert "gridlist" in sim_files
        assert "rasterorigin.txt" in sim_files

    def test_write_inputs_invalid_version(self):
        sim_inputs = self.get_test_object()
        with pytest.raises(ValueError):
            sim_inputs.write_inputs(TMP_DIR, version="invalid_version")

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
        assert len(transient_winds.wind_sensors) == 1
        assert transient_winds.wind_sensors[0].wind_times == [1653321600, 1653321700]
        assert transient_winds.wind_sensors[0].wind_speeds == [6, 6]
        assert transient_winds.wind_sensors[0].wind_directions == [270, 180]
        assert transient_winds.wind_sensors[0].sensor_heights == [10.0, 10.0]

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
