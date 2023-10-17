"""
Test module for the inputs module of the quicfire_tools package.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from quicfire_tools.inputs import *

# Create a tmp/ directory to store the temporary test files
Path("tmp/").mkdir(exist_ok=True)


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
        assert result_dict['n'] == 10
        assert result_dict['m'] == 10
        assert result_dict['l'] == 10
        assert result_dict['dx'] == 1
        assert result_dict['dy'] == 1
        assert result_dict['dz'] == 1
        assert '_validate_inputs' not in result_dict

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
        assert result_dict['utm_x'] == raster_origin.utm_x
        assert result_dict['utm_y'] == raster_origin.utm_y

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
        raster_origin.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/rasterorigin.txt", 'r') as file:
            lines = file.readlines()
            assert float(lines[0].strip()) == raster_origin.utm_x
            assert float(lines[1].strip()) == raster_origin.utm_y

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            raster_origin.to_file("/non_existent_path/rasterorigin.txt")

    def test_from_file(self):
        """Test initializing a class from a rasterorigin.txt file."""
        raster_origin = RasterOrigin()
        raster_origin.to_file("tmp/")
        test_object = RasterOrigin.from_file("tmp/")
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
        qu_buildings = QU_Buildings(wall_roughness_length=1.0,
                                    number_of_buildings=0,
                                    number_of_polygon_nodes=0)
        assert qu_buildings.wall_roughness_length == 1.0
        assert qu_buildings.number_of_buildings == 0
        assert qu_buildings.number_of_polygon_nodes == 0

        # Test data type casting
        qu_buildings = QU_Buildings(wall_roughness_length="1.0",
                                    number_of_buildings=1.0)
        assert isinstance(qu_buildings.wall_roughness_length, float)
        assert qu_buildings.wall_roughness_length == 1.0
        assert isinstance(qu_buildings.number_of_buildings, int)
        assert qu_buildings.number_of_buildings == 1

        # Pass bad parameters
        with pytest.raises(ValidationError):
            QU_Buildings(wall_roughness_length=-1, number_of_buildings=0,
                         number_of_polygon_nodes=0)
        with pytest.raises(ValidationError):
            QU_Buildings(wall_roughness_length=1, number_of_buildings=-1,
                         number_of_polygon_nodes=0)
        with pytest.raises(ValidationError):
            QU_Buildings(wall_roughness_length=0)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_buildings = QU_Buildings()
        result_dict = qu_buildings.to_dict()
        assert result_dict[
                   'wall_roughness_length'] == qu_buildings.wall_roughness_length
        assert result_dict[
                   'number_of_buildings'] == qu_buildings.number_of_buildings
        assert result_dict[
                   'number_of_polygon_nodes'] == qu_buildings.number_of_polygon_nodes

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
        qu_buildings.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/QU_buildings.inp", 'r') as file:
            lines = file.readlines()
            assert float(lines[1].strip().split("\t")[
                             0]) == qu_buildings.wall_roughness_length
            assert int(lines[2].strip().split("\t")[
                           0]) == qu_buildings.number_of_buildings
            assert int(lines[3].strip().split("\t")[
                           0]) == qu_buildings.number_of_polygon_nodes

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qu_buildings.to_file("/non_existent_path/QU_buildings.inp")

    def test_from_file(self):
        """Test initializing a class from a QU_buildings.inp file."""
        qu_buildings = QU_Buildings()
        qu_buildings.to_file("tmp/")
        test_object = QU_Buildings.from_file("tmp/")
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
        for invalid_flag in [-1, 0, 5, "1", 1., 1.5]:
            with pytest.raises(ValidationError):
                QU_Fileoptions(output_data_file_format_flag=invalid_flag)

        # Test invalid non_mass_conserved_initial_field_flag flag
        for invalid_flag in [-1, 0., "1", 2]:
            with pytest.raises(ValidationError):
                QU_Fileoptions(
                    non_mass_conserved_initial_field_flag=invalid_flag)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_fileoptions = QU_Fileoptions()
        result_dict = qu_fileoptions.to_dict()
        assert result_dict[
                   'output_data_file_format_flag'] == qu_fileoptions.output_data_file_format_flag
        assert result_dict[
                   'non_mass_conserved_initial_field_flag'] == qu_fileoptions.non_mass_conserved_initial_field_flag
        assert result_dict[
                   'initial_sensor_velocity_field_flag'] == qu_fileoptions.initial_sensor_velocity_field_flag
        assert result_dict[
                   'qu_staggered_velocity_file_flag'] == qu_fileoptions.qu_staggered_velocity_file_flag
        assert result_dict[
                   'generate_wind_startup_files_flag'] == qu_fileoptions.generate_wind_startup_files_flag

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
        qu_fileoptions.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/QU_fileoptions.inp", 'r') as file:
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
        qu_fileoptions.to_file("tmp/")
        test_object = QU_Fileoptions.from_file("tmp/")
        assert isinstance(test_object, QU_Fileoptions)
        assert qu_fileoptions == test_object


class TestQU_Simparams:
    @staticmethod
    def get_test_object():
        return QU_Simparams(nx=100, ny=100, nz=26, dx=2., dy=2,
                            quic_domain_height=250)

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
        assert len(qu_simparams.dz_array) == 26
        assert len(qu_simparams.vertical_grid_lines.split("\n")) == 29

        # Test changing the default values
        qu_simparams.nx = 150
        assert qu_simparams.nx == 150

        # Test property setters
        qu_simparams.nz = 30
        assert qu_simparams.nz == 30
        assert len(qu_simparams.dz_array) == 30
        assert len(qu_simparams.vertical_grid_lines.split("\n")) == 33

        # Test data type casting
        qu_simparams = QU_Simparams(nx="100", ny=100, nz=26, dx=2, dy=2,
                                    quic_domain_height=5)
        assert isinstance(qu_simparams.nx, int)
        assert qu_simparams.nx == 100

        # Test with custom dz_array
        # TODO: Come back to this tests
        # qu_simparams = QU_Simparams(nx=100, ny=100, nz=26, dx=2, dy=2,
        #                             custom_dz_array=[1] * 26,
        #                             quic_domain_height=250)
        # assert qu_simparams.dz_array == [
        #     qu_simparams.surface_vertical_cell_size] * 26

        # Test invalid stretch_grid_flags
        for invalid_flag in [-1, 4, "1", 1., 1.5, 2]:
            with pytest.raises(ValidationError):
                QU_Simparams(stretch_grid_flag=invalid_flag)

    def test_dz_array(self):
        # Test with stretch_grid_flag = 0
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        assert qu_simparams.dz_array == [
            qu_simparams.surface_vertical_cell_size] * qu_simparams.nz

        # Test with stretch_grid_flag = 1
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.custom_dz_array = [0.5] * qu_simparams.nz
        assert qu_simparams.dz_array == [0.5] * qu_simparams.nz

        # Test with stretch_grid_flag = 3
        qu_simparams = self.get_test_object()
        assert len(qu_simparams.dz_array) == qu_simparams.nz

    def test_stretch_grid_flag_0(self):
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        vertical_grid_lines = qu_simparams._stretch_grid_flag_0()
        with open("data/test-templates/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_stretch_grid_flag_1(self):
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 1

        # Test with no dz_array input
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with 19 custom_dz_array inputs
        qu_simparams.custom_dz_array = [1] * (qu_simparams.nz - 1)
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with dz inputs that don't match the surface values
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        qu_simparams.custom_dz_array[0] = 2
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test valid case
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        vertical_grid_lines = qu_simparams._stretch_grid_flag_1()
        with open("data/test-templates/stretchgrid_1.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_stretch_grid_flag_3(self):
        qu_simparams = self.get_test_object()
        vertical_grid_lines = qu_simparams._stretch_grid_flag_3()
        with open("data/test-templates/stretchgrid_3.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_generate_vertical_grid(self):
        qu_simparams = self.get_test_object()

        # Test stretch_grid_flag = 0
        qu_simparams.stretch_grid_flag = 0
        with open("data/test-templates/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams.vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 1
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        with open("data/test-templates/stretchgrid_1.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams.vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 3
        qu_simparams.stretch_grid_flag = 3
        with open("data/test-templates/stretchgrid_3.txt") as f:
            expected_lines = f.readlines()
        assert qu_simparams.vertical_grid_lines == "".join(expected_lines)

    def test_generate_wind_times(self):
        # Test valid wind_step_times
        qu_simparams = self.get_test_object()
        qu_simparams.wind_times = [0]
        wind_times_lines = qu_simparams._generate_wind_time_lines()
        with open("data/test-templates/wind_times.txt") as f:
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
        assert result_dict['nx'] == qu_simparams.nx
        assert result_dict['ny'] == qu_simparams.ny
        assert result_dict['nz'] == qu_simparams.nz
        assert result_dict['dx'] == qu_simparams.dx
        assert result_dict['dy'] == qu_simparams.dy
        assert result_dict[
                   'surface_vertical_cell_size'] == qu_simparams.surface_vertical_cell_size
        assert result_dict[
                   'number_surface_cells'] == qu_simparams.number_surface_cells

        # Test the default parameters
        assert result_dict[
                   'surface_vertical_cell_size'] == qu_simparams.surface_vertical_cell_size
        assert result_dict[
                   'number_surface_cells'] == qu_simparams.number_surface_cells
        assert result_dict[
                   'stretch_grid_flag'] == qu_simparams.stretch_grid_flag
        assert result_dict['dz_array'] == qu_simparams.dz_array
        assert result_dict['utc_offset'] == qu_simparams.utc_offset
        assert result_dict['wind_times'] == qu_simparams.wind_times
        assert result_dict['sor_iter_max'] == qu_simparams.sor_iter_max
        assert result_dict[
                   'sor_residual_reduction'] == qu_simparams.sor_residual_reduction
        assert result_dict[
                   'use_diffusion_flag'] == qu_simparams.use_diffusion_flag
        assert result_dict[
                   'number_diffusion_iterations'] == qu_simparams.number_diffusion_iterations
        assert result_dict['domain_rotation'] == qu_simparams.domain_rotation
        assert result_dict['utm_x'] == qu_simparams.utm_x
        assert result_dict['utm_y'] == qu_simparams.utm_y
        assert result_dict['utm_zone_number'] == qu_simparams.utm_zone_number
        assert result_dict['utm_zone_letter'] == qu_simparams.utm_zone_letter
        assert result_dict['quic_cfd_flag'] == qu_simparams.quic_cfd_flag
        assert result_dict[
                   'explosive_bldg_flag'] == qu_simparams.explosive_bldg_flag
        assert result_dict['bldg_array_flag'] == qu_simparams.bldg_array_flag

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
            if key in ["vertical_grid_lines", "wind_time_lines",
                       "custom_dz_array"]:
                continue
            assert key in result_docs
        for key in result_docs:
            assert key in result_dict

    def test_to_file(self):
        """
        Test the to_file method of a QU_Simparams object.
        """
        qu_simparams = self.get_test_object()
        qu_simparams.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/QU_simparams.inp", 'r') as file:
            lines = file.readlines()

        # Check nx, ny, nz, dx, dy
        assert int(lines[1].strip().split("!")[0]) == qu_simparams.nx
        assert int(lines[2].strip().split("!")[0]) == qu_simparams.ny
        assert int(lines[3].strip().split("!")[0]) == qu_simparams.nz
        assert float(lines[4].strip().split("!")[0]) == qu_simparams.dx
        assert float(lines[5].strip().split("!")[0]) == qu_simparams.dy

        # Check stretch_grid_flag, surface_vertical_cell_size,
        # number_surface_cells
        assert int(
            lines[6].strip().split("!")[0]) == qu_simparams.stretch_grid_flag
        assert float(lines[7].strip().split("!")[
                         0]) == qu_simparams.surface_vertical_cell_size
        assert int(
            lines[8].strip().split("!")[0]) == qu_simparams.number_surface_cells

        # Check dz_array
        assert lines[9] == "! DZ array [m]\n"
        for i in range(qu_simparams.nz):
            index = i + 10
            dz = qu_simparams.dz_array[i]
            assert float(lines[index].strip()) == dz

        # Update lines index
        i_current = 10 + qu_simparams.nz

        # Check number of time increments, utc_offset
        assert int(lines[i_current].strip().split("!")[0]) == len(
            qu_simparams.wind_times)
        assert int(lines[i_current + 1].strip().split("!")[
                       0]) == qu_simparams.utc_offset

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
        assert int(
            lines[i_current].strip().split("!")[0]) == qu_simparams.sor_iter_max
        assert int(lines[i_current + 1].strip().split("!")[
                       0]) == qu_simparams.sor_residual_reduction

        # Check use_diffusion_flag, number_diffusion_iterations, domain_rotation
        # utm_x, utm_y, utm_zone_number, utm_zone_letter, quic_cfd_flag,
        # explosive_bldg_flag, bldg_array_flag
        assert int(lines[i_current + 2].strip().split("!")[
                       0]) == qu_simparams.use_diffusion_flag
        assert int(lines[i_current + 3].strip().split("!")[
                       0]) == qu_simparams.number_diffusion_iterations
        assert float(lines[i_current + 4].strip().split("!")[
                         0]) == qu_simparams.domain_rotation
        assert float(
            lines[i_current + 5].strip().split("!")[0]) == qu_simparams.utm_x
        assert float(
            lines[i_current + 6].strip().split("!")[0]) == qu_simparams.utm_y
        assert int(lines[i_current + 7].strip().split("!")[
                       0]) == qu_simparams.utm_zone_number
        assert int(lines[i_current + 8].strip().split("!")[
                       0]) == qu_simparams.utm_zone_letter
        assert int(lines[i_current + 9].strip().split("!")[
                       0]) == qu_simparams.quic_cfd_flag
        assert int(lines[i_current + 10].strip().split("!")[
                       0]) == qu_simparams.explosive_bldg_flag
        assert int(lines[i_current + 11].strip().split("!")[
                       0]) == qu_simparams.bldg_array_flag

    def test_from_file(self):
        """
        Test initializing a class from a QU_simparams.inp file.
        """
        # Test stretch grid flag = 3
        qu_simparams = self.get_test_object()
        qu_simparams.to_file("tmp/")
        test_object = QU_Simparams.from_file("tmp/")
        assert isinstance(test_object, QU_Simparams)
        assert qu_simparams == test_object

        # Test stretch grid flag = 0
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        qu_simparams.to_file("tmp/")
        test_object = QU_Simparams.from_file("tmp/")
        assert isinstance(test_object, QU_Simparams)
        assert qu_simparams == test_object

        # Test stretch grid flag = 1
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.custom_dz_array = [1] * qu_simparams.nz
        qu_simparams.to_file("tmp/")
        test_object = QU_Simparams.from_file("tmp/")
        assert isinstance(test_object, QU_Simparams)
        assert qu_simparams == test_object


class TestQFire_Advanced_User_Inputs:
    def test_init(self):
        """Test the initialization of a QFire_Advanced_User_Inputs object."""
        # Test the default initialization
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        assert qfire_advanced_user_inputs.fraction_cells_launch_firebrands == 0.05

        # Test custom initialization
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs(
            fraction_cells_launch_firebrands=0.1)
        assert qfire_advanced_user_inputs.fraction_cells_launch_firebrands == 0.1

        # Test data type casting
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs(
            fraction_cells_launch_firebrands="0.1")
        assert isinstance(
            qfire_advanced_user_inputs.fraction_cells_launch_firebrands, float)
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
        assert result_dict[
                   'fraction_cells_launch_firebrands'] == qfire_advanced_user_inputs.fraction_cells_launch_firebrands
        assert result_dict[
                   'firebrand_radius_scale_factor'] == qfire_advanced_user_inputs.firebrand_radius_scale_factor
        assert result_dict[
                   'firebrand_trajectory_time_step'] == qfire_advanced_user_inputs.firebrand_trajectory_time_step
        assert result_dict[
                   'firebrand_launch_interval'] == qfire_advanced_user_inputs.firebrand_launch_interval
        assert result_dict[
                   'firebrands_per_deposition'] == qfire_advanced_user_inputs.firebrands_per_deposition
        assert result_dict[
                   'firebrand_area_ratio'] == qfire_advanced_user_inputs.firebrand_area_ratio
        assert result_dict[
                   'minimum_burn_rate_coefficient'] == qfire_advanced_user_inputs.minimum_burn_rate_coefficient
        assert result_dict[
                   'max_firebrand_thickness_fraction'] == qfire_advanced_user_inputs.max_firebrand_thickness_fraction
        assert result_dict[
                   'firebrand_germination_delay'] == qfire_advanced_user_inputs.firebrand_germination_delay
        assert result_dict[
                   'vertical_velocity_scale_factor'] == qfire_advanced_user_inputs.vertical_velocity_scale_factor
        assert result_dict[
                   'minimum_firebrand_ignitions'] == qfire_advanced_user_inputs.minimum_firebrand_ignitions
        assert result_dict[
                   'maximum_firebrand_ignitions'] == qfire_advanced_user_inputs.maximum_firebrand_ignitions
        assert result_dict[
                   'minimum_landing_angle'] == qfire_advanced_user_inputs.minimum_landing_angle
        assert result_dict[
                   'maximum_firebrand_thickness'] == qfire_advanced_user_inputs.maximum_firebrand_thickness

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
            fraction_cells_launch_firebrands=0.1)
        qfire_advanced_user_inputs.to_file("tmp/")

        # Read the content of the file and check for correctness
        with open("tmp/QFIRE_advanced_user_inputs.inp", 'r') as file:
            lines = file.readlines()
            assert float(lines[0].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.fraction_cells_launch_firebrands
            assert float(lines[1].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrand_radius_scale_factor
            assert float(lines[2].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrand_trajectory_time_step
            assert float(lines[3].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrand_launch_interval
            assert float(lines[4].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrands_per_deposition
            assert float(lines[5].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrand_area_ratio
            assert float(lines[6].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.minimum_burn_rate_coefficient
            assert float(lines[7].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.max_firebrand_thickness_fraction
            assert float(lines[8].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.firebrand_germination_delay
            assert float(lines[9].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.vertical_velocity_scale_factor
            assert float(lines[10].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.minimum_firebrand_ignitions
            assert float(lines[11].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.maximum_firebrand_ignitions
            assert float(lines[12].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.minimum_landing_angle
            assert float(lines[13].strip().split("!")[
                             0]) == qfire_advanced_user_inputs.maximum_firebrand_thickness

        # Test writing to a non-existent directory
        with pytest.raises(FileNotFoundError):
            qfire_advanced_user_inputs.to_file(
                "/non_existent_path/QFIRE_advanced_user_inputs.inp")

    def test_from_file(self):
        """Test initializing a class from a QFIRE_advanced_user_inputs.inp
        file."""
        qfire_advanced_user_inputs = QFire_Advanced_User_Inputs()
        qfire_advanced_user_inputs.to_file("tmp/")
        test_object = QFire_Advanced_User_Inputs.from_file("tmp/")
        assert isinstance(test_object, QFire_Advanced_User_Inputs)
        assert qfire_advanced_user_inputs == test_object


class TestQUIC_fire:
    @staticmethod
    def get_test_object():
        return QUIC_fire(nz=26,
                         sim_time=60, time_now=1695311421)

    def test_init(self):
        # Test default initialization
        quic_fire = self.get_test_object()
        assert quic_fire.nz == 26
        assert quic_fire.sim_time == 60

        # Test changing the default values
        quic_fire.nz = 27
        assert quic_fire.nz == 27

        # Test data type casting
        quic_fire = QUIC_fire(nz="26",
                              sim_time=60, time_now=1695311421)
        assert isinstance(quic_fire.nz, int)
        assert quic_fire.nz == 26

        # Test stretch grid input
        assert quic_fire.stretch_grid_flag == 0
        assert quic_fire.stretch_grid_input == "1"
        assert quic_fire.dz == 1
        quic_fire.nz = 5
        quic_fire.dz_array = [1, 2, 3, 4, 5]
        quic_fire.stretch_grid_flag = 1
        assert quic_fire.stretch_grid_input == "1.0\n2.0\n3.0\n4.0\n5.0\n"

        # Test invalid dz array
        quic_fire = QUIC_fire(nz=26,
                              sim_time=60, time_now=1695311421,
                              stretch_grid_flag=1,
                              dz_array=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            assert quic_fire.stretch_grid_input == "1.0\n2.0\n3.0\n4.0\n5.0\n"
        
        # Test invalid random_seed
        quic_fire = self.get_test_object()
        with pytest.raises(ValidationError):
            quic_fire.random_seed = 0

        # Test fuel inputs
        quic_fire = QUIC_fire(nz=26,
                              sim_time=60, time_now=1695311421)
        assert quic_fire.fuel_params is None
        quic_fire.fuel_flag = 1
        quic_fire.fuel_params = FuelParams(fuel_density=0.5,
                                           fuel_moisture=1,
                                           fuel_height=0.75)
        assert quic_fire.fuel_lines == (
            f"{quic_fire.fuel_flag}\t! fuel density flag: 1 = uniform; "
            f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
            f" files for quic grid, 4 = Firetech files for "
            f"different grid (need interpolation)"
            f"\n0.5"
            f"\n{quic_fire.fuel_flag}\t! fuel moisture flag: 1 = uniform; "
            f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
            f" files for quic grid, 4 = Firetech files for "
            f"different grid (need interpolation)"
            f"\n1.0"
            f"\n{quic_fire.fuel_flag}\t! fuel height flag: 1 = uniform; "
            f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
            f" files for quic grid, 4 = Firetech files for "
            f"different grid (need interpolation)"
            f"\n0.75")

    def test_to_dict(self):
        """Test the to_dict method of a QUIC_fire object."""
        quic_fire = self.get_test_object()
        result_dict = quic_fire.to_dict()

        assert result_dict['nz'] == quic_fire.nz
        # TODO: revisit this one
        # assert result_dict["output_times"] == quic_fire.output_times
        assert result_dict['time_now'] == quic_fire.time_now
        assert result_dict['sim_time'] == quic_fire.sim_time
        assert result_dict['fire_flag'] == quic_fire.fire_flag
        assert result_dict['random_seed'] == quic_fire.random_seed
        assert result_dict['fire_time_step'] == quic_fire.fire_time_step
        assert result_dict['quic_time_step'] == quic_fire.quic_time_step
        assert result_dict['out_time_fire'] == quic_fire.out_time_fire
        assert result_dict['out_time_wind'] == quic_fire.out_time_wind
        assert result_dict['out_time_emis_rad'] == quic_fire.out_time_emis_rad
        assert result_dict['out_time_wind_avg'] == quic_fire.out_time_wind_avg
        assert result_dict['stretch_grid_flag'] == quic_fire.stretch_grid_flag
        assert result_dict['dz'] == quic_fire.dz
        assert result_dict['dz_array'] == quic_fire.dz_array
        assert result_dict['fuel_flag'] == quic_fire.fuel_flag
        assert result_dict['fuel_params'] == quic_fire.fuel_params
        assert result_dict['fuel_lines'] == quic_fire.fuel_lines
        assert result_dict['ignitions_per_cell'] == quic_fire.ignitions_per_cell
        assert result_dict['firebrand_flag'] == quic_fire.firebrand_flag
        assert result_dict['auto_kill'] == quic_fire.auto_kill
        assert result_dict['eng_to_atm_out'] == quic_fire.eng_to_atm_out
        assert result_dict['react_rate_out'] == quic_fire.react_rate_out
        assert result_dict['fuel_dens_out'] == quic_fire.fuel_dens_out
        assert result_dict['QF_wind_out'] == quic_fire.QF_wind_out
        assert result_dict['QU_wind_inst_out'] == quic_fire.QU_wind_inst_out
        assert result_dict['QU_wind_avg_out'] == quic_fire.QU_wind_avg_out
        assert result_dict['fuel_moist_out'] == quic_fire.fuel_moist_out
        assert result_dict['mass_burnt_out'] == quic_fire.mass_burnt_out
        assert result_dict['firebrand_out'] == quic_fire.firebrand_out
        assert result_dict['emissions_out'] == quic_fire.emissions_out
        assert result_dict['radiation_out'] == quic_fire.radiation_out
        assert result_dict['intensity_out'] == quic_fire.intensity_out

        # Computed fields
        assert result_dict['stretch_grid_input'] == quic_fire.stretch_grid_input
        assert result_dict['ignition_lines'] == quic_fire.ignition_lines
        assert result_dict['fuel_lines'] == quic_fire.fuel_lines

    # TODO: Add test for from_dict
    # TODO: Add test for to_docs
    # TODO: Add test for to_file

    def test_from_file(self):
        """Test initializing a class from a QFIRE_advanced_user_inputs.inp
        file."""
        quic_fire = self.get_test_object()
        quic_fire.to_file("tmp/")
        test_object = QUIC_fire.from_file("tmp/")
        assert isinstance(test_object, QUIC_fire)
        assert quic_fire == test_object
