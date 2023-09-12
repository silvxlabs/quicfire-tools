import sys

sys.path.append("../quicfire_tools")
from quicfire_tools.inputs import *

# External Imports
import pytest


class TestGridList:
    def test_init(self):
        """Test the initialization of a Gridlist object."""
        # Test the default initialization
        gridlist = Gridlist(n=10, m=10, l=10, dx=1, dy=1, dz=1, aa1=1)
        assert isinstance(gridlist, Gridlist)
        assert gridlist.n == 10
        for i in ["n", "m", "l", "dx", "dy", "dz", "aa1"]:
            assert i in gridlist.list_parameters()

        # Pass bad parameters: non-real numbers
        with pytest.raises(TypeError):
            Gridlist(n=2.5, m=10, l=10, dx="", dy=1, dz=1, aa1=1)
        with pytest.raises(TypeError):
            Gridlist(n="10", m=10, l=10, dx=1, dy=1, dz=1, aa1=1)

        # Pass bad parameters: zero or negative values
        with pytest.raises(TypeError):
            Gridlist(n=-10, m=10, l=10, dx=0, dy=1, dz=1, aa1=1)
        with pytest.raises(TypeError):
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
        with pytest.raises(TypeError):
            QU_Buildings(wall_roughness_length=-1, number_of_buildings=0,
                         number_of_polygon_nodes=0)
        with pytest.raises(TypeError):
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


class TestQU_Fileoptions:
    def test_init_default(self):
        """Test the default initialization of a QU_Fileoptions object."""
        file_options = QU_Fileoptions()

        assert isinstance(file_options, QU_Fileoptions)
        assert file_options.output_data_file_format_flag == 2
        assert file_options.non_mass_conserved_initial_field_flag == 0
        assert file_options.initial_sensor_velocity_field_flag == 0
        assert file_options.qu_staggered_velocity_file_flag == 0
        assert file_options.generate_wind_startup_files_flag == 0

    def test_init_custom(self):
        """Test custom initialization of a QU_Fileoptions object."""
        file_options = QU_Fileoptions(output_data_file_format_flag=1,
                                      non_mass_conserved_initial_field_flag=1,
                                      initial_sensor_velocity_field_flag=1,
                                      qu_staggered_velocity_file_flag=1,
                                      generate_wind_startup_files_flag=1)

        assert file_options.output_data_file_format_flag == 1
        assert file_options.non_mass_conserved_initial_field_flag == 1
        assert file_options.initial_sensor_velocity_field_flag == 1
        assert file_options.qu_staggered_velocity_file_flag == 1
        assert file_options.generate_wind_startup_files_flag == 1

    def test_validate_inputs_type(self):
        """Test input validation for type constraints."""
        with pytest.raises(TypeError):
            QU_Fileoptions(output_data_file_format_flag="a")
        with pytest.raises(TypeError):
            QU_Fileoptions(non_mass_conserved_initial_field_flag="b")
        with pytest.raises(TypeError):
            QU_Fileoptions(initial_sensor_velocity_field_flag="c")
        with pytest.raises(TypeError):
            QU_Fileoptions(qu_staggered_velocity_file_flag="d")
        with pytest.raises(TypeError):
            QU_Fileoptions(generate_wind_startup_files_flag="e")

    def test_validate_inputs_value_for_flags(self):
        """Test input validation for value constraints for flags."""
        with pytest.raises(TypeError):
            QU_Fileoptions(non_mass_conserved_initial_field_flag=4)
        with pytest.raises(TypeError):
            QU_Fileoptions(initial_sensor_velocity_field_flag=3)
        with pytest.raises(TypeError):
            QU_Fileoptions(qu_staggered_velocity_file_flag=4)
        with pytest.raises(TypeError):
            QU_Fileoptions(generate_wind_startup_files_flag=5)

    def test_validate_inputs_value_for_output_format(self):
        """
        Test input validation for value constraints for
        output_data_file_format_flag.
        """
        with pytest.raises(TypeError):
            QU_Fileoptions(output_data_file_format_flag=5)
        with pytest.raises(TypeError):
            QU_Fileoptions(output_data_file_format_flag=-1)

    def test_to_dict(self):
        """Test the to_dict method of a QU_Buildings object."""
        qu_fileoptions = QU_Fileoptions()
        result_dict = qu_fileoptions.to_dict()
        assert result_dict['output_data_file_format_flag'] == 2
        assert result_dict['non_mass_conserved_initial_field_flag'] == 0
        assert result_dict['initial_sensor_velocity_field_flag'] == 0
        assert result_dict['qu_staggered_velocity_file_flag'] == 0
        assert result_dict['generate_wind_startup_files_flag'] == 0
        assert '_validate_inputs' not in result_dict

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


class TestQU_Simparams:
    @staticmethod
    def get_test_object():
        return QU_Simparams(nx=100, ny=100, nz=26, dx=2, dy=2,
                            surface_vertical_cell_size=1,
                            number_surface_cells=5,
                            )

    def test_stretch_grid_flag_0(self):
        qu_simparams = self.get_test_object()
        qu_simparams.stretch_grid_flag = 0
        qu_simparams.dz_array = []
        vertical_grid_lines = qu_simparams._stretch_grid_flag_0()
        with open("data/test-templates/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

    def test_stretch_grid_flag_1(self):
        qu_simparams = self.get_test_object()

        # Test with no dz_array input
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with 19 dz_array inputs
        qu_simparams.dz_array = [1] * 19
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test with dz inputs that don't match the surface values
        qu_simparams.dz_array = [1] * 20
        qu_simparams.dz_array[0] = 2
        with pytest.raises(ValueError):
            qu_simparams._stretch_grid_flag_1()

        # Test valid case
        qu_simparams.dz_array = [1] * 26
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
        qu_simparams.dz_array = []
        vertical_grid_lines = qu_simparams._generate_vertical_grid()
        with open("data/test-templates/stretchgrid_0.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 1
        qu_simparams.stretch_grid_flag = 1
        qu_simparams.dz_array = [1] * 26
        vertical_grid_lines = qu_simparams._generate_vertical_grid()
        with open("data/test-templates/stretchgrid_1.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 2
        qu_simparams.stretch_grid_flag = 2
        with pytest.raises(ValueError):
            qu_simparams._generate_vertical_grid()

        # Test stretch_grid_flag = 3
        qu_simparams.stretch_grid_flag = 3
        vertical_grid_lines = qu_simparams._generate_vertical_grid()
        with open("data/test-templates/stretchgrid_3.txt") as f:
            expected_lines = f.readlines()
        assert vertical_grid_lines == "".join(expected_lines)

        # Test stretch_grid_flag = 4
        qu_simparams.stretch_grid_flag = 4
        with pytest.raises(ValueError):
            qu_simparams._generate_vertical_grid()

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
        assert result_dict['nx'] == 100
        assert result_dict['ny'] == 100
        assert result_dict['nz'] == 26
        assert result_dict['dx'] == 2
        assert result_dict['dy'] == 2
        assert result_dict['surface_vertical_cell_size'] == 1
        assert result_dict['number_surface_cells'] == 5

        # Test the default parameters
        assert result_dict['surface_vertical_cell_size'] == 1.
        assert result_dict['number_surface_cells'] == 5
        assert result_dict['stretch_grid_flag'] == 3
        assert len(result_dict['dz_array']) == 26
        assert result_dict['utc_offset'] == 0
        assert len(result_dict['wind_times']) == 1
        assert result_dict['sor_iter_max'] == 10
        assert result_dict['sor_residual_reduction'] == 3
        assert result_dict['use_diffusion_flag'] == 0
        assert result_dict['number_diffusion_iterations'] == 10
        assert result_dict['domain_rotation'] == 0.
        assert result_dict['utm_x'] == 0.
        assert result_dict['utm_y'] == 0.
        assert result_dict['utm_zone_number'] == 1
        assert result_dict['utm_zone_letter'] == 1
        assert result_dict['quic_cfd_flag'] == 0
        assert result_dict['explosive_bldg_flag'] == 0
        assert result_dict['bldg_array_flag'] == 0

        # Test that the stuff we don't want is not there
        assert '_validate_inputs' not in result_dict

    def test_to_docs(self):
        qu_simparams = self.get_test_object()
        result_dict = qu_simparams.to_dict()
        result_docs = qu_simparams.get_documentation()
        for key in result_dict:
            if key in ["filename", "param_info", "vertical_grid", "wind_lines"]:
                continue
            assert key in result_docs

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
        assert int(lines[6].strip().split("!")[0]) == qu_simparams.stretch_grid_flag
        assert float(lines[7].strip().split("!")[0]) == qu_simparams.surface_vertical_cell_size
        assert int(lines[8].strip().split("!")[0]) == qu_simparams.number_surface_cells

        # Check dz_array
        assert lines[9] == "! DZ array [m]\n"
        for i in range(qu_simparams.nz):
            index = i + 10
            dz = qu_simparams.dz_array[i]
            assert float(lines[index].strip()) == dz

        # Update lines index
        i_current = 10 + qu_simparams.nz

        # Check number of time increments, utc_offset
        assert int(lines[i_current].strip().split("!")[0]) == len(qu_simparams.wind_times)
        assert int(lines[i_current + 1].strip().split("!")[0]) == qu_simparams.utc_offset

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
        assert int(lines[i_current + 1].strip().split("!")[0]) == qu_simparams.sor_residual_reduction

        # Check use_diffusion_flag, number_diffusion_iterations, domain_rotation
        # utm_x, utm_y, utm_zone_number, utm_zone_letter, quic_cfd_flag,
        # explosive_bldg_flag, bldg_array_flag
        assert int(lines[i_current + 2].strip().split("!")[0]) == qu_simparams.use_diffusion_flag
        assert int(lines[i_current + 3].strip().split("!")[0]) == qu_simparams.number_diffusion_iterations
        assert float(lines[i_current + 4].strip().split("!")[0]) == qu_simparams.domain_rotation
        assert float(lines[i_current + 5].strip().split("!")[0]) == qu_simparams.utm_x
        assert float(lines[i_current + 6].strip().split("!")[0]) == qu_simparams.utm_y
        assert int(lines[i_current + 7].strip().split("!")[0]) == qu_simparams.utm_zone_number
        assert int(lines[i_current + 8].strip().split("!")[0]) == qu_simparams.utm_zone_letter
        assert int(lines[i_current + 9].strip().split("!")[0]) == qu_simparams.quic_cfd_flag
        assert int(lines[i_current + 10].strip().split("!")[0]) == qu_simparams.explosive_bldg_flag
        assert int(lines[i_current + 11].strip().split("!")[0]) == qu_simparams.bldg_array_flag
