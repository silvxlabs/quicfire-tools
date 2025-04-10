# Core imports
from __future__ import annotations
from pathlib import Path

# Internal imports
from quicfire_tools.utils import read_dat_file
from quicfire_tools.inputs import SimulationInputs
from quicfire_tools.outputs import (
    _process_grid_info,
    _get_resolution_from_coords,
    _process_fire_indexes,
    _process_compressed_bin,
    _process_gridded_bin,
    SimulationOutputs,
    OutputFile,
)

# External imports
import zarr
import numpy as np
import netCDF4 as nc

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
SAMPLE_DATA_DIR = TEST_DATA_DIR / "samples"
LINE_FIRE_DIR = SAMPLE_DATA_DIR / "LineFire"
EG_CANOPY_DIR = SAMPLE_DATA_DIR / "EglinCanopyTest"
COS_HILL_DIR = SAMPLE_DATA_DIR / "CosHill"
TMP_DIR = TEST_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)


class TestProcessGridInfo:
    def test_read_grid_info_line_fire(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        quic_nz, quic_grid, en2atmos_nz, en2atmos_grid = _process_grid_info(
            LINE_FIRE_DIR / "Output" / "grid.bin",
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
        )
        assert quic_nz == line_fire.qu_simparams.nz
        assert len(quic_grid) == line_fire.qu_simparams.nz + 2
        quic_dz = _get_resolution_from_coords(quic_grid)
        assert np.allclose(quic_dz[:-1], line_fire.qu_simparams._dz_array)

        assert en2atmos_nz == 21
        en2atmos_dz = _get_resolution_from_coords(en2atmos_grid)
        assert en2atmos_dz == line_fire.quic_fire.dz

    def test_read_grid_info_eglin_canopy(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        quic_nz, quic_grid, en2atmos_nz, en2atmos_grid = _process_grid_info(
            EG_CANOPY_DIR / "Output" / "grid.bin",
            eglin_canopy.qu_simparams.ny,
            eglin_canopy.qu_simparams.nx,
        )
        assert quic_nz == eglin_canopy.qu_simparams.nz
        assert len(quic_grid) == eglin_canopy.qu_simparams.nz + 2
        quic_dz = _get_resolution_from_coords(quic_grid)
        assert np.allclose(quic_dz[:-1], eglin_canopy.qu_simparams._dz_array)

        assert en2atmos_nz == 43
        en2atmos_dz = _get_resolution_from_coords(en2atmos_grid)
        assert en2atmos_dz == eglin_canopy.quic_fire.dz


class TestProcessedCompressedBin:
    line_fire_sim = SimulationInputs.from_directory(LINE_FIRE_DIR)
    line_fire_indexes = _process_fire_indexes(
        LINE_FIRE_DIR / "Output" / "fire_indexes.bin"
    )

    eglin_canopy_sim = SimulationInputs.from_directory(EG_CANOPY_DIR)
    eglin_canopy_indexes = _process_fire_indexes(
        EG_CANOPY_DIR / "Output" / "fire_indexes.bin"
    )

    def test_line_fire_fuel_dens(self):
        fuels_dens_start = _process_compressed_bin(
            LINE_FIRE_DIR / "Output" / "fuels-dens-00000.bin",
            (
                self.line_fire_sim.quic_fire.nz,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
            self.line_fire_indexes,
        )
        assert fuels_dens_start.shape == (
            1,
            self.line_fire_sim.quic_fire.nz,
            self.line_fire_sim.qu_simparams.ny,
            self.line_fire_sim.qu_simparams.nx,
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "fuels-dens-00000.npz")
        reference_fuel_dens = npz["var"]
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 0, 2)

        assert np.allclose(fuels_dens_start[0], reference_fuel_dens)

        # trim xy edges of the array
        fuels_dens_start = fuels_dens_start[0, 0, 1:-1, 1:-1]
        assert np.allclose(fuels_dens_start, self.line_fire_sim.quic_fire.fuel_density)

        # Load the next time step
        fuels_dens_end = _process_compressed_bin(
            LINE_FIRE_DIR / "Output" / "fuels-dens-00060.bin",
            (
                self.line_fire_sim.quic_fire.nz,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
            self.line_fire_indexes,
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "fuels-dens-00060.npz")
        reference_fuel_dens = npz["var"]
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 0, 2)
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 1, 2)

        assert np.allclose(fuels_dens_end[0, 0, ...], reference_fuel_dens)

    def test_eglin_fuels_dens(self):
        # Load the first time step model output of fuel density
        fuels_dens_start = _process_compressed_bin(
            EG_CANOPY_DIR / "Output" / "fuels-dens-00000.bin",
            (
                self.eglin_canopy_sim.quic_fire.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
            self.eglin_canopy_indexes,
        )
        assert fuels_dens_start.shape == (
            1,
            self.eglin_canopy_sim.quic_fire.nz,
            self.eglin_canopy_sim.qu_simparams.ny,
            self.eglin_canopy_sim.qu_simparams.nx,
        )

        # Load in the input .dat file for comparison
        fuels_dens_dat = read_dat_file(
            EG_CANOPY_DIR / "treesrhof.dat",
            nx=self.eglin_canopy_sim.qu_simparams.nx,
            ny=self.eglin_canopy_sim.qu_simparams.ny,
            nz=self.eglin_canopy_sim.quic_fire.nz,
        )

        # QUIC-Fire removes fuel from the edges of the outputs
        # I don't know why but David confirmed this 4/1/25 - NJT
        assert np.allclose(
            fuels_dens_start[0, :, 1:-1, 2:-1], fuels_dens_dat[:, 1:-1, 1:-2], atol=1e-3
        )

        # Load the first time step of the reference data
        npz = np.load(EG_CANOPY_DIR / "Reference" / "fuels-dens-00000.npz")
        reference_fuel_dens = npz["var"]
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 0, 2)
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 1, 2)

        assert np.allclose(fuels_dens_start[0, ...], reference_fuel_dens)

        # Load the next time step
        fuels_dens_end = _process_compressed_bin(
            EG_CANOPY_DIR / "Output" / "fuels-dens-00600.bin",
            (
                self.eglin_canopy_sim.quic_fire.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
            self.eglin_canopy_indexes,
        )

        # Load the next time step of the reference data
        npz = np.load(EG_CANOPY_DIR / "Reference" / "fuels-dens-00600.npz")
        reference_fuel_dens = npz["var"]
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 0, 2)
        reference_fuel_dens = np.swapaxes(reference_fuel_dens, 1, 2)

        assert np.allclose(fuels_dens_end[0, ...], reference_fuel_dens)

    def test_eglin_co_emissions(self):
        # Load the model output
        co_emissions = _process_compressed_bin(
            EG_CANOPY_DIR / "Output" / "co_emissions-00600.bin",
            (
                self.eglin_canopy_sim.quic_fire.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
            self.eglin_canopy_indexes,
        )

        # Load the reference data
        npz = np.load(EG_CANOPY_DIR / "Reference" / "co_emissions-00600.npz")
        reference_co_emissions = npz["var"]
        reference_co_emissions = np.swapaxes(reference_co_emissions, 0, 2)
        reference_co_emissions = np.swapaxes(reference_co_emissions, 1, 2)

        assert np.allclose(co_emissions[0, ...], reference_co_emissions)

    def test_eglin_thermal_radiation(self):
        # Load the model output
        thermal_radiation = _process_compressed_bin(
            EG_CANOPY_DIR / "Output" / "thermalradiation-00600.bin",
            (
                self.eglin_canopy_sim.quic_fire.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
            self.eglin_canopy_indexes,
        )

        # Load the reference data
        npz = np.load(EG_CANOPY_DIR / "Reference" / "thermalradiation-00600.npz")
        reference_thermal_radiation = npz["var"]
        reference_thermal_radiation = np.swapaxes(reference_thermal_radiation, 0, 2)
        reference_thermal_radiation = np.swapaxes(reference_thermal_radiation, 1, 2)

        assert np.allclose(thermal_radiation[0, ...], reference_thermal_radiation)


class TestProcessGriddedBin:
    line_fire_sim = SimulationInputs.from_directory(LINE_FIRE_DIR)
    line_fire_indexes = _process_fire_indexes(
        LINE_FIRE_DIR / "Output" / "fire_indexes.bin"
    )

    eglin_canopy_sim = SimulationInputs.from_directory(EG_CANOPY_DIR)
    eglin_canopy_indexes = _process_fire_indexes(
        EG_CANOPY_DIR / "Output" / "fire_indexes.bin"
    )

    def test_line_fire_eng2atmos(self):
        _, _, line_fire_eng2atmos_nz, _ = _process_grid_info(
            LINE_FIRE_DIR / "Output" / "grid.bin",
            self.line_fire_sim.qu_simparams.ny,
            self.line_fire_sim.qu_simparams.nx,
        )

        # Load first time step
        eng2atmos = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "fire-energy_to_atmos-00000.bin",
            (
                line_fire_eng2atmos_nz,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "fire-energy_to_atmos-00000.npz")
        reference_eng2atmos = npz["var"]
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 0, 2)
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 1, 2)

        assert np.allclose(eng2atmos[0], reference_eng2atmos)

        # Load next time step
        eng2atmos = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "fire-energy_to_atmos-00060.bin",
            (
                line_fire_eng2atmos_nz,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
            self.line_fire_indexes,
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "fire-energy_to_atmos-00060.npz")
        reference_eng2atmos = npz["var"]
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 0, 2)
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 1, 2)

        assert np.allclose(eng2atmos[0], reference_eng2atmos)

    def test_line_fire_surf_energy(self):
        # Load first time step
        surf_energy = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "surfEnergy00009.bin",
            (
                1,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
        )

        assert surf_energy.shape == (1, 1, 200, 200)
        assert surf_energy.max() > 0

    def test_line_fire_mburnt_integ(self):
        # Load first time step
        mburnt_integ = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "mburnt_integ-00000.bin",
            (
                1,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "mburnt_integ-00000.npz")
        reference_mburnt_integ = npz["var"]
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 0, 2)
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 1, 2)

        assert np.allclose(mburnt_integ[0], reference_mburnt_integ)

        # Load next time step
        mburnt_integ = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "mburnt_integ-00060.bin",
            (
                1,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
            self.line_fire_indexes,
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "mburnt_integ-00060.npz")
        reference_mburnt_integ = npz["var"]
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 0, 2)
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 1, 2)

        assert np.allclose(mburnt_integ[0], reference_mburnt_integ)

    def test_line_fire_groundfuelheight(self):
        # Load model output
        groundfuelheight = _process_gridded_bin(
            LINE_FIRE_DIR / "Output" / "groundfuelheight.bin",
            (
                1,
                self.line_fire_sim.qu_simparams.ny,
                self.line_fire_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(LINE_FIRE_DIR / "Reference" / "groundfuelheight.npz")
        reference_groundfuelheight = npz["var"]

        assert np.allclose(groundfuelheight[0], reference_groundfuelheight)

    def test_eglin_eng2atmos(self):
        _, _, eglin_eng2atmos_nz, _ = _process_grid_info(
            EG_CANOPY_DIR / "Output" / "grid.bin",
            self.eglin_canopy_sim.qu_simparams.ny,
            self.eglin_canopy_sim.qu_simparams.nx,
        )

        # Load first time step
        eng2atmos = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "fire-energy_to_atmos-00000.bin",
            (
                eglin_eng2atmos_nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "fire-energy_to_atmos-00000.npz")
        reference_eng2atmos = npz["var"]
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 0, 2)
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 1, 2)

        assert eng2atmos[0].shape == reference_eng2atmos.shape
        assert np.allclose(eng2atmos[0], reference_eng2atmos)

        # Load next time step
        eng2atmos = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "fire-energy_to_atmos-00600.bin",
            (
                eglin_eng2atmos_nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
            self.eglin_canopy_indexes,
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "fire-energy_to_atmos-00600.npz")
        reference_eng2atmos = npz["var"]
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 0, 2)
        reference_eng2atmos = np.swapaxes(reference_eng2atmos, 1, 2)

        assert eng2atmos[0].shape == reference_eng2atmos.shape
        assert np.allclose(eng2atmos[0], reference_eng2atmos)

    def test_eglin_qu_windu(self):
        # Load first time step
        qu_windu = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "qu_windu00600.bin",
            (
                self.eglin_canopy_sim.qu_simparams.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "qu_windu00600.npz")
        reference_qu_windu = npz["var"]
        reference_qu_windu = np.swapaxes(reference_qu_windu, 0, 2)
        reference_qu_windu = np.swapaxes(reference_qu_windu, 1, 2)
        reference_qu_windu = reference_qu_windu[:-1, ...]

        assert np.allclose(qu_windu[0], reference_qu_windu)

    def test_eglin_windu(self):
        # Load first time step
        windu = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "windu00600.bin",
            (
                self.eglin_canopy_sim.quic_fire.nz,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "windu00600.npz")
        reference_windu = npz["var"]
        reference_windu = np.swapaxes(reference_windu, 0, 2)
        reference_windu = np.swapaxes(reference_windu, 1, 2)
        reference_windu = reference_windu[:-1, ...]

        assert np.allclose(windu[0], reference_windu)

    def test_eglin_mburnt_integ(self):
        # Load first time step
        mburnt_integ = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "mburnt_integ-00000.bin",
            (
                1,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "mburnt_integ-00000.npz")
        reference_mburnt_integ = npz["var"]
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 0, 2)
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 1, 2)

        assert np.allclose(mburnt_integ[0], reference_mburnt_integ)

        # Load next time step
        mburnt_integ = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "mburnt_integ-00600.bin",
            (
                1,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "mburnt_integ-00600.npz")
        reference_mburnt_integ = npz["var"]
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 0, 2)
        reference_mburnt_integ = np.swapaxes(reference_mburnt_integ, 1, 2)

        assert np.allclose(mburnt_integ[0], reference_mburnt_integ)

    def test_eglin_groundfuelheight(self):
        # Load model output
        groundfuelheight = _process_gridded_bin(
            EG_CANOPY_DIR / "Output" / "groundfuelheight.bin",
            (
                1,
                self.eglin_canopy_sim.qu_simparams.ny,
                self.eglin_canopy_sim.qu_simparams.nx,
            ),
        )

        # Load reference data for comparison
        npz = np.load(EG_CANOPY_DIR / "Reference" / "groundfuelheight.npz")
        reference_groundfuelheight = npz["var"]

        assert np.allclose(groundfuelheight[0], reference_groundfuelheight)


class TestSimulationOutputs:
    def test_line_fire(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        outputs = SimulationOutputs(
            LINE_FIRE_DIR / "Output",
            line_fire.quic_fire.nz,
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
            line_fire.qu_simparams.dy,
            line_fire.qu_simparams.dx,
        )

        # Check that the grid info is correct
        assert outputs.fire_nz == line_fire.quic_fire.nz
        assert outputs.quic_nz == line_fire.qu_simparams.nz
        assert outputs.fire_dz == line_fire.quic_fire.dz
        assert np.allclose(outputs.quic_dz, line_fire.qu_simparams._dz_array[:-1])

    def test_line_fire_georeferenced(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        utm_zones = range(1, 16)
        epsg_codes = [32600 + utm_zone for utm_zone in utm_zones]
        for zone, epsg in zip(utm_zones, epsg_codes):
            outputs = SimulationOutputs(
                LINE_FIRE_DIR / "Output",
                line_fire.quic_fire.nz,
                line_fire.qu_simparams.ny,
                line_fire.qu_simparams.nx,
                line_fire.qu_simparams.dy,
                line_fire.qu_simparams.dx,
                utm_x=np.random.randint(1, 1e8),
                utm_y=np.random.randint(1, 1e8),
                utm_zone=zone,
            )

            assert outputs.crs == f"EPSG:{epsg}"

    def test_line_fire_list_outputs(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        outputs = SimulationOutputs(
            LINE_FIRE_DIR / "Output",
            line_fire.quic_fire.nz,
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
            line_fire.qu_simparams.dy,
            line_fire.qu_simparams.dx,
        )
        outputs_list = outputs.list_outputs()

        assert "fire-energy_to_atmos" in outputs_list
        assert "fuels-dens" in outputs_list
        assert "groundfuelheight" in outputs_list
        assert "mburnt_integ" in outputs_list
        assert "surfEnergy" in outputs_list

    def test_line_fire_get_output(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        outputs = SimulationOutputs(
            LINE_FIRE_DIR / "Output",
            line_fire.quic_fire.nz,
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
            line_fire.qu_simparams.dy,
            line_fire.qu_simparams.dx,
        )

        fire_energy = outputs.get_output("fire-energy_to_atmos")
        assert fire_energy.times == [0, 60]

        fuels_dens = outputs.get_output("fuels-dens")
        assert fuels_dens.times == [0, 60]

        groundfuelheight = outputs.get_output("groundfuelheight")
        assert groundfuelheight.times == [0]

        mburnt_integ = outputs.get_output("mburnt_integ")
        assert mburnt_integ.times == [0, 60]

        surf_energy = outputs.get_output("surfEnergy")
        assert surf_energy.times == [0, 9]

    def test_line_fire_from_simulation(self):
        # Create the SimulationOutputs object with the normal constructor
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        outputs = SimulationOutputs(
            LINE_FIRE_DIR / "Output",
            line_fire.quic_fire.nz,
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
            line_fire.qu_simparams.dy,
            line_fire.qu_simparams.dx,
        )

        # Create the SimulationOutputs object from the SimulationInputs object
        outputs_from_sim_inputs = SimulationOutputs.from_simulation_inputs(
            LINE_FIRE_DIR / "Output", line_fire
        )

        # The two objects should be the same
        assert outputs == outputs_from_sim_inputs

    def test_eglin_canopy(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        outputs = SimulationOutputs(
            EG_CANOPY_DIR / "Output",
            eglin_canopy.quic_fire.nz,
            eglin_canopy.qu_simparams.ny,
            eglin_canopy.qu_simparams.nx,
            eglin_canopy.qu_simparams.dy,
            eglin_canopy.qu_simparams.dx,
        )

        # Check that the grid info is correct
        assert outputs.fire_nz == eglin_canopy.quic_fire.nz
        assert outputs.quic_nz == eglin_canopy.qu_simparams.nz
        assert outputs.fire_dz == eglin_canopy.quic_fire.dz
        assert np.allclose(outputs.quic_dz, eglin_canopy.qu_simparams._dz_array[:-1])

    def test_eglin_canopy_list_outputs(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        outputs = SimulationOutputs(
            EG_CANOPY_DIR / "Output",
            eglin_canopy.quic_fire.nz,
            eglin_canopy.qu_simparams.ny,
            eglin_canopy.qu_simparams.nx,
            eglin_canopy.qu_simparams.dy,
            eglin_canopy.qu_simparams.dx,
        )

        assert "co_emissions" in outputs.list_outputs()
        assert "fire-energy_to_atmos" in outputs.list_outputs()
        assert "fuels-dens" in outputs.list_outputs()
        assert "groundfuelheight" in outputs.list_outputs()
        assert "mburnt_integ" in outputs.list_outputs()
        assert "qu_windu" in outputs.list_outputs()
        assert "thermalradiation" in outputs.list_outputs()
        assert "windu" in outputs.list_outputs()

    def test_eglin_canopy_get_output(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        outputs = SimulationOutputs(
            EG_CANOPY_DIR / "Output",
            eglin_canopy.quic_fire.nz,
            eglin_canopy.qu_simparams.ny,
            eglin_canopy.qu_simparams.nx,
            eglin_canopy.qu_simparams.dy,
            eglin_canopy.qu_simparams.dx,
        )

        co_emissions = outputs.get_output("co_emissions")
        assert co_emissions.times == [600]
        assert co_emissions.shape == (outputs.fire_nz, outputs.ny, outputs.nx)

        fire_energy = outputs.get_output("fire-energy_to_atmos")
        assert fire_energy.times == [0, 600]
        assert fire_energy.shape == (outputs.en2atmos_nz, outputs.ny, outputs.nx)

        groundfuelheight = outputs.get_output("groundfuelheight")
        assert groundfuelheight.times == [0]
        assert groundfuelheight.shape == (1, outputs.ny, outputs.nx)

        mburnt_integ = outputs.get_output("mburnt_integ")
        assert mburnt_integ.times == [0, 600]
        assert mburnt_integ.shape == (1, outputs.ny, outputs.nx)

        qu_windu = outputs.get_output("qu_windu")
        assert qu_windu.times == [600]
        assert qu_windu.shape == (outputs.quic_nz, outputs.ny, outputs.nx)

        thermal_rad = outputs.get_output("thermalradiation")
        assert thermal_rad.times == [600]
        assert thermal_rad.shape == (outputs.fire_nz, outputs.ny, outputs.nx)

        wind = outputs.get_output("windu")
        assert wind.times == [600]
        assert wind.shape == (outputs.fire_nz, outputs.ny, outputs.nx)

    def test_object_equality(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        line_fire_outputs = SimulationOutputs(
            LINE_FIRE_DIR / "Output",
            line_fire.quic_fire.nz,
            line_fire.qu_simparams.ny,
            line_fire.qu_simparams.nx,
            line_fire.qu_simparams.dy,
            line_fire.qu_simparams.dx,
        )

        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        eglin_canopy_outputs = SimulationOutputs(
            EG_CANOPY_DIR / "Output",
            eglin_canopy.quic_fire.nz,
            eglin_canopy.qu_simparams.ny,
            eglin_canopy.qu_simparams.nx,
            eglin_canopy.qu_simparams.dy,
            eglin_canopy.qu_simparams.dx,
        )

        assert line_fire_outputs != eglin_canopy_outputs


class TestOutputFileToNumpy:
    line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
    line_fire_outputs = SimulationOutputs.from_simulation_inputs(
        LINE_FIRE_DIR / "Output", line_fire
    )

    eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
    eglin_canopy_outputs = SimulationOutputs.from_simulation_inputs(
        EG_CANOPY_DIR / "Output", eglin_canopy
    )

    @staticmethod
    def _test_to_numpy(output: OutputFile, times, nz, ny, nx):
        # Get all time steps
        data_all = output.to_numpy()
        assert data_all.shape == (len(times), nz, ny, nx)

        # Get all time steps with range
        data_all = output.to_numpy(range(len(times)))
        assert data_all.shape == (len(times), nz, ny, nx)

        # Get every other time step with range
        data_all = output.to_numpy(range(0, len(times), 2))
        assert data_all.shape == (max(len(times) // 2, 1), nz, ny, nx)

        # Get the first time step
        data_first = output.to_numpy(timestep=0)
        assert data_first.shape == (1, nz, ny, nx)

        # Get the last time step
        data_last = output.to_numpy(timestep=-1)
        assert data_last.shape == (1, nz, ny, nx)

    def test_lf_eng2atmos(self):
        eng2atmos = self.line_fire_outputs.get_output("fire-energy_to_atmos")
        self._test_to_numpy(eng2atmos, eng2atmos.times, *eng2atmos.shape)

    def test_lf_fuels_dens(self):
        fuels_dens = self.line_fire_outputs.get_output("fuels-dens")
        self._test_to_numpy(fuels_dens, fuels_dens.times, *fuels_dens.shape)

    def test_lf_groundfuelheight(self):
        groundfuelheight = self.line_fire_outputs.get_output("groundfuelheight")
        self._test_to_numpy(
            groundfuelheight,
            groundfuelheight.times,
            *groundfuelheight.shape,
        )

    def test_lf_mburnt_integ(self):
        mburnt_integ = self.line_fire_outputs.get_output("mburnt_integ")
        self._test_to_numpy(
            mburnt_integ,
            mburnt_integ.times,
            *mburnt_integ.shape,
        )

    def test_lf_surf_energy(self):
        surf_energy = self.line_fire_outputs.get_output("surfEnergy")
        self._test_to_numpy(
            surf_energy,
            surf_energy.times,
            *surf_energy.shape,
        )

    def test_ec_co_emissions(self):
        co_emissions = self.eglin_canopy_outputs.get_output("co_emissions")
        self._test_to_numpy(co_emissions, co_emissions.times, *co_emissions.shape)

    def test_ec_eng2atmos(self):
        eng2atmos = self.eglin_canopy_outputs.get_output("fire-energy_to_atmos")
        self._test_to_numpy(eng2atmos, eng2atmos.times, *eng2atmos.shape)

    def test_ec_fuel_dens(self):
        fuels_dens = self.eglin_canopy_outputs.get_output("fuels-dens")
        self._test_to_numpy(fuels_dens, fuels_dens.times, *fuels_dens.shape)

    def test_ec_groundfuelheight(self):
        groundfuelheight = self.eglin_canopy_outputs.get_output("groundfuelheight")
        self._test_to_numpy(
            groundfuelheight,
            groundfuelheight.times,
            *groundfuelheight.shape,
        )

    def test_ec_mburnt_integ(self):
        mburnt_integ = self.eglin_canopy_outputs.get_output("mburnt_integ")
        self._test_to_numpy(
            mburnt_integ,
            mburnt_integ.times,
            *mburnt_integ.shape,
        )

    def test_ec_qu_windu_to_numpy(self):
        qu_windu = self.eglin_canopy_outputs.get_output("qu_windu")
        self._test_to_numpy(
            qu_windu,
            qu_windu.times,
            *qu_windu.shape,
        )

    def test_ec_thermal_radiation_to_numpy(self):
        thermal_radiation = self.eglin_canopy_outputs.get_output("thermalradiation")
        self._test_to_numpy(
            thermal_radiation,
            thermal_radiation.times,
            *thermal_radiation.shape,
        )

    def test_ec_windu_to_numpy(self):
        windu = self.eglin_canopy_outputs.get_output("windu")
        self._test_to_numpy(
            windu,
            windu.times,
            *windu.shape,
        )


class TestOutputFileToNetCDF:
    line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
    line_fire_outputs = SimulationOutputs.from_simulation_inputs(
        LINE_FIRE_DIR / "Output", line_fire
    )

    eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
    eglin_canopy_outputs = SimulationOutputs.from_simulation_inputs(
        EG_CANOPY_DIR / "Output", eglin_canopy
    )

    @staticmethod
    def _test_to_netcdf(output: OutputFile, directory: Path, times, nz, ny, nx):
        # Get all time steps
        data_all = output.to_numpy()
        output.to_netcdf(directory)
        test = nc.Dataset(directory / f"{output.name}.nc", "r")
        data_all = test.variables[output.name]
        assert data_all.shape == (len(times), nz, ny, nx)
        test.close()

        # Get all time steps with range
        data_all = output.to_numpy(range(len(times)))
        output.to_netcdf(directory, range(len(times)))
        test = nc.Dataset(directory / f"{output.name}.nc", "r")
        data_all = test.variables[output.name]
        assert data_all.shape == (len(times), nz, ny, nx)
        test.close()

        # Get every other time step with range
        data_all = output.to_numpy(range(0, len(times), 2))
        output.to_netcdf(directory, range(0, len(times), 2))
        test = nc.Dataset(directory / f"{output.name}.nc", "r")
        data_all = test.variables[output.name]
        assert data_all.shape == (max(len(times) // 2, 1), nz, ny, nx)
        test.close()

        # Get the first time step
        data_first = output.to_numpy(timestep=0)
        output.to_netcdf(directory, timestep=0)
        test = nc.Dataset(directory / f"{output.name}.nc", "r")
        data_first = test.variables[output.name]
        assert data_first.shape == (1, nz, ny, nx)
        test.close()

        # Get the last time step
        data_last = output.to_numpy(timestep=-1)
        output.to_netcdf(directory, timestep=-1)
        test = nc.Dataset(directory / f"{output.name}.nc", "r")
        data_last = test.variables[output.name]
        assert data_last.shape == (1, nz, ny, nx)
        test.close()

    def test_lf_eng2atmos(self):
        eng2atmos = self.line_fire_outputs.get_output("fire-energy_to_atmos")
        self._test_to_netcdf(eng2atmos, TMP_DIR, eng2atmos.times, *eng2atmos.shape)

    def test_lf_fuels_dens(self):
        fuels_dens = self.line_fire_outputs.get_output("fuels-dens")
        self._test_to_netcdf(fuels_dens, TMP_DIR, fuels_dens.times, *fuels_dens.shape)

    def test_lf_groundfuelheight(self):
        groundfuelheight = self.line_fire_outputs.get_output("groundfuelheight")
        self._test_to_netcdf(
            groundfuelheight,
            TMP_DIR,
            groundfuelheight.times,
            *groundfuelheight.shape,
        )

    def test_lf_mburnt_integ(self):
        mburnt_integ = self.line_fire_outputs.get_output("mburnt_integ")
        self._test_to_netcdf(
            mburnt_integ,
            TMP_DIR,
            mburnt_integ.times,
            *mburnt_integ.shape,
        )

    def test_lf_surf_energy(self):
        surf_energy = self.line_fire_outputs.get_output("surfEnergy")
        self._test_to_netcdf(
            surf_energy,
            TMP_DIR,
            surf_energy.times,
            *surf_energy.shape,
        )

    def test_ec_co_emissions(self):
        co_emissions = self.eglin_canopy_outputs.get_output("co_emissions")
        self._test_to_netcdf(
            co_emissions, TMP_DIR, co_emissions.times, *co_emissions.shape
        )

    def test_ec_eng2atmos(self):
        eng2atmos = self.eglin_canopy_outputs.get_output("fire-energy_to_atmos")
        self._test_to_netcdf(eng2atmos, TMP_DIR, eng2atmos.times, *eng2atmos.shape)

    def test_ec_fuel_dens(self):
        fuels_dens = self.eglin_canopy_outputs.get_output("fuels-dens")
        self._test_to_netcdf(fuels_dens, TMP_DIR, fuels_dens.times, *fuels_dens.shape)

    def test_ec_groundfuelheight(self):
        groundfuelheight = self.eglin_canopy_outputs.get_output("groundfuelheight")
        self._test_to_netcdf(
            groundfuelheight,
            TMP_DIR,
            groundfuelheight.times,
            *groundfuelheight.shape,
        )

    def test_ec_mburnt_integ(self):
        mburnt_integ = self.eglin_canopy_outputs.get_output("mburnt_integ")
        self._test_to_netcdf(
            mburnt_integ,
            TMP_DIR,
            mburnt_integ.times,
            *mburnt_integ.shape,
        )

    def test_ec_qu_windu_to_numpy(self):
        qu_windu = self.eglin_canopy_outputs.get_output("qu_windu")
        self._test_to_netcdf(
            qu_windu,
            TMP_DIR,
            qu_windu.times,
            *qu_windu.shape,
        )

    def test_ec_thermal_radiation_to_numpy(self):
        thermal_radiation = self.eglin_canopy_outputs.get_output("thermalradiation")
        self._test_to_netcdf(
            thermal_radiation,
            TMP_DIR,
            thermal_radiation.times,
            *thermal_radiation.shape,
        )

    def test_ec_windu_to_numpy(self):
        windu = self.eglin_canopy_outputs.get_output("windu")
        self._test_to_netcdf(
            windu,
            TMP_DIR,
            windu.times,
            *windu.shape,
        )


class TestOutputFileToZarr:
    def test_lf_fuel_dens(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        line_fire_outputs = SimulationOutputs.from_simulation_inputs(
            LINE_FIRE_DIR / "Output", line_fire
        )

        fuels_dens = line_fire_outputs.get_output("fuels-dens")
        fuels_dens.to_zarr(TMP_DIR)

        zroot = zarr.open(str(TMP_DIR / "fuels-dens.zarr"), mode="r")
        assert np.allclose(zroot["time"][:], fuels_dens.times)
        assert np.allclose(zroot["x"][:], fuels_dens.x_coords)
        assert np.allclose(zroot["y"][:], fuels_dens.y_coords)
        assert np.allclose(zroot["z"][:], fuels_dens.z_coords)

        z_array = zroot["fuels-dens"]
        assert np.allclose(z_array[:], fuels_dens.to_numpy())
        assert z_array.chunks == (1, *fuels_dens.shape)

    def test_lf_fuel_dens_custom_time_chunks(self):
        line_fire = SimulationInputs.from_directory(LINE_FIRE_DIR)
        line_fire_outputs = SimulationOutputs.from_simulation_inputs(
            LINE_FIRE_DIR / "Output", line_fire
        )

        fuels_dens = line_fire_outputs.get_output("fuels-dens")
        fuels_dens.to_zarr(TMP_DIR, chunk_size={"time": 100})

        zroot = zarr.open(str(TMP_DIR / "fuels-dens.zarr"), mode="r")
        z_array = zroot["fuels-dens"]
        print(z_array.chunks)

        assert z_array.chunks == (100, *fuels_dens.shape)

    def test_ec_qu_windu(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        eglin_canopy_outputs = SimulationOutputs.from_simulation_inputs(
            EG_CANOPY_DIR / "Output", eglin_canopy
        )

        qu_windu = eglin_canopy_outputs.get_output("qu_windu")
        qu_windu.to_zarr(TMP_DIR)

        zroot = zarr.open(str(TMP_DIR / "qu_windu.zarr"), mode="r")
        assert np.allclose(zroot["time"][:], qu_windu.times)
        assert np.allclose(zroot["x"][:], qu_windu.x_coords)
        assert np.allclose(zroot["y"][:], qu_windu.y_coords)
        assert np.allclose(zroot["z"][:], qu_windu.z_coords)

        z_array = zroot["qu_windu"]
        assert np.allclose(z_array[:], qu_windu.to_numpy())
        assert z_array.chunks == (1, *qu_windu.shape)

    def test_ec_qu_windu_custom_chunks(self):
        eglin_canopy = SimulationInputs.from_directory(EG_CANOPY_DIR)
        eglin_canopy_outputs = SimulationOutputs.from_simulation_inputs(
            EG_CANOPY_DIR / "Output", eglin_canopy
        )

        qu_windu = eglin_canopy_outputs.get_output("qu_windu")
        qu_windu.to_zarr(
            TMP_DIR,
            chunk_size={
                "time": 10,
                "z": qu_windu.shape[0] // 2,
                "y": qu_windu.shape[1] // 2,
                "x": qu_windu.shape[2] // 2,
            },
        )

        zroot = zarr.open(str(TMP_DIR / "qu_windu.zarr"), mode="r")
        z_array = zroot["qu_windu"]

        assert z_array.chunks == (
            10,
            qu_windu.shape[0] // 2,
            qu_windu.shape[1] // 2,
            qu_windu.shape[2] // 2,
        )
        assert z_array.cdata_shape == (1, 2, 2, 2)
