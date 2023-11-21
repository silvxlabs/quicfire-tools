from __future__ import annotations

from quicfire_tools.inputs import SimulationInputs


def create_line_fire() -> SimulationInputs:
    # Set up the basic simulation data
    sim_inputs = SimulationInputs.create_simulation(
        nx=200,
        ny=200,
        simulation_time=600,
        wind_speed=1.7,
        wind_direction=90,
    )

    # Set uniform fuel values
    sim_inputs.set_uniform_fuels(
        fuel_density=0.7,
        fuel_moisture=0.05,
        fuel_height=1.0,
    )

    # Set a line ignitions
    sim_inputs.set_rectangle_ignition(
        x_min=150,
        y_min=100,
        x_length=10,
        y_length=100,
    )

    # Select which binary files to output
    sim_inputs.set_output_files(
        fuel_dens=True,
        pm_emissions=True,
        radiation=True,
    )

    # Make modifications to the QUIC_Fire input file
    sim_inputs.quic_fire.random_seed = 222
    sim_inputs.quic_fire.time_now = 1653321600
    sim_inputs.quic_fire.sim_time = 300
    sim_inputs.quic_fire.out_time_fire = 50
    sim_inputs.quic_fire.out_time_wind = 50
    sim_inputs.quic_fire.out_time_emis_rad = 50
    sim_inputs.quic_fire.out_time_wind_avg = 50
    sim_inputs.quic_fire.ignitions_per_cell = 100

    # Write the input file deck to a directory
    sim_inputs.write_inputs("/tmp")


line_fire_simulation = create_line_fire()
