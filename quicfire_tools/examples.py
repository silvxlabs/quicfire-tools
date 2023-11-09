from pathlib import Path
from quicfire_tools.inputs import SimulationInputs, QUIC_fire
from quicfire_tools.ignitions import RectangleIgnition


def create_line_fire() -> SimulationInputs:
    # Set up the basic simulation inputs
    sim_inputs = SimulationInputs.custom_domain(
        nx=200,
        ny=200,
        fire_nz=1,
    )

    # Make modifications to the QUIC_Fire input file
    quic_fire: QUIC_fire = sim_inputs.get_input("QUIC_fire")
    quic_fire.random_seed = 222
    quic_fire.time_now = 1653321600
    quic_fire.sim_time = 300
    quic_fire.out_time_fire = 50
    quic_fire.out_time_wind = 50
    quic_fire.out_time_emis_rad = 50
    quic_fire.out_time_wind_avg = 50
    quic_fire.nz = 1
    quic_fire.dz = 1.0
    quic_fire.fuel_flag = 1
    quic_fire.fuel_density = 0.7
    quic_fire.fuel_moisture = 0.05
    quic_fire.fuel_height = 1.0
    line_ignition = RectangleIgnition(
        x_min=150,
        y_min=100,
        x_length=100,
        y_length=100,
    )
    quic_fire.ignitions_per_cell = 100

    # Write the input file deck to a directory
    sim_inputs.write_inputs("tmp")


line_fire_simulation = create_line_fire()
