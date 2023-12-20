from __future__ import annotations

from quicfire_tools.inputs import SimulationInputs


def create_line_fire() -> SimulationInputs:
    # Set up the basic simulation data
    sim_inputs = SimulationInputs.create_simulation(
        nx=200,
        ny=200,
        fire_nz=1,
        simulation_time=600,
        wind_speed=1.7,
        wind_direction=270,
    )

    # Set uniform fuel values
    sim_inputs.set_uniform_fuels(
        fuel_density=0.7,
        fuel_moisture=0.10,
        fuel_height=1.0,
    )

    # Set a line ignitions
    sim_inputs.set_rectangle_ignition(
        x_min=10,
        y_min=50,
        x_length=10,
        y_length=100,
    )

    # Select which binary files to output
    sim_inputs.set_output_files(
        fuel_dens=True,
        mass_burnt=True,
        qf_wind=True,
    )

    return sim_inputs


# TODO: Implement a FastFuels custom .dat file example
def create_custom_fuels_simulation() -> SimulationInputs:
    return None


line_fire_simulation = create_line_fire()
custom_fuels_simulation = create_custom_fuels_simulation()
