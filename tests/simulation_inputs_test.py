import os
import sys

sys.path.insert(0, "/Users/ntutland/Documents/Projects/quicfire-tools")
import quicfire_tools as qft

test = qft.SimulationInputs.setup_simulation(
    nx=200,
    ny=100,
    fire_nz=1,
    quic_nz=26,
    quic_height=100,
    dx=2,
    dy=2,
    fire_dz=1,
    wind_speed=5,
    wind_direction=270,
    simulation_time=60,
    output_time=30,
)

print(test.list_inputs())

test_QUIC_fire = test.get_input("QUIC_fire")
print(
    test_QUIC_fire.nz,
    test_QUIC_fire.fuel_dens_out,
    test_QUIC_fire.fuel_density,
)

test_QUIC_fire.fuel_flag = 1
test_QUIC_fire.fuel_density = 0.8
test_QUIC_fire.fuel_moisture = 1
test_QUIC_fire.fuel_height = 0.5

test.write_inputs("/Users/ntutland/Documents/Projects/quicfire-tools/tests/tmp")
