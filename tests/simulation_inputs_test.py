import os
import sys

sys.path.insert(0, "/Users/ntutland/Documents/Projects/quicfire-tools")
import quicfire_tools as qft

test = qft.SimulationInputs.setup_simulation()

print(test.list_inputs())

test_QUIC_fire = test.get_input("QUIC_fire")
print(
    test_QUIC_fire.nz,
    test_QUIC_fire.fuel_dens_out,
    test_QUIC_fire.fuel_flag,
    test_QUIC_fire.fuel_density,
    test_QUIC_fire.fuel_moisture,
    test_QUIC_fire.fuel_height,
    test_QUIC_fire.ignition_type,
)


test.write_inputs("/Users/ntutland/Documents/Projects/quicfire-tools/tests/tmp")
