import sys

sys.path.insert(0, "/Users/ntutland/Documents/Projects/quicfire-tools")
import quicfire_tools as qft

test = qft.SimulationInputs.custom_domain(nx=100, ny=100, fire_nz=40)

print(test.list_inputs())

test_QUIC_fire = test.get_input("QUIC_fire")
print(
    test_QUIC_fire.nz,
    "\n",
    test_QUIC_fire.fuel_dens_out,
    "\n",
    test_QUIC_fire.fuel_flag,
    "\n",
    test_QUIC_fire.fuel_density,
    "\n",
    test_QUIC_fire.fuel_moisture,
    "\n",
    test_QUIC_fire.fuel_height,
    "\n",
    test_QUIC_fire.ignition_type,
    "\n",
)


test.write_inputs("/Users/ntutland/Documents/Projects/quicfire-tools/tests/tmp")
