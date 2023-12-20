from pathlib import Path
from quicfire_tools.examples import line_fire_simulation, custom_fuels_simulation


TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"

# Write simulation inputs to the test data directory
line_fire_sim_path = TEST_DATA_PATH / "test-simulation-line-fire"
line_fire_sim_path.mkdir(exist_ok=True)
line_fire_simulation.write_inputs(line_fire_sim_path)
line_fire_simulation.to_json(line_fire_sim_path / "line-fire-simulation.json")

custom_fuels_sim_path = TEST_DATA_PATH / "test-simulation-custom-fuels"
custom_fuels_sim_path.mkdir(exist_ok=True)
custom_fuels_simulation.write_inputs(custom_fuels_sim_path)
custom_fuels_simulation.to_json(custom_fuels_sim_path / "custom-fuels-simulation.json")
