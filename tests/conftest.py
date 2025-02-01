"""
tests/conftest.py
"""

import pytest
from pathlib import Path
from shutil import rmtree

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data" / "test-inputs"
SAMPLES_DIR = TEST_DIR / "data" / "samples"
TMP_DIR = TEST_DIR / "tmp"


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_tmpdir():
    """
    Everytime a test function is run, this fixture will automatically delete the
    existing tmp directory and create a new one before the test function is run.
    This solves any potential side effects from previous tests that may have left
    files in the tmp directory.

    I elected to keep the tmp directory after the test function is run so that
    the user can inspect the contents of the directory after the test is run.
    If it becomes necessary to delete the directory in the future, add the line
    `rmtree(TMP_DIR)` to the teardown (after the yield statement).
    """
    if TMP_DIR.exists():
        rmtree(TMP_DIR)
    TMP_DIR.mkdir()
    yield
