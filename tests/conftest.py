"""
tests/conftest.py
"""

import pytest
from pathlib import Path
from shutil import rmtree
import math
import json
from types import SimpleNamespace

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


@pytest.fixture
def assert_valid_ignite_dat():
    def _check(path, igntype=5):
        lines = Path(path).read_text().splitlines()
        while lines and not lines[-1].strip():  # tolerate trailing blank lines
            lines.pop()

        assert len(lines) >= 6, "file is shorter than the 6-line header"

        # Header
        assert lines[0].strip() == f"igntype={igntype}", f"bad line 1: {lines[0]!r}"
        assert lines[1].strip() == "&atvlist", f"bad line 2: {lines[1]!r}"
        assert lines[2].startswith("natv="), f"bad line 3: {lines[2]!r}"
        natv = int(lines[2].split("=")[1])
        assert lines[3].startswith("targettemp="), f"bad line 4: {lines[3]!r}"
        float(lines[3].split("=")[1])
        assert lines[4].startswith("flamedistance="), f"bad line 5: {lines[4]!r}"
        float(lines[4].split("=")[1])
        assert lines[5].strip() == "/", f"bad line 6: {lines[5]!r}"

        # Data rows
        rows = lines[6:]
        assert len(rows) == natv, f"natv={natv} but found {len(rows)} data rows"
        for i, row in enumerate(rows, start=7):
            parts = row.split()
            assert len(parts) == 6, f"line {i}: expected 6 columns, got {len(parts)}"
            for p in parts:
                assert math.isfinite(float(p)), f"line {i}: non-finite value {p!r}"

    return _check


CRS_5070 = {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:EPSG::5070"},
}

# --- Geometry definitions (projected meters) --------------------------------
DOMAIN_ORIGIN = (1000.0, 2000.0)  # SW corner of the domain
DOMAIN_SIZE = 200.0  # square, meters per side

UNIT_BOUNDS = (1040.0, 2050.0, 1160.0, 2150.0)  # xmin, ymin, xmax, ymax

# Multi-vertex lines, both fully inside the unit
LINE_A = [(1050.0, 2060.0), (1150.0, 2060.0), (1150.0, 2080.0)]  # 100 + 20 = 120 m
LINE_B = [
    (1150.0, 2100.0),
    (1050.0, 2100.0),
    (1050.0, 2120.0),
    (1150.0, 2120.0),
]  # 100 + 20 + 100 = 220 m

# Both lines are 1 m/s tracks: A takes 120 s, B takes 220 s
LINE_A_PROPS = {
    "fire_order": 1,
    "TimeStart": "4/18/2025 14:30:00 (UTC)",
    "TimeEnd": "4/18/2025 14:32:00 (UTC)",
}
LINE_B_PROPS = {
    "fire_order": 2,
    "TimeStart": "4/18/2025 14:31:00 (UTC)",
    "TimeEnd": "4/18/2025 14:34:40 (UTC)",
}

POINTS = [
    (1050.0, 2060.0),
    (1075.0, 2080.0),
    (1100.0, 2100.0),
    (1125.0, 2120.0),
    (1150.0, 2140.0),
]
POINT_START_TIMES = [
    "4/18/2025 14:30:00 (UTC)",
    "4/18/2025 14:31:00 (UTC)",
    "4/18/2025 14:31:30 (UTC)",
    "4/18/2025 14:35:00 (UTC)",
    "4/18/2025 14:36:00 (UTC)",
]


# --- Helpers ----------------------------------------------------------------
def _rect_ring(xmin, ymin, xmax, ymax):
    """Closed polygon ring (first vertex repeated at the end)."""
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]


def _feature(geometry, properties=None):
    return {"type": "Feature", "properties": properties or {}, "geometry": geometry}


def _write_geojson(path, features):
    path.write_text(
        json.dumps({"type": "FeatureCollection", "crs": CRS_5070, "features": features})
    )


# --- Fixtures ---------------------------------------------------------------
@pytest.fixture
def geom():
    """Constants and expected values for use in assertions."""
    ox, oy = DOMAIN_ORIGIN

    def to_local(x, y):
        """Convert projected coordinates to domain-local coordinates."""
        return x - ox, y - oy

    return SimpleNamespace(
        domain_origin=DOMAIN_ORIGIN,
        domain_size=DOMAIN_SIZE,
        unit_bounds=UNIT_BOUNDS,
        line_a=LINE_A,
        line_b=LINE_B,
        line_a_length=120.0,
        line_b_length=220.0,
        points=POINTS,
        to_local=to_local,
    )


@pytest.fixture
def domain_file(tmp_path):
    """200 m x 200 m square domain. Returns the file name (in tmp_path)."""
    ox, oy = DOMAIN_ORIGIN
    ring = _rect_ring(ox, oy, ox + DOMAIN_SIZE, oy + DOMAIN_SIZE)
    _write_geojson(
        tmp_path / "domain.geojson",
        [_feature({"type": "Polygon", "coordinates": [ring]})],
    )
    return "domain.geojson"


@pytest.fixture
def unit_file(tmp_path):
    """Rectangular burn unit inside the domain."""
    ring = _rect_ring(*UNIT_BOUNDS)
    _write_geojson(
        tmp_path / "unit.geojson",
        [_feature({"type": "Polygon", "coordinates": [ring]}, {"name": "unit1"})],
    )
    return "unit.geojson"


@pytest.fixture
def lines_file(tmp_path):
    """Two multi-vertex LineStrings inside the unit.

    Carries both fire_order (for order_time='order') and
    TimeStart/TimeEnd (for order_time='time').
    """
    _write_geojson(
        tmp_path / "lines.geojson",
        [
            _feature(
                {"type": "LineString", "coordinates": [list(p) for p in LINE_A]},
                LINE_A_PROPS,
            ),
            _feature(
                {"type": "LineString", "coordinates": [list(p) for p in LINE_B]},
                LINE_B_PROPS,
            ),
        ],
    )
    return "lines.geojson"


@pytest.fixture
def points_file(tmp_path):
    """Five points inside the unit, each with a StartTime in seconds."""
    _write_geojson(
        tmp_path / "points.geojson",
        [
            _feature(
                {"type": "Point", "coordinates": list(p)},
                {"id": i, "StartTime": t},
            )
            for i, (p, t) in enumerate(zip(POINTS, POINT_START_TIMES), start=1)
        ],
    )
    return "points.geojson"


@pytest.fixture
def expected_ignite_dat(tmp_path):
    """Writes expected_ignite.dat (order mode, 1 m/s) into tmp_path and
    returns its Path. Built only from the geometry constants above."""
    speed_mps = 1.0
    ox, oy = DOMAIN_ORIGIN
    lines = [(LINE_A, LINE_A_PROPS), (LINE_B, LINE_B_PROPS)]

    rows = []
    group_start = 0.0
    for order in sorted({props["fire_order"] for _, props in lines}):
        group_end = group_start
        for coords, props in lines:
            if props["fire_order"] != order:
                continue
            t = group_start
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                t_end = t + math.hypot(x2 - x1, y2 - y1) / speed_mps
                rows.append((x1 - ox, y1 - oy, x2 - ox, y2 - oy, t, t_end))
                t = t_end
            group_end = max(group_end, t)
        group_start = group_end

    header = [
        "igntype=5",
        "&atvlist",
        f"natv={len(rows)}",
        "targettemp=1000.0",
        "flamedistance=4.00",
        "/",
    ]
    body = [" ".join(f"{v:.2f}" for v in row) for row in rows]

    path = tmp_path / "expected_ignite.dat"
    path.write_text("\n".join(header + body) + "\n")
    return path
