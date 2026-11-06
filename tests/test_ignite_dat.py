"""
Test module for ignite dat files module of quicfire_tools package
"""
import pytest
from quicfire_tools.ignite_dat_files import (
    geojson_lines_to_dat,
    geojson_points_to_dat,
    delay_dat_times,
)


def _build_geojson_order(tmp_path):
    geojson_lines_to_dat(
        tmp_path,
        "domain.geojson",
        "lines.geojson",
        "ignite.dat",
        order_time="order",
        speed_mps=1,
    )
    return tmp_path / "ignite.dat"


def _build_geojson_time(tmp_path):
    geojson_lines_to_dat(
        tmp_path,
        "domain.geojson",
        "lines.geojson",
        "ignite.dat",
        order_time="time",
        start_t_name="TimeStart",
        end_t_name="TimeEnd",
    )
    return tmp_path / "ignite.dat"


def _build_geojson_points(tmp_path):
    geojson_points_to_dat(
        tmp_path,
        "domain.geojson",
        "points.geojson",
        "ignite.dat",
        line_length_m=1.0,
        line_speed_mps=0.5,
        start_t_name="StartTime",
    )
    return tmp_path / "ignite.dat"


def _build_delay_ignite(tmp_path):
    delay_dat_times(
        tmp_path,
        "expected_ignite.dat",
        20,
        "ignite.dat",
    )
    return tmp_path / "ignite.dat"


BUILDERS = [
    _build_geojson_order,
    _build_geojson_time,
    _build_geojson_points,
    _build_delay_ignite,
]  #


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_output_matches_dat_format(
    builder,
    tmp_path,
    domain_file,
    lines_file,
    points_file,
    expected_ignite_dat,
    assert_valid_ignite_dat,
):
    path = builder(tmp_path)
    assert_valid_ignite_dat(path)


# only checking lines. haven't made example for points
def test_lines_time_output_matches_expected(
    tmp_path, domain_file, lines_file, expected_ignite_dat
):
    geojson_lines_to_dat(
        tmp_path,
        domain_file,
        lines_file,
        "ignite.dat",
        order_time="time",
        speed_mps=1,
        start_t_name="TimeStart",
        end_t_name="TimeEnd",
    )
    actual = (tmp_path / "ignite.dat").read_text().splitlines()
    expected = expected_ignite_dat.read_text().splitlines()
    # Same number of lines (header + data rows)
    assert len(actual) == len(
        expected
    ), f"expected {len(expected)} lines, got {len(actual)}"
    # Header (first 6 lines) must match exactly, including natv
    assert actual[:6] == expected[:6]
    # Data rows: compare numerically, so tiny float differences don't fail
    for i, (a_row, e_row) in enumerate(zip(actual[6:], expected[6:]), start=7):
        a_vals = [float(v) for v in a_row.split()]
        e_vals = [float(v) for v in e_row.split()]
        assert len(a_vals) == 6, f"line {i}: expected 6 columns, got {len(a_vals)}"
        assert a_vals == pytest.approx(
            e_vals, abs=0.01
        ), f"line {i}: got {a_row!r}, expected {e_row!r}"


def test_lines_order_output_matches_expected(
    tmp_path, domain_file, lines_file, expected_ignite_dat
):
    geojson_lines_to_dat(
        tmp_path,
        domain_file,
        lines_file,
        "ignite.dat",
        order_time="order",
        speed_mps=1,
    )
    actual = (tmp_path / "ignite.dat").read_text().splitlines()
    expected = expected_ignite_dat.read_text().splitlines()
    # Same number of lines (header + data rows)
    assert len(actual) == len(
        expected
    ), f"expected {len(expected)} lines, got {len(actual)}"
    # Header (first 6 lines) must match exactly, including natv
    assert actual[:6] == expected[:6]
    # Data rows: compare numerically, so tiny float differences don't fail
    for i, (a_row, e_row) in enumerate(zip(actual[6:], expected[6:]), start=7):
        a_vals = [float(v) for v in a_row.split()]
        e_vals = [float(v) for v in e_row.split()]
        assert len(a_vals) == 6, f"line {i}: expected 6 columns, got {len(a_vals)}"
        assert a_vals == pytest.approx(
            e_vals, abs=0.01
        ), f"line {i}: got {a_row!r}, expected {e_row!r}"
