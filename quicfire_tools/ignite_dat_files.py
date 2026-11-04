"""
QUIC-Fire Tools ignite.dat files Module to create and visualize QUIC-Fire ignitions
"""

from __future__ import annotations

# Core Imports
import re
import os
import io
import json
import math
from datetime import datetime
from pathlib import Path
from PIL import Image

# External Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from shapely.geometry import shape, Polygon, MultiPolygon

__all__ = [
    "geojson_lines_to_dat",
    "geojson_points_to_dat",
    "geojsons_from_track_csv",
    "delay_dat_times",
    "plot_ignition",
    "plot_ignition_deluxe",
    "animate_geojson_ignition",
]

# ----------------------------------------
# Section 1: Functions to create and modify .dat ignition files
# ----------------------------------------


def geojson_lines_to_dat(
    prj_path: str | Path,
    domain_fname: str,
    geojson_fname: str,
    ign_fname: str,
    order_time: str,
    speed_mps: int | float | None = None,
    start_t_name: str = "TimeStart",
    end_t_name: str = "TimeEnd",
):
    """
    Convert ignition lines in a geojson to a QUIC-Fire .dat ignition file.
    ONLY WORKS WITH METER BASED CRSs

    Each feature in the geojson must be a LineString (single- or multi-vertex).
    Multi-vertex lines are split into consecutive segments in the .dat file

    Ordering/timing is controlled by order_time:

    order_time='order'
        Lines with the same fire_order are treated as simultaneous:
        they all start at the same time (accounting for any set delay).
        The next fire_order group starts after previous group has completely finished.
        This option is likely what you want for a geojson created by hand in a GIS

    order_time='time'
        This option is better for working with data from timestamped tracks.
        Each line starts at its own start timestamp (start_t_name), measured
        relative to the earliest start timestamp in the file (t = 0). Lines are
        independent: a line does not wait for earlier lines to finish. Each
        line's speed is derived from its length and its start/end timestamps,
        so it finishes at its end timestamp (end_t_name)

    REQUIRED attribute table fields (must be present on every feature in geojson)
    -----------------------------------------------------------------------
    fire_order : int   (only when order_time='order')
        Firing sequence number. Features with the same value ignite
        simultaneously. Groups are processed in ascending order.
        Example: three lines with fire_order 1, 1, 2 → lines 1 & 2 light
        together, then line 3 starts after both of the first two have finished.

    OR

    Time start and end   (only when order_time='time')
        Start and end timestamps formatted like '4/18/2025 14:32:32 (UTC)'.
        start_t_name: If using 'time' option for input type, this sets the name of the starting time attribute

        end_t_name: If using 'time' option, this sets the name of the ending time attribute

    OPTIONAL attribute table fields
    -----------------------------------------------------------------------
    order_time='order' only; omit column or leave cell NULL/empty to accept the default

    reverse : bool / int (0 or 1)  [default: 0, FALSE]
        Set to 1 to traverse the line in reverse (end→start) instead of
        digitized order (start→end). Useful if line drawn in the
        wrong direction

    delay : float  [default: 0.0]
        Extra delay in seconds added to this line's start time on top of
        whatever the group's base time already is. Use to stagger lines
        within the same fire_order group

    speed : float  [default: None → uses global speed_mps]
        Travel speed in m/s for this line only. Overrides the global
        speed_mps argument

    Parameters
    ----------
    prj_path : str|Path,
        Path to the project directory containing domain, geojson, and where the ignition .dat file will be saved.
    domain_fname : str
        GeoJSON filename of the domain polygon. Used to establish the local
        coordinate origin (SW corner of its bounding box).
    geojson_fname : str
        GeoJSON filename of the manually digitized ignition lines or track.
    speed_mps : float|int|None
        Default travel speed in meters per second used for all lines unless
        overridden by a per-feature speed attribute ('order' mode).
    ign_fname : str
        name of output file. Include .dat. Saves to prj_path directory
    order_time : str
        'order' to sequence lines by the fire_order attribute (default),
        or 'time' to sequence and time the lines by the timestamp attributes.
    start_t_name : str
        Start timestamp attribute used when order_time='time'. Default 'TimeStart'.
    end_t_name : str
        End timestamp attribute used when order_time='time'. Default 'TimeEnd'.

    Examples
    --------
    from quicfire_tools.ignite_dat_files import geojson_lines_to_dat
    geojson_lines_to_dat(prj_path="file/path/to/project", domain_fname="domain.geojson", geojson_fname="my_ignition_plan.geojson",
    ign_fname="ignite.dat", order_time="order", speed_mps=1)

    geojson_lines_to_dat(prj_path="file/path/to/project", domain_fname="domain.geojson", geojson_fname="ignition_track.geojson",
    ign_fname="ignite.dat", order_time="time", start_t_name="TimeStart", end_t_name="TimeEnd")

    """
    if order_time not in ("order", "time"):
        raise ValueError("order_time must be either 'order' or 'time'")
    if order_time == "order":
        if speed_mps is None:
            raise ValueError(
                "speed_mps must have a value for order_time='order'. If each feature in your geojson has a specified time, speed_mps will not be used but you still need to enter a value"
            )
    if speed_mps is not None:
        if speed_mps <= 0:
            raise ValueError("speed_mps must be greater than 0")
    # Load files and build local coordinate origin from domain SW corner
    with open(os.path.join(prj_path, domain_fname), "r") as f:
        domain_geojson = json.load(f)
    with open(os.path.join(prj_path, geojson_fname), "r") as f:
        lines_geojson = json.load(f)
    crs_lines = lines_geojson["crs"]["properties"]["name"]
    crs_lines = crs_lines.split(":")[-1]
    crs_domain = domain_geojson["crs"]["properties"]["name"]
    crs_domain = crs_domain.split(":")[-1]
    not_valid = (
        "CRS84",
        "4269",
        "6318",
        "4326",
    )  # there are definitely more invalid codes but these are (hopefully) the most common
    if crs_lines != crs_domain:
        raise ValueError(
            "The crs for your domain file and your ignition lines do not match"
        )
    if crs_lines in not_valid:
        raise ValueError(
            f"Your crs ({crs_lines}) is not valid. Please reproject to a meter based system"
        )
    if crs_lines == "5070" or (len(crs_lines) == 5 and crs_lines[0:3] == "326"):
        pass
    else:
        print(
            f"\nWarning: your crs {crs_lines} was not recognized. If it is meter based units, you should be fine but you should probably check\n"
        )
    domain_coords = _extract_all_coords(domain_geojson["features"][0]["geometry"])
    origin_x = min(c[0] for c in domain_coords)
    origin_y = min(c[1] for c in domain_coords)

    def to_local(x, y):
        return x - origin_x, y - origin_y

    # Parse and validate features
    features = lines_geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_fname}.")
    parsed = _parse_features(
        features=features,
        to_local=to_local,
        speed_mps=speed_mps,
        order_time=order_time,
        start_t_name=start_t_name,
        end_t_name=end_t_name,
    )

    all_records = []
    if order_time == "order":
        # Sort by fire_order, then group simultaneous lines
        parsed.sort(key=lambda f: f["fire_order"])
        groups = {}
        for feat in parsed:
            groups.setdefault(feat["fire_order"], []).append(feat)
        # Build .dat group by group
        current_time = 0.0  # base time for the next group
        for order in sorted(groups.keys()):
            group_feats = groups[order]
            group_end_time = current_time

            for feat in group_feats:
                line_start = current_time + feat["start_delay"]
                records, line_end = _vertices_to_records(
                    feat["vertices"], line_start, feat["speed"]
                )
                all_records.extend(records)
                group_end_time = max(group_end_time, line_end)
            # Next group starts only after the slowest line in this group finishes
            current_time = group_end_time
    else:
        # Each line starts at its timestamp relative to the earliest one
        parsed.sort(key=lambda f: f["start_time"])
        t0 = parsed[0]["start_time"]
        for feat in parsed:
            line_start = (feat["start_time"] - t0).total_seconds()
            records, _ = _vertices_to_records(
                feat["vertices"], line_start, feat["speed"]
            )
            all_records.extend(records)
    # Save
    _write_ign_dat(all_records, prj_path, ign_fname)
    print(f"{ign_fname} written to {prj_path}  ({len(all_records)} segment records)")
    return None


def geojson_points_to_dat(
    prj_path: str | Path,
    domain_fname: str,
    geojson_fname: str,
    ign_fname: str,
    line_length_m: int | float = 1.0,
    line_speed_mps: int | float = 1.0,
    start_t_name: str = "DateTime",
):
    """
    Convert a geojson of timestamped points to a
    QUIC-Fire .dat ignition file. ONLY WORKS WITH METER BASED CRSs

    This function works well for aerial psd types of ignitions. Each point becomes
    a short line (default 1 m) centered on the point and oriented along the direction
    of travel (taken from the neighboring points in time order). Each line starts at
    its point's time relative to the earliest timestamp in the file (t = 0 at the
    first point). delay_dat_times can be used after the ignition dat file is created
    to delay all ignition times by a set amount, if you don't want ignitions starting
    at t = 0.

    REQUIRED attribute table fields
    -----------------------------------------------------------------------
    DateTime : str  (can have a different name set by start_t_name)
        Timestamp formatted like '4/18/2025 14:32:32 (UTC)' or iso format.
        Sets both the order and the start time of each line. Points with
        identical timestamps ignite simultaneously.

    Parameters
    -----------------------------------------------------------------------
    prj_path : str|Path,
        Path to the project directory.
    domain_fname : str
        GeoJSON filename of the domain polygon (sets the local origin at
        the SW corner of its bounding box).
    geojson_fname : str
        GeoJSON filename of the timestamped points. Must be in the same
        meter-based CRS as the domain.
    ign_fname : str
        Name of output file. Include .dat. Saves to prj_path directory.
    line_length_m : float
        Length of the line created at each point. Default 1.0 m.
    line_speed_mps : float
        Speed used to traverse each short line. Each line takes
        line_length_m / line_speed_mps seconds to ignite. Default 1.0 m/s.
    start_t_name : str
        Name of the timestamp attribute. Default 'DateTime'.

    Examples
    --------
    from quicfire_tools.ignite_dat_files import geojson_points_to_dat
    geojson_points_to_dat(prj_path="file/path/to/project", domain_fname="domain.geojson", geojson_fname="aerial_ign_track.geojson",
    ign_fname="ignite.dat", line_length_m=1, line_speed_mps=0.2, start_t_name='DateTime')
    """
    if line_length_m <= 0:
        raise ValueError("line_length_m must be greater than 0")
    if line_speed_mps <= 0:
        raise ValueError("line_speed_mps must be greater than 0")

    # Load files and build local coordinate origin from domain SW corner
    with open(os.path.join(prj_path, domain_fname), "r") as f:
        domain_geojson = json.load(f)
    with open(os.path.join(prj_path, geojson_fname), "r") as f:
        points_geojson = json.load(f)
    crs_points = points_geojson["crs"]["properties"]["name"].split(":")[-1]
    crs_domain = domain_geojson["crs"]["properties"]["name"].split(":")[-1]
    not_valid = ("CRS84", "4269", "6318", "4326")
    if crs_points != crs_domain:
        raise ValueError("The crs for your domain file and your points do not match")
    if crs_points in not_valid:
        raise ValueError(
            f"Your crs ({crs_points}) is not valid. Please reproject to a meter based system"
        )
    if crs_points == "5070" or (len(crs_points) == 5 and crs_points[0:3] == "326"):
        pass
    else:
        print(
            f"\nWarning: your crs {crs_points} was not recognized. If it is meter based units, you should be fine but you should probably check\n"
        )
    domain_coords = _extract_all_coords(domain_geojson["features"][0]["geometry"])
    origin_x = min(c[0] for c in domain_coords)
    origin_y = min(c[1] for c in domain_coords)

    def to_local(x, y):
        return x - origin_x, y - origin_y

    # Parse and validate point features
    features = points_geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_fname}.")

    pts = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            raise ValueError(f"Feature {i} is not a Point (got {geom.get('type')}).")
        props = feat.get("properties") or {}
        if props.get(start_t_name) in (None, ""):
            raise ValueError(f"Feature {i} is missing required field '{start_t_name}'.")
        x, y = to_local(*geom["coordinates"][:2])
        pts.append({"t": _parse_dtstr(props[start_t_name]), "x": x, "y": y})

    # Sort by time and convert to seconds since the first point
    pts.sort(key=lambda p: p["t"])
    t0 = pts[0]["t"]
    for p in pts:
        p["sec"] = (p["t"] - t0).total_seconds()

    # Heading at each point from its neighbors (central difference; one-sided at ends)
    n = len(pts)
    headings = []
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        norm = math.hypot(dx, dy)
        headings.append((dx / norm, dy / norm) if norm > 0 else None)

    # Stationary points (no direction): borrow the nearest earlier heading,
    # then back-fill any leading ones; fall back to east if nothing moved at all
    last = next((h for h in headings if h is not None), (1.0, 0.0))
    for i in range(n):
        if headings[i] is None:
            headings[i] = last
        else:
            last = headings[i]

    # Build records: one short line per point, starting at that point's time
    half = line_length_m / 2.0
    all_records = []
    for p, (ux, uy) in zip(pts, headings):
        vertices = [
            (p["x"] - ux * half, p["y"] - uy * half),
            (p["x"] + ux * half, p["y"] + uy * half),
        ]
        records, _ = _vertices_to_records(vertices, p["sec"], line_speed_mps)
        all_records.extend(records)
    _write_ign_dat(all_records, prj_path, ign_fname)
    print(f"{ign_fname} written to {prj_path}  ({len(all_records)} segment records)")
    return None


def delay_dat_times(
    prj_path: str | Path,
    dat_fname: str = "ignite.dat",
    delay_s: float = None,
    out_fname=None,
):
    """
    Add a constant delay (in seconds) to the time columns of a .dat ignition file.
    Each edited value keeps its original number of decimal places.

    Parameters
    ----------
    prj_path : str|Path,
        Path to the project directory containing ignition dat file
    dat_fname : str
        Name of the .dat file to read.
    delay_s : float
        Seconds to add to every time value (e.g. 100).
    out_fname : str or None
        Output filename. None overwrites the original file in place.

    Examples
    --------
    from quicfire_tools.ignite_dat_files import delay_dat_times
    delay_dat_times(prj_path="path/to/project", dat_fname="ignite.dat", delay_s=100)
    """
    if delay_s is None:
        raise ValueError("you must set a value for delay_s")
    time_cols = (4, 5)  # assumes standard quicfire ignite.dat formatting
    in_path = os.path.join(prj_path, dat_fname)
    out_path = os.path.join(prj_path, out_fname) if out_fname else in_path
    with open(in_path, "r") as f:
        lines = f.read().splitlines()
    new_lines = []
    n_edited = 0
    first_example = None

    for line in lines:
        # Split but keep separators so spacing preserved
        col_values = re.split(r"(\s+|,)", line)  # spaces as columns
        # Fields are the even positions (odd positions are separators)
        fields = [t for i, t in enumerate(col_values) if i % 2 == 0 and t != ""]
        is_data_row = (
            len(fields) == 6
        )  # number of columns in data row -- header formatted differently
        if is_data_row:
            try:
                [float(t) for t in fields]
            except ValueError:
                is_data_row = False
        if not is_data_row:
            new_lines.append(line)
            continue
        # Edit columns in place
        field_idx = -1
        for i, col in enumerate(col_values):
            if i % 2 == 1 or col == "":
                continue  # skip the spaces
            field_idx += 1  # spaces as columns are not counted in fields
            if field_idx in time_cols:
                value = float(col) + delay_s
                if "." in col:
                    # Keep og number of decimal places
                    decimals = len(col.split(".")[1])
                    col_values[i] = f"{value:.{decimals}f}"
                elif float(delay_s).is_integer():
                    col_values[i] = str(int(value))
                else:
                    col_values[i] = f"{value:g}"
        new_line = "".join(col_values)
        if first_example is None:
            first_example = (line, new_line)
        new_lines.append(new_line)
        n_edited += 1

    if n_edited == 0:
        raise ValueError("No data rows found?? Check yer file")
    with open(out_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    return None


def geojsons_from_track_csv(
    prj_path: str | Path,
    csv_fname: str,
    input_time: str,
    input_y: str,
    input_x: str,
    epsg: int,
    resample_interval: int | float,
):
    """
    Convert a high frequency gps track, saved as a csv, to a resampled set of points. From those points create a geojson of linestrings
    following the track (useful for blacklining, or other continuous ignitions) and a geojson of just the resampled
    points (useful for aerial/hand ignition with known drop frequency). Both the linestrings and the points geojsons
    can be converted to QUIC-Fire ignition files using geojson_lines_to_dat and geojson_points_to_dat

    New files created (3):
    resampled_track.csv is a csv of timestamps and coordinates at the resample interval.
    resampled_lines.geojson is linestrings connecting all the resampled coordinates with start and end times.
    resampled_points.geojson is the resampled coordinates with start times.

    Parameters
    ----------
    prj_path : str|Path
        Path to the project directory containing the csv file, and it is where the resampled csv and new geojson files will be saved
    csv_fname : str
        full file path to a csv with columns for coordinate_y, coordinate_x, and timestamp
    input_time : str
        name of timestamp column -- will be used in output files
    input_y : str
        name of latitude/northing/y column
    input_x : str
        name of longitude/easting/x column
    epsg : int
        epsg code of the coordinates
    resample_interval : int|float
        number of seconds between resampled points

    Examples
    --------
    from quicfire_tools.ignite_dat_files import geojsons_from_track_csv
    geojsons_from_track_csv(prj_path="path/to/project", csv_fname="heli_track.csv", input_time="DateTime", input_y="Northing", input_x="Easting", epsg=32617, resample_interval=10)
    """
    prj_path = Path(prj_path)
    csv_path = os.path.join(prj_path, csv_fname)
    df, time_col, lat_col, lon_col, is_datetime = _load_and_prepare(
        csv_path, input_time, input_y, input_x
    )
    resampled = _resample(df, lat_col, lon_col, resample_interval)
    t0 = df["_t_display"].iloc[0] if is_datetime else None

    out_csv_path = os.path.join(prj_path, "resampled_track.csv")
    out_lines_path = os.path.join(prj_path, "resampled_lines.geojson")
    out_points_path = os.path.join(prj_path, "resampled_points.geojson")
    # save csv
    csv_out = resampled.copy()
    if is_datetime:
        csv_out.insert(
            0, time_col, t0 + pd.to_timedelta(csv_out["_t_seconds"], unit="s")
        )
    else:
        csv_out.insert(0, time_col, csv_out["_t_seconds"])
    csv_out = csv_out.drop(columns=["_t_seconds"])
    out_path = Path(out_csv_path)
    csv_out.to_csv(out_path, index=False)
    # save linestring geojson
    if len(csv_out) < 2:
        raise ValueError("Need at least 2 resampled points to build line segments.")
    times = csv_out[time_col].tolist()
    lons = csv_out[lon_col].to_numpy()
    lats = csv_out[lat_col].to_numpy()
    features = []
    for i in range(len(csv_out) - 1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "segment": i,
                    f"{input_time}": _format_timestamp(
                        times[i]
                    ),  # keep time column names consistent
                    "TimeEnd": _format_timestamp(times[i + 1]),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lons[i], lats[i]], [lons[i + 1], lats[i + 1]]],
                },
            }
        )

    line_geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        },
        "features": features,
    }
    with open(out_lines_path, "w") as f:
        json.dump(line_geojson, f)
    # save resampled points as points geojson
    csv_out = csv_out.rename(
        columns={time_col: input_time}
    )  # keep time column names consistent
    features = []
    for _, row in csv_out.iterrows():
        props = {
            c: (row[c] if not isinstance(row[c], pd.Timestamp) else row[c].isoformat())
            for c in csv_out.columns
            if c not in (lat_col, lon_col)
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [row[lon_col], row[lat_col]],
                },
            }
        )
    points_geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        },
        "features": features,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_points_path, "w") as f:
        json.dump(points_geojson, f)

    print(f"Original points: {len(df)}")
    print(f"Resampled points: {len(resampled)} (every {resample_interval}s)")


def _extract_all_coords(
    geometry,
):  # also used in creating the dat files (geojson to dat)
    """Flat list of all [x, y] pairs from a GeoJSON geometry."""
    gtype = geometry["type"]
    if gtype == "Polygon":
        return [c for ring in geometry["coordinates"] for c in ring]
    elif gtype == "MultiPolygon":
        return [c for poly in geometry["coordinates"] for ring in poly for c in ring]
    elif gtype in ("LineString", "MultiPoint"):
        return geometry["coordinates"]
    elif gtype == "MultiLineString":
        return [c for line in geometry["coordinates"] for c in line]
    elif gtype == "Point":
        return [geometry["coordinates"]]
    else:
        raise ValueError(f"Unsupported domain geometry type: {gtype}")


def _parse_dtstr(timestring):
    """Parse strings like '4/18/2025 14:32:32 (UTC)' (fractional seconds optional)."""
    try:
        timestamp = datetime.fromisoformat(timestring)
        return timestamp
    except ValueError:
        timestring = str(timestring).replace("(UTC)", "").strip()
        for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(timestring, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse datetime string: '{timestring}'")


def _parse_features(
    features,
    to_local,
    speed_mps,
    order_time,
    start_t_name="DateTimeS",
    end_t_name="TimeEnd",
):
    """
    Minimal feature parser

    order_time='order' : reads fire_order, reverse, speed, delay
    order_time='time'  : reads start_t_name and end_t_name timestamps;
                         per-segment speed is derived as length / (end - start).
                         reverse, delay, speed and fire_order are ignored.

    Each parsed dict has: vertices, speed, start_delay, plus
    fire_order (order mode) or start_time (time mode).
    """
    parsed = []
    for idx, feat in enumerate(features):
        props = feat.get("properties") or {}
        geom = feat.get("geometry", {})
        geom_type = geom.get("type")

        # geometry - same for time and order
        if geom_type == "LineString":
            parts = [geom["coordinates"]]
        elif geom_type == "MultiLineString":
            parts = geom["coordinates"]
        else:
            raise ValueError(f"Feature {idx}: unsupported geometry type {geom_type!r}")
        if not parts or any(len(p) < 2 for p in parts):
            raise ValueError(
                f"Feature {idx}: every part must have at least 2 vertices."
            )

        raw_verts = list(parts[0])
        for part in parts[1:]:
            if raw_verts[-1] == part[0]:
                raw_verts.extend(part[1:])
            else:
                raw_verts.extend(part)

        local_verts = [to_local(c[0], c[1]) for c in raw_verts]
        # attributes vary by input file type -- linestrings from a GIS or a GPS track
        if order_time == "order":
            try:
                fire_order = int(
                    props.get("fire_order", idx + 1)
                )  # default kicks in if key is not in geojson (props)
            except TypeError:  # covers if key is in geojson but value None
                fire_order = idx + 1
            try:
                reverse = bool(int(props.get("reverse", 0)))
            except TypeError:
                reverse = False
            try:
                speed = float(
                    props.get("speed") or speed_mps
                )  # originally speed_override but too long for easy attribute table add in QGIS
                if speed <= 0:
                    raise ValueError(
                        "speed in line properties/attribute table must be greater than 0"
                    )  # speed_mps already checked for >0
            except TypeError:
                speed = speed_mps
            try:
                start_delay = float(
                    props.get("delay") or 0
                )  # originally start_delay was name in table/properties but too long for easy attribute table add in QGIS
            except TypeError:
                start_delay = 0
            if reverse:
                local_verts = local_verts[::-1]
            parsed.append(
                {
                    "fire_order": fire_order,
                    "vertices": local_verts,
                    "speed": speed,
                    "start_delay": start_delay,
                }
            )
        else:  # order_time == 'time'
            start_raw = props.get(start_t_name)
            end_raw = props.get(end_t_name)
            if start_raw in (None, "") or end_raw in (None, ""):
                raise ValueError(
                    f"Feature {idx}: both '{start_t_name}' and '{end_t_name}' are required when order_time='time'"
                )
            t_start = _parse_dtstr(start_raw)
            t_end = _parse_dtstr(end_raw)
            duration = (t_end - t_start).total_seconds()
            if duration <= 0:
                raise ValueError(
                    f"Feature {idx}: '{end_t_name}' must be later than '{start_t_name}' (got {start_raw!r} and {end_raw!r}).\n"
                    f"Timestamps with whole-second resolution can have issues if the resample interval is under 1 s."
                )
            length = sum(
                np.hypot(ex - sx, ey - sy)
                for (sx, sy), (ex, ey) in zip(local_verts[:-1], local_verts[1:])
            )
            # Zero-length segment (stationary GPS): speed is irrelevant (dist = 0),
            # but must be > 0 to avoid dividing by zero downstream.
            speed = length / duration if length > 0 else speed_mps
            parsed.append(
                {
                    "start_time": t_start,
                    "vertices": local_verts,
                    "speed": speed,
                    "start_delay": 0.0,  # never a dealy, using the timestamps as they stand
                }
            )
    return parsed


def _write_ign_dat(records, prj_path, ign_fname):
    """Write records to a QUIC-Fire ignition .dat file."""
    out_path = os.path.join(prj_path, ign_fname)
    with open(out_path, "w") as f:
        f.write("igntype=5\n")
        f.write("&atvlist\n")
        f.write(f"natv={len(records)}\n")
        f.write("targettemp=1000.0\n")
        f.write("flamedistance=4.00\n")
        f.write("/\n")
        for r in records:
            f.write(
                f"{r['start_x']:.6f} {r['start_y']:.6f} "
                f"{r['end_x']:.6f} {r['end_y']:.6f} "
                f"{r['start_time']:.6f} {r['end_time']:.6f}\n"
            )


def _vertices_to_records(vertices, base_time, speed_mps):
    """
    Convert an ordered list of (x, y) vertices into consecutive .dat segment
    records, chaining timing so the line is traversed as one continuous walk.

    Parameters
    ----------
    vertices : list of (float, float)
        Local-coordinate vertices in traversal order.
    base_time : float
        Time (seconds) at which traversal of the first vertex begins.
    speed_mps : float
        Travel speed in m/s for this line.

    Returns
    -------
    records : list of dict
    end_time : float
        Time at which the last vertex is reached.
    """
    records = []
    t = base_time
    for i in range(len(vertices) - 1):
        sx, sy = vertices[i]
        ex, ey = vertices[i + 1]
        dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        dt = dist / speed_mps
        records.append(
            {
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "start_time": t,
                "end_time": t + dt,
            }
        )
        t += dt
    return records, t


def _load_and_prepare(csv_path, time_col=None, lat_col=None, lon_col=None):
    """
    Load the CSV, and add
    two helper columns:
      _t_seconds  - numeric elapsed seconds from the first row (used for
                    interpolation)
      _t_display  - the value to show back to the user (a real datetime
                    if the input was timestamps, otherwise the same
                    elapsed seconds)
    Returns (df, time_col, lat_col, lon_col, is_datetime).
    """
    df = pd.read_csv(csv_path)
    missing = [
        name
        for name, val in [("time", time_col), ("lat", lat_col), ("lon", lon_col)]
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Could not auto-detect column(s): {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Set TIME_COL/LAT_COL/LON_COL explicitly."
        )
    # Decide whether the time column is real timestamps (text/datetime) or
    # elapsed seconds. Check dtype/content FIRST, b/c pd.to_datetime will
    # reinterpret plain floats as nanosecond-since-epoch offsets
    is_datetime = False
    raw_col = df[time_col]
    looks_numeric = pd.api.types.is_numeric_dtype(raw_col) or (
        raw_col.astype(str).str.match(r"^-?\d+(\.\d+)?$").all()
    )
    if looks_numeric:
        df["_t_seconds"] = pd.to_numeric(raw_col, errors="raise")
        df["_t_display"] = df["_t_seconds"]
    else:
        try:
            parsed = pd.to_datetime(raw_col)
            if parsed.notna().all():
                df["_t_seconds"] = (parsed - parsed.iloc[0]).dt.total_seconds()
                df["_t_display"] = parsed
                is_datetime = True
            else:
                raise ValueError
        except Exception:
            df["_t_seconds"] = pd.to_numeric(raw_col, errors="raise")
            df["_t_display"] = df["_t_seconds"]

    df = df.sort_values("_t_seconds").reset_index(drop=True)
    # Drop exact-duplicate timestamps (interpolation needs strictly increasing x)
    df = df.drop_duplicates(subset="_t_seconds", keep="first").reset_index(drop=True)
    return df, time_col, lat_col, lon_col, is_datetime


def _resample(df, lat_col, lon_col, interval):
    """
    Linearly interpolate lat/lon  at a fixed time
    interval, from the first to the last timestamp in df.

    Returns a DataFrame with columns: _t_seconds, lat_col, lon_col, *extra_cols
    """
    t = df["_t_seconds"].to_numpy()
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    t_start, t_end = t[0], t[-1]
    n_steps = int(np.floor((t_end - t_start) / interval))
    target_t = t_start + np.arange(n_steps + 1) * interval
    target_t = target_t[
        target_t <= t_end
    ]  # make sure not to interpolate slightly past the last sample
    out = pd.DataFrame({"_t_seconds": target_t})
    out[lat_col] = np.interp(target_t, t, lat)
    out[lon_col] = np.interp(target_t, t, lon)
    return out


def _format_timestamp(t):
    """
    Format a pandas Timestamp like '4/18/2025 14:32:32 (UTC)'.
    Fractional seconds are appended only when nonzero (e.g. '14:32:32.5 (UTC)'),
    which _parse_dtstr can read back. Non-timestamps (e.g. elapsed seconds)
    are returned unchanged.
    """
    if not isinstance(t, pd.Timestamp):
        return t
    s = f"{t.month}/{t.day}/{t.year} {t:%H:%M:%S}"
    if t.microsecond:
        s += f".{t.microsecond:06d}".rstrip("0")
    return s + " (UTC)"


# ----------------------------------------
# Section 2: Functions to visualize .dat and .geojson ignition files
# ----------------------------------------


def plot_ignition(
    prj_path: str,
    domain_fname: str,
    ign_fname: str,
    unit_fname: str | None = None,
    out_fname: str | None = None,
):
    """
    Plot domain and burn unit polygons and ignition lines from .dat file.
    Serves as a quick sanity check. Optional to save the plot.

    Parameters:
    -----------
    prj_path : str|Path,
        path to project directory containing domain, ignition, and optional unit files
    domain_fname : str
        file name of the domain geojson (reference polygon)
    ign_fname : str
        filename of the ignition .dat file
    unit_fname : str|None
        file name of the burn unit (polygon or multipolygon). WIll be saved in prj_path. optional - default None will not plot unit boundary
    out_fname : str or None
            Output image filename (include .png/.jpg/etc). optional - default None displays but does not save the plot

    Examples
    --------
    from quicfire_tools.ignite_dat_files import plot_ignition
    plot_ignition(prj_path="path/to/project", domain_fname="domain.geojson", ign_fname="ignite.dat", unit_fname="unit.geojson")
    """
    # Load and transform polygons to local coords
    with open(os.path.join(prj_path, domain_fname), "r") as f:
        geojson_a = json.load(f)
    polygon_a = shape(geojson_a["features"][0]["geometry"])
    bounds_a = polygon_a.bounds
    origin_x, origin_y = bounds_a[0], bounds_a[1]
    fig, ax = plt.subplots(figsize=(10, 10))

    def to_local(coords):
        return [(x - origin_x, y - origin_y) for x, y in coords]

    # Plot domain polygon
    if isinstance(polygon_a, Polygon):
        domain_polys = [polygon_a]
    elif isinstance(polygon_a, MultiPolygon):
        domain_polys = list(polygon_a.geoms)
    else:
        raise ValueError("is your domain a polygon or multipolygon??")
    for poly in domain_polys:
        local_a = to_local(poly.exterior.coords)
        xs, ys = zip(*local_a)
        ax.plot(xs, ys, color="blue", linewidth=1.5, label="Simulation Domain")

    # Plot burn unit
    if unit_fname is not None:
        with open(os.path.join(prj_path, unit_fname), "r") as f:
            geojson_b = json.load(f)
        geometry_b = shape(geojson_b["features"][0]["geometry"])
        if isinstance(geometry_b, Polygon):
            burn_polys = [geometry_b]
        elif isinstance(geometry_b, MultiPolygon):
            burn_polys = list(geometry_b.geoms)
        else:
            raise ValueError("is your unit a polygon or multipolygon??")
        for i, poly in enumerate(burn_polys):
            local_b = to_local(poly.exterior.coords)
            xs, ys = zip(*local_b)
            ax.plot(
                xs,
                ys,
                color="black",
                linewidth=1.5,
                label="Burn Unit" if i == 0 else None,
            )
            for interior in poly.interiors:
                local_i = to_local(interior.coords)
                xs, ys = zip(*local_i)
                ax.plot(xs, ys, color="black", linewidth=1.0)

    # Parse and plot ignition lines
    ignition_segments = []
    end_time = 0
    with open(os.path.join(prj_path, ign_fname), "r") as f:
        for line in f:
            line = line.strip()
            # Skip header lines
            try:
                parts = [float(v) for v in line.split()]
                if len(parts) == 6:
                    sx, sy, ex, ey, _, end_t = parts
                    end_time = max(
                        end_time, end_t
                    )  # overwrite end_time to get max of full ignition plan
                    ignition_segments.append([(sx, sy), (ex, ey)])
            except ValueError:
                continue

    lc = LineCollection(
        ignition_segments, colors="red", linewidth=0.8, label="Ignition lines"
    )
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_xlabel("East/West (m)")
    ax.set_ylabel("North/South (m)")
    ax.set_title(f"Ignition Plan. Time: {round(end_time / 60, 2)} minutes")
    ax.legend(
        handles=[
            mpatches.Patch(color="black", label="Burn Unit"),
            mpatches.Patch(color="red", label="Ignition Lines"),
            mpatches.Patch(color="blue", label="Domain"),
        ]
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    if out_fname is None:
        return print(f"Done. \nPlot not saved to file")
    out_path = os.path.join(prj_path, out_fname)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Done. \nPlot saved to {out_path}")
    return None


def plot_ignition_deluxe(
    prj_path: str | Path,
    ign_fname,
    out_fname=None,
    unit_fname: str | None = None,
    domain_fname: str | None = None,
    plot_dir: bool = True,
):
    """
    Read a QUIC-Fire ignition .dat file and produce a 2D plot. Optional to save plot

    Each fire_order group, is drawn in a distinct color. Direction of
    travel is shown with arrowheads at the end of each line.
    A circle is placed at the start of the line

    Fire_order groups are inferred from timing: lines whose start_time
    occurs during previous line's time range are considered simultaneous and assigned to the
    same group. This coloring nicely highlights burn groups in strip-firing and similar
    situations. It may not show much useful information in other scenarios.

    Parameters
    ----------
    prj_path : str|Path,
        path to project directory containing ignition file, domain (optional), unit (optional)
    ign_fname : str
        filename of the ignition .dat file
    out_fname : str or None
        Output image filename (include .png/.jpg/etc). Will be saved in prj_path. optional - default None displays but does not save the plot
    unit_fname : str|None
        file name of the burn unit (polygon or multipolygon). optional - default None will not plot unit boundary
    domain_fname : str
        file name of the domain geojson (reference polygon). optional but must be used if plotting unit boundary
    plot_dir : bool
        if True, plots direction arrows on the lines. default True

    Examples
    --------
    from quicfire_tools.ignite_dat_files import plot_ignition_deluxe
    plot_ignition_deluxe(prj_path="path/to/project", ign_fname="ignite.dat", unit_fname="unit.geojson", domain_fname="domain.geojson")
    """
    if unit_fname is not None:
        if domain_fname is None:
            print(
                "WARNING: you must list a domain file name if you wish to plot the unit boundary \n\tThe domain is used to build the local coordinate system \n\tPlotting without the unit boundary"
            )
            unit_fname = None
    # Read .dat file -- skip header lines
    print("Reading .dat file...", end="")
    records = _read_ign_dat(os.path.join(prj_path, ign_fname))  # list of dictionaries
    if not records:
        raise ValueError(f"No ignition records found in {ign_fname}.")
    print("Done. \nProcessing line segments into continuous lines...", end="")
    records = sorted(records, key=lambda r: r["start_time"])

    # chain segments into continuous lines
    # A segment belongs to the current line if its start point matches
    # the previous segment's end point (within floating point tolerance).
    # assumes that line segments from a single line are listed in order in the .dat file
    lines = []  # each entry: list of record indices (line segments) forming one line
    for i, rec in enumerate(records):
        placed = False
        for line in lines:
            last = records[line[-1]]
            if np.isclose(last["end_x"], rec["start_x"], atol=1e-4) and np.isclose(
                last["end_y"], rec["start_y"], atol=1e-4
            ):
                line.append(i)
                placed = True
                break
        if not placed:
            lines.append([i])  # new line
    # first and last segments of lines -- used for plotting
    ends = [[segment[0], segment[-1]] for segment in lines]
    start_segs = [
        a[0] for a in ends
    ]  # record indexes for line segments that start a line
    end_segs = [a[1] for a in ends]

    # time window for each line (start of first seg → end of last for each record)
    line_windows = []
    final_t = 0
    for line in lines:
        t_start = records[line[0]]["start_time"]
        t_end = max(records[j]["end_time"] for j in line)
        line_windows.append((t_start, t_end))
        final_t = max(final_t, t_end)

    # group lines by overlapping time windows
    line_groups = []  # each entry: list of lines that overlap in time
    for li, (t0, t1) in enumerate(
        line_windows
    ):  # for each line group, get the start and end times
        placed = False
        for grp in line_groups:  # for each group already added
            for lj in grp:  # for each line already added to each group
                s0, s1 = line_windows[lj]  # get start and end
                if (
                    t0 < s1 and s0 < t1
                ):  # if current line times overlap, add it to the group
                    grp.append(li)
                    placed = True
                    break
            # if placed:
            #    break
        if (
            not placed
        ):  # if time didn't fit in an existing group, start a new group (or first time through start a new group)
            line_groups.append([li])

    # Map every line from .dat to group number (groups indexed to 1)
    group_num = {}  # line segment index from record dict : group number
    for g_idx, grp in enumerate(line_groups):  # for each group of lines
        for li in grp:  # for each line
            for rec_idx in lines[li]:  # for each segment within the line
                group_num[rec_idx] = g_idx + 1  # set group number
    print("Done. \nBeginning plotting...", end="")
    # Plot
    n_groups = len(line_groups)
    palette = _group_palette(n_groups)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xlabel("East/West (m)", color="white", fontsize=9)
    ax.set_ylabel("North/South (m)", color="white", fontsize=9)
    if final_t > 120:
        final_t = round(final_t / 60, 2)
        ax.set_title(
            f"Ignition lines by firing group. Total time: {final_t} min",
            color="white",
            fontsize=10,
        )
    else:
        ax.set_title(
            f"Ignition lines by firing group. Total time: {round(final_t, 2)} s",
            color="white",
            fontsize=10,
        )

    # Plot burn unit (handles Polygon and MultiPolygon)
    if unit_fname is not None:
        # build local coord sys
        with open(os.path.join(prj_path, domain_fname), "r") as f:
            geojson_a = json.load(f)
        polygon_a = shape(geojson_a["features"][0]["geometry"])
        bounds_a = polygon_a.bounds
        origin_x, origin_y = bounds_a[0], bounds_a[1]

        def to_local(coords):
            return [(x - origin_x, y - origin_y) for x, y in coords]

        with open(os.path.join(prj_path, unit_fname), "r") as f:
            geojson_b = json.load(f)
        geometry_b = shape(geojson_b["features"][0]["geometry"])
        if isinstance(geometry_b, Polygon):
            burn_polys = [geometry_b]
        elif isinstance(geometry_b, MultiPolygon):
            burn_polys = list(geometry_b.geoms)
        else:
            raise ValueError("is your unit a polygon or multipolygon??")
        for i, poly in enumerate(burn_polys):
            local_b = to_local(poly.exterior.coords)
            xs, ys = zip(*local_b)
            ax.plot(
                xs,
                ys,
                color="#555555",
                linewidth=1.5,
                label="Burn Unit" if i == 0 else None,
            )
            for interior in poly.interiors:
                local_i = to_local(interior.coords)
                xs, ys = zip(*local_i)
                ax.plot(xs, ys, color="black", linewidth=1.0)

    drawn_groups = set()
    for i, rec in enumerate(records):  # for each line segment from .dat
        g = group_num[i]
        col = palette[g - 1]
        sx, sy = rec["start_x"], rec["start_y"]
        ex, ey = rec["end_x"], rec["end_y"]
        label = (
            f"Group {g}" if g not in drawn_groups else None
        )  # if already added, don't add label again
        if (
            np.linalg.norm(np.array([sx, sy]) - np.array([ex, ey])) < 4
        ):  # 4 chosen somewhat arbitrarily as short enough line to represent as a point
            ax.scatter(sx, sy, color=col, s=1, zorder=2)
        else:
            ax.plot([sx, ex], [sy, ey], color=col, linewidth=2, label=label, zorder=2)
            if plot_dir:
                # Start dot on the very first segment of each continuous line and arrow at the end
                if i in start_segs:
                    ax.plot(sx, sy, "o", color=col, markersize=5, zorder=4)
                if i in end_segs:
                    ax.annotate(
                        "",
                        xy=(ex, ey),
                        xytext=(sx, sy - 1),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2),
                        zorder=3,
                    )
        drawn_groups.add(g)
        # print(f"completed drawing line {i}/{len(records)-1}")
    if len(drawn_groups) <= 20:
        ax.legend(
            loc="upper right",
            facecolor="#2a2a2a",
            edgecolor="#555555",
            labelcolor="white",
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()
    if out_fname is None:
        print(f"Done. \nPlot not saved to file")
        return None
    out_path = os.path.join(prj_path, out_fname)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Done. \nPlot saved to {out_path}")
    return None


def animate_geojson_ignition(
    prj_path: str | Path,
    domain_fname: str,
    unit_fname: str | None,
    lines_fname: str,
    speed_mps: int | float,
    gif_fname: str,
    seconds_per_frame: int = 20,
    fps: int = 12,
):
    """
    Produce a gif of an ignition sequence saved in a geojson file. Geojson must be line/linestrings with attribute table/properties that
    includes: fire_order (for groups) and optional delay (delay in s after first start in group), reverse (1 will ignite backwards of
    how it was drawn), speed (m/s, if a value is entered it will override speed_m/s parameter for that line only)

    Fire-order groups animate sequentially: group 2 does not start until
    every line in group 1 has finished drawing.

    This function makes it easy to visually assess timing for hand-digitized ignition plans. Particularly useful for situations like
    strip ignitions with staggered start times within a burn group

    Parameters
    ----------
    prj_path : str|Path,
        Project directory.
    domain_fname : str
        file name of the domain polygon (sets coordinate origin).
    unit_fname : str
        file name of the unit polygon. Optional -- used only for plotting boundary
    lines_fname : str
        file name of the ignition lines (must have fire_order attribute).
    speed_mps : float
        Default travel speed m/s (used when speed is absent from attribute table/properties).
    gif_fname : str
        Output GIF filename written into prj_path.
    seconds_per_frame : float
        How many seconds pass between frames (default 20).
        Lower = smoother but larger file; higher = faster but choppier.
    fps : int
        Frames per second of the output GIF (default 12).

    Examples
    --------
    from quicfire_tools.ignite_dat_files import animate_geojson_ignition
    animate_geojson_ignition(prj_path="path/to/project", domain_fname="domain.geojson", unit_fname="unit.geojson",
    lines_fname="my_ignition_plan.geojson", speed_mps=0.5, gif_fname="plan_animation.gif", seconds_per_frame=15)

    """
    if speed_mps <= 0:
        raise ValueError("speed_mps must be greater than 0")
    # read in files
    with open(os.path.join(prj_path, domain_fname)) as f:
        domain_geojson = json.load(f)
    with open(os.path.join(prj_path, lines_fname)) as f:
        lines_geojson = json.load(f)
    if unit_fname is not None:
        with open(os.path.join(prj_path, unit_fname)) as f:
            unit_geojson = json.load(f)
    else:
        unit_geojson = None
    # get coords for domain and lines
    domain_coords = _extract_all_coords(
        domain_geojson["features"][0]["geometry"]
    )  # list
    origin_x = min(c[0] for c in domain_coords)
    origin_y = min(c[1] for c in domain_coords)
    domain_x = [
        c[0] - origin_x for c in domain_coords
    ]  # list of corners in local (m from sw corner)
    domain_y = [c[1] - origin_y for c in domain_coords]

    def to_local(x, y):  # define in place here for access to origin_x and _y
        return x - origin_x, y - origin_y

    features = lines_geojson.get("features", [])
    parsed = _parse_features(
        features=features, to_local=to_local, speed_mps=speed_mps, order_time="order"
    )  # lines to dictionary of lines with order and speed
    groups = {}
    for feat in parsed:
        groups.setdefault(feat["fire_order"], []).append(feat)
    sorted_orders = sorted(groups.keys())
    # Figure
    # set up
    palette = _group_palette(len(sorted_orders))  # color selection
    order_to_color = {o: palette[i] for i, o in enumerate(sorted_orders)}
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    # Domain outline
    ax.plot(
        domain_x + [domain_x[0]],
        domain_y + [domain_y[0]],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )
    if unit_geojson is not None:
        unit_coords = _extract_all_coords(unit_geojson["features"][0]["geometry"])
        unit_coords_local = [to_local(ux, uy) for ux, uy in unit_coords]
        unit_x = [c[0] for c in unit_coords_local]
        unit_y = [c[1] for c in unit_coords_local]
        ax.plot(
            unit_x,
            unit_y,
            label="Burn Unit",
            color="#EBDFED",
            linewidth=0.8,
            linestyle="-",
            zorder=1,
        )
    # Legend
    legend_patches = [
        mpatches.Patch(color=order_to_color[o], label=f"Group {o}")
        for o in sorted_orders
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        facecolor="#2a2a2a",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=7,
    )
    title_text = ax.set_title("Ignition sequence", color="white", fontsize=9)

    # Frames
    frames = []
    completed_line_objects = []  # matplotlib Line2D objects for faded completed lines

    def capture_frame(title_str):
        title_text.set_text(title_str)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    total_elapsed = 0.0
    for group_idx, order in enumerate(sorted_orders):
        group_feats = groups[order]
        color = order_to_color[order]
        faded_color = (*mcolors.to_rgb(color), 0.25)  # RGBA with low alpha
        # Build per-feature state: list of vertices + distance walked so far
        states = []
        for feat in group_feats:
            verts = feat["vertices"]  # list of coords
            # Cumulative distances along the polyline
            cum_dist = [0.0]
            for i in range(len(verts) - 1):
                d = np.sqrt(
                    (verts[i + 1][0] - verts[i][0]) ** 2
                    + (verts[i + 1][1] - verts[i][1]) ** 2
                )
                cum_dist.append(cum_dist[-1] + d)  # list of progressive total distance
            states.append(
                {
                    "vertices": verts,
                    "cum_dist": cum_dist,
                    "total": cum_dist[-1],
                    "walked": 0.0,
                    "speed": feat["speed"],
                    "start_delay": feat["start_delay"],
                    "color": color,
                    "line_obj": None,  # active drawing line
                    "arrow_obj": None,  # direction arrowhead
                }
            )
        # Animate until every line in this group is fully drawn
        any_active = True
        elapsed_time = 0.0
        true_group_duration = max(
            st["start_delay"] + st["total"] / st["speed"] for st in states
        )  # these two lines and 'display' below circumvent added time from faded and final hold frames
        true_group_start = total_elapsed  # snapshot before loop runs
        while any_active:
            any_active = False
            for st in states:  # for each line
                if (
                    elapsed_time < st["start_delay"]
                ):  # Don't start drawing until elapsed time exceeds this line's delay
                    any_active = True  # make sure not to exit loop if there is still a line waiting to start
                    continue
                if st["walked"] < st["total"]:  # if line is not finished
                    any_active = True
                    # Scale advance by this line's own speed
                    advance = seconds_per_frame * st["speed"]
                    st["walked"] = min(st["walked"] + advance, st["total"])
            # Redraw active lines
            for st in states:
                walked = st["walked"]
                xs, ys = _interpolate_polyline(st["vertices"], st["cum_dist"], walked)
                if st["line_obj"] is not None:
                    st["line_obj"].remove()
                if st["arrow_obj"] is not None:
                    st["arrow_obj"].remove()
                (line_obj,) = ax.plot(xs, ys, color=st["color"], linewidth=2, zorder=3)
                st["line_obj"] = line_obj
            elapsed_time += seconds_per_frame
            total_elapsed += seconds_per_frame
            # Clamp display to true values so frame-count drift can't accumulate
            display_group_t = min(elapsed_time, true_group_duration)
            display_total_t = min(total_elapsed, true_group_start + true_group_duration)
            capture_frame(
                f"Group {order} | group t = {display_group_t:.0f} s | total t = {display_total_t:.0f} s"
            )
        total_elapsed = true_group_start + true_group_duration
        # Group finished, now fade lines
        for st in states:
            if st["line_obj"] is not None:
                st["line_obj"].remove()
            if st["arrow_obj"] is not None:
                st["arrow_obj"].remove()
            # Draw full faded line
            xs = [v[0] for v in st["vertices"]]
            ys = [v[1] for v in st["vertices"]]
            (faded,) = ax.plot(xs, ys, color=faded_color, linewidth=1.5, zorder=2)
            # Static end arrow on completed line
            if len(xs) >= 2:
                ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=faded_color, lw=1.2),
                    zorder=2,
                )
            completed_line_objects.append(faded)

        # Hold on the last group completion frame for 1 second
        if order == len(order_to_color):
            hold_frames = max(1, fps)
            for _ in range(hold_frames):
                capture_frame(
                    f"Group {order} | group t = {display_group_t:.0f} s | total t = {total_elapsed:.0f} s"
                )
        print(f"Group {order}/{len(order_to_color)} complete")
    # save
    out_path = os.path.join(prj_path, gif_fname)
    frame_duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    print(
        f"GIF written to {out_path}  ({len(frames)} frames) \nStill image left open but not saved to a file"
    )
    return None


def _group_palette(n):
    """Return n visually distinct colors."""
    base = [
        "#FF4500",
        "#FFD700",
        "#FF8C00",
        "#ADFF2F",
        "#00CED1",
        "#DA70D6",
        "#1E90FF",
        "#FF69B4",
        "#7FFF00",
        "#FF6347",
    ]
    if n <= len(base):
        return base[:n]
    # Fall back to matplotlib tab20 for large n
    cmap = plt.get_cmap("tab20")
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]


def _interpolate_polyline(vertices, cum_dist, distance):
    """
    Return the (xs, ys) coordinate lists of a polyline truncated at
    `distance` meters
    """
    if distance <= 0:
        return [vertices[0][0]], [vertices[0][1]]
    xs = [vertices[0][0]]
    ys = [vertices[0][1]]
    for i in range(1, len(vertices)):
        seg_start = cum_dist[i - 1]
        seg_end = cum_dist[i]
        if distance <= seg_start:
            break
        if distance >= seg_end:
            xs.append(vertices[i][0])
            ys.append(vertices[i][1])
        else:
            t = (distance - seg_start) / (seg_end - seg_start)
            ix = vertices[i - 1][0] + t * (vertices[i][0] - vertices[i - 1][0])
            iy = vertices[i - 1][1] + t * (vertices[i][1] - vertices[i - 1][1])
            xs.append(ix)
            ys.append(iy)
            break
    return xs, ys


def _read_ign_dat(filepath):
    """
    Parse a QUIC-Fire ignition .dat file
    Skips all header lines (those not starting with a digit or '-').
    Returns a list of dicts with start_x, start_y, end_x, end_y,
    start_time, end_time.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Data lines start with a digit or minus sign
            if not (line[0].isdigit() or line[0] == "-"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                sx, sy, ex, ey, st, et = (float(p) for p in parts[:6])
            except ValueError:
                continue
            records.append(
                {
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "start_time": st,
                    "end_time": et,
                }
            )
    return records
