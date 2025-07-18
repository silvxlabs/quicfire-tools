"""
QUIC-Fire Tools Visualization Module
"""

from __future__ import annotations

## Core imports
from pathlib import Path

## External imports
import imageio
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Colormap, ListedColormap
import numpy as np
import pylab

## Internal imports
from quicfire_tools import SimulationOutputs
from quicfire_tools.outputs import OutputFile


def plot_fuel_density(
    simulation_outputs: SimulationOutputs,
    save_directory: Path,
    png: bool,
    gif: bool,
    z_layers: int | list[int] = None,
    integrated: bool = False,
    output_times: int | list[int] = None,
    visualize_fire: bool = True,
    framerate: int = 4,
) -> None:
    """"""
    if isinstance(save_directory, str):
        save_directory = Path(save_directory)
    nx = simulation_outputs.nx
    ny = simulation_outputs.ny
    dx = simulation_outputs.dx
    dy = simulation_outputs.dy
    dz = simulation_outputs.fire_dz
    dens = simulation_outputs.get_output("fuels-dens")
    output_times = (
        _verify_output_times(output_times, dens)
        if output_times is not None
        else dens.times
    )
    z_layers = _verify_z_layers(z_layers, simulation_outputs, grid="fire")
    timesteps = _timesteps_from_times(output_times, dens)
    dens_init = dens.to_numpy(timestep=0)
    extent = [1, nx * dx, 1, ny * dy]
    figsize = set_figure_size(simulation_outputs)
    if integrated:
        if len(z_layers) < 2:
            raise ValueError(
                "Vertical integration must be performed over multiple z layers"
            )
        surface = False
        image_list = []
        for t in timesteps:
            dens_t = dens.to_numpy(timestep=t)
            integ_init = np.sum(dens_init[0, z_layers, :, :], axis=0)
            integ_t = np.sum(dens_t[0, z_layers, :, :], axis=0)
            if t > 0:
                dens_prev = dens.to_numpy(timestep=t - 1)
                integ_prev = np.sum(dens_prev[0, z_layers, :, :], axis=0)
            else:
                integ_prev = integ_init.copy()
            burning, burnt = set_fire_vis(
                visualize_fire, integ_t, integ_init, integ_prev
            )
            title = f"Time = {output_times[t]:.0f} s"
            file_name = (
                f"fuel_density_{min(z_layers)}-{max(z_layers)}m_{output_times[t]:.0f}s"
            )
            save_path = save_directory / file_name
            create_fuel_dens_image(
                integ_t,
                integ_init,
                burning,
                burnt,
                title,
                save_path,
                figsize,
                extent,
                surface,
                integrated,
            )
            image_list.append(f"{str(save_path)}.png")
        if gif:
            gif_path = (
                save_directory / f"fuel_density_{min(z_layers)}-{max(z_layers)}m.gif"
            )
            make_gif(gif_path, image_list, framerate)
        if png == False:
            for png in image_list:
                Path(png).unlink()
    else:
        for z in z_layers:
            init_z = dens_init[0, z, :, :]
            surface = True if z == 0 else False
            image_list = []
            print(f"Creating images for {z*dz:.0f}m")
            for t in timesteps:
                dens_t = dens.to_numpy(timestep=t)
                dens_t_z = dens_t[0, z, :, :]
                if t > 0:
                    dens_prev = dens.to_numpy(timestep=t - 1)
                    prev_z = dens_prev[0, z, :, :]
                else:
                    prev_z = init_z.copy()
                burning, burnt = set_fire_vis(visualize_fire, dens_t_z, init_z, prev_z)
                title = f"Time = {output_times[t]:.0f} s"
                file_name = f"fuel_density_{z*dz:.0f}m_{output_times[t]:.0f}s"
                save_path = save_directory / file_name
                create_fuel_dens_image(
                    dens_t_z,
                    init_z,
                    burning,
                    burnt,
                    title,
                    save_path,
                    figsize,
                    extent,
                    surface,
                    integrated,
                )
                image_list.append(f"{str(save_path)}.png")
            if gif:
                gif_path = save_directory / f"fuel_density_{z*dz:.0f}m.gif"
                make_gif(gif_path, image_list, framerate)
            if png == False:
                for png in image_list:
                    Path(png).unlink()


def make_gif(gif_path: Path, image_list: list, framerate: int) -> None:
    duration = 1 / framerate
    with imageio.get_writer(gif_path, mode="I", duration=duration) as writer:
        for image in image_list:
            frame = plt.imread(image) * 255
            writer.append_data(frame.astype(np.uint8))


def create_fuel_dens_image(
    current_fuels: np.ndarray,
    initial_fuels: np.ndarray,
    active_fire_cells: np.ndarray,
    burned_cells: np.ndarray,
    title: str,
    save_path: Path,
    figure_size: tuple,
    extent: list,
    surface: bool,
    integrated: bool,
):
    """"""
    font_dict = set_fonts()
    cmap, vmin, vmax, log_scale = set_fuel_dens_colors(
        surface, integrated, initial_fuels
    )
    fig = pylab.figure(num=1, clear=True, figsize=figure_size)
    ax = fig.add_subplot(111)  # why 111?

    # remove burning and burned cells, if provided
    current_fuels = np.where(active_fire_cells == 1, 0, current_fuels)
    current_fuels = np.where(burned_cells == 1, 0, current_fuels)

    # Plot fuel density without fire visualization
    panel = ax.imshow(
        current_fuels,
        cmap=cmap,
        norm=log_scale,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )
    add_text_ticks_colorbar(fig, ax, panel, font_dict, title)
    # Add fire visualization if provided
    add_binary_color(ax, active_fire_cells, "red", extent)
    add_binary_color(ax, burned_cells, "black", extent)

    pylab.savefig(save_path)
    pylab.close()


def set_fuel_dens_colors(
    surface: bool, integrated: bool, initial_fuels: np.ndarray
) -> tuple[Colormap, float, float, LogNorm]:
    if surface:
        gist_earth = plt.get_cmap("gist_earth_r")
        cmap = ListedColormap(gist_earth(np.linspace(0.05, 0.6, 256)))
        vmin = np.min(initial_fuels)
        vmax = np.max(initial_fuels)
        log_scale = None
    else:
        cmap = plt.get_cmap("Greens")
        vmin = np.min(initial_fuels[np.where(initial_fuels > 0)])
        if integrated:
            vmax = np.percentile(initial_fuels[np.where(initial_fuels > 0)], 75)
        else:
            vmax = np.max(initial_fuels)
        log_scale = LogNorm(vmin=vmin, vmax=vmax)
        vmin = None
        vmax = None
    return cmap, vmin, vmax, log_scale


def set_figure_size(sim: SimulationOutputs, y: float = 8.5, f: float = 1.5) -> tuple:
    """
    Sets image size for matplotlib. y and f defaults are from drawfire.
    Perhaps we can have things like y and f be set from kwargs in the main functions.
    """
    x = y / (sim.ny * sim.dy) * (sim.nx * sim.dx) * f
    return (x, y)


def set_fonts(font: str = "Arial", base_size: float = 14) -> dict:
    font_dict = {}
    font_dict["axis"] = {"fontname": font, "size": str(base_size)}
    font_dict["title"] = {
        "fontname": font,
        "size": str(base_size + 2),
        "fontweight": "bold",
    }
    font_dict["colorbar"] = {"fontname": font, "size": str(base_size - 2)}
    return font_dict


def set_fire_vis(
    visualize_fire: bool, current: np.ndarray, initial: np.ndarray, previous: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """"""
    if visualize_fire:
        burning = _find_active_fire(current, previous)
        burnt = _find_burned_cells(current, initial)
    else:
        burning = np.zeros(initial.shape)
        burnt = np.zeros(initial.shape)
    return (burning, burnt)


def add_text_ticks_colorbar(
    fig: pylab.Figure,
    ax: pylab.Axes,
    panel: pylab.AxesImage,
    font_dict: dict,
    title: str,
) -> None:
    cbar = fig.colorbar(panel, ax=ax)
    cbar.set_label("Fuel Density (kg m$^{-3}$)", **font_dict["colorbar"])
    cbar.ax.tick_params(labelsize="14")
    pylab.xlabel("X (m)", **font_dict["axis"])
    pylab.ylabel("Y (m)", **font_dict["axis"])
    ax.set_aspect("equal")
    pylab.title(title, font_dict["title"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font_dict["axis"]["fontname"])
        label.set_fontsize(font_dict["axis"]["size"])


def add_binary_color(
    ax: pylab.Axes, binary_array: np.ndarray, color: str, extent: list
) -> None:
    ax.imshow(
        binary_array,
        cmap=ListedColormap([(0, 0, 0, 0), color]),
        vmin=0,
        vmax=1,
        extent=extent,
        interpolation="none",
        origin="lower",
        aspect="equal",
    )


def _find_active_fire(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """"""
    burning = np.where(previous - current > 0, 1, 0)
    return burning


def _find_burned_cells(current: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """"""
    no_fuel = np.where(initial == 0, 0, 1)
    consumed = np.where(initial - current > 0, 1, 0)
    burned = np.where((current < 0.1) & (consumed == 1), 1, 0)
    burned *= no_fuel
    return burned


def _timesteps_from_times(times: list, output: OutputFile) -> list:
    """"""
    if times == output.times:
        return list(range(len(times)))
    else:
        return [output.times.index(time) for time in times]


def _verify_output_times(output_times: int | list, output: OutputFile) -> list:
    """"""
    if isinstance(output_times, int):
        output_times = [output_times]
    for t in output_times:
        if t not in output.times:
            raise IndexError(f"Timestep {t} not present in {output.name} output")
    return output_times


def _verify_z_layers(
    z_layers: int | list | None, simulation_outputs: SimulationOutputs, grid: str
) -> list:
    """"""
    grids = ["quic", "fire"]
    if grid not in grids:
        raise AttributeError(f"Grid must be one of {grids}")
    if isinstance(z_layers, int):
        z_layers = [z_layers]
    nz = simulation_outputs.fire_nz if grid == "fire" else simulation_outputs.quic_nz
    nz_list = list(range(nz))
    if z_layers is None:
        return nz_list
    else:
        for z in z_layers:
            if z not in nz_list:
                raise IndexError(f"Z layer {z} not present in simulation")
        return z_layers
