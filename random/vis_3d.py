#!/usr/bin/env python3
"""
3D cross-plane (XY & YZ) visualization with temperature contours and velocity glyphs.

- Uses domain-center plane origins for symmetric intersection
- Robust to U:[n_cells,3], T:[n_cells,1] in cell_data (converts cell->point)
- Adjustable glyph color/opacity
- Adds a corner orientation axes widget
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pyvista as pv
import os

# ---- headless-safe preamble ----
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
# If VTK supports EGL on your machine, uncomment:
# os.environ.setdefault("PYVISTA_USE_EGL", "true")
try:
    pv.start_xvfb()
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 3D XY/YZ planes of temperature with velocity glyphs."
    )
    p.add_argument( "--input", type=str, default="/home/shilaj/shilaj_data/repitframework/repitframework/Solvers/natural_convection_case1_3D/VTK/natural_convection_case1_3D_1000.vtk", required=False, help="Path to .vtu/.vtk file (OpenFOAM → foamToVTK output)." )
    p.add_argument("--scalar", type=str, default="T", help="Scalar field name (default: T).")
    p.add_argument("--vector", type=str, default="U", help="Vector field name (default: U).")
    p.add_argument("--xy_z", type=float, default=None, help="Z of XY slice (default: domain mid).")
    p.add_argument("--yz_x", type=float, default=None, help="X of YZ slice (default: domain mid).")
    p.add_argument(
        "--glyph_density",
        type=float,
        default=0.005,
        help="Fraction of points used for glyphs (0–1], default 0.02.",
    )
    p.add_argument(
        "--glyph_factor",
        type=float,
        default=0.5,
        help="Base glyph size factor (scaled by |U|).",
    )
    p.add_argument("--glyph_color", type=str, default="#FC7F91", help="Glyph color (hex or name).")
    p.add_argument("--glyph_opacity", type=float, default=1.0, help="Glyph opacity (0–1).")
    p.add_argument("--cmap", type=str, default="coolwarm", help="Colormap for temperature.")
    p.add_argument("--out", type=str, default="figure.png", help="Output PNG filename.")
    p.add_argument("--dpi", type=int, default=600, help="DPI hint (PNG).")
    p.add_argument("--bg", type=str, default="white", help="Background color.")
    p.add_argument(
        "--show_axes",
        action="store_true",
        help="Show a corner orientation axes widget (recommended).",
    )
    return p.parse_args()


def bounds_and_center(grid: pv.DataSet) -> Tuple[Tuple[float, float, float, float, float, float], Tuple[float, float, float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    return (xmin, xmax, ymin, ymax, zmin, zmax), (cx, cy, cz)


def choose_slice_positions(
    grid: pv.DataSet, xy_z: float | None, yz_x: float | None
) -> Tuple[float, float]:
    (xmin, xmax, ymin, ymax, zmin, zmax), _ = bounds_and_center(grid)
    z_mid = 0.5 * (zmin + zmax) if xy_z is None else xy_z
    x_mid = 0.5 * (xmin + xmax) if yz_x is None else yz_x
    return z_mid, x_mid


def ensure_point_arrays(mesh: pv.DataSet, scalar_name: str, vector_name: str) -> pv.DataSet:
    """Ensure scalar & vector are point_data; if not, convert cell -> point and squeeze [n,1] to [n]."""
    have_scalar_pt = scalar_name in mesh.point_data
    have_vector_pt = vector_name in mesh.point_data
    if not (have_scalar_pt and have_vector_pt):
        mesh = mesh.cell_data_to_point_data()

    # Squeeze [n,1] scalars to [n]
    if scalar_name in mesh.point_data:
        arr = np.asarray(mesh.point_data[scalar_name])
        if arr.ndim == 2 and arr.shape[1] == 1:
            mesh.point_data[scalar_name] = arr.ravel()

    # Validate vector shape
    if vector_name not in mesh.point_data:
        raise KeyError(f"Vector '{vector_name}' not found in point_data after conversion.")
    v = np.asarray(mesh.point_data[vector_name])
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Vector '{vector_name}' must be (N,3) after conversion; got {v.shape}")
    return mesh


def add_velocity_magnitude(mesh: pv.DataSet, vec_name: str, out_name: str = "velocity_mag") -> None:
    vec = np.asarray(mesh.point_data[vec_name])
    mesh.point_data[out_name] = np.linalg.norm(vec, axis=1)


def subsample_points_with_arrays(mesh: pv.DataSet, fraction: float) -> pv.PolyData:
    """Subsample points and copy point_data arrays into the subsampled polydata."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("glyph_density must be in (0, 1].")
    n = mesh.points.shape[0]
    k = max(1, int(n * fraction))
    idx = np.random.default_rng(42).choice(n, size=k, replace=False)

    poly = pv.PolyData(mesh.points[idx])
    for name, arr in mesh.point_data.items():
        arr = np.asarray(arr)
        if arr.shape[0] == n:
            poly.point_data[name] = arr[idx]
    return poly


def main() -> None:
    args = parse_args()
    infile = Path(args.input)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    mesh = pv.read(str(infile))

    # Check arrays exist (point or cell)
    if (args.scalar not in list(mesh.point_data) and args.scalar not in list(mesh.cell_data)) or \
       (args.vector not in list(mesh.point_data) and args.vector not in list(mesh.cell_data)):
        raise KeyError(
            f"Required arrays not found. Point data: {list(mesh.point_data.keys())}, "
            f"Cell data: {list(mesh.cell_data.keys())}"
        )

    # Normalize arrays and compute |U|
    mesh = ensure_point_arrays(mesh, args.scalar, args.vector)
    add_velocity_magnitude(mesh, args.vector, out_name="velocity_mag")

    # Domain info
    bounds, center = bounds_and_center(mesh)
    (xmin, xmax, ymin, ymax, zmin, zmax) = bounds
    (cx, cy, cz) = center

    # Slice positions + symmetric origins through the domain center
    z_mid, x_mid = choose_slice_positions(mesh, args.xy_z, args.yz_x)
    xy_slice = mesh.slice(normal="z", origin=(cx, cy, z_mid))      # XY plane at z=z_mid through (cx,cy)
    yz_slice = mesh.slice(normal="x", origin=(x_mid, cy, cz))      # YZ plane at x=x_mid through (cy,cz)

    # Subsample and glyph
    glyph_points = subsample_points_with_arrays(mesh, args.glyph_density)
    glyphs = glyph_points.glyph(
        orient=args.vector,
        scale="velocity_mag",
        factor=args.glyph_factor,
        geom=pv.Arrow(),
    )

    # Plotter
    pv.global_theme.background = args.bg
    pv.global_theme.font.size = 10

    plotter = pv.Plotter(off_screen=True, window_size=(2200, 1600))
    plotter.enable_anti_aliasing("ssaa")

    # Domain wireframe
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, style="wireframe", color="black", line_width=0.6, opacity=0.15)

    # Temperature on slices
    scalar_kwargs = dict(cmap=args.cmap, opacity=0.98, show_scalar_bar=True)
    plotter.add_mesh(xy_slice, scalars=args.scalar, **scalar_kwargs)
    plotter.add_mesh(yz_slice, scalars=args.scalar, **scalar_kwargs)

    # Velocity glyphs (customizable color & opacity)
    plotter.add_mesh(
        glyphs,
        color=args.glyph_color,
        opacity=float(args.glyph_opacity),
        smooth_shading=True,
        ambient=0.2,
        specular=0.1,
    )

    # Camera
    cx, cy, cz = center
    plotter.camera_position = [
        (cx + 1.8 * (xmax - xmin), cy + 1.4 * (ymax - ymin), cz + 1.2 * (zmax - zmin)),
        (cx, cy, cz),
        (0, 0, 1),
    ]
    plotter.enable_eye_dome_lighting()

    # Corner orientation axes widget
    if args.show_axes:
        plotter.add_axes()  # orientation triad in a corner (non-intrusive)

    # Save
    outfile = Path(args.out)
    plotter.show(screenshot=str(outfile), window_size=(2200, 1600))
    plotter.close()
    print(f"Saved figure to: {outfile.resolve()} (DPI hint: {args.dpi})")


if __name__ == "__main__":
    main()
