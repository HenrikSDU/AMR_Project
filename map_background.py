from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import math
import urllib.request
import warnings

import matplotlib.image as mpimg
import numpy as np


WEB_MERCATOR_RADIUS_M = 6378137.0
WEB_MERCATOR_HALF_WORLD_M = math.pi * WEB_MERCATOR_RADIUS_M
WEB_MERCATOR_MAX_LAT_DEG = 85.05112878


@dataclass(frozen=True)
class WebMapConfig:
    """Configuration for drawing web-map tiles in a local N/E metre frame."""

    origin_lat_deg: float
    origin_lon_deg: float
    origin_north_m: float = 0.0
    origin_east_m: float = 0.0
    zoom: int = 16
    alpha: float = 0.82
    cache_dir: Path = Path(".tile_cache/osm")
    tile_url_template: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    user_agent: str = "AMR_Project real-data tracking plot"
    attribution: str = "(C) OpenStreetMap contributors"
    timeout_s: float = 10.0
    max_tiles: int = 100
    zorder: float = -20.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _latlon_to_web_mercator(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat_deg = _clamp(lat_deg, -WEB_MERCATOR_MAX_LAT_DEG, WEB_MERCATOR_MAX_LAT_DEG)
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = WEB_MERCATOR_RADIUS_M * lon_rad
    y = WEB_MERCATOR_RADIUS_M * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))
    return x, y


def _local_to_web_mercator(
    east_m: float,
    north_m: float,
    config: WebMapConfig,
) -> tuple[float, float]:
    origin_x, origin_y = _latlon_to_web_mercator(
        config.origin_lat_deg,
        config.origin_lon_deg,
    )
    scale = math.cos(math.radians(config.origin_lat_deg))
    if abs(scale) < 1e-6:
        raise ValueError("Map origin latitude is too close to the Web Mercator pole.")

    return (
        origin_x + (east_m - config.origin_east_m) / scale,
        origin_y + (north_m - config.origin_north_m) / scale,
    )


def _web_mercator_to_local(
    x_m: float,
    y_m: float,
    config: WebMapConfig,
) -> tuple[float, float]:
    origin_x, origin_y = _latlon_to_web_mercator(
        config.origin_lat_deg,
        config.origin_lon_deg,
    )
    scale = math.cos(math.radians(config.origin_lat_deg))
    return (
        config.origin_east_m + (x_m - origin_x) * scale,
        config.origin_north_m + (y_m - origin_y) * scale,
    )


def _tile_bounds_web_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2**z
    tile_size_m = 2.0 * WEB_MERCATOR_HALF_WORLD_M / n
    left = -WEB_MERCATOR_HALF_WORLD_M + x * tile_size_m
    right = left + tile_size_m
    top = WEB_MERCATOR_HALF_WORLD_M - y * tile_size_m
    bottom = top - tile_size_m
    return left, right, bottom, top


def _tile_for_web_mercator(x_m: float, y_m: float, z: int) -> tuple[int, int]:
    n = 2**z
    xtile = int(math.floor((x_m + WEB_MERCATOR_HALF_WORLD_M) / (2.0 * WEB_MERCATOR_HALF_WORLD_M) * n))
    ytile = int(math.floor((WEB_MERCATOR_HALF_WORLD_M - y_m) / (2.0 * WEB_MERCATOR_HALF_WORLD_M) * n))
    return (
        int(_clamp(xtile, 0, n - 1)),
        int(_clamp(ytile, 0, n - 1)),
    )


def _tile_ranges_for_bounds(
    left_m: float,
    right_m: float,
    bottom_m: float,
    top_m: float,
    requested_zoom: int,
    max_tiles: int,
) -> tuple[int, range, range]:
    for z in range(int(requested_zoom), -1, -1):
        x0, y_top = _tile_for_web_mercator(left_m, top_m, z)
        x1, y_bottom = _tile_for_web_mercator(right_m, bottom_m, z)
        x_start, x_stop = sorted((x0, x1))
        y_start, y_stop = sorted((y_top, y_bottom))
        tile_count = (x_stop - x_start + 1) * (y_stop - y_start + 1)
        if tile_count <= max_tiles:
            return z, range(x_start, x_stop + 1), range(y_start, y_stop + 1)

    return 0, range(0, 1), range(0, 1)


def _fetch_tile(z: int, x: int, y: int, config: WebMapConfig) -> np.ndarray:
    cache_path = config.cache_dir / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        data = cache_path.read_bytes()
    else:
        url = config.tile_url_template.format(z=z, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": config.user_agent})
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            data = response.read()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)

    return mpimg.imread(io.BytesIO(data), format="png")


def _blank_tile(reference_tile: np.ndarray | None = None) -> np.ndarray:
    if reference_tile is None:
        tile = np.ones((256, 256, 4), dtype=float)
        tile[:, :, 3] = 0.0
        return tile

    tile = np.ones_like(reference_tile)
    if tile.ndim == 3 and tile.shape[2] == 4:
        tile[:, :, 3] = 0.0
    return tile


def _load_tile_mosaic(
    z: int,
    x_range: range,
    y_range: range,
    config: WebMapConfig,
) -> np.ndarray:
    rows = []
    reference_tile = None
    first_error: Exception | None = None

    for y in y_range:
        row_tiles = []
        for x in x_range:
            try:
                tile = _fetch_tile(z, x, y, config)
                reference_tile = tile
            except Exception as exc:  # network/cache failures should not break plotting
                if first_error is None:
                    first_error = exc
                tile = None
            row_tiles.append(tile)
        rows.append(row_tiles)

    if reference_tile is None and first_error is not None:
        raise RuntimeError(f"Could not load any map tiles: {first_error}") from first_error

    rendered_rows = []
    for row_tiles in rows:
        rendered_rows.append(
            np.concatenate(
                [
                    tile if tile is not None else _blank_tile(reference_tile)
                    for tile in row_tiles
                ],
                axis=1,
            )
        )

    return np.concatenate(rendered_rows, axis=0)


def add_osm_background(ax, config: WebMapConfig) -> bool:
    """Draw an OSM tile mosaic behind the current local East/North axes."""

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    corners = [
        _local_to_web_mercator(xlim[0], ylim[0], config),
        _local_to_web_mercator(xlim[0], ylim[1], config),
        _local_to_web_mercator(xlim[1], ylim[0], config),
        _local_to_web_mercator(xlim[1], ylim[1], config),
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    left_m, right_m = min(xs), max(xs)
    bottom_m, top_m = min(ys), max(ys)

    z, x_range, y_range = _tile_ranges_for_bounds(
        left_m,
        right_m,
        bottom_m,
        top_m,
        config.zoom,
        config.max_tiles,
    )

    try:
        image = _load_tile_mosaic(z, x_range, y_range, config)
    except Exception as exc:
        warnings.warn(f"Skipping map background: {exc}", RuntimeWarning, stacklevel=2)
        return False

    tile_left, _, _, tile_top = _tile_bounds_web_mercator(z, x_range.start, y_range.start)
    _, tile_right, tile_bottom, _ = _tile_bounds_web_mercator(z, x_range.stop - 1, y_range.stop - 1)
    extent_left, extent_bottom = _web_mercator_to_local(tile_left, tile_bottom, config)
    extent_right, extent_top = _web_mercator_to_local(tile_right, tile_top, config)

    ax.imshow(
        image,
        extent=[extent_left, extent_right, extent_bottom, extent_top],
        origin="upper",
        alpha=config.alpha,
        zorder=config.zorder,
        interpolation="bilinear",
    )
    if config.attribution:
        ax.text(
            0.01,
            0.01,
            config.attribution,
            transform=ax.transAxes,
            fontsize=7,
            color="0.15",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 2.0},
            zorder=10,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return True
