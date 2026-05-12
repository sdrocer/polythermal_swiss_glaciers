# import necessary libraries
import numpy as np
import rasterio
from rasterio.merge import merge
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
from matplotlib.transforms import offset_copy
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from typing import List, Tuple, Optional

def imshow_tif(ax, path):
    with rasterio.open(path) as src:
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        if src.count >= 3:
            data = src.read([1, 2, 3])
            img = np.moveaxis(data, 0, -1)
        else:
            img = src.read(1)
        ax.imshow(img, extent=extent, origin='upper')
    return extent

# --- NEW: hillshade renderer from DEM tiles -----------------------------------
def imshow_hillshade(ax, dem_tiles, plot_extent=None, merge_bbox=None, cmap='gray', blend_alpha=1.0,
                      zfactor=1.0, azimuth=315, altitude=45, vmin=None, vmax=None):
    srcs = [rasterio.open(p) for p in dem_tiles]
    try:
        if merge_bbox is not None:
            # merge_bbox must be (minx, miny, maxx, maxy)
            mosaic, out_trans = merge(srcs, bounds=merge_bbox)
        else:
            mosaic, out_trans = merge(srcs)
    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass

    dem = mosaic[0].astype(float)
    h, w = dem.shape


    # Mask/fill no-data values
    dem = np.where(np.isnan(dem) | (dem < -9999), np.nan, dem)
    fill_value = np.nanmedian(dem) if np.isnan(dem).any() else np.min(dem)
    dem_filled = np.where(np.isnan(dem), fill_value, dem)

    # Compute hillshade
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    dx = abs(out_trans.a) if hasattr(out_trans, "a") else 1.0
    dy = abs(out_trans.e) if hasattr(out_trans, "e") else 1.0
    hs = ls.hillshade(dem_filled * zfactor, vert_exag=1.0, dx=dx, dy=dy)

    # Normalize hillshade to [0, 1]
    hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-8)

    # Use plot_extent for imshow, not for merging
    if plot_extent is not None:
        extent = plot_extent
    else:
        minx = out_trans.c
        maxy = out_trans.f
        maxx = minx + out_trans.a * w
        miny = maxy + out_trans.e * h
        extent = (minx, maxx, miny, maxy)

    img = ax.imshow(hs, cmap=cmap, extent=extent, origin='upper', alpha=blend_alpha, vmin=0, vmax=1)
    return img

def add_panel_outline(ax, bbox, color='black', linewidth=2):
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    rect = Rectangle(
        (minx, miny), width, height,
        fill=False, edgecolor=color, linewidth=linewidth, zorder=100
    )
    ax.add_patch(rect)

def annotate_borehole_with_line(ax, x, y, text, dx=30, dy=10, color='k', fontsize=11):
    """
    Annotate a borehole with a label and a connecting line.
    dx, dy: offset in points for the annotation.
    """
    # Convert offset in points to data coordinates
    from matplotlib.transforms import offset_copy
    tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units='points')
    # Draw the annotation text
    txt = ax.text(x, y, text, transform=tr, fontsize=fontsize, fontweight='bold', color=color,
                  ha='left', va='bottom', zorder=20,
                  bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    # Draw a line from the borehole to the annotation
    # Get display coordinates for both points
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x, y))
    p1 = (p1[0] + dx, p1[1] + dy)
    # Convert back to data coordinates
    p1_data = ax.transData.inverted().transform(p1)
    ax.plot([x, p1_data[0]], [y, p1_data[1]], color=color, lw=1.2, zorder=19)
    return txt

def add_bbox(ax, bbox, color, label, where='bottom-left', offset=0):
    xmin, ymin, xmax, ymax = bbox
    # Draw white buffer rectangle first
    buffer_lw = 3.2
    rect_buffer = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, edgecolor='white', linewidth=buffer_lw, zorder=14)
    ax.add_patch(rect_buffer)
    # Draw colored rectangle on top
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                     fill=False, edgecolor=color, linewidth=1.6, zorder=15)
    ax.add_patch(rect)
    # Place label just OUTSIDE the chosen corner (default: upper-right)
    if where == 'top-right':
        x, y = xmax, ymax; ha, va = 'left', 'bottom'; dx, dy = offset, offset
    elif where == 'top-left':
        x, y = xmin, ymax; ha, va = 'right', 'bottom'; dx, dy = -offset, offset
    elif where == 'bottom-right':
        x, y = xmax, ymin; ha, va = 'left', 'top'; dx, dy = offset, -offset
    else:  # bottom-left
        x, y = xmin, ymin; ha, va = 'right', 'top'; dx, dy = -offset, -offset
    tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units='points')
    ax.text(x, y, label, color=color, fontsize=16, fontweight='bold',
            ha=ha, va=va, transform=tr, zorder=16,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.8))

def add_topo_points(ax, points, zorder=25):
    """
    Draw named locations on the overview map.
    points: list of dicts with keys: name, x, y, kind(optional), dx, dy, ha, va.
    """
    for p in points:
        x, y = p["x"], p["y"]
        kind = p.get("kind", "peak")
        # style per kind
        if kind == "village":
            mkw = dict(marker="o", ms=5, mfc="#ff7f0e", mec="white", mew=0.8)
        else:  # peaks/stations
            mkw = dict(marker="^", ms=5, mfc="black", mec="white", mew=0.8)

        ax.plot(x, y, zorder=zorder, **mkw)
        # label with small offset so it doesn't sit on the marker
        dx = p.get("dx", 4); dy = p.get("dy", 4)
        ha = p.get("ha", "left"); va = p.get("va", "bottom")
        tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        ax.text(
            x, y, p["name"], transform=tr, ha=ha, va=va, fontsize=9, color="white",
            zorder=zorder+1,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black", alpha=0.85)]
        )

def annotate_gpr_profile(ax, gdf_pts, profile_id, text=None, where='mid',
                         offset_pts=(4, 4), color='crimson', fontsize=9, bg=True, zorder=30):
    """
    Place a small label on a GPR profile composed of points.
    where: 'start' | 'mid' | 'end' (along-profile)
    offset_pts: (dx, dy) in points from anchor
    """
    sub = gdf_pts[gdf_pts['profile'] == profile_id]
    if sub.empty:
        return None
    coords, order = _profile_pca_order(sub)
    if where == 'start':
        i = order[0]
    elif where == 'end':
        i = order[-1]
    else:
        i = order[len(order)//2]
    x, y = coords[i]
    tr = offset_copy(ax.transData, fig=ax.figure, x=offset_pts[0], y=offset_pts[1], units='points')
    if text is None:
        text = f"P{profile_id}"
    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7) if bg else None
    return ax.text(x, y, text, color=color, fontsize=fontsize, ha='left', va='bottom',
                   transform=tr, zorder=zorder, bbox=bbox)

def add_north_arrow_compass(ax, xy=(0.92, 0.08), size=0.08, width=None,
                              colors=('white', 'k'), edge='black',
                              text_color='0.2', lw=1.0):
    """
    Draw a simple north-arrow compass in axes fraction coordinates.

    Uses Matplotlib patches (explicitly imported here) to avoid name collisions
    with shapely.geometry.Polygon that may be present in the module namespace.
    """
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    if width is None:
        width = size * 0.6
    x, y = xy
    tip        = (x, y + size)
    base_left  = (x - width/2, y)
    base_right = (x + width/2, y)
    spine      = (x, y + size*0.25)

    tri_left  = MplPolygon([tip, spine, base_left], closed=True,
                          facecolor=colors[0], edgecolor=edge, linewidth=lw,
                          transform=ax.transAxes, zorder=50, joinstyle='miter')
    tri_right = MplPolygon([tip, base_right, spine], closed=True,
                           facecolor=colors[1], edgecolor=edge, linewidth=lw,
                           transform=ax.transAxes, zorder=50, joinstyle='miter')
    ax.add_patch(tri_left)
    ax.add_patch(tri_right)

    ax.text(x, y - 0.025, 'N', transform=ax.transAxes, ha='center', va='top',
            fontsize=11, color=text_color, zorder=51,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white', alpha=0.95)])

def add_topo_points(ax, points, zorder=25):
    for p in points:
        x, y = p["x"], p["y"]
        kind = p.get("kind", "peak")
        dx = p.get("dx", 4)
        dy = p.get("dy", 4)
        ha = p.get("ha", "left")
        va = p.get("va", "bottom")
        tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        if kind != "village":
            mkw = dict(marker="^", ms=5, mfc="black", mec="white", mew=0.8)
            ax.plot(x, y, zorder=zorder, **mkw)
        ax.text(
            x, y, p["name"], transform=tr, ha=ha, va=va, fontsize=11, color="white",
            zorder=zorder+1,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black", alpha=0.85)]
        )

def read_xyzn_to_gdf(path: str, crs: Optional[int] = 2056) -> gpd.GeoDataFrame:
    """
    Read a .xyzn outline file and return a GeoDataFrame.

    The parser is permissive:
    - lines with three or more numeric columns are interpreted as x y [z ...]
    - blank lines or non-numeric lines are treated as polygon separators; non-numeric
      lines are used as polygon 'name' where available
    - polygons with < 3 vertices become LineStrings

    Parameters
    ----------
    path : str
        Path to the .xyzn file.
    crs : int | None
        EPSG code for the returned GeoDataFrame (default: 2056 / LV95). If None,
        CRS will be left unset.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns: geometry, name
    """
    polys: List[Tuple[List[Tuple[float, float]], Optional[str]]] = []
    current: List[Tuple[float, float]] = []
    current_name: Optional[str] = None

    def _flush():
        nonlocal current, current_name
        if current:
            polys.append((current, current_name))
            current = []
            current_name = None

    with open(path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if line == "":
                _flush()
                continue
            parts = line.split()
            # try parse first two tokens as floats
            try:
                x = float(parts[0])
                y = float(parts[1])
                current.append((x, y))
            except Exception:
                # non-numeric line -> treat as a name / separator
                # if we already have points, flush them first and treat this line as next name
                if current:
                    _flush()
                current_name = line
                # continue reading points for this named polygon
        # end for
    # final flush
    _flush()

    geoms = []
    names = []
    for coords, name in polys:
        # ensure at least 2 points
        if len(coords) < 2:
            continue
        # close polygon if enough points and not closed
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            try:
                geom = Polygon(coords)
            except Exception:
                geom = LineString(coords)
        else:
            geom = LineString(coords)
        geoms.append(geom)
        names.append(name if name is not None else '')

    gdf = gpd.GeoDataFrame({'name': names}, geometry=geoms)
    if crs is not None:
        try:
            gdf.set_crs(epsg=int(crs), inplace=True)
        except Exception:
            # leave CRS unset if invalid
            pass
    return gdf

def geometric_properties_from_xyzn_demtiles(xyzn_path, dem_tile_paths):
    """
    Calculate area, min/max elevation, min/max/mean slope for an outline in .xyzn,
    using elevations sampled from a list of DEM GeoTIFF tiles.
    """
    xs, ys = [], []
    with open(xyzn_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
                xs.append(x)
                ys.append(y)
            except Exception:
                continue

    coords = np.column_stack([xs, ys])
    polygon = Polygon(coords)
    area = polygon.area

    # Mosaic DEM tiles
    srcs = [rasterio.open(p) for p in dem_tile_paths]
    mosaic, out_trans = merge(srcs)
    dem_arr = mosaic[0]
    dem_nodata = srcs[0].nodata
    dem_crs = srcs[0].crs
    for s in srcs:
        s.close()

    # Sample DEM at each XY location
    dem_elevs = []
    for x, y in zip(xs, ys):
        # Convert map coordinates to row/col in mosaic
        col, row = ~out_trans * (x, y)
        col, row = int(round(col)), int(round(row))
        if 0 <= row < dem_arr.shape[0] and 0 <= col < dem_arr.shape[1]:
            val = dem_arr[row, col]
            if dem_nodata is not None and val == dem_nodata:
                continue
            if val < -9999 or np.isnan(val):
                continue
            dem_elevs.append(val)
    dem_elevs = np.array(dem_elevs)

    min_elev = float(np.min(dem_elevs)) if dem_elevs.size else np.nan
    max_elev = float(np.max(dem_elevs)) if dem_elevs.size else np.nan

    # Slope calculation: between consecutive points
    slopes = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dz = dem_elevs[i] - dem_elevs[i-1] if i < len(dem_elevs) else 0
        horiz_dist = np.hypot(dx, dy)
        if horiz_dist > 0:
            slope_deg = np.degrees(np.arctan2(dz, horiz_dist))
            slopes.append(slope_deg)
    slopes = np.abs(np.array(slopes))  # <-- Use absolute steepness

    min_slope = float(np.min(slopes)) if slopes.size else np.nan
    max_slope = float(np.max(slopes)) if slopes.size else np.nan
    mean_slope = float(np.mean(slopes)) if slopes.size else np.nan

    return {
        "area": area,
        "min_elevation": min_elev,
        "max_elevation": max_elev,
        "min_slope_deg": min_slope,
        "max_slope_deg": max_slope,
        "mean_slope_deg": mean_slope,
    }