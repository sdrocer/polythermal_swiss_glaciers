import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge as rio_merge
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, FuncFormatter

def format_plot(ax=None, title=None, legend_loc='upper right',
                base_fontsize=22, base_linewidth=4, font_family='Arial',
                x_tick_rotation=45, y_tick_rotation=0, cbar=None):
    """
    Simple, size-aware plot styling helper (no class needed).
    - ax: target axes (defaults to current)
    - title: plot title
    - legend_loc: legend location if <=4 items; otherwise placed outside
    - base_fontsize/base_linewidth: scaled relative to ~12-inch fig
    - font_family: e.g., 'Arial'
    - x_tick_rotation: rotation angle for x tick labels
    - cbar: optional matplotlib Colorbar to style
    """
    ax = ax if ax is not None else plt.gca()
    fig = ax.figure

    # Scale with figure size (reference ~12 inches)
    fig_w, fig_h = fig.get_size_inches()
    scale = (fig_w + fig_h) / 2.0
    fontsize = int(base_fontsize * scale / 12.0)
    linewidth = base_linewidth * scale / 12.0

    # Global-ish rc tweaks
    plt.rcParams['font.sans-serif'] = font_family
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['lines.linewidth'] = linewidth-4

    # Labels and title
    ax.set_title(title if title else '', fontsize=fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.xticks(rotation=x_tick_rotation)

    if y_tick_rotation == 90:
        # Rotate y ticks and center them on the tick mark (keep outside)
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_rotation_mode('anchor')
            lbl.set_va('center')      # center along the tick
            lbl.set_ha('center')      # tick goes through the middle of the label
    else:
        plt.yticks(rotation=y_tick_rotation)

    # Line widths for existing lines
    for line in ax.get_lines():
        line.set_linewidth(linewidth)

    # Legend logic
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) == 1:
        pass
    elif len(labels) > 4:
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                  loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    elif len(labels) > 1:
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                  loc=legend_loc, ncol=1)

    # Colorbar styling (optional)
    if cbar is not None:
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=fontsize)

    ax.grid(True)
    plt.tight_layout()

def compute_extent(transform, shape):
    """
    Return (xmin, xmax, ymin, ymax) extent for imshow from a transform and 2D shape.
    """
    nrows, ncols = shape
    xmin = transform.c
    xmax = transform.c + transform.a * ncols
    ymax = transform.f
    ymin = transform.f + transform.e * nrows
    return (xmin, xmax, ymin, ymax)

def imshow_grid(ax, grid, transform, *, cmap=cmc.batlow_r, alpha=0.7, vmin=None, vmax=None, zorder=3):
    """
    Shorthand to imshow a georeferenced grid. Returns (image, extent).
    """
    extent = compute_extent(transform, grid.shape)
    im = ax.imshow(grid, extent=extent, origin='upper', cmap=cmap, alpha=alpha,
                   vmin=vmin, vmax=vmax, zorder=zorder)
    return im, extent

def grid_coords_from_transform(transform, shape):
    """
    Return meshgrid (X, Y) in map coords for a raster with given transform and shape (rows, cols).
    """
    nrows, ncols = shape
    xs = transform.c + np.arange(ncols) * transform.a
    ys = transform.f + np.arange(nrows) * transform.e
    return np.meshgrid(xs, ys)

def mosaic_dems(
    dem_paths: list[str],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    pixel_size: float | None = None,
    fill_value: float = np.nan,
    dtype: str = "float32",
):
    """
    Mosaic multiple DEM tiles into a single array and transform.
    Returns (arr(masked), transform, crs).
    """
    if not dem_paths:
        raise ValueError("dem_paths is empty")
    srcs = [rasterio.open(p) for p in dem_paths]
    try:
        target_res = (float(pixel_size), float(pixel_size)) if pixel_size else None
        mosaic, transform = rio_merge(
            srcs, bounds=bbox, res=target_res, nodata=fill_value, dtype=dtype
        )
        crs = srcs[0].crs
        arr = mosaic[0].astype(dtype)
        mask = np.isnan(arr)
        arr = np.ma.array(arr, mask=mask)
        return arr, transform, crs
    finally:
        for s in srcs:
            s.close()

def plot_dem_contours(
    ax,
    dem,
    transform=None,
    *,
    levels=None,
    minor_step=10.0,
    major_step=50.0,
    minor_kwargs=None,
    major_kwargs=None,
    label=False,
    label_fmt="%.0f m",
    label_kwargs=None,
    zorder_minor=5,
    zorder_major=6,
):
    """
    Draw elevation contours on ax. dem can be a path or a 2D array (+ transform).
    Returns (cs_minor, cs_major).
    """
    close_src = False
    if isinstance(dem, (str, os.PathLike)):
        src = rasterio.open(dem)
        arr = src.read(1, masked=True).astype(float)
        tfm = src.transform
        close_src = True
    else:
        if transform is None:
            raise ValueError("transform is required when passing a DEM array")
        arr = np.ma.masked_invalid(np.asarray(dem, dtype=float))
        tfm = transform

    X, Y = grid_coords_from_transform(tfm, arr.shape)

    minor_kwargs = {'colors': 'k', 'linewidths': 0.25, 'alpha': 0.5} | (minor_kwargs or {})
    major_kwargs = {'colors': 'k', 'linewidths': 0.6,  'alpha': 0.85} | (major_kwargs or {})
    label_kwargs = {'inline': True, 'fontsize': 7, 'fmt': label_fmt, 'colors': 'k'} | (label_kwargs or {})

    cs_minor = None
    cs_major = None

    if levels is None:
        if np.ma.getmaskarray(arr).all():
            if close_src: src.close()
            return None, None
        vmin = float(arr.min())
        vmax = float(arr.max())
        if minor_step and minor_step > 0:
            minor_lvls = np.arange(np.floor(vmin/minor_step)*minor_step,
                                   np.ceil(vmax/minor_step)*minor_step + 0.1, minor_step)
            cs_minor = ax.contour(X, Y, arr, levels=minor_lvls, zorder=zorder_minor, **minor_kwargs)
        if major_step and major_step > 0:
            major_lvls = np.arange(np.floor(vmin/major_step)*major_step,
                                   np.ceil(vmax/major_step)*major_step + 0.1, major_step)
            cs_major = ax.contour(X, Y, arr, levels=major_lvls, zorder=zorder_major, **major_kwargs)
            if label:
                ax.clabel(cs_major, **label_kwargs)
    else:
        cs_major = ax.contour(X, Y, arr, levels=levels, zorder=zorder_major, **major_kwargs)
        if label:
            ax.clabel(cs_major, **label_kwargs)

    if close_src:
        src.close()
    return cs_minor, cs_major

def plot_dem_contours_from_tiles(
    ax,
    dem_paths: list[str],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    pixel_size: float | None = None,
    levels=None,
    minor_step=10.0,
    major_step=50.0,
    minor_kwargs=None,
    major_kwargs=None,
    label=False,
    label_fmt="%.0f m",
    label_kwargs=None,
    zorder_minor=5,
    zorder_major=6,
):
    """
    Mosaic tiles (optionally clipped/resampled) and draw contours.
    """
    arr, tfm, _ = mosaic_dems(dem_paths, bbox=bbox, pixel_size=pixel_size)
    return plot_dem_contours(
        ax,
        dem=arr,
        transform=tfm,
        levels=levels,
        minor_step=minor_step,
        major_step=major_step,
        minor_kwargs=minor_kwargs,
        major_kwargs=major_kwargs,
        label=label,
        label_fmt=label_fmt,
        label_kwargs=label_kwargs,
        zorder_minor=zorder_minor,
        zorder_major=zorder_major,
    )

def draw_gpr_line_points(ax, gdf, *, size=3, color='k', alpha=0.5, zorder=6, label='_nolegend_', rasterized=True):
    """
    Draw points or line vertices as small dots for GPR samples.
    """
    xs, ys = [], []
    for geom in gdf.geometry:
        if geom is None:
            continue
        gt = geom.geom_type
        if gt == 'MultiPoint':
            xs.extend([p.x for p in geom.geoms]); ys.extend([p.y for p in geom.geoms])
        elif gt == 'Point':
            xs.append(geom.x); ys.append(geom.y)
        elif gt in ('LineString', 'LinearRing'):
            cx, cy = zip(*list(geom.coords))
            xs.extend(cx); ys.extend(cy)
        elif gt == 'MultiLineString':
            for ln in geom.geoms:
                cx, cy = zip(*list(ln.coords))
                xs.extend(cx); ys.extend(cy)
    if xs:
        ax.scatter(xs, ys, s=size, c=color, marker='o', linewidths=0, alpha=alpha,
                   zorder=zorder, label=label, rasterized=rasterized)

def annotate_coordinates_on_grid(
    ax,
    *,
    sides=('top', 'right'),       # any of: 'top','bottom','left','right'
    thousands='apostrophe',
    unit='m',
    decimals=0,
    fontsize=8,
    color='k',
    clear=True
):
    """
    Place small coordinate labels inside the plot along the current MAJOR grid lines.
    - sides: where to draw labels inside the axes
    - thousands: 'none'|'space'|'comma'|'apostrophe'
    Hides the default tick labels so only the inside labels remain.
    """
    import matplotlib.transforms as mtransforms

    def _fmt(v):
        if decimals == 0:
            base = f"{int(round(v))}"
        else:
            base = f"{v:.{decimals}f}"
        if thousands != 'none' and decimals == 0:
            if thousands == 'space':
                base = f"{int(round(v)):,}".replace(',', ' ')
            elif thousands == 'comma':
                base = f"{int(round(v)):,}"
            elif thousands == 'apostrophe':
                base = f"{int(round(v)):,}".replace(',', "'")
        return f"{base} {unit}" if unit else base

    # Remove previous labels from earlier calls
    if clear:
        for t in list(ax.texts):
            if getattr(t, 'gid', '') == 'gridlabel':
                t.remove()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xticks = [t for t in ax.get_xticks() if xmin < t < xmax]
    yticks = [t for t in ax.get_yticks() if ymin < t < ymax]

    # Hide default outer tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_visible(False)

    # Tiny offsets in points so text sits just inside the frame
    top_tr    = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=0,  y=-3, units='points')
    bottom_tr = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=0,  y=+3, units='points')
    left_tr   = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=+3, y=0,  units='points')
    right_tr  = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=-3, y=0,  units='points')

    # X labels on top/bottom
    if 'top' in sides:
        for x in xticks:
            ax.text(x, ymax, _fmt(x), ha='center', va='top', fontsize=fontsize,
                    color=color, transform=top_tr, clip_on=False, gid='gridlabel')
    if 'bottom' in sides:
        for x in xticks:
            ax.text(x, ymin, _fmt(x), ha='center', va='bottom', fontsize=fontsize,
                    color=color, transform=bottom_tr, clip_on=False, gid='gridlabel')

    # Y labels on left/right (rotated)
    if 'left' in sides:
        for y in yticks:
            ax.text(xmin, y, _fmt(y), ha='left', va='center', rotation=90, fontsize=fontsize,
                    color=color, transform=left_tr, clip_on=False, gid='gridlabel')
    if 'right' in sides:
        for y in yticks:
            ax.text(xmax, y, _fmt(y), ha='right', va='center', rotation=90, fontsize=fontsize,
                    color=color, transform=right_tr, clip_on=False, gid='gridlabel')

def format_axes_coords(ax=None, x_step=None, y_step=None, thousands='none', unit=None,
                       decimals=0, tick_len=6, tick_pad=6,
                       hide_lower_edge=True, avoid_overlap=True, overlap_pad_px=2):
    """
    Format axis ticks for projected map coordinates (e.g., EPSG:2056).
    - x_step/y_step: major tick spacing in map units (e.g., 200)
    - thousands: 'none' | 'space' | 'comma' | 'apostrophe'
    - unit: append unit string (e.g., 'm') or None
    - decimals: number of decimals (0 -> clean integers)
    - tick_len/tick_pad: keep ticks outside the plot
    - hide_lower_edge: hide tick labels sitting exactly on left/bottom border
    - avoid_overlap: hide tick labels that would overlap neighboring labels
    - overlap_pad_px: padding (px) used when checking overlap
    """
    ax = ax or plt.gca()

    if x_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))

    def _fmt(v):
        if decimals == 0:
            base = f"{int(round(v))}"
        else:
            base = f"{v:.{decimals}f}"
        if thousands != 'none' and decimals == 0:
            if thousands == 'space':
                base = f"{int(round(v)):,}".replace(',', ' ')
            elif thousands == 'comma':
                base = f"{int(round(v)):,}"
            elif thousands == 'apostrophe':
                base = f"{int(round(v)):,}".replace(',', "'")
        return f"{base} {unit}" if unit else base

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: _fmt(val)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: _fmt(val)))

    # Keep ticks outside and suppress offset/scientific notation
    ax.tick_params(axis='both', which='both', direction='out', length=tick_len, pad=tick_pad)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    # Hide labels at the lower/left border to avoid corner overlap
    if hide_lower_edge:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        atol_x = (x_step or 1.0) * 0.01
        atol_y = (y_step or 1.0) * 0.01
        for xv, lbl in zip(xticks, ax.get_xticklabels()):
            if np.isclose(xv, min(xmin, xmax), atol=atol_x):
                lbl.set_visible(False)
        for yv, lbl in zip(yticks, ax.get_yticklabels()):
            if np.isclose(yv, min(ymin, ymax), atol=atol_y):
                lbl.set_visible(False)

    # Hide overlapping tick labels (both axes)
    if avoid_overlap:
        fig = ax.figure
        try:
            fig.canvas.draw()  # realize text positions
            renderer = fig.canvas.get_renderer()
            # X labels
            kept = []
            for lbl in ax.get_xticklabels():
                if not lbl.get_visible() or not lbl.get_text():
                    continue
                bb = lbl.get_window_extent(renderer=renderer)
                bb = bb.expanded((bb.width + overlap_pad_px)/bb.width,
                                 (bb.height + overlap_pad_px)/bb.height)
                if any(bb.overlaps(b) for b in kept):
                    lbl.set_visible(False)
                else:
                    kept.append(bb)
            # Y labels
            kept = []
            for lbl in ax.get_yticklabels():
                if not lbl.get_visible() or not lbl.get_text():
                    continue
                bb = lbl.get_window_extent(renderer=renderer)
                bb = bb.expanded((bb.width + overlap_pad_px)/bb.width,
                                 (bb.height + overlap_pad_px)/bb.height)
                if any(bb.overlaps(b) for b in kept):
                    lbl.set_visible(False)
                else:
                    kept.append(bb)
        except Exception:
            pass

def plot_heatmap(grid, transform, title=None, points_gdf=None, polygon_gdf=None, cmap=cmc.batlow, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    extent = (transform.c, transform.c + transform.a*grid.shape[1],
              transform.f + transform.e*grid.shape[0], transform.f)
    im = ax.imshow(grid, extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    if polygon_gdf is not None and not polygon_gdf.empty:
        polygon_gdf.boundary.plot(ax=ax, color='k', linewidth=1.0)
    if points_gdf is not None and not points_gdf.empty:
        points_gdf.plot(ax=ax, markersize=6, column='thickness', cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolor='k', linewidth=0.2, alpha=0.8)
    cb = fig.colorbar(im, ax=ax, label='Ice thickness [m]')
    ax.set_xlabel('Easting (EPSG:2056)')
    ax.set_ylabel('Northing (EPSG:2056)')
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def cairomakie_cmap(name: str = "Blues", n: int = 256, reverse: bool = False) -> ListedColormap:
    """
    Create a Matplotlib colormap from a ColorSchemes.jl palette (CairoMakie uses these).
    Requires `pip install juliacall` and a working Julia. Falls back to Matplotlib if Julia not available.
    """
    try:
        from juliacall import Main as jl
        jl.seval("""
        try
            using ColorSchemes, Colors
        catch
            import Pkg
            Pkg.add(["ColorSchemes","Colors"])
            using ColorSchemes, Colors
        end
        """)
        cs = jl.seval(f"getfield(ColorSchemes, Symbol(\"{name}\"))")
        jl.cs = cs
        hexs = list(jl.seval(f"[\"#\" * hex(Colors.RGB(get(cs, t))) for t in range(0,1,length={n})]"))
        if reverse:
            hexs = hexs[::-1]
        return ListedColormap(hexs, name=f"{name}{'_r' if reverse else ''}")
    except Exception:
        # Fallback to a similarly named Matplotlib map if Julia/juliacall not available
        import matplotlib.pyplot as plt
        try:
            base = plt.get_cmap(name)
        except ValueError:
            base = plt.get_cmap("Blues")
        return base.reversed() if reverse else base

def plot_thickness_profile(profile_df: pd.DataFrame, *, ax=None, title=None,
                           flip='auto',                      # NEW
                           facecolor="#7db7d8", edgecolor="k", alpha=0.85,
                           surface_kwargs=None, bed_kwargs=None):
    """
    Plot a 2D glacier cross-section (distance vs elevation).
    flip:
      - 'auto' -> reverse so surface increases from left to right
      - True   -> always reverse
      - False  -> as-is
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Flip so it starts low and goes up (left->right)
    d = d0
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    # Fill glacier body
    ax.fill_between(d, z_b, z_s, color=facecolor, alpha=alpha, linewidth=0)

    # Lines
    surface_kwargs = {'color': 'k', 'linewidth': 1.2} | (surface_kwargs or {})
    bed_kwargs     = {'color': 'k', 'linewidth': 1.0, 'linestyle': ':'} | (bed_kwargs or {})
    ax.plot(d, z_s, **surface_kwargs, label='Surface')
    ax.plot(d, z_b, **bed_kwargs,     label='Bed')

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    fig = ax.figure
    fig.tight_layout()
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left')
    return fig, ax

def plot_thickness_icetemp_heatmap(profile_df, borehole_coords_df, temp_data_dict, depth_dict, ax=None, title=None,
                                   cmap=cmc.batlow_r, vmin=None, vmax=None, flip='auto'):
    """
    Plots glacier cross-section with borehole positions and interpolated temperature heatmap.
    - profile_df: DataFrame with 'distance', 'zsurf', 'zbed'
    - borehole_coords_df: DataFrame with borehole coordinates ('name', 'x', 'y', ...)
    - temp_data_dict: dict {borehole_name: pandas.Series of temperatures}
    - depth_dict: dict {borehole_name: {probe: depth}}
    - flip: 'auto' | True | False (reverse so surface increases left to right)
    """
    from scipy.interpolate import griddata

    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Flip so it starts low and goes up (left->right)
    d = d0
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    # Plot surface and bed
    ax.plot(d, z_s, color='k', linewidth=1.2, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.0, linestyle=':', label='Bed')

    # Collect borehole positions and sensor depths/temps for interpolation
    interp_points = []
    interp_temps = []
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if 'x' in profile_df and 'y' in profile_df else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            profile_dist = d[profile_dist_idx]
            surface_elev = z_s[profile_dist_idx]
        else:
            profile_dist = bh_x  # fallback
            surface_elev = np.interp(profile_dist, d, z_s)

        # For each sensor in borehole
        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            # Plot each thermistor as a small horizontal line at its elevation
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    interp_points.append([profile_dist, therm_elev])
                    interp_temps.append(temps[probe])
                    # Draw a small horizontal line ("-") for the thermistor
                    ax.plot([profile_dist-1.5, profile_dist+1.5], [therm_elev, therm_elev], color='black', linewidth=1.2, zorder=10)
            # Draw vertical line for borehole (from shallowest to deepest thermistor)
            min_elev = surface_elev - max(depths.values())
            max_elev = surface_elev - min(depths.values())
            ax.axvline(profile_dist, color='k', linestyle='solid', alpha=0.7, zorder=5, linewidth=0.5)
            # Annotate borehole name above glacier surface
            ax.text(profile_dist, surface_elev+6, name, color='red', fontsize=10, va='bottom', ha='center', zorder=12)

    # Interpolate temperature heatmap
    interp_points = np.array(interp_points)
    interp_temps = np.array(interp_temps)
    # Create grid for heatmap
    grid_x = d
    grid_y = np.linspace(z_b.min(), z_s.max(), 200)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    # Interpolate (linear, can change to 'nearest' or 'cubic')
    grid_temp = griddata(interp_points, interp_temps, (grid_xx, grid_yy), method='nearest')

    # Mask grid_temp outside glacier body
    for i, x in enumerate(grid_x):
        bed = np.interp(x, d, z_b)
        surf = np.interp(x, d, z_s)
        for j, y in enumerate(grid_y):
            if not (bed < y < surf):
                grid_temp[j, i] = np.nan

    # Plot heatmap (now masked)
    im = ax.imshow(grid_temp, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                   origin='lower', aspect='auto', cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax, zorder=0)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')

    # --- Set y-axis limits to glacier body only ---
    ax.set_ylim(np.min(z_b)-2, np.max(z_s)+2)

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left')
    plt.tight_layout()
    return ax.figure, ax