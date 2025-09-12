import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge as rio_merge
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.lines import Line2D
from scipy.interpolate import Rbf
from processing.thermistor_processing import *


def format_plot(ax=None, title=None, legend_loc='upper right',
                base_fontsize=22, base_linewidth=4, font_family='Arial',
                x_tick_rotation=45, y_tick_rotation=0, cbar=None,
                adjust_linewidths=True):  # <-- new parameter
    """
    Simple, size-aware plot styling helper (no class needed).
    ...
    - adjust_linewidths: if True, adjust all line widths in the plot
    """
    ax = ax if ax is not None else plt.gca()
    fig = ax.figure

    # Scale with figure size (reference ~12 inches)
    fig_w, fig_h = fig.get_size_inches()
    scale = (fig_w + fig_h) / 2.0
    fontsize = int(base_fontsize * scale / 12.0)
    linewidth = (base_linewidth * scale / 12.0)

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
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_rotation_mode('anchor')
            lbl.set_va('center')
            lbl.set_ha('center')
    else:
        plt.yticks(rotation=y_tick_rotation)

    # Line widths for existing lines (optional)
    if adjust_linewidths:
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

def icetemp_cmap(n=256, red_fraction=0.08):
    """
    Custom colormap for ice temperatures:
    - Blue shades for T < 0°C (from cmcrameri.vik)
    - Sharp transition to red for T >= 0°C (reddest part of vik)
    - No white in the middle
    """
    base = cmc.vik(np.linspace(0, 1, n))
    # Take blue part (left half)
    blue = base[:n//2]
    # Take only the reddest part for >=0°C
    n_red = max(2, int(n * red_fraction))-5
    red = base[-n_red:]
    # Stack: blue for <0, red for >=0
    colors = np.vstack([blue, red])
    return ListedColormap(colors, name="icetemp")

def discrete_icetemp_cmap(levels):
    """
    Discretize the icetemp_cmap so that all intervals except the highest use the blue spectrum of cmc.vik,
    and only the highest interval (last) uses the red color.
    """
    n = len(levels) - 1
    # Blue spectrum: left half of cmc.vik
    blue_colors = cmc.vik(np.linspace(0, 0.5, n-1)) if n > 1 else np.empty((0, 4))
    red_color = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([blue_colors, red_color])
    return ListedColormap(colors, name="icetemp_discrete")

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

def draw_gpr_line_points(ax, gdf, *, size=3, color='k', alpha=0.5, zorder=6, label='_nolegend_', rasterized=True, highlight_id=None, highlight_color='crimson', highlight_size=8):
    """
    Draw GPR points as small dots.
    Optionally highlight all points belonging to a specific profile.
    """
    # Plot highlighted profile points first (if requested)
    if highlight_id is not None and 'profile' in gdf.columns:
        mask = gdf['profile'] == highlight_id
        if mask.any():
            gdf_highlight = gdf[mask]
            ax.scatter(
                gdf_highlight['x'], gdf_highlight['y'],
                s=highlight_size, c=highlight_color, marker='o',
                alpha=0.9, zorder=zorder+2, label=f'Profile {highlight_id}', rasterized=rasterized
            )
        # Plot the rest as normal (excluding highlighted)
        gdf = gdf[~mask]

    # Plot all remaining points
    if not gdf.empty:
        ax.scatter(
            gdf['x'], gdf['y'],
            s=size, c=color, marker='o',
            alpha=alpha, zorder=zorder, label=label, rasterized=rasterized
        )
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
                           flip='auto',                      
                           facecolor="#7db7d8", edgecolor="k", alpha=0.85,
                           surface_kwargs=None, bed_kwargs=None,
                           break_threshold=50.0,
                           smooth_sigma=2.0):  # NEW: smoothing parameter
    """
    Plot a 2D glacier cross-section (distance vs elevation).
    Detects and visualizes breaks in the profile where there are gaps in data.
    Creates proper visual cuts so lines don't connect across breaks.
    
    Args:
        break_threshold: Distance threshold (in meters) to detect profile breaks
        smooth_sigma: Gaussian smoothing sigma for surface/bed lines (0 = no smoothing)
        flip: 'auto' -> reverse so surface increases from left to right
              True   -> always reverse
              False  -> as-is
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage

    def smooth_line(x, y, sigma):
        """Apply Gaussian smoothing to y values, preserving x spacing"""
        if sigma <= 0 or len(y) < 3:
            return y
        return ndimage.gaussian_filter1d(y, sigma=sigma, mode='nearest')

    ax = ax or plt.subplots(figsize=(11, 5), dpi=150)[1]
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

    # Detect breaks in the profile
    dist_diffs = np.diff(d)
    break_indices = np.where(dist_diffs > break_threshold)[0]
    
    # Split into continuous segments
    segments = []
    start_idx = 0
    
    if len(break_indices) == 0:
        # No breaks - single segment
        segments.append((0, len(d)))
    else:
        # Multiple segments separated by breaks
        print(f"Profile has {len(break_indices)} break(s) at distances: {d[break_indices + 1]}")
        
        for break_idx in break_indices:
            end_idx = break_idx + 1
            segments.append((start_idx, end_idx))
            start_idx = end_idx
        segments.append((start_idx, len(d)))  # Last segment
    
    # Plot each segment separately (this ensures no connections across breaks)
    surface_kwargs = {'color': 'k', 'linewidth': 1.5} | (surface_kwargs or {})
    bed_kwargs     = {'color': 'k', 'linewidth': 1.5, 'linestyle': ':'} | (bed_kwargs or {})
    
    for i, (start, end) in enumerate(segments):
        if start >= end:
            continue
            
        # Extract segment data
        d_seg = d[start:end]
        z_s_seg = z_s[start:end]
        z_b_seg = z_b[start:end]
        
        # Skip segments that are too small
        if len(d_seg) < 2:
            continue
        
        # Apply smoothing to each segment
        z_s_seg_smooth = smooth_line(d_seg, z_s_seg, smooth_sigma)
        z_b_seg_smooth = smooth_line(d_seg, z_b_seg, smooth_sigma)
        
        # Fill glacier body for this segment (use smoothed data)
        ax.fill_between(d_seg, z_b_seg_smooth, z_s_seg_smooth, color=facecolor, alpha=alpha, linewidth=0)
        
        # Plot lines (only add labels for first segment to avoid duplicate legend entries)
        label_surface = 'Surface' if i == 0 else None
        label_bed = 'Bed' if i == 0 else None
        
        ax.plot(d_seg, z_s_seg_smooth, **surface_kwargs, label=label_surface)
        ax.plot(d_seg, z_b_seg_smooth, **bed_kwargs, label=label_bed)
    
    # Mark the breaks with vertical dashed lines (optional visual indicator)
    if len(break_indices) > 0:
        y_min = min(np.min(z_b), np.min(z_s)) - 5
        y_max = max(np.max(z_b), np.max(z_s)) + 5
        
        for i, break_idx in enumerate(break_indices):
            break_distance = d[break_idx + 1]
            ax.axvline(break_distance, color='red', linestyle='--', linewidth=1.0, 
                      alpha=0.5, label='Profile break' if i == 0 else None)

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    fig = ax.figure
    fig.tight_layout()
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    return fig, ax

def plot_icetemp_profile(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    flip='auto',
    n_depth=500,
    n_elev=600,
    plot_contours=True,
    break_threshold=50.0,
    smooth_sigma=0.0,
    temp_step=0.1        # NEW: controls contour interval AND colorbar tick spacing
):
    """
    Plot glacier cross-section with borehole temps interpolated to a 2D temperature field.
    - temp_step sets both the contour interval and colorbar tick spacing (default 0.1°C).
      The final positive bin is [0.0, temp_step] and is colored red.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter1d

    # Helpers
    def _smooth(y, s):
        y = np.asarray(y, float)
        if s and s > 0 and y.size >= 3:
            return gaussian_filter1d(y, sigma=float(s), mode='nearest')
        return y

    def _edges(nodes):
        x = np.asarray(nodes, float)
        if x.size == 1:
            dx = 0.5
            return np.array([x[0] - dx, x[0] + dx])
        dx = np.diff(x)
        return np.r_[x[0] - dx[0]/2, 0.5*(x[:-1] + x[1:]), x[-1] + dx[-1]/2]

    def _unique_sorted(vals, tol=1e-12):
        v = np.sort(np.asarray(vals, float))
        if v.size == 0:
            return v
        out = [v[0]]
        for t in v[1:]:
            if abs(t - out[-1]) > tol:
                out.append(t)
        return np.array(out, float)

    def _decimals_from_step(step):
        if step is None or step <= 0:
            return 1
        return int(min(6, max(0, np.ceil(-np.log10(step)))))

    # Prepare and optional flip
    d0 = np.asarray(profile_df['distance'].to_numpy(), float)
    z_s = np.asarray(profile_df['zsurf'].to_numpy(), float)
    z_b = np.asarray(profile_df['zbed'].to_numpy(), float)

    d = d0.copy()
    prof = profile_df.copy()
    flip_true = (flip is True) or (isinstance(flip, str) and flip.lower() == 'true')
    if flip_true or (flip == 'auto' and z_s[-1] < z_s[0]):
        d = d0.max() - d0[::-1]
        z_s = z_s[::-1]
        z_b = z_b[::-1]
        prof = profile_df.iloc[::-1].copy()
        prof['distance'] = d
        prof['zsurf'] = z_s
        prof['zbed'] = z_b

    # Breaks (use for segmenting only)
    diffs = np.diff(d)
    brk_idx = np.where(diffs > float(break_threshold))[0]
    segments = []
    start = 0
    if brk_idx.size == 0:
        segments.append((0, len(d)))
    else:
        for bi in brk_idx:
            segments.append((start, bi + 1))
            start = bi + 1
        segments.append((start, len(d)))

    # Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    else:
        fig = ax.figure

    # Collect measured values
    measured_vals = []
    for _, s in temp_data_dict.items():
        if s is None:
            continue
        vals = list(s.values) if hasattr(s, 'values') else list(getattr(s, 'values', []) or getattr(s, 'values()', []))
        if not vals and isinstance(s, dict):
            vals = list(s.values())
        measured_vals.extend([float(v) for v in vals if np.isfinite(v)])
    if not measured_vals:
        measured_vals = [-1.0, 0.0]

    # Unified temperature step for bins, contours and colorbar ticks
    step = float(temp_step) if temp_step and temp_step > 0 else 0.1
    lo_levels = np.floor(np.nanmin(measured_vals) / step) * step
    any_positive = (np.nanmax(measured_vals) > 0.0 + 1e-12)

    # Discrete bins from lo to 0 by step, plus one positive edge at +step
    neg_to_zero_bins = np.arange(lo_levels, 0.0 + step, step)  # includes 0.0
    levels = _unique_sorted(np.append(neg_to_zero_bins, step))

    # Colormap: blue bins for negatives, single red bin for [0.0, step]
    n_intervals = len(levels) - 1
    if n_intervals <= 0:
        colors = np.array(cmc.vik(1.0)).reshape(1, -1)
    else:
        if n_intervals > 1:
            blue = cmc.vik(np.linspace(0.0, 0.5, n_intervals - 1))
            red = np.array(cmc.vik(1.0)).reshape(1, -1)
            colors = np.vstack([blue, red])
        else:
            colors = np.array(cmc.vik(1.0)).reshape(1, -1)
    cmap_use = ListedColormap(colors, name="icetemp_discrete_runtime") if cmap is None else cmap

    norm = BoundaryNorm(levels, cmap_use.N, clip=True)
    im = None

    # Plot per segment
    for si, (i0, i1) in enumerate(segments):
        if i1 - i0 < 2:
            continue

        d_seg = d[i0:i1]
        zs_seg = _smooth(z_s[i0:i1], smooth_sigma)
        zb_seg = _smooth(z_b[i0:i1], smooth_sigma)

        prof_seg = prof.iloc[i0:i1].copy()
        try:
            T_seg, elevs_seg, xnodes_seg = interpolate_glacier_temperature_field_2d(
                prof_seg, borehole_coords_df, temp_data_dict, depth_dict,
                n_depth=n_depth, n_elev=n_elev
            )

            # Mask using the same x-grid as T
            zs_on_x = np.interp(xnodes_seg, d_seg, zs_seg)
            zb_on_x = np.interp(xnodes_seg, d_seg, zb_seg)
            inside = ((elevs_seg[:, None] >= zb_on_x[None, :]) &
                      (elevs_seg[:, None] <= zs_on_x[None, :]))
            T_masked = np.ma.masked_where(~inside, T_seg)

            # Draw on true, non-uniform grid
            x_edges = _edges(xnodes_seg)
            y_edges = _edges(elevs_seg)
            im = ax.pcolormesh(
                x_edges, y_edges, T_masked,
                cmap=cmap_use, norm=norm, shading='auto', alpha=0.85, zorder=1
            )

            # Contours and CTS (use the same step)
            if plot_contours:
                XX, YY = np.meshgrid(xnodes_seg, elevs_seg)
                ax.contour(XX, YY, T_masked, levels=levels,
                           colors='k', linewidths=0.6, alpha=0.5, zorder=5)
                ax.contour(XX, YY, T_masked, levels=[0.0],
                           colors='red', linewidths=2.2, alpha=0.95, zorder=6)

        except Exception as e:
            print(f"Warning: temperature interpolation for segment {si} failed: {e}")

        # Smoothed outlines
        ax.plot(d_seg, zs_seg, color='k', lw=1.5, label='Surface' if si == 0 else None, zorder=10)
        ax.plot(d_seg, zb_seg, color='k', lw=1.5, ls='--', label='Bed' if si == 0 else None, zorder=10)

    # Boreholes and sensors (unchanged)
    prof_has_xy = ('x' in prof.columns) and ('y' in prof.columns)
    prof_xy = np.column_stack([np.asarray(prof['x'], float), np.asarray(prof['y'], float)]) if prof_has_xy else None
    for _, r in borehole_coords_df.iterrows():
        name = r['name']
        bx = float(str(r['x']).replace(',', '.'))
        by = float(str(r['y']).replace(',', '.'))
        if prof_xy is not None:
            j = int(np.argmin(np.hypot(prof_xy[:, 0] - bx, prof_xy[:, 1] - by)))
            loc = float(prof['distance'].iloc[j])
            surf_elev = float(prof['zsurf'].iloc[j])
        else:
            loc = bx
            surf_elev = np.interp(loc, d, z_s)
        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            pairs = []
            for k in depths.keys():
                try:
                    tval = float(temps[k] if hasattr(temps, '__getitem__') else temps.get(k))
                except Exception:
                    continue
                if np.isfinite(tval):
                    pairs.append((float(depths[k]), tval))
            if pairs:
                pairs.sort(key=lambda p: p[0])
                dep = np.array([p[0] for p in pairs], float)
                ax.plot([loc, loc], [surf_elev - dep.max(), surf_elev], color='k', lw=1.2, zorder=11)
                for dd in dep:
                    ax.plot(loc, surf_elev - dd, 'ko', ms=4, zorder=12)
                ax.text(loc, surf_elev + 6, name, color='red', fontsize=10,
                        ha='center', va='bottom', zorder=13)

    # Colorbar ticks (use the same step)
    if im is not None:
        tick_step = step
        tick_dec = _decimals_from_step(tick_step)
        lo_ticks = np.floor(np.nanmin(measured_vals) / tick_step) * tick_step
        neg_ticks = np.arange(lo_ticks, 0.0 + tick_step, tick_step)

        ticks = list(neg_ticks)
        if not np.isclose(ticks[-1], 0.0, atol=1e-12):
            ticks.append(0.0)
        ticks = [t for t in ticks if t <= 0.0 + 1e-12]
        if any_positive:
            ticks.append(step)

        final_ticks, final_labels = [], []
        seen = set()
        for t in ticks:
            lbl = f"{t:.{tick_dec}f}"
            if lbl not in seen:
                final_ticks.append(t); final_labels.append(lbl); seen.add(lbl)

        cb = fig.colorbar(im, ax=ax, label='Ice Temperature [°C]')
        cb.set_ticks(final_ticks)
        cb.set_ticklabels(final_labels)

    # Axes and formatting
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    ax.set_ylim(np.nanmin(z_b) - 2, np.nanmax(z_s) + 2)
    if title:
        ax.set_title(title)

    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)

    # Legend (add CTS handle explicitly so it always appears)
    cts_handle = Line2D([0], [0], color='red', linewidth=2.2, label='CTS (0°C isotherm)')
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles + [cts_handle], labels + ['CTS (0°C isotherm)']):
        if l and l not in dedup:
            dedup[l] = h
    ax.legend(list(dedup.values()), list(dedup.keys()),
              frameon=True, edgecolor='black', framealpha=1, facecolor='white',
              loc='upper left', ncol=1)

    plt.tight_layout()
    return fig, ax