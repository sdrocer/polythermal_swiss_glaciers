import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge as rio_merge
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter

# For smoothing
from scipy.ndimage import gaussian_filter1d

# ice temperature processing
from processing.thermistor_processing import (
    melting_point_at_pressure,
    interpolate_glacier_temperature_field_2d
)

def _smooth(y, s):
    y = np.asarray(y, float)
    if s and s > 0 and y.size >= 3:
        return gaussian_filter1d(y, s, mode='nearest')
    return y

def _edges(nodes):
    x = np.asarray(nodes, float)
    if x.size == 1:
        return np.array([x[0]-0.5, x[0]+0.5])
    dx = np.diff(x)
    return np.r_[x[0]-dx[0]/2, 0.5*(x[:-1]+x[1:]), x[-1]+dx[-1]/2]

def _decimals(step):
    if step <= 0: return 1
    return int(min(6, max(0, np.ceil(-np.log10(step)))))

def _segment_indices(dist, thr):
    dd = np.diff(dist)
    brk = np.where(dd > thr)[0]
    if brk.size == 0:
        return [(0, dist.size)]
    segs = []
    start = 0
    for b in brk:
        segs.append((start, b+1))
        start = b+1
    segs.append((start, dist.size))
    return segs


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

## icetemp profile plotting helpers ----

def _prepare_profile(profile_df, flip):
    d = np.asarray(profile_df['distance'], float)
    zsurf = np.asarray(profile_df['zsurf'], float)
    zbed  = np.asarray(profile_df['zbed'], float)
    do_flip = (
        (flip is True) or
        (isinstance(flip, str) and str(flip).lower() == 'true') or
        (flip == 'auto' and zsurf[-1] < zsurf[0])
    )
    if do_flip:
        d_new = d.max() - d[::-1]
        pf = profile_df.iloc[::-1].copy()
        pf['distance'] = d_new
        pf['zsurf'] = zsurf[::-1]
        pf['zbed']  = zbed[::-1]
        return d_new, pf['zsurf'].to_numpy(), pf['zbed'].to_numpy(), pf
    return d, zsurf, zbed, profile_df.copy()

def _collect_measured_values(temp_data_dict):
    vals = []
    for td in temp_data_dict.values():
        if td is None:
            continue
        try:
            it = td.values() if isinstance(td, dict) else td
            for v in it:
                try:
                    f = float(v)
                    if np.isfinite(f):
                        vals.append(f)
                except:
                    pass
        except:
            pass
    if not vals:
        vals = [-1.0, 0.0]
    return np.array(vals, float)

def _temperature_levels(measured, step):
    step = step if step > 0 else 0.1
    lo_edge = np.floor(measured.min()/step)*step
    neg_bins = np.arange(lo_edge, 0.0+step, step)
    levels = np.unique(np.append(neg_bins, step))
    return levels

def _discrete_icetemp_cmap_from_levels(levels):
    n_int = len(levels) - 1
    if n_int <= 0:
        colors = np.array(cmc.vik(1.0)).reshape(1, -1)
    else:
        if n_int > 1:
            blue = cmc.vik(np.linspace(0.0, 0.5, n_int-1))
            red  = np.array(cmc.vik(1.0)).reshape(1, -1)
            colors = np.vstack([blue, red])
        else:
            colors = np.array(cmc.vik(1.0)).reshape(1, -1)
    return ListedColormap(colors, name='icetemp_discrete')

def _interpolate_segment(prof_seg, borehole_coords_df, temp_data_dict, depth_dict, n_depth, n_elev):
    return interpolate_glacier_temperature_field_2d(
        prof_seg, borehole_coords_df, temp_data_dict, depth_dict,
        n_depth=n_depth, n_elev=n_elev
    )

def _mask_inside(T, elevs, xnodes, zs_on_x, zb_on_x):
    inside = ((elevs[:, None] >= zb_on_x[None, :]) &
              (elevs[:, None] <= zs_on_x[None, :]))
    return np.ma.masked_where(~inside, T)

def _extract_temperate_layers(T_masked, elevs, xnodes, zs_on_x, zb_on_x,
                              adjust_cts_for_pressure, cts_tol):
    if adjust_cts_for_pressure:
        XX, YY = np.meshgrid(xnodes, elevs)
        h_over = (zs_on_x[None, :] - YY)
        Tm = melting_point_at_pressure(h_over)
        diff = T_masked - Tm
    else:
        diff = T_masked
    temperate_mask = (~T_masked.mask) & (np.abs(diff.data) <= cts_tol)
    if not temperate_mask.any():
        return None
    try:
        from scipy.ndimage import binary_closing, binary_opening
        temperate_mask = binary_closing(temperate_mask, structure=np.ones((3,3)))
        temperate_mask = binary_opening(temperate_mask, structure=np.ones((2,2)))
    except Exception:
        pass

    dz = np.median(np.diff(elevs)) if elevs.size > 1 else 1.0
    near_bed_tol = 1.5 * dz
    min_thick_m  = 2.0 * dz

    top_cts = np.full(xnodes.shape, np.nan)
    bot_cts = np.full(xnodes.shape, np.nan)
    touches_bed = np.zeros(xnodes.shape, bool)

    for j in range(temperate_mask.shape[1]):
        idx = np.where(temperate_mask[:, j])[0]
        if idx.size == 0:
            continue
        z_bot = elevs[idx.min()]
        z_top = elevs[idx.max()]
        if (z_top - z_bot) < min_thick_m:
            continue
        bot_cts[j] = z_bot
        top_cts[j] = z_top
        if abs(z_bot - zb_on_x[j]) <= near_bed_tol:
            touches_bed[j] = True

    return {
        'top': top_cts,
        'bot': bot_cts,
        'touches_bed': touches_bed,
        'diff': diff
    }

def _runs_from_valid(valid_bool):
    vidx = np.where(valid_bool)[0]
    if vidx.size == 0:
        return []
    runs = []
    start = vidx[0]
    for a,b in zip(vidx[:-1], vidx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, vidx[-1]))
    return runs

def _plot_cts_layers(ax, elevs, xnodes, zs_on_x, zb_on_x, info, cts_tol,
                     adjust_cts_for_pressure, label_once_flag):
    top_cts = info['top']
    bot_cts = info['bot']
    touches_bed = info['touches_bed']
    diff = info['diff']

    valid = np.isfinite(top_cts)
    if np.count_nonzero(valid) < 3:
        return label_once_flag

    # bridge single‑point gaps
    v_idx = np.where(valid)[0]
    gaps = np.diff(v_idx)
    for gpos in np.where(gaps == 2)[0]:
        mid = v_idx[gpos] + 1
        valid[mid] = True

    runs = _runs_from_valid(valid)
    basal_frac_threshold = 0.7
    cold_under_frac_threshold = 0.4

    for r0, r1 in runs:
        cols = slice(r0, r1+1)
        cols_idx = np.arange(r0, r1+1)
        if cols_idx.size < 3:
            continue
        run_top = top_cts[cols]
        run_bot = bot_cts[cols]
        run_touches = touches_bed[cols]
        frac_basal = np.count_nonzero(run_touches) / cols_idx.size
        is_basal = frac_basal >= basal_frac_threshold

        if not label_once_flag:
            lbl = ('Estimated CTS (p-adjusted)' if adjust_cts_for_pressure
                   else 'Estimated CTS (≈0°C)')
            label_once_flag = True
        else:
            lbl = None

        if is_basal:
            ax.plot(xnodes[cols], run_top, color='black',
                    linestyle=':', linewidth=2.0, label=lbl, zorder=15)
            continue

        # internal lens: check cold ice below
        cold_under = []
        for jj, col in enumerate(cols_idx):
            z_bot_this = run_bot[jj]
            if not np.isfinite(z_bot_this):
                cold_under.append(False); continue
            below_mask = (elevs < z_bot_this) & (elevs >= zb_on_x[col])
            if not np.any(below_mask):
                cold_under.append(False); continue
            col_diff = diff[:, col]
            if isinstance(col_diff, np.ma.MaskedArray):
                valid_cells = ~col_diff.mask & below_mask
                if not np.any(valid_cells):
                    cold_under.append(False); continue
                vals = col_diff.data[valid_cells]
            else:
                vals = col_diff[below_mask]
            cold_under.append(np.any(vals < -cts_tol))
        cold_frac = np.count_nonzero(cold_under)/len(cold_under) if cold_under else 0.0

        ax.plot(xnodes[cols], run_top, color='red',
                linestyle=':', linewidth=2.0, label=lbl, zorder=15)
        if cold_frac >= cold_under_frac_threshold:
            ax.plot(xnodes[cols], run_bot, color='red',
                    linestyle=':', linewidth=2.0, zorder=14)
    return label_once_flag

## Main function to plot ice temperature profile ----

def plot_icetemp_profile(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    *,
    ax=None,
    title=None,
    flip='auto',
    n_depth=500,
    n_elev=600,
    temp_step=0.1,
    plot_contours=True,
    break_threshold=50.0,
    smooth_sigma=0.0,
    show_cts=True,
    adjust_cts_for_pressure=True,
    cts_tol=0.05,
    cmap=None,
    **deprecated
):
    # Warn on deprecated args
    for k in list(deprecated.keys()):
        print(f"[plot_icetemp_profile] Ignoring unknown/deprecated argument: {k}")

    dist, zsurf, zbed, prof = _prepare_profile(profile_df, flip)
    segments = _segment_indices(dist, float(break_threshold))
    measured = _collect_measured_values(temp_data_dict)
    levels = _temperature_levels(measured, temp_step)
    if cmap is None:
        cmap = _discrete_icetemp_cmap_from_levels(levels)
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    any_positive = (measured.max() > 0 + 1e-12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    else:
        fig = ax.figure

    im = None
    first_seg = True
    cts_label_done = False

    for si, (i0, i1) in enumerate(segments):
        if i1 - i0 < 2:
            continue

        seg_dist = dist[i0:i1]
        zs_seg = _smooth(zsurf[i0:i1], smooth_sigma)
        zb_seg = _smooth(zbed[i0:i1], smooth_sigma)
        prof_seg = prof.iloc[i0:i1].copy()

        try:
            T_seg, elevs_seg, xnodes_seg = _interpolate_segment(
                prof_seg, borehole_coords_df, temp_data_dict, depth_dict,
                n_depth, n_elev
            )
        except Exception as e:
            print(f"Interpolation failed (segment {si}): {e}")
            continue

        zs_on_x = np.interp(xnodes_seg, seg_dist, zs_seg)
        zb_on_x = np.interp(xnodes_seg, seg_dist, zb_seg)
        T_masked = _mask_inside(T_seg, elevs_seg, xnodes_seg, zs_on_x, zb_on_x)

        x_edges = _edges(xnodes_seg)
        y_edges = _edges(elevs_seg)
        im = ax.pcolormesh(
            x_edges, y_edges, T_masked,
            cmap=cmap, norm=norm, shading='auto',
            alpha=0.85, zorder=1
        )

        if plot_contours:
            XX, YY = np.meshgrid(xnodes_seg, elevs_seg)
            ax.contour(
                XX, YY, T_masked,
                levels=levels[:-1],
                colors='k', linewidths=0.6, alpha=0.45, zorder=5
            )

        if show_cts:
            info = _extract_temperate_layers(
                T_masked, elevs_seg, xnodes_seg, zs_on_x, zb_on_x,
                adjust_cts_for_pressure, cts_tol
            )
            if info is not None:
                cts_label_done = _plot_cts_layers(
                    ax, elevs_seg, xnodes_seg, zs_on_x, zb_on_x,
                    info, cts_tol, adjust_cts_for_pressure, cts_label_done
                )

        ax.plot(seg_dist, zs_seg, color='k', lw=1.4,
                label='Surface' if first_seg else None, zorder=10)
        ax.plot(seg_dist, zb_seg, color='k', lw=1.4, ls='--',
                label='Bed' if first_seg else None, zorder=10)
        first_seg = False

    # Borehole markers (same logic retained)
    prof_has_xy = ('x' in prof.columns) and ('y' in prof.columns)
    prof_xy = np.column_stack([prof['x'], prof['y']]) if prof_has_xy else None
    for _, r in borehole_coords_df.iterrows():
        name = r.get('name')
        if name not in temp_data_dict or name not in depth_dict:
            continue
        try:
            bx = float(str(r['x']).replace(',', '.'))
            by = float(str(r['y']).replace(',', '.'))
        except:
            continue
        if prof_xy is not None:
            j = int(np.argmin(np.hypot(prof_xy[:, 0] - bx, prof_xy[:, 1] - by)))
            loc = float(prof['distance'].iloc[j])
            surf_elev = float(prof['zsurf'].iloc[j])
        else:
            loc = bx
            surf_elev = np.interp(loc, dist, zsurf)

        temps  = temp_data_dict[name]
        depths = depth_dict[name]
        pairs = []
        for k, dep in getattr(depths, 'items', lambda: depths.items())():
            try:
                tval = temps[k] if hasattr(temps, '__getitem__') else temps.get(k)
                dnum = float(dep); tnum = float(tval)
                if np.isfinite(dnum) and np.isfinite(tnum):
                    pairs.append((dnum, tnum))
            except:
                continue
        if not pairs:
            continue
        pairs.sort()
        dep_arr = np.array([p[0] for p in pairs])
        t_arr   = np.array([p[1] for p in pairs])

        ax.plot([loc, loc], [surf_elev - dep_arr.max(), surf_elev],
                color='k', lw=1.0, zorder=20)
        for dd, tv in zip(dep_arr, t_arr):
            col = cmap(norm(tv))
            ax.plot(
                loc, surf_elev - dd, marker='o', linestyle='None', markersize=5,
                markerfacecolor=col, markeredgecolor='black', markeredgewidth=0.4,
                zorder=21
            )
        ax.text(loc, surf_elev + 6, name, color='red', fontsize=10,
                ha='center', va='bottom', zorder=22)

    # Colorbar
    if im is not None:
        dec = _decimals(temp_step)
        lo_ticks = np.floor(measured.min()/temp_step)*temp_step
        neg_ticks = np.arange(lo_ticks, 0.0+temp_step, temp_step)
        ticks = list(neg_ticks)
        if not np.any(np.isclose(ticks, 0.0)):
            ticks.append(0.0)
        if any_positive:
            ticks.append(temp_step)
        final = []
        seen = set()
        for t in ticks:
            key = f"{t:.{dec}f}"
            if key not in seen:
                final.append(t); seen.add(key)
        cb = fig.colorbar(im, ax=ax, label='Ice Temperature [°C]')
        cb.set_ticks(final)
        cb.set_ticklabels([f"{t:.{dec}f}" for t in final])

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    ax.set_ylim(np.nanmin(zbed)-2, np.nanmax(zsurf)+2)
    if title:
        ax.set_title(title)

    format_plot(ax=ax, title=title, x_tick_rotation=0,
                legend_loc='upper left', adjust_linewidths=False)

    # Deduplicate legend
    h, l = ax.get_legend_handles_labels()
    uniq = {}
    for hi, li in zip(h, l):
        if li and li not in uniq:
            uniq[li] = hi
    ax.legend(uniq.values(), uniq.keys(),
              frameon=True, edgecolor='black', framealpha=1,
              facecolor='white', loc='upper left', ncol=1)
    plt.tight_layout()
    return fig, ax