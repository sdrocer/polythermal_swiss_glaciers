# import necessary modules
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import re
import os

from shapely.vectorized import contains

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc
from PIL import Image
from pathlib import Path
from typing import Union, Sequence

import subprocess

# Import required geodata helpers (centralized in download_geodata.py)
from processing.geodata_download import (
    SWISS_CRS,
    download_swisstopo_wms,
    _load_vector,
    _points_to_gdf
)

from processing import gpr_plotting as gprp
from processing.geodata_processing import *

def plot_switzerland_glacier_overview(
    *,
    pixel_size: float = 120.0,
    glacier_extent_layer: str = "ch.swisstopo.geologie-gletscherausdehnung",
    lakes_layer: str = "ch.bafu.vec25-seen",
    canton_layer: str = "ch.swisstopo.swissboundaries3d-kanton-flaeche.fill",
    country_layer: str = "ch.swisstopo.swissboundaries3d-land-flaeche.fill",
    glacier_color: str = "#b3d9ff", lakes_color: str = "#c0c0c0", outline_color: str = "black",
    country_outline_width: int = 3,
    canton_outline_width: int = 1,
    figsize=(12, 7),
    show_scale: bool = True,
    scalebar_km: float = 50,
    font_family: str = "Arial",
    dpi: int | None = None,
    trim_whitespace: bool = True,
    trim_margin_km: float = 4.0,
    city_labels: dict | None = None,
    city_fontsize: int = 28,
    city_marker_size: int = 70,
    field_site_markers: Union[list, dict, None] = None,
    savefig_path: Union[str, Path, None] = None,
    max_wms_dim: int = 6000,
    fallback_pixel_sizes: tuple = (80, 100, 120, 150),
    upscale_to_first: bool = True,
    upscale_interpolation: str = "lanczos",
    verbose_wms: bool = True,
    annotate_field_sites: bool = True,
    field_site_label_fontsize: int = 18,
    field_site_label_color: str = "black",
    field_site_label_weight: str = "bold",
    field_site_label_offset: tuple[float, float] | None = None,
    field_site_label_ha: str = "left",
    field_site_label_va: str = "bottom",
    field_site_label_box: dict | None = None,
    field_site_label_offsets: dict | None = None,
    field_site_label_line: bool = True,
    field_site_label_line_color: str = "black",
    field_site_label_line_width: float = 1.5,
    field_site_label_line_anchor: dict | None = None,
    transparent_background: bool = False,
):
    try:
        plt.rcParams["font.family"] = font_family
    except Exception:
        pass

    def _anchor_offset(anchor, box_w, box_h):
        if anchor == "ul":
            return (0, box_h)
        elif anchor == "ur":
            return (box_w, box_h)
        elif anchor == "ll":
            return (0, 0)
        elif anchor == "lr":
            return (box_w, 0)
        elif anchor == "c":
            return (box_w/2, box_h/2)
        else:
            return (0, box_h)

    full_bbox = (2420000, 1030000, 2900000, 1350000)

    import PIL.Image as _PILImage
    from PIL import UnidentifiedImageError

    def _download(layer_id, bbox):
        target_px = pixel_size
        def _compute_dims(px_size):
            w_m = bbox[2] - bbox[0]
            h_m = bbox[3] - bbox[1]
            w_px = int(round(w_m / px_size))
            h_px = int(round(h_m / px_size))
            scale = 1.0
            if max(w_px, h_px) > max_wms_dim:
                scale = max(w_px / max_wms_dim, h_px / max_wms_dim)
                w_px = int(round(w_px / scale))
                h_px = int(round(h_px / scale))
            return w_px, h_px, scale
        tried = [target_px] + [ps for ps in fallback_pixel_sizes if ps != target_px]
        first_dims = None
        first_arr = None
        first_extent = None
        for i, px in enumerate(tried):
            w_px, h_px, clamp_scale = _compute_dims(px)
            req_px_size = (bbox[2]-bbox[0]) / w_px
            try:
                arr, ext, _ = download_swisstopo_wms(
                    layer=layer_id,
                    bbox=bbox,
                    pixel_size=req_px_size,
                    img_format="image/png",
                    transparent=True
                )
                if i == 0:
                    first_arr, first_extent = arr, ext
                    first_dims = (w_px, h_px)
                if i == 0 or not upscale_to_first:
                    return arr, ext
                if upscale_to_first and first_dims:
                    im_mode = "RGBA" if arr.shape[2] == 4 else "RGB"
                    pil_im = _PILImage.fromarray(arr, mode=im_mode)
                    pil_im = pil_im.resize(first_dims, _PILImage.Resampling.LANCZOS if upscale_interpolation.lower()=="lanczos" else _PILImage.Resampling.BILINEAR)
                    up_arr = np.array(pil_im)
                    return up_arr, ext
            except (UnidentifiedImageError, OSError, RuntimeError) as e:
                if verbose_wms:
                    print(f"WMS fail for {layer_id} at pixel_size≈{px} (requested dims ~{w_px}x{h_px}): {e}")
                continue
        if first_arr is not None:
            if verbose_wms:
                print(f"Using degraded map for {layer_id} (only initial partial success).")
            return first_arr, first_extent
        raise RuntimeError(f"All WMS attempts failed for layer {layer_id}")

    def _solid_fill(src_rgba, hex_color):
        mask = src_rgba[..., 3] > 0
        out = np.zeros_like(src_rgba)
        if hex_color.startswith("#"):
            r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
        else:
            r, g, b = (179, 217, 255)
        out[mask, 0] = r; out[mask, 1] = g; out[mask, 2] = b; out[mask, 3] = 255
        return out

    def _extract_edge(mask: np.ndarray, width: int, use_scipy: bool = True):
        if width < 1:
            return np.zeros_like(mask, dtype=bool)
        if use_scipy:
            try:
                from scipy.ndimage import binary_erosion, binary_dilation
                eroded = binary_erosion(mask)
                edge = mask & (~eroded)
                if width > 1:
                    edge = binary_dilation(edge, iterations=width - 1)
                return edge
            except Exception:
                pass
        up = np.zeros_like(mask); up[1:] = mask[:-1]
        down = np.zeros_like(mask); down[:-1] = mask[1:]
        left = np.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = np.zeros_like(mask); right[:, :-1] = mask[:, 1:]
        edge = mask & (~(up & down & left & right))
        for _ in range(max(0, width - 1)):
            dil = edge.copy()
            dil[:-1] |= edge[1:]
            dil[1:] |= edge[:-1]
            dil[:, :-1] |= edge[:, 1:]
            dil[:, 1:] |= edge[:, :-1]
            edge = dil
        return edge

    canton_rgba, extent = _download(canton_layer, full_bbox)
    country_rgba, _ = _download(country_layer, full_bbox)
    glaciers_rgba, _ = _download(glacier_extent_layer, full_bbox)
    lakes_rgba, _ = _download(lakes_layer, full_bbox)

    minx, maxx, miny, maxy = extent
    h, w = canton_rgba.shape[:2]
    px_w = (maxx - minx) / w
    px_h = (maxy - miny) / h

    if trim_whitespace:
        mask = country_rgba[..., 3] > 0
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size and cols.size:
            r0, r1 = rows[0], rows[-1]
            c0, c1 = cols[0], cols[-1]
            mpx_x = int((trim_margin_km * 1000) / px_w)
            mpx_y = int((trim_margin_km * 1000) / px_h)
            r0 = max(0, r0 - mpx_y); r1 = min(h - 1, r1 + mpx_y)
            c0 = max(0, c0 - mpx_x); c1 = min(w - 1, c1 + mpx_x)
            def _crop(a): return a[r0:r1+1, c0:c1+1, :]
            canton_rgba = _crop(canton_rgba)
            country_rgba = _crop(country_rgba)
            glaciers_rgba = _crop(glaciers_rgba)
            lakes_rgba = _crop(lakes_rgba)
            new_minx = minx + c0 * px_w
            new_maxx = minx + (c1 + 1) * px_w
            new_maxy = maxy - r0 * px_h
            new_miny = maxy - (r1 + 1) * px_h
            extent = (new_minx, new_maxx, new_miny, new_maxy)

    swiss_gdf = gpd.read_file(
        "/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/polythermal_swiss_glaciers/products/figures/maps/switzerland_shapefile/pays.shp"
    ).to_crs(SWISS_CRS)

    glaciers_fill = _solid_fill(glaciers_rgba, glacier_color)
    lakes_fill = _solid_fill(lakes_rgba, lakes_color)

    h, w = canton_rgba.shape[:2]
    minx, maxx, miny, maxy = extent
    x = np.linspace(minx, maxx, w)
    y = np.linspace(miny, maxy, h)
    xx, yy = np.meshgrid(x, y)
    mask = contains(swiss_gdf.geometry.unary_union, xx, yy)
    mask = np.flipud(mask)

    if transparent_background:
        white_country = np.zeros_like(canton_rgba)
        white_country[..., 3] = 0
        white_country[mask, 0:3] = 255
        white_country[mask, 3] = 255
        country_rgba = white_country


    alpha_thresh = 180
    canton_mask = canton_rgba[..., 3] >= alpha_thresh
    country_mask = country_rgba[..., 3] >= alpha_thresh

    country_edge = _extract_edge(country_mask, country_outline_width)
    canton_edge = _extract_edge(canton_mask, canton_outline_width)
    canton_edge &= ~country_edge

    def _edge_to_rgba(edge_mask: np.ndarray) -> np.ndarray:
        out = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 4), dtype=np.uint8)
        out[edge_mask, 0:3] = 0
        out[edge_mask, 3] = 255
        return out

    canton_outline = _edge_to_rgba(canton_edge)
    country_outline = _edge_to_rgba(country_edge)

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='none' if transparent_background else 'white')
    ax.set_facecolor('none' if transparent_background else 'white')
    ax.imshow(country_rgba, extent=extent, origin="upper", zorder=1)
    ax.imshow(glaciers_fill, extent=extent, origin="upper", zorder=2)
    ax.imshow(canton_outline, extent=extent, origin="upper", zorder=3)
    ax.imshow(country_outline, extent=extent, origin="upper", zorder=4)
    ax.imshow(lakes_fill, extent=extent, origin="upper", zorder=5)

    if city_labels is None:
        city_labels = {
            # "Genève": (2483000, 1118000),
            "Bern":   (2600500, 1196500),
            "Zürich": (2683000, 1244000)
        }
    for name, (cx, cy) in city_labels.items():
        if extent[0] <= cx <= extent[1] and extent[2] <= cy <= extent[3]:
            ax.scatter([cx], [cy], s=city_marker_size, zorder=10, color="black")
            ax.text(cx + (extent[1]-extent[0]) * 0.007,
                    cy + (extent[3]-extent[2]) * 0.006,
                    name, ha="left", va="bottom",
                    fontsize=city_fontsize, color="black", zorder=11)

    canton_labels = {
        "VS": (2620000, 1125000),
        "GR": (2765000, 1165000),
        "VD": (2569332, 1133979,)
    }
    for label, (cx, cy) in canton_labels.items():
        if extent[0] <= cx <= extent[1] and extent[2] <= cy <= extent[3]:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=city_fontsize + 2, color="gray", weight="bold", style="italic", zorder=12)

    # rect_colors = {"AH": "#1f77b4", "CJ": "#2ca02c", "HS": "#ff7f0e"}

    if field_site_markers:
        if isinstance(field_site_markers, dict):
            coords_iter = field_site_markers.items()
            xs = []; ys = []
            for code, (fx, fy) in coords_iter:
                if extent[0] <= fx <= extent[1] and extent[2] <= fy <= extent[3]:
                    xs.append(fx); ys.append(fy)
            if xs:
                ax.scatter(xs, ys, s=city_marker_size * 1.5,
                           zorder=10, color="red", marker="D", antialiased=False)
            if annotate_field_sites:
                if field_site_label_box is None:
                    field_site_label_box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
                if field_site_label_offset is None:
                    base_dx = (extent[1] - extent[0]) * 0.014
                    base_dy = (extent[3] - extent[2]) * 0.014
                else:
                    base_dx, base_dy = field_site_label_offset
                for code, (fx, fy) in field_site_markers.items():
                    if not (extent[0] <= fx <= extent[1] and extent[2] <= fy <= extent[3]):
                        continue
                    if field_site_label_offsets and code in field_site_label_offsets:
                        dx, dy = field_site_label_offsets[code]
                    else:
                        dx, dy = base_dx, base_dy
                    label_color = 'k'
                    txt = ax.text(
                        fx + dx, fy + dy, code,
                        ha=field_site_label_ha,
                        va=field_site_label_va,
                        fontsize=field_site_label_fontsize,
                        color=label_color,
                        weight=field_site_label_weight,
                        zorder=11,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=label_color, alpha=0.7)
                    )
                    fig.canvas.draw()
                    if field_site_label_line:
                        renderer = ax.figure.canvas.get_renderer()
                        bbox = txt.get_window_extent(renderer=renderer)
                        inv = ax.transData.inverted()
                        anchor = "ul"
                        if field_site_label_line_anchor and code in field_site_label_line_anchor:
                            anchor = field_site_label_line_anchor[code]
                        box_w = bbox.width
                        box_h = bbox.height
                        off_x, off_y = _anchor_offset(anchor, box_w, box_h)
                        anchor_disp = (bbox.x0 + off_x, bbox.y0 + off_y)
                        anchor_data = inv.transform(anchor_disp)
                        ax.plot([fx, anchor_data[0]], [fy, anchor_data[1]],
                                color=field_site_label_line_color,
                                lw=field_site_label_line_width,
                                zorder=10)
        else:
            for (fx, fy) in field_site_markers:
                if extent[0] <= fx <= extent[1] and extent[2] <= fy <= extent[3]:
                    ax.scatter([fx], [fy], s=city_marker_size * 1.5,
                               zorder=10, color="red", marker="D", antialiased=False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if show_scale:
        sb_len = scalebar_km * 1000
        x0 = extent[0] + (extent[1] - extent[0]) * 0.045
        y0 = extent[2] + (extent[3] - extent[2]) * 0.085
        ax.plot([x0, x0 + sb_len], [y0, y0],
                color=outline_color, lw=3, solid_capstyle="butt", zorder=20)
        ax.text(x0 + sb_len / 2, y0 + sb_len * 0.17,
                f"{int(scalebar_km)} km",
                ha="center", va="bottom",
                fontsize=15, color=outline_color)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Figure saved to: {savefig_path}")

    return fig, ax, extent

def _figure_to_array(fig, dpi=None):
    """Render a Matplotlib Figure to RGBA numpy array."""
    if dpi is not None:
        fig.set_dpi(dpi)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    rgba = np.empty_like(buf)
    rgba[..., 0] = buf[..., 1]
    rgba[..., 1] = buf[..., 2]
    rgba[..., 2] = buf[..., 3]
    rgba[..., 3] = buf[..., 0]
    return rgba

def create_mosaic_figure(
    *,
    mosaic: str,
    panels: dict,
    figsize=(12, 8),
    dpi: int = 300,
    panel_labels: dict | None = None,
    label_style: str = "{label}",
    label_fontsize: int = 12,
    label_weight: str = "bold",
    label_xy: tuple[float, float] = (0.01, 0.99),
    label_ha: str = "left",
    label_va: str = "top",
    label_box: dict | None = None,
    background: str = "white",
    pad_inches: float = 0.02,
    tight: bool = True,
    savefig_path: Union[str, Path, None] = None
):
    """
    Generic mosaic compositor without captions.

    NOTE:
      Removed unsupported 'layout' argument to subplot_mosaic for older Matplotlib.
      If tight=True we use constrained_layout; else we call tight_layout at end.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background, constrained_layout=tight)
    axs = fig.subplot_mosaic(mosaic)  # no 'layout' kw (caused TypeError)

    # Auto labels if not provided
    if panel_labels is None:
        seq = []
        for ch in mosaic.replace("\n", ""):
            if ch != " " and ch not in seq:
                seq.append(ch)
        panel_labels = {k: k for k in seq}

    if label_box is None:
        label_box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)

    for key, ax in axs.items():
        src = panels.get(key, None)
        arr = None
        if isinstance(src, Figure):
            arr = _figure_to_array(src)
        elif isinstance(src, (str, Path)):
            try:
                arr = np.array(Image.open(src).convert("RGBA"))
            except Exception as e:
                ax.text(0.5, 0.5, f"Image error:\n{e}", ha="center", va="center", fontsize=8)
        elif isinstance(src, np.ndarray):
            arr = src
        elif src is None:
            pass
        else:
            ax.text(0.5, 0.5, f"Unsupported: {type(src)}", ha="center", va="center", fontsize=8)

        if arr is not None:
            ax.imshow(arr)

        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        if key in panel_labels:
            lab = label_style.format(label=panel_labels[key])
            ax.text(
                label_xy[0], label_xy[1], lab,
                transform=ax.transAxes,
                ha=label_ha, va=label_va,
                fontsize=label_fontsize, weight=label_weight,
                bbox=label_box
            )

    # If not using constrained_layout, apply tight_layout manually
    if not tight:
        fig.tight_layout(pad=pad_inches)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        print(f"Mosaic saved: {savefig_path}")

    return fig, axs

def build_glacier_map_mosaic(
    image_dir: Union[str, Path],
    field_sites: dict,
    map_kwargs: dict | None = None,
    right_sites: list[str] | None = None,
    bottom_sites: list[str] | None = None,
    figsize=(13, 9),
    dpi: int = 300,
    pixel_size_map: float | None = None,
    savefig_path: Union[str, Path, None] = None,
    label_fontsize: int = 16,  # +4 (was 12)
    label_box: dict | None = None,
    trim_map_border: bool = True,
    trim_tolerance: int = 5,
    gutter: float = 0.05,
    map_render_dpi: int = 450,
    map_interpolation: str = "none",
    override_map_pixel_size: float | None = None,
    map_row_span: int = 2,
    right_col_width_factor: float = 1.25,
    highlight_bottom: bool = False,
    highlight_right: bool = True,
    highlight_group_edgecolor: str = "red",
    highlight_group_linewidth: float = 3.0,
    highlight_group_pad: float = 0.0,
    export_pad_inches: float = 0.01,
    highlight_inset: float = 0.001
):
    """
    Composite layout:
      - Map spans first two rows (rows 0 & 1) columns 0-2.
      - Right column: 3 images (rows 0,1,2) column 3.
      - Bottom row (row 2) columns 0-2: focus images.
      - Single red rectangle around entire bottom row if highlight_bottom=True.
    """
    if map_row_span < 2:
        raise ValueError("map_row_span must be >=2 to make sense for this layout.")

    image_dir = Path(image_dir)
    if map_kwargs is None:
        map_kwargs = {}
    if pixel_size_map is not None:
        map_kwargs.setdefault("pixel_size", pixel_size_map)
    if override_map_pixel_size is not None:
        map_kwargs["pixel_size"] = override_map_pixel_size

    if right_sites is None:
        right_sites = ["AH", "CJ", "HS"]
    if bottom_sites is None:
        bottom_sites = ["SR", "TO", "CV"]
    if len(right_sites) != 3 or len(bottom_sites) != 3:
        raise ValueError("right_sites and bottom_sites must each contain exactly 3 codes.")

    # Find images
    def _find_image(code: str) -> Path | None:
        for p in image_dir.glob(f"{code}*lowres.jpg"):
            return p
        return None

    codes_all = right_sites + bottom_sites
    img_paths = {c: _find_image(c) for c in codes_all}

    # Reference (middle bottom image)
    ref_code = bottom_sites[1]
    ref_path = img_paths.get(ref_code)
    if ref_path is None:
        raise FileNotFoundError(f"Reference image '{ref_code}*lowres.jpg' not found.")

    ref_img = Image.open(ref_path)
    ref_w, ref_h = ref_img.size
    ref_ratio = ref_w / ref_h

    def _crop_to_ratio(im: Image.Image, r: float):
        w, h = im.size
        cr = w / h
        if abs(cr - r) < 1e-3:
            return im
        if cr > r:
            new_w = int(h * r)
            x0 = (w - new_w) // 2
            return im.crop((x0, 0, x0 + new_w, h))
        else:
            new_h = int(w / r)
            y0 = (h - new_h) // 2
            return im.crop((0, y0, w, y0 + new_h))

    def _prep(code: str):
        p = img_paths.get(code)
        if p is None:
            return None
        try:
            im = Image.open(p)
            im = _crop_to_ratio(im, ref_ratio).resize((ref_w, ref_h), Image.LANCZOS)
            return np.array(im.convert("RGBA"))
        except Exception:
            return None

    img_arrays = {c: _prep(c) for c in codes_all}

    # Map (high-res)
    map_fig, _ = plot_switzerland_glacier_overview(**map_kwargs)
    map_arr = _figure_to_array(map_fig, dpi=map_render_dpi)
    plt.close(map_fig)

    # Trim white
    def _trim_rgba_border(arr: np.ndarray, tol: int = 5):
        if arr.shape[2] == 4:
            alpha = arr[..., 3] > 0
        else:
            alpha = np.ones(arr.shape[:2], bool)
        rgb = arr[..., :3]
        diff = np.abs(rgb.astype(int) - 255)
        mask = (diff.max(axis=2) > tol) & alpha
        coords = np.argwhere(mask)
        if coords.size == 0:
            return arr
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return arr[y0:y1+1, x0:x1+1, :]

    if trim_map_border:
        map_arr = _trim_rgba_border(map_arr, tol=trim_tolerance)

    # Layout: 3 rows, 4 columns (right column scaled)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        nrows=3,
        ncols=4,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1, right_col_width_factor],
        wspace=gutter,
        hspace=gutter,
        figure=fig
    )

    # Map
    ax_map = fig.add_subplot(gs[0:map_row_span, 0:3])
    ax_map.imshow(map_arr, interpolation=map_interpolation)
    ax_map.set_axis_off()

    # Right column images
    right_axes = []
    for i, code in enumerate(right_sites):
        ax_r = fig.add_subplot(gs[i, 3])
        right_axes.append(ax_r)
        arr = img_arrays.get(code)
        if arr is not None:
            ax_r.imshow(arr)
        else:
            ax_r.text(0.5, 0.5, f"Missing\n{code}", ha="center", va="center", fontsize=8)
        ax_r.set_axis_off()
        _add_corner_label(ax_r, code, fontsize=label_fontsize, box=label_box)

    # Bottom row images
    bottom_axes = []
    for j, code in enumerate(bottom_sites):
        ax_b = fig.add_subplot(gs[2, j])
        bottom_axes.append(ax_b)
        arr = img_arrays.get(code)
        if arr is not None:
            ax_b.imshow(arr)
        else:
            ax_b.text(0.5, 0.5, f"Missing\n{code}", ha="center", va="center", fontsize=8)
        ax_b.set_axis_off()
        _add_corner_label(ax_b, code, fontsize=label_fontsize, box=label_box)

    # Adjust outer padding so positions are final
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    def _add_group_highlight(axes_list):
        if not axes_list:
            return
        renderer = fig.canvas.get_renderer()
        x0 = float("inf"); y0 = float("inf")
        x1 = float("-inf"); y1 = float("-inf")
        for ax in axes_list:
            try:
                bb = ax.get_tightbbox(renderer)
            except Exception:
                bb = ax.get_window_extent(renderer)
            x0 = min(x0, bb.x0); y0 = min(y0, bb.y0)
            x1 = max(x1, bb.x1); y1 = max(y1, bb.y1)
        fig_bb = fig.bbox
        fx0 = (x0 - fig_bb.x0) / fig_bb.width
        fx1 = (x1 - fig_bb.x0) / fig_bb.width
        fy0 = (y0 - fig_bb.y0) / fig_bb.height
        fy1 = (y1 - fig_bb.y0) / fig_bb.height
        pad = highlight_group_pad
        fx0p = max(0.0, fx0 - pad + highlight_inset)
        fx1p = min(1.0, fx1 + pad - highlight_inset)
        fy0p = max(0.0, fy0 - pad + highlight_inset)
        fy1p = min(1.0, fy1 + pad - highlight_inset)
        rect = Rectangle(
            (fx0p, fy0p), fx1p - fx0p, fy1p - fy0p,
            transform=fig.transFigure, fill=False,
            edgecolor=highlight_group_edgecolor,
            linewidth=highlight_group_linewidth,
            joinstyle="miter", zorder=50
        )
        fig.add_artist(rect)

    # Draw group highlight (priority: right column if requested)
    if highlight_right and not highlight_bottom:
        _add_group_highlight(right_axes)
    elif highlight_bottom and not highlight_right:
        _add_group_highlight(bottom_axes)
    elif highlight_right and highlight_bottom:
        # If both True, highlight both (can change logic if undesired)
        _add_group_highlight(right_axes)
        _add_group_highlight(bottom_axes)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a slight pad to avoid cutting outer pixels / red frame
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight", pad_inches=export_pad_inches)
        print(f"Composite saved: {savefig_path}")

    return fig

def build_tyntag_timeseries_one_row(
    *,
    order: tuple[str, str, str],
    plotters: dict,
    depths_current: dict,
    depths_initial: dict,
    labels: dict,
    lower_y_limit: float = -3.0,
    figsize=(15, 4.8),
    dpi: int = 300,
    panel_labels = {"AH": "a", "CJ": "b", "HS": "c"},
    legend_position: str = "lower_right",
    legend_fontsize: int = 11,
    legend_labels: tuple[str, str] = ("1TT", "2TT"),
    legend_anchor: tuple[float, float] = (0.985, -0.04),
    base_fontsize: int = 18,
    smooth_days: float | int | None = 0,
    savefig_path: Union[str, Path, None] = None,
    annotation_y=None,
    annotation_spacing: float = 0.08,
    annotation_arrow_hide_dy: float = 0.06,
    annotation_dx_pts: int = 6,
    annotation_positions=None,
    annotation_fontsize: int | None = None,
    xlabel_y: float = 0.012,
    wspace: float = 0.12,
    exclude_sensors: dict | None = None  # NEW argument
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from processing.thermistor_plotting import build_profile_color_map

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=1, ncols=3, figure=fig, wspace=wspace, hspace=0.0)

    shared_y = None
    for j, code in enumerate(order):
        ax = fig.add_subplot(gs[0, j], sharey=shared_y)
        if shared_y is None:
            shared_y = ax

        pl = plotters[code]
        dcur = depths_current[code]
        dinit = depths_initial.get(code, None)
        blabs = labels[code]

        # For each glacier in the loop:
        if exclude_sensors and code in exclude_sensors:
            sensors_to_exclude = set(exclude_sensors[code])
            blabs = [lab for lab in blabs if lab not in sensors_to_exclude]
            # Remove depths for excluded borehole
            if len(blabs) == 1 and isinstance(dcur, list) and len(dcur) == 4:
                orig_labels = labels[code]
                idx = orig_labels.index(blabs[0])  # 0 or 1
                if idx == 0:
                    dcur = dcur[:2]
                    if dinit is not None and isinstance(dinit, list) and len(dinit) == 4:
                        dinit = dinit[:2]
                else:
                    dcur = dcur[2:]
                    if dinit is not None and isinstance(dinit, list) and len(dinit) == 4:
                        dinit = dinit[2:]
            # --- MANUAL ANNOTATION OVERRIDE FOR GT1TT ---
            if code == "GT" and "GT2TT" in sensors_to_exclude and len(blabs) == 1 and blabs[0] == "GT1TT":
                # Override current and initial depths for annotation
                dcur = [1.4, 6.4]
                dinit = [4.2, 9.2]

        # Resolve per-panel annotation positions
        if isinstance(annotation_positions, dict):
            ann_pos_panel = annotation_positions.get(code, None)
        else:
            ann_pos_panel = annotation_positions  # could be list/tuple[4] or None

        pl.plot_multiple_ntc_boreholes(
            savepath=None,
            depths=dcur,
            borehole_labels=blabs,
            lower_y_limit=lower_y_limit,
            initial_depths=dinit,
            ax=ax,
            show_title=False,
            show_legend=False,
            legend_outside=False,
            show_xlabel=False,
            show_xticklabels=True,
            base_fontsize=base_fontsize,
            smooth_days=smooth_days,
            annotation_y=annotation_y,
            annotation_spacing=annotation_spacing,
            annotation_arrow_hide_dy=annotation_arrow_hide_dy,
            annotation_dx_pts=annotation_dx_pts,
            annotation_positions=ann_pos_panel,
            annotation_fontsize=annotation_fontsize,
        )

        # Ensure only the left panel shows the y‑axis label and tick labels
        if j > 0:
            ax.set_ylabel("")                 # remove duplicated ylabel
            ax.tick_params(labelleft=False)   # hide y tick labels

        # Lower left glacier code (colored, white box)
        label_color = "k"
        ax.text(0.02, 0.02, code, transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=base_fontsize, weight="bold",
                color=label_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=label_color, alpha=0.7))

        # Upper left panel label (a/b/c, black, white box)
        panel_labels = panel_labels
        ax.text(0.02, 0.98, f"({panel_labels.get(code, '')})", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=base_fontsize, weight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0))

    # Figure-level legend with consistent colors
    cmap = build_profile_color_map(list(legend_labels))
    col1 = cmap[legend_labels[0]]
    col2 = cmap[legend_labels[1]]
    handles = [
        Line2D([0], [0], color=col1, lw=3, linestyle='-'),
        Line2D([0], [0], color=col2, lw=3, linestyle='-'),
    ]
    if legend_position == "lower_right":
        fig.legend(
            handles, list(legend_labels),
            loc="lower right", bbox_to_anchor=legend_anchor,
            ncol=2,
            frameon=True, fancybox=False, edgecolor="black", framealpha=1, facecolor="white",
            fontsize=legend_fontsize
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.96,
                            bottom=max(0.26, xlabel_y + 0.14))
    else:
        fig.legend(
            handles, list(legend_labels),
            loc="upper center", ncol=2,
            frameon=True, fancybox=False, edgecolor="black", framealpha=1, facecolor="white",
            fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.985)
        )
        fig.subplots_adjust(top=0.91, left=0.07, right=0.98,
                            bottom=max(0.18, xlabel_y + 0.12))

    fig.supxlabel("Time", fontsize=base_fontsize, y=xlabel_y)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight")

    return fig

def build_tyntag_timeseries_two_row(
    *,
    top_row: tuple[str, str, str],
    bottom_row: tuple[str, str, str],
    plotters: dict,
    depths_current: dict,
    depths_initial: dict,
    labels: dict,
    lower_y_limit_top: float = -3.0,
    lower_y_limit_bottom: float = -3.0,
    figsize=(15, 9.6),
    dpi: int = 300,
    panel_labels: dict = {
        "AH": "a", "CJ": "b", "HS": "c",
        "SR": "d", "GT": "e", "CV": "f"
    },
    legend_position: str = "lower_right",
    legend_fontsize: int = 11,
    legend_labels: tuple[str, str] = ("1TT", "2TT"),
    legend_anchor: tuple[float, float] = (0.985, -0.04),
    base_fontsize: int = 18,
    smooth_days: float | int | None = 0,
    savefig_path: Union[str, Path, None] = None,
    annotation_y=None,
    annotation_spacing: float = 0.08,
    annotation_arrow_hide_dy: float = 0.06,
    annotation_dx_pts: int = 6,
    annotation_positions=None,
    annotation_fontsize: int | None = None,
    xlabel_y: float = 0.012,
    wspace: float = 0.12,
    hspace: float = 0.25,
    exclude_sensors: dict | None = None
):
    """
    Create a 2-row × 3-column grid of TinyTag timeseries plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator  # <-- NEW IMPORT
    from processing.thermistor_plotting import build_profile_color_map

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=2, ncols=3, figure=fig, wspace=wspace, hspace=hspace)

    shared_y_top = None
    shared_y_bottom = None
    
    all_codes = top_row + bottom_row
    
    for row_idx, row_codes in enumerate([top_row, bottom_row]):
        shared_y = shared_y_top if row_idx == 0 else shared_y_bottom
        lower_y_limit = lower_y_limit_top if row_idx == 0 else lower_y_limit_bottom
        
        for col_idx, code in enumerate(row_codes):
            ax = fig.add_subplot(gs[row_idx, col_idx], sharey=shared_y)
            
            if shared_y is None:
                if row_idx == 0:
                    shared_y_top = ax
                else:
                    shared_y_bottom = ax
                shared_y = ax

            pl = plotters[code]
            dcur = depths_current[code]
            dinit = depths_initial.get(code, None)
            blabs = labels[code]

            # Handle sensor exclusion
            if exclude_sensors and code in exclude_sensors:
                sensors_to_exclude = set(exclude_sensors[code])
                blabs = [lab for lab in blabs if lab not in sensors_to_exclude]
                
                if len(blabs) == 1 and isinstance(dcur, list) and len(dcur) == 4:
                    orig_labels = labels[code]
                    idx = orig_labels.index(blabs[0])
                    if idx == 0:
                        dcur = dcur[:2]
                        if dinit is not None and isinstance(dinit, list) and len(dinit) == 4:
                            dinit = dinit[:2]
                    else:
                        dcur = dcur[2:]
                        if dinit is not None and isinstance(dinit, list) and len(dinit) == 4:
                            dinit = dinit[2:]
                
                if code == "GT" and "GT2TT" in sensors_to_exclude and len(blabs) == 1 and blabs[0] == "GT1TT":
                    dcur = [1.4, 6.4]
                    dinit = [4.2, 9.2]

            if isinstance(annotation_positions, dict):
                ann_pos_panel = annotation_positions.get(code, None)
            else:
                ann_pos_panel = annotation_positions

            pl.plot_multiple_ntc_boreholes(
                savepath=None,
                depths=dcur,
                borehole_labels=blabs,
                lower_y_limit=lower_y_limit,
                initial_depths=dinit,
                ax=ax,
                show_title=False,
                show_legend=False,
                legend_outside=False,
                show_xlabel=False,
                show_xticklabels=True,
                base_fontsize=base_fontsize,
                smooth_days=smooth_days,
                annotation_y=annotation_y,
                annotation_spacing=annotation_spacing,
                annotation_arrow_hide_dy=annotation_arrow_hide_dy,
                annotation_dx_pts=annotation_dx_pts,
                annotation_positions=ann_pos_panel,
                annotation_fontsize=annotation_fontsize,
            )

            # Only leftmost column shows y-axis label
            if col_idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            
            # <-- NEW: Force integer-only y-axis ticks
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))

            # Glacier code label (lower left)
            label_color = "k"
            ax.text(0.02, 0.02, code, transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=base_fontsize, weight="bold",
                    color=label_color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=label_color, alpha=0.7))

            # Panel label (upper left)
            ax.text(0.02, 0.98, f"({panel_labels.get(code, '')})", transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=base_fontsize, weight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0))

    # Figure-level legend with consistent colors
    cmap = build_profile_color_map(list(legend_labels))
    col1 = cmap[legend_labels[0]]
    col2 = cmap[legend_labels[1]]
    handles = [
        Line2D([0], [0], color=col1, lw=3, linestyle='-'),
        Line2D([0], [0], color=col2, lw=3, linestyle='-'),
    ]
    
    if legend_position == "lower_right":
        fig.legend(
            handles, list(legend_labels),
            loc="lower right", bbox_to_anchor=legend_anchor,
            ncol=2,
            frameon=True, fancybox=False, edgecolor="black", framealpha=1, facecolor="white",
            fontsize=legend_fontsize
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.96,
                            bottom=max(0.14, xlabel_y + 0.08))
    else:
        fig.legend(
            handles, list(legend_labels),
            loc="upper center", ncol=2,
            frameon=True, fancybox=False, edgecolor="black", framealpha=1, facecolor="white",
            fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.985)
        )
        fig.subplots_adjust(top=0.91, left=0.07, right=0.98,
                            bottom=max(0.10, xlabel_y + 0.06))

    fig.supxlabel("Time", fontsize=base_fontsize, y=xlabel_y)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight")

    return fig

def _add_corner_label(ax, text, fontsize=16, box=None):  # default +4
    if box is None:
        box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)
    ax.text(0.01, 0.985, f"({text})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, weight="bold",
            bbox=box)

def compose_icetemp_and_vertical_profiles(
    left,
    right,
    *,
    radargram=None,
    figsize=(14, 5.6),
    dpi: int = 300,
    width_ratios: Union[str, tuple[float, float]] = "auto",
    wspace: float = 0.05,
    hspace: float = 0.10,
    labels: tuple[str, str, str] = ("a", "b", "c"),
    label_fontsize: int = 16,
    label_box: dict | None = None,
    render_dpi: int | None = 300,
    trim_borders: bool = True,
    trim_tolerance: int = 6,
    savefig_path: Union[str, Path, None] = None,
):
    """
    Compose a 2‑ or 3‑panel figure:
      - left  panel (a): ice temperature profile
      - right panel (b): multiple vertical profiles
      - optional bottom panel (c): radargram spanning full width (preserving aspect ratio)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    from PIL import Image

    def _to_rgba(obj):
        if obj is None:
            return None
        if isinstance(obj, Figure):
            return _figure_to_array(obj, dpi=render_dpi)
        if isinstance(obj, (str, Path)):
            return np.array(Image.open(obj).convert("RGBA"))
        if isinstance(obj, np.ndarray):
            if obj.ndim == 3 and obj.shape[2] in (3, 4):
                return obj if obj.shape[2] == 4 else np.dstack([obj, 255*np.ones(obj.shape[:2], np.uint8)])
        raise TypeError(f"Unsupported type for panel: {type(obj)}")

    def _trim_rgba_border(arr: np.ndarray, tol: int = 5) -> np.ndarray:
        rgb = arr[..., :3].astype(int)
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            nonwhite = (np.abs(rgb - 255).max(axis=2) > tol)
            mask = (alpha > 0) & nonwhite
        else:
            mask = (np.abs(rgb - 255).max(axis=2) > tol)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return arr
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return arr[y0:y1+1, x0:x1+1, :]

    arr_left = _to_rgba(left)
    arr_right = _to_rgba(right)
    arr_radargram = _to_rgba(radargram)

    if trim_borders:
        arr_left = _trim_rgba_border(arr_left, tol=trim_tolerance)
        arr_right = _trim_rgba_border(arr_right, tol=trim_tolerance)
        if arr_radargram is not None:
            arr_radargram = _trim_rgba_border(arr_radargram, tol=trim_tolerance)

    # Compute widths from native aspect
    if isinstance(width_ratios, str) and width_ratios.lower() == "auto":
        wl = max(1e-3, arr_left.shape[1] / arr_left.shape[0])
        wr = max(1e-3, arr_right.shape[1] / arr_right.shape[0])
        width_ratios = (wl, wr)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Determine layout
    if arr_radargram is not None:
        # Calculate radargram aspect ratio (height/width)
        radar_h, radar_w = arr_radargram.shape[:2]
        radar_aspect = radar_h / radar_w
        
        # Combined width ratio = sum of width_ratios
        combined_width_ratio = sum(width_ratios)
        
        # Height needed to preserve aspect when spanning that combined width
        bottom_height_ratio = radar_aspect * combined_width_ratio * 0.8  # scale factor for visual balance
        
        height_ratios = (1.0, bottom_height_ratio)
        
        # Create outer grid with wspace=0 to avoid shifting
        gs_outer = GridSpec(nrows=2, ncols=1, figure=fig,
                            height_ratios=height_ratios,
                            hspace=hspace,
                            wspace=0)  # no horizontal space in outer grid
        
        # Top row: nested grid with wspace for panels A and B
        gs_top = gs_outer[0].subgridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
        axA = fig.add_subplot(gs_top[0, 0])
        axB = fig.add_subplot(gs_top[0, 1])
        
        # Bottom row: single axis spanning full width
        # Use left=0, right=1 positioning to ignore any column structure
        axC = fig.add_subplot(gs_outer[1, 0])
    else:
        gs = GridSpec(nrows=1, ncols=2, figure=fig,
                      width_ratios=width_ratios,
                      wspace=wspace)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = None

    # Panel A
    axA.imshow(arr_left, aspect='equal')
    axA.set_axis_off()

    # Panel B
    axB.imshow(arr_right, aspect='equal')
    axB.set_axis_off()

    # Labels
    if label_box is None:
        label_box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)
    _add_corner_label(axA, labels[0], fontsize=label_fontsize, box=label_box)
    _add_corner_label(axB, labels[1], fontsize=label_fontsize, box=label_box)

    result_axes = {"A": axA, "B": axB}

    # Panel C (radargram) - preserve aspect ratio and force full width
    if axC is not None:
        # Manually set position to span exact combined width of A+B
        # Get positions of A and B after layout
        fig.canvas.draw()  # ensure layout is computed
        bbox_a = axA.get_position()
        bbox_b = axB.get_position()
        
        # Panel C should span from left edge of A to right edge of B
        left_edge = bbox_a.x0
        right_edge = bbox_b.x1
        bottom_edge = gs_outer[1].get_position(fig).y0
        top_edge = gs_outer[1].get_position(fig).y1
        
        # Set new position for axC
        axC.set_position([left_edge, bottom_edge, right_edge - left_edge, top_edge - bottom_edge])
        
        axC.imshow(arr_radargram, aspect='equal')
        axC.set_axis_off()
        _add_corner_label(axC, labels[2], fontsize=label_fontsize, box=label_box)
        result_axes["C"] = axC

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)

    return fig, result_axes

def format_axes_coords(ax, x_step=5000, y_step=5000, thousands="'", decimals=0):
    """
    Format axes ticks for Swiss coordinates (LV95) with thousands separator.
    """
    def fmt(val, pos):
        sval = f"{int(round(val)):,}".replace(",", thousands)
        return f"{sval}"
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

def plot_cropped_map_with_grid(
    image_path, extent, crop_top_px=0, crop_bottom_px=0, grid_interval_km=10, fontsize=12, ax=None,
    xlabel=True, ylabel=True
):
    """
    Crop a map image from the top/bottom, update extent, and plot with Swiss-style axes and grid.

    Args:
        image_path (str or Path): Path to PNG image (from plot_switzerland_glacier_overview).
        extent (tuple): (minx, maxx, miny, maxy) in map coordinates.
        crop_top_px (int): Number of pixels to crop from the top.
        crop_bottom_px (int): Number of pixels to crop from the bottom.
        grid_interval_km (int): Grid spacing in kilometers.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new fig/ax.
        xlabel (bool): Whether to show x-axis label.
        ylabel (bool): Whether to show y-axis label.
    """
    img = Image.open(image_path)
    arr = np.array(img)
    h, w = arr.shape[0], arr.shape[1]

    # Crop from top and/or bottom
    y0 = crop_top_px
    y1 = h - crop_bottom_px
    arr_cropped = arr[y0:y1, :, :]

    # Update extent based on cropping
    minx, maxx, miny, maxy = extent
    px_h = (maxy - miny) / h
    new_maxy = maxy - y0 * px_h
    new_miny = miny + crop_bottom_px * px_h
    new_extent = (minx, maxx, new_miny, new_maxy)

    # Calculate aspect ratio from extent
    map_width = new_extent[1] - new_extent[0]
    map_height = new_extent[3] - new_extent[2]
    aspect = map_width / map_height
    base_height = 7  # or any preferred height
    figsize = (base_height * aspect, base_height)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(arr_cropped, extent=new_extent, origin="upper")
    ax.set_xlim(new_extent[0], new_extent[1])
    ax.set_ylim(new_extent[2], new_extent[3])
    ax.set_aspect('equal')

    # Format axes ticks and labels
    format_axes_coords(
        ax,
        x_step=grid_interval_km * 1000,
        y_step=grid_interval_km * 1000,
        thousands="'",
        decimals=0
    )
    ax.tick_params(axis='both', which='both', direction='out', pad=2.0, labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(90)
        lbl.set_verticalalignment('center')

    if xlabel:
        ax.set_xlabel("Easting (LV95) [m]", labelpad=8, fontsize=fontsize)
    else:
        ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel("Northing (LV95) [m]", labelpad=10, fontsize=fontsize)
    else:
        ax.set_ylabel("")

    # Only save/show if we created a new figure
    if ax is None:
        fig.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.show()

    return fig, ax, new_extent

def hide_edge_map_labels(ax, margin_frac=0.02):
    import re
    pat_dem   = re.compile(r"^\d+(?:\.\d+)?\s*m$")
    pat_coord = re.compile(r"^\d{4,}(?:'\d{3})+\s*m$")
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    xm0, xm1 = x0 + dx*margin_frac, x1 - dx*margin_frac
    ym0, ym1 = y0 + dy*margin_frac, y1 - dy*margin_frac
    for t in list(ax.texts):
        s = t.get_text().strip()
        if not (pat_dem.match(s) or pat_coord.match(s)):
            continue
        tx, ty = t.get_position()
        if tx < xm0 or tx > xm1 or ty < ym0 or ty > ym1:
            t.set_visible(False)

# Glacier abbreviations and colors
abbr = {"Alphubel": "AH", "Chessjen": "CJ", "Hohsaas": "HS", "Sex Rouge": "SR", "Tortin": "GT", "Corvatsch": "CV"}
rect_colors = {"AH": "#1f77b4", "CJ": "#2ca02c", "HS": "#ff7f0e", "SR": "#d62728", "GT": "#9467bd", "CT": "#8c564b"}

def draw_glacier_map(
    ax, ortho_path, bbox, gdf_pts, dem_tiles, boreholes, title,
    ANNO_FONTSIZE, ABBR_FONTSIZE,
    add_label_color=False, xlabel=True, ylabel=True, panel=None,
    background="ortho",
    highlight_ids: Union[int, Sequence[int], None] = None,
    highlight_color: str = "crimson",
    highlight_size: float = 12.0,
    annotations: dict | None = None,
    outlines: gpd.GeoDataFrame | str | None = None,
    outline_color: str | None = None,
    outline_linewidth: float = 2.0,
    outline_alpha: float = 0.85,
    tick_labelsize : int = 8,
    show_borehole_labels: bool = True,
    show_bedrock_depth: bool = False,  # NEW ARGUMENT
    gdf_pts_2: gpd.GeoDataFrame | None = None,
    show_contours: bool = True,
    show_flow_arrow: bool = True,
    flow_arrow_angle_deg: float | None = None,   # override computed direction (0=east, 90=north)
    flow_arrow_pos_offset: tuple | None = None,  # (dx, dy) in map units to shift arrow start
    flow_arrow_label: str = ""
):
    """
    Draw a single glacier map on `ax`.

    - draws orthophoto or hillshade background
    - plots DEM contours, GPR points, boreholes
    - optionally highlights specific GPR profile(s) and annotates profiles
    - optionally overlays glacier outline(s) (GeoDataFrame or path)

    outlines may be:
      - a GeoDataFrame with a 'geometry' column,
      - a path to a vector file readable by geopandas,
      - a pandas-like DataFrame with a 'geometry' column containing shapely geometries or WKT strings.

    Note: default outline color is black ('k') unless an explicit outline_color is provided.
    """
    if outline_color is None:
        outline_color = 'darkblue'
    plt.rcParams['font.family'] = 'Arial'

    hs_kwargs = getattr(draw_glacier_map, "_hillshade_defaults", None) or {}
    hs_kwargs.pop("background", None)

    # Get orthophoto extent using imshow_tif
    ortho_extent = imshow_tif(ax, ortho_path)  # returns (minx, maxx, miny, maxy)

    # extent ortho extent by 15 m towards the east otherwise weird black stripes occur
    ortho_extent = (
        ortho_extent[0],
        ortho_extent[1] + 15.0,
        ortho_extent[2],
        ortho_extent[3]
    )

    # Background logic
    if background in ("hillshade", "shade", "dem"):
        hs_img = imshow_hillshade(ax, dem_tiles, plot_extent=ortho_extent, merge_bbox=bbox, **hs_kwargs)
        try:
            if hasattr(hs_img, "set_zorder"):
                hs_img.set_zorder(5)
        except Exception:
            pass

    elif background in ("ortho_hillshade", "ortho+hillshade", "ortho+hs"):
        # Draw ortho first
        imshow_tif(ax, ortho_path)
        try:
            if hasattr(ax.images[-1], "set_zorder"):
                ax.images[-1].set_zorder(1)
        except Exception:
            pass

        # Overlay hillshade on top with semi-transparency
        hs_alpha = hs_kwargs.pop("alpha", 0.2)
        hs_img = imshow_hillshade(ax, dem_tiles, plot_extent=ortho_extent, merge_bbox=bbox)
        try:
            if hs_img is not None and hasattr(hs_img, "set_alpha"):
                hs_img.set_alpha(hs_alpha)
                hs_img.set_zorder(4)
            else:
                ax.images[-1].set_alpha(hs_alpha)
                ax.images[-1].set_zorder(4)
        except Exception:
            try:
                ax.images[-1].set_alpha(hs_alpha)
                ax.images[-1].set_zorder(4)
            except Exception:
                pass

    else:
        imshow_tif(ax, ortho_path)

    # DEM contours
    if show_contours:
        gprp.plot_dem_contours_from_tiles(
            ax, dem_tiles, bbox=bbox, pixel_size=2.0,
            minor_step=25.0, major_step=50.0,
            minor_kwargs={'linewidths':0.25, 'colors':'k', 'alpha':0.4},
            major_kwargs={'linewidths':0.6,  'colors':'k', 'alpha':0.55},
            zorder_minor=6, zorder_major=7, label=True, label_fmt="%.0f m"
        )
    hide_edge_map_labels(ax, margin_frac=0.02)

    # Will hold the reprojected/cleaned outline polygon GDF for flow-arrow DEM masking
    _outline_gdf = None

    # Optional glacier outlines (draw before GPR points so points stay visible)
    if outlines is not None:
        try:
            # Accept path -> read with geopandas
            if isinstance(outlines, (str, Path)):
                gdf_outline = gpd.read_file(outlines)
            elif isinstance(outlines, gpd.GeoDataFrame):
                gdf_outline = outlines.copy()
            else:
                # DataFrame-like: ensure geometry column contains shapely geometries
                gdf_outline = gpd.GeoDataFrame(outlines)
                if 'geometry' in gdf_outline.columns:
                    geom_col = gdf_outline['geometry']
                    # convert WKT strings to geometries if needed
                    if geom_col.dtype == object and any(isinstance(v, str) for v in geom_col.dropna().values):
                        try:
                            from shapely import wkt
                            gdf_outline['geometry'] = gdf_outline['geometry'].apply(lambda s: wkt.loads(s) if isinstance(s, str) else s)
                        except Exception:
                            pass
                    gdf_outline = gdf_outline.set_geometry('geometry', inplace=False)

            # If no CRS provided but coords look like lon/lat, assume EPSG:4326
            try:
                if getattr(gdf_outline, "crs", None) is None:
                    coords_sample = list(gdf_outline.geometry.dropna().head(3).apply(lambda g: list(g.bounds) if g is not None else None))
                    flat = [c for b in coords_sample if b for c in b]
                    if flat and all(-180.0 <= v <= 180.0 for v in flat):
                        gdf_outline.set_crs(epsg=4326, inplace=True)
            except Exception:
                pass

            # Reproject to LV95 (2056) if needed
            try:
                if hasattr(gdf_outline, "crs") and gdf_outline.crs is not None:
                    if getattr(gdf_outline.crs, 'to_epsg', lambda: None)() != 2056:
                        gdf_outline = gdf_outline.to_crs(epsg=2056)
            except Exception:
                pass

            # Clean / fix invalid geometries and optionally simplify a little
            try:
                gdf_outline['geometry'] = gdf_outline.geometry.apply(lambda g: (g.buffer(0) if (g is not None and not g.is_valid) else g))
            except Exception:
                pass

            # Save reprojected+cleaned outline for flow-arrow DEM masking
            try:
                _outline_gdf = gdf_outline.copy()
            except Exception:
                pass

            # Clip to bbox to avoid drawing long spurious lines
            try:
                from shapely.geometry import box as _box
                clip_box = _box(x0 := bbox[0], y0 := bbox[1], x1 := bbox[2], y1 := bbox[3])
                gdf_outline['geom_clipped'] = gdf_outline.geometry.apply(lambda g: g.intersection(clip_box) if g is not None else None)
                # keep only non-empty geometries
                gdf_plot = gpd.GeoDataFrame(geometry=gdf_outline['geom_clipped'])
                gdf_plot = gdf_plot[~gdf_plot.geometry.is_empty & gdf_plot.geometry.notna()]
                if gdf_plot.empty:
                    # if clipping removed everything, fall back to full geometries (but still cleaned)
                    gdf_plot = gpd.GeoDataFrame(geometry=gdf_outline.geometry.dropna())
            except Exception:
                gdf_plot = gdf_outline.copy()

            # Filter tiny slivers (area for polygons, length for lines)
            try:
                def _keep_geom(g):
                    if g is None or g.is_empty:
                        return False
                    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
                    if isinstance(g, (Polygon, MultiPolygon)):
                        return g.area > 1.0  # keep polygons > 1 m^2
                    if isinstance(g, (LineString, MultiLineString)):
                        return g.length > 1.0  # keep lines > 1 m
                    return True
                gdf_plot = gdf_plot[gdf_plot.geometry.apply(_keep_geom)]
            except Exception:
                pass

            # Determine color: override param -> rect_colors by title abbreviation -> fallback 'k'
            outline_color_use = outline_color if outline_color is not None else rect_colors.get(abbr.get(title, title[:2].upper()), '#000000')

            # Prepare line segments for plotting (avoid polygon-filling or unexpected connectors)
            try:
                from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
                from matplotlib.collections import LineCollection
                segs = []
                for geom in gdf_plot.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    # geometry collections: extract contained polygons/lines
                    if isinstance(geom, GeometryCollection):
                        geoms = [g for g in geom.geoms]
                    else:
                        geoms = [geom]
                    for g in geoms:
                        if isinstance(g, (Polygon, MultiPolygon)):
                            # handle multipolygon by iterating polygons
                            if isinstance(g, MultiPolygon):
                                parts = list(g.geoms)
                            else:
                                parts = [g]
                            for p in parts:
                                # append exterior ring only (no fill) to avoid interior odd lines
                                ext = p.exterior
                                segs.append(list(ext.coords))
                                # optionally plot interiors (holes) as dashed
                                for interior in p.interiors:
                                    segs.append(list(interior.coords))
                        elif isinstance(g, (LineString, MultiLineString)):
                            if isinstance(g, MultiLineString):
                                for part in g.geoms:
                                    segs.append(list(part.coords))
                            else:
                                segs.append(list(g.coords))
                        else:
                            # fallback: try to extract boundary
                            try:
                                b = g.boundary
                                if not b.is_empty:
                                    if hasattr(b, "__iter__"):
                                        for part in b:
                                            segs.append(list(part.coords))
                                    else:
                                        segs.append(list(b.coords))
                            except Exception:
                                continue

                if segs:
                    # create LineCollection
                    lc = LineCollection(segs, colors=outline_color_use, linewidths=outline_linewidth, alpha=outline_alpha, zorder=11, linestyles='-')
                    ax.add_collection(lc)
            except Exception:
                # fallback: use GeoDataFrame plotting (last resort)
                try:
                    gdf_plot.plot(ax=ax, facecolor='none', edgecolor=outline_color_use, linewidth=outline_linewidth, alpha=outline_alpha, zorder=11, linestyle=':')
                except Exception:
                    pass

        except Exception:
            # don't fail the whole plot if outlines can't be parsed
            pass

    # Draw all GPR points (base layer) -- ensure fully opaque and on top of hillshade
    gprp.draw_gpr_line_points(ax, gdf_pts, size=3, color='k', alpha=1.0, zorder=8)

    # Optionally draw second set of GPR points (e.g., different survey)
    if gdf_pts_2 is not None:
        gprp.draw_gpr_line_points(ax, gdf_pts_2, size=3, color='darkorange', alpha=1.0, zorder=8)

    # If highlight_ids provided, draw them on top explicitly (robust regardless of gprp implementation)
    if highlight_ids is not None:
        if isinstance(highlight_ids, (int, np.integer)):
            hids = [int(highlight_ids)]
        else:
            hids = [int(h) for h in highlight_ids]
        # subset points belonging to highlighted profiles
        try:
            sub = gdf_pts[gdf_pts['profile'].isin(hids)]
        except Exception:
            sub = gdf_pts[gdf_pts.get('profile', -999).astype(int).isin(hids)] if 'profile' in gdf_pts.columns else gdf_pts.iloc[0:0]
        if not sub.empty:
            ax.scatter(
                sub.geometry.x, sub.geometry.y,
                s=highlight_size, marker='o',
                facecolor=highlight_color, edgecolors='none',
                zorder=9, alpha=1.0
            )

    # Annotation helper: replicate notebook-style PCA ordering and place label
    def _annotate_profile(ax_local, gdf_pts_local, profile_id, text=None, where='mid',
                          offset_pts=(3, 3), color=None, fontsize=None, bg=True, zorder=30):
        if 'profile' not in gdf_pts_local.columns:
            return None
        sub = gdf_pts_local[gdf_pts_local['profile'] == profile_id]
        if sub.empty:
            return None
        coords = np.c_[sub.geometry.x.values, sub.geometry.y.values]
        if coords.shape[0] < 2:
            idx = 0
        else:
            mean = coords.mean(0)
            _, _, Vt = np.linalg.svd(coords - mean, full_matrices=False)
            t = (coords - mean) @ Vt[0]
            order = np.argsort(t)
            if where == 'start':
                idx = order[0]
            elif where == 'end':
                idx = order[-1]
            else:
                idx = order[len(order)//2]
        x, y = coords[idx]
        if color is None:
            color = 'crimson'
        if fontsize is None:
            fontsize = ANNO_FONTSIZE
        if offset_pts is None:
            offset_pts = (3, 3)
        tr = plt.transforms.offset_copy(ax_local.transData, fig=ax_local.figure, x=offset_pts[0], y=offset_pts[1], units='points')
        if text is None:
            text = f"P{profile_id}"
        bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=1.0) if bg else None
        return ax_local.text(x, y, text, color=color, fontsize=fontsize, ha='left', va='bottom',
                             transform=tr, zorder=zorder, bbox=bbox)

    # Handle annotations dict (profile_id -> opts)
    if annotations:
        # prefer gprp.annotate_gpr_profile if available
        annot_fn = getattr(gprp, "annotate_gpr_profile", None)
        for pid, opts in annotations.items():
            if callable(annot_fn):
                try:
                    # many implementations expect (ax, gdf_pts, profile_id, **opts)
                    annot_fn(ax, gdf_pts, pid, **(opts or {}))
                except Exception:
                    # fallback to internal helper
                    try:
                        _annotate_profile(ax, gdf_pts, pid, **(opts or {}))
                    except Exception:
                        pass
            else:
                try:
                    _annotate_profile(ax, gdf_pts, pid, **(opts or {}))
                except Exception:
                    pass

    # Boreholes + labels
    if (boreholes is not None) and (not boreholes.empty):
        for _, r in boreholes.iterrows():
            # Determine base marker color
            if isinstance(r.get('name', ''), str) and r['name'].endswith('G'):
                bh_color = 'red'
            elif isinstance(r.get('name', ''), str) and r['name'].endswith('TT'):
                bh_color = 'white'
            else:
                bh_color = 'white'
            
            # Check if this borehole reached bedrock
            reached_bedrock = r.get('reached_bedrock', False)
            
            # Determine label position ONCE (before any conditional blocks that need it)
            txt_va = 'top' if ((title == "Chessjen" and r.get('name') == "CJ1G") or 
                               (title == "Hohsaas" and r.get('name') == "HS3G")) else 'bottom'
            
            # Draw standard borehole marker
            ax.scatter(r.geometry.x, r.geometry.y,
                       s=40, marker='o', color=bh_color, edgecolors='black',
                       linewidths=1, alpha=0.95, zorder=15, label='Boreholes')

            # Borehole name label (strip glacier abbreviation prefix, e.g. "AH1G" -> "1G")
            if show_borehole_labels:
                bh_name = r.get('name', '')
                glacier_abbr = abbr.get(title, title[:2].upper() if isinstance(title, str) and title else '')
                if glacier_abbr and isinstance(bh_name, str) and bh_name.startswith(glacier_abbr):
                    bh_label = bh_name[len(glacier_abbr):]
                else:
                    bh_label = bh_name
                import matplotlib.transforms as _transforms
                # Per-borehole label position overrides: (ha, va, offset_x, offset_y)
                _bh_label_pos = {
                    ("Hohsaas",   "HS2TT"): ('right', 'bottom', -2,  2),
                    ("Chessjen",  "CJ3TT"): ('right', 'bottom', -2,  2),
                    ("Corvatsch", "CV1TT"): ('right', 'bottom', -2,  2),
                    ("Alphubel",  "AH3TT"): ('right', 'bottom', -2,  2),
                    ("Alphubel",  "AH2TT"): ('left',  'top',     2, -2),
                    ("Alphubel",  "AH1TT"): ('left',  'top',     2, -2),
                }
                _pos = _bh_label_pos.get((title, bh_name))
                if _pos:
                    _txt_ha, _lbl_va, _offset_x, offset_y = _pos
                else:
                    _txt_ha, _lbl_va, _offset_x = 'left', txt_va, 2
                    offset_y = -2 if txt_va == 'top' else 2
                tr = _transforms.offset_copy(ax.transData, fig=ax.figure, x=_offset_x, y=offset_y, units='points')
                ax.text(r.geometry.x, r.geometry.y, bh_label,
                        fontsize=ANNO_FONTSIZE-2, color='k', ha=_txt_ha, va=_lbl_va,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.95),
                        transform=tr, zorder=16)
            
            # Bedrock depth annotation (if enabled and reached bedrock)
            if show_bedrock_depth and reached_bedrock:
                depth = r.get('borehole depth (m)', None)
                if depth is not None:
                    # Convert to float if it's a string
                    try:
                        depth_val = float(depth)
                    except (ValueError, TypeError):
                        continue  # Skip if conversion fails
                    
                    # Format depth value
                    depth_str = f"{depth_val:.1f} m"
                    
                    # Calculate annotation position (outside the main area)
                    x0, y0, x1, y1 = bbox
                    bh_x, bh_y = r.geometry.x, r.geometry.y
                    
                    # Determine optimal placement (away from center)
                    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
                    dx_from_center = bh_x - center_x
                    dy_from_center = bh_y - center_y
                    
                    # Special handling for specific glaciers/boreholes to avoid overlap
                    bh_name = r.get('name', '')
                    
                    # Default placement distance
                    offset_dist = 80  # meters in map units
                    
                    # Glacier-specific positioning rules
                    if title == "Chessjen":
                        # For Chessjen, use diagonal positioning for both bedrock boreholes
                        if bh_name == "CJ1G":
                            # Place CJ1G at 45° angle (upper-right)
                            angle = np.radians(45)
                            anno_x = bh_x + offset_dist * np.cos(angle)
                            anno_y = bh_y + offset_dist * np.sin(angle)
                            ha = 'left'
                            va = 'bottom'
                        elif bh_name == "CJ2G":
                            # Place CJ2G at 135° angle (upper-left) 
                            angle = np.radians(135)
                            anno_x = bh_x + offset_dist * np.cos(angle)
                            anno_y = bh_y + offset_dist * np.sin(angle)
                            ha = 'right'
                            va = 'bottom'
                        else:
                            # Fallback for any other boreholes
                            anno_x = bh_x + offset_dist
                            anno_y = bh_y
                            ha = 'left'
                            va = 'center'
                    else:
                        # Default placement for other glaciers (perpendicular from center)
                        if abs(dx_from_center) > abs(dy_from_center):
                            # Place horizontally
                            anno_x = bh_x + (offset_dist if dx_from_center > 0 else -offset_dist)
                            anno_y = bh_y
                            ha = 'left' if dx_from_center > 0 else 'right'
                            va = 'center'
                        else:
                            # Place vertically
                            anno_x = bh_x
                            anno_y = bh_y + (offset_dist if dy_from_center > 0 else -offset_dist)
                            ha = 'center'
                            va = 'bottom' if dy_from_center > 0 else 'top'
                    
                    # Draw connecting line from borehole to annotation
                    ax.plot([bh_x, anno_x], [bh_y, anno_y],
                            color='k', linewidth=1.0, linestyle='-',
                            zorder=10, alpha=0.8)
                    
                    # Place annotation text
                    ax.text(anno_x, anno_y, depth_str,
                            fontsize=ANNO_FONTSIZE-3, color='k',
                            ha=ha, va=va,
                            weight='bold',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                    ec='k', lw=1.0, alpha=0.95),
                            zorder=11)

    # --- Previously measured borehole reference annotations ------------------
    # Sex Rouge: "Fischer (2018)" with lines to SR1TT and SR2TT
    # Corvatsch: "Haeberli et al. (2004)" with line to CV2TT
    if (boreholes is not None) and (not boreholes.empty):
        _ref_targets = {}
        if title == "Sex Rouge":
            _ref_targets["Fischer (2018)"] = {"SR1TT", "SR2TT"}
        elif title == "Corvatsch":
            _ref_targets["Haeberli et al. (2004)"] = {"CV2TT"}

        for _ref_label, _bh_names in _ref_targets.items():
            # Collect positions of target boreholes
            _bh_pos = {}
            for _, r in boreholes.iterrows():
                if r.get('name', '') in _bh_names:
                    _bh_pos[r['name']] = (r.geometry.x, r.geometry.y)
            if not _bh_pos:
                continue

            # Place text at centroid of target boreholes, offset toward bbox edge
            _xs = [p[0] for p in _bh_pos.values()]
            _ys = [p[1] for p in _bh_pos.values()]
            _mid_x = sum(_xs) / len(_xs)
            _mid_y = sum(_ys) / len(_ys)
            _x0b, _y0b, _x1b, _y1b = bbox
            _cx, _cy = (_x0b + _x1b) / 2, (_y0b + _y1b) / 2
            _panel_w = _x1b - _x0b
            # Offset text away from glacier center
            _dx = _mid_x - _cx
            _dy = _mid_y - _cy
            _off = _panel_w * 0.18
            _norm = (_dx**2 + _dy**2) ** 0.5 or 1
            _txt_x = _mid_x + _dx / _norm * _off
            _txt_y = _mid_y + _dy / _norm * _off
            # Manual fine-tuning offsets (map units)
            _fine_tune = {
                "Fischer (2018)":        ( 40,  100),
                "Haeberli et al. (2004)": (120,  150),
            }
            _ftx, _fty = _fine_tune.get(_ref_label, (0, 0))
            _txt_x += _ftx
            _txt_y += _fty
            _ha = 'left' if _dx >= 0 else 'right'
            _va = 'bottom' if _dy >= 0 else 'top'

            # Draw text box
            ax.text(_txt_x, _txt_y, _ref_label,
                    fontsize=ANNO_FONTSIZE - 3, color='k', ha=_ha, va=_va,
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9),
                    zorder=18)
            # Draw leader lines from text anchor to each borehole
            for _bx, _by in _bh_pos.values():
                ax.plot([_txt_x, _bx], [_txt_y, _by],
                        color='k', linewidth=0.7, linestyle='-', zorder=17, alpha=0.8)

    # --- Flow direction arrow ------------------------------------------------
    # Draws a small downslope arrow near the glacier terminus, derived from the
    # DEM elevation gradient.  Disabled by passing show_flow_arrow=False.
    if show_flow_arrow and dem_tiles:
        try:
            import rasterio as _rasterio
            from rasterio.merge import merge as _rio_merge

            _x0b, _y0b, _x1b, _y1b = bbox

            # Load + merge DEM tiles clipped to the panel bbox
            _srcs = [_rasterio.open(t) for t in dem_tiles]
            try:
                _mosaic, _dem_tf = _rio_merge(_srcs, bounds=(_x0b, _y0b, _x1b, _y1b))
            finally:
                for _s in _srcs:
                    try:
                        _s.close()
                    except Exception:
                        pass

            _dem = _mosaic[0].astype(float)
            _dem[_dem < -9000] = np.nan

            # Mask DEM to glacier outline polygon(s) if available
            if _outline_gdf is not None:
                try:
                    from rasterio.features import rasterize as _rasterize
                    from shapely.geometry import mapping as _mapping
                    _shapes = [
                        (_mapping(g), 1) for g in _outline_gdf.geometry
                        if g is not None and not g.is_empty
                    ]
                    if _shapes:
                        _mask = _rasterize(
                            _shapes, out_shape=_dem.shape,
                            transform=_dem_tf, fill=0, dtype='uint8'
                        )
                        _dem[_mask == 0] = np.nan
                except Exception:
                    pass  # proceed with unmasked DEM

            _valid = ~np.isnan(_dem)
            if _valid.sum() >= 30:
                # Fill NaN for gradient computation (gradient can't handle NaN natively)
                _dem_fill = _dem.copy()
                _dem_fill[~_valid] = np.nanmedian(_dem)

                _gr, _gc = np.gradient(_dem_fill)
                # Zero out gradient at invalid (outside-outline) pixels
                _gr[~_valid] = np.nan
                _gc[~_valid] = np.nan

                _psx = abs(_dem_tf.a)   # metres per pixel (east direction)
                _psy = abs(_dem_tf.e)   # metres per pixel (south direction)

                # Flow = downslope = negative of gradient in map (east, north) coords
                # - _gc / _psx  →  west if slope rises east
                # +_gr / _psy  →  north if slope rises going south (increasing row)
                _fe = -_gc / _psx
                _fn = _gr / _psy

                # Terminus region = lowest 25 % of valid elevations within glacier
                _low_thresh = np.percentile(_dem[_valid], 25)
                _term = _valid & (_dem <= _low_thresh)

                if _term.sum() > 0:
                    _rows_t, _cols_t = np.where(_term)
                    _cr = int(np.mean(_rows_t))
                    _cc = int(np.mean(_cols_t))

                    # Convert pixel centroid to map coordinates
                    _ax0 = _dem_tf.c + (_cc + 0.5) * _dem_tf.a
                    _ay0 = _dem_tf.f + (_cr + 0.5) * _dem_tf.e

                    # Mean flow direction at terminus
                    _fe_m = float(np.nanmean(_fe[_term]))
                    _fn_m = float(np.nanmean(_fn[_term]))
                    _norm = np.hypot(_fe_m, _fn_m)

                    # Allow manual angle override
                    if flow_arrow_angle_deg is not None:
                        _ang = np.radians(flow_arrow_angle_deg)
                        _fe_m = np.cos(_ang)
                        _fn_m = np.sin(_ang)
                        _norm = 1.0

                    if _norm > 0 and not np.isnan(_norm):
                        _alen = (_x1b - _x0b) * 0.10   # arrow length = 10 % of panel width
                        _dx = _fe_m / _norm * _alen
                        _dy = _fn_m / _norm * _alen

                        # Apply manual position offset if given
                        if flow_arrow_pos_offset is not None:
                            _ax0 += flow_arrow_pos_offset[0]
                            _ay0 += flow_arrow_pos_offset[1]

                        _ax1, _ay1 = _ax0 + _dx, _ay0 + _dy

                        # Solid block arrow (thick body + wide triangular head)
                        from matplotlib.patches import FancyArrow as _FancyArrow
                        _body_w = _alen * 0.18
                        _head_w = _alen * 0.42
                        _head_l = _alen * 0.38
                        _arrow_patch = _FancyArrow(
                            _ax0, _ay0, _dx, _dy,
                            width=_body_w, head_width=_head_w,
                            head_length=_head_l, length_includes_head=True,
                            color='red', zorder=25
                        )
                        ax.add_patch(_arrow_patch)

                        # Label in-line with arrow: placed just past tip, rotated to match
                        if flow_arrow_label:
                            _angle_deg = float(np.degrees(np.arctan2(_fn_m, _fe_m)))
                            # small gap between tip and text start (in map units)
                            _gap = _alen * 0.05
                            _lx = _ax1 + np.cos(np.radians(_angle_deg)) * _gap
                            _ly = _ay1 + np.sin(np.radians(_angle_deg)) * _gap
                            ax.text(
                                _lx, _ly, flow_arrow_label,
                                fontsize=ANNO_FONTSIZE, color='red',
                                ha='left', va='center',
                                fontweight='bold', fontfamily='Arial',
                                rotation=_angle_deg, rotation_mode='anchor',
                                zorder=26,
                                bbox=dict(boxstyle='square,pad=0.1', fc='white',
                                          ec='none', alpha=0.7)
                            )
        except Exception:
            pass   # never let a failed arrow crash the figure
    # -------------------------------------------------------------------------

    # Limits & formatting
    x0, y0, x1, y1 = bbox
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    gprp.format_axes_coords(ax, x_step=200, y_step=200, thousands='apostrophe', unit='m', decimals=0)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.tick_params(axis='both', which='both', direction='out', pad=1.0, labelsize=tick_labelsize)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(90); lbl.set_verticalalignment('center')

    if xlabel:
        ax.set_xlabel("Easting (LV95) [m]", labelpad=2, fontsize=ANNO_FONTSIZE)
    if ylabel:
        ax.set_ylabel("Northing (LV95) [m]", labelpad=2, fontsize=ANNO_FONTSIZE)

    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    # Abbreviation tag (use color if requested)
    default_abbr = abbr.get(title, (title.split()[0][:2].upper() if isinstance(title, str) and title else ""))
    label_abbr = default_abbr
    label_color = rect_colors.get(label_abbr, 'k') if add_label_color else 'k'
    label_display = f"{title} ({label_abbr})" if title else label_abbr
    ax.text(0.02, 0.02, label_display, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=ABBR_FONTSIZE, fontweight='bold',
            color=label_color,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=label_color, alpha=0.7),
            zorder=20)

    # Panel label
    if panel:
        ax.text(
            0.02, 0.98, f"({panel})",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=ABBR_FONTSIZE, fontweight='bold',
            color='black', zorder=21,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=1.0)
        )

    # Color-code axes spines and ticks to match glacier color
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', colors=label_color)

def compress_figure_inplace(input_path, max_size_mb=5):
    """
    Compress a PDF or PNG figure so the output file size is ≤ max_size_mb.
    Overwrites the input file.

    Adjusted to be less aggressive to avoid visible pixelation:
      - PNG: prefer lossless optimizations and quantization before resizing;
             limit downscaling and use smaller steps.
      - PDF: try higher-quality Ghostscript presets first, then explicit DPI
             downsampling (150 → 120 → 96 dpi) if presets are insufficient.
    """
    ext = str(input_path).lower().split('.')[-1]
    max_bytes = int(max_size_mb * 1024 * 1024)

    if ext == "png":
        img = Image.open(input_path)
        # ensure RGBA for consistent handling
        if img.mode not in ("RGBA", "RGB"):
            img = img.convert("RGBA")
        w, h = img.size
        tmp_path = str(input_path) + ".tmp.png"

        # 1) Try optimized PNG write (lossless) first
        img.save(tmp_path, optimize=True, compress_level=9)
        if os.path.getsize(tmp_path) <= max_bytes:
            os.replace(tmp_path, str(input_path))
            print(f"Compressed file saved: {input_path}")
            return input_path

        # 2) Try mild quantization (preserve as many colors as possible)
        for colors in (256, 192, 128):
            try:
                # quantize returns a P image which often saves smaller
                q = img.convert("RGB").quantize(colors=colors, method=Image.FASTOCTREE)
                q.save(tmp_path, optimize=True)
                if os.path.getsize(tmp_path) <= max_bytes:
                    os.replace(tmp_path, str(input_path))
                    print(f"Compressed (quantized to {colors} colors) file saved: {input_path}")
                    return input_path
            except Exception:
                continue

        # 3) As last resort, downscale in small steps but stop earlier to avoid pixelation
        min_width = 1200  # don't downscale below this to preserve detail
        scale_step = 0.95  # mild downscale per iteration
        while os.path.getsize(tmp_path) > max_bytes and w > min_width:
            w = int(w * scale_step)
            h = int(h * scale_step)
            img = img.resize((w, h), Image.LANCZOS)
            img.save(tmp_path, optimize=True, compress_level=9)
            # try a quick quantize pass at moderate colors if needed
            if os.path.getsize(tmp_path) > max_bytes:
                try:
                    q = img.convert("RGB").quantize(colors=192, method=Image.FASTOCTREE)
                    q.save(tmp_path, optimize=True)
                except Exception:
                    pass

        if os.path.getsize(tmp_path) <= max_bytes:
            os.replace(tmp_path, str(input_path))
            print(f"Compressed file saved: {input_path}")
        else:
            # keep the best-effort temporary file if smaller, otherwise keep original
            final_size = os.path.getsize(tmp_path)
            orig_size = os.path.getsize(str(input_path))
            if final_size < orig_size:
                os.replace(tmp_path, str(input_path))
                print(f"Warning: Could not reach target size, saved smaller file: {input_path}")
            else:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                print(f"Warning: Could not compress below {max_size_mb} MB without heavy quality loss.")
        return input_path

    elif ext == "pdf":
        # Strategy: only downsample embedded raster images (orthophotos) using
        # Ghostscript — vector elements (axes, labels, lines) are never rasterized
        # and stay sharp. We try progressively lower image DPI / JPEG quality until
        # the file fits.
        tmp_path = str(input_path) + ".tmp.pdf"

        def _gs_image_only(preset, dpi):
            """Run Ghostscript compressing only embedded raster images.
            Uses a PDFSETTINGS preset for JPEG quality baseline, then
            overrides DPI. Vectors (text, lines, axes) are never rasterized.
            """
            cmd = [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.5",
                "-dNOPAUSE", "-dQUIET", "-dBATCH",
                f"-dPDFSETTINGS={preset}",
                # Override image resolution
                "-dDownsampleColorImages=true",
                "-dDownsampleGrayImages=true",
                "-dColorImageDownsampleType=/Bicubic",
                "-dGrayImageDownsampleType=/Bicubic",
                f"-dColorImageResolution={dpi}",
                f"-dGrayImageResolution={dpi}",
                f"-sOutputFile={tmp_path}",
                str(input_path),
            ]
            subprocess.run(cmd, check=True)

        # Try progressively more aggressive image compression; vectors stay sharp.
        # preset controls JPEG quality: /prepress~95, /ebook~85, /screen~70
        attempts = [
            ("/prepress", 200),  # high quality, 200 dpi images
            ("/ebook",    150),  # good quality, 150 dpi
            ("/ebook",    120),
            ("/screen",   120),
            ("/screen",    96),
        ]
        best_tmp = None
        best_size = None
        for preset, dpi in attempts:
            try:
                _gs_image_only(preset, dpi)
                size = os.path.getsize(tmp_path)
                if best_size is None or size < best_size:
                    best_size = size
                    best_tmp = tmp_path + ".best"
                    import shutil
                    shutil.copy2(tmp_path, best_tmp)
                if size <= max_bytes:
                    print(f"Compressed (images at {dpi} dpi, preset={preset}) saved: {input_path}")
                    os.replace(tmp_path, str(input_path))
                    if best_tmp and os.path.exists(best_tmp):
                        os.remove(best_tmp)
                    return input_path
            except Exception:
                continue

        # Use best result achieved
        src = best_tmp if (best_tmp and os.path.exists(best_tmp)) else tmp_path
        try:
            orig_size = os.path.getsize(str(input_path))
            if best_size and best_size < orig_size:
                os.replace(src, str(input_path))
                print(f"Warning: target {max_size_mb} MB not reached; "
                      f"saved best result ({best_size/1024/1024:.1f} MB): {input_path}")
            else:
                print(f"Warning: Could not compress below {max_size_mb} MB without quality loss.")
        except Exception:
            print("Warning: Ghostscript compression failed.")
        for p in [tmp_path, best_tmp]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        return input_path

    else:
        raise ValueError("Only PNG and PDF files are supported.")