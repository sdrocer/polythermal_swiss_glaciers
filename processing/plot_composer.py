# import necessary modules
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import re

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc
from PIL import Image
from pathlib import Path
from typing import Union, Sequence

# Import required geodata helpers (centralized in download_geodata.py)
from processing.download_geodata import (
    SWISS_CRS,
    download_swisstopo_wms,
    _load_vector,
    _points_to_gdf
)


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
    field_site_label_fontsize: int = 18,   # match ABBR_FONTSIZE
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
    field_site_label_line_anchor: dict | None = None,  # NEW: per-site anchor ("ur", "ll", etc.)
    cut_to_country_outline: bool = False,
):
    """
    Switzerland glacier overview.
    Downloads WMS layers from Swisstopo and composes a map with glaciers, lakes, cantonal and national borders.
    """
    try:
        plt.rcParams["font.family"] = font_family
    except Exception:
        pass

    # Helper: get anchor offset for label box
    def _anchor_offset(anchor, box_w, box_h):
        # anchor: "ul"=upper left, "ur"=upper right, "ll"=lower left, "lr"=lower right, "c"=center
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
            return (0, box_h)  # default upper left

    full_bbox = (2420000, 1030000, 2900000, 1350000)

    import PIL.Image as _PILImage
    from PIL import UnidentifiedImageError

    # --- UPDATED download helper with dimension clamp + fallback ladder ---
    def _download(layer_id, bbox):
        target_px = pixel_size  # initial desired ground resolution (m/pixel)

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
            return w_px, h_px, scale  # scale >1 means we shrank request

        tried = [target_px] + [ps for ps in fallback_pixel_sizes if ps != target_px]
        first_dims = None
        first_arr = None
        first_extent = None

        for i, px in enumerate(tried):
            w_px, h_px, clamp_scale = _compute_dims(px)
            req_px_size = (bbox[2]-bbox[0]) / w_px  # actual ground pixel after clamp
            try:
                arr, ext, _ = download_swisstopo_wms(
                    layer=layer_id,
                    bbox=bbox,
                    pixel_size=req_px_size,
                    img_format="image/png",
                    transparent=True
                )
                # keep first successful (at target) or current
                if i == 0:
                    first_arr, first_extent = arr, ext
                    first_dims = (w_px, h_px)
                # If this is first attempt success OR we are on fallback and not upscaling:
                if i == 0 or not upscale_to_first:
                    return arr, ext
                # Need to upscale to first target dims
                if upscale_to_first and first_dims:
                    # upscale current arr to first_dims using PIL
                    im_mode = "RGBA" if arr.shape[2] == 4 else "RGB"
                    pil_im = _PILImage.fromarray(arr, mode=im_mode)
                    pil_im = pil_im.resize(first_dims, _PILImage.Resampling.LANCZOS if upscale_interpolation.lower()=="lanczos" else _PILImage.Resampling.BILINEAR)
                    up_arr = np.array(pil_im)
                    return up_arr, ext
            except (UnidentifiedImageError, OSError, RuntimeError) as e:
                if verbose_wms:
                    print(f"WMS fail for {layer_id} at pixel_size≈{px} (requested dims ~{w_px}x{h_px}): {e}")
                continue

        # If all fallbacks failed but we captured first_arr before upscale logic
        if first_arr is not None:
            if verbose_wms:
                print(f"Using degraded map for {layer_id} (only initial partial success).")
            return first_arr, first_extent
        raise RuntimeError(f"All WMS attempts failed for layer {layer_id}")

    # Download
    canton_rgba, extent = _download(canton_layer, full_bbox)
    country_rgba, _ = _download(country_layer, full_bbox)
    glaciers_rgba, _ = _download(glacier_extent_layer, full_bbox)
    lakes_rgba, _ = _download(lakes_layer, full_bbox)

    minx, maxx, miny, maxy = extent
    h, w = canton_rgba.shape[:2]
    px_w = (maxx - minx) / w
    px_h = (maxy - miny) / h

    # Tight crop using country alpha
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

    # Solid recolor
    def _solid_fill(src_rgba, hex_color):
        mask = src_rgba[..., 3] > 0
        out = np.zeros_like(src_rgba)
        if hex_color.startswith("#"):
            r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
        else:
            r, g, b = (179, 217, 255)
        out[mask, 0] = r; out[mask, 1] = g; out[mask, 2] = b; out[mask, 3] = 255
        return out

    glaciers_fill = _solid_fill(glaciers_rgba, glacier_color)
    lakes_fill = _solid_fill(lakes_rgba, lakes_color)

    # ------------------------------------------------------------------
    # Improved outline extraction (solid, no double edges)
    # ------------------------------------------------------------------
    def _extract_edge(mask: np.ndarray, width: int, use_scipy: bool = True):
        """
        Return a boolean edge mask (single rim, dilated to width).
        mask: boolean polygon mask
        width: target line thickness in pixels
        """
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
                pass  # fall back if scipy missing
        # Fallback (numpy shifts)
        up = np.zeros_like(mask); up[1:] = mask[:-1]
        down = np.zeros_like(mask); down[:-1] = mask[1:]
        left = np.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = np.zeros_like(mask); right[:, :-1] = mask[:, 1:]
        edge = mask & (~(up & down & left & right))
        # Dilate rudimentarily
        for _ in range(max(0, width - 1)):
            dil = edge.copy()
            dil[:-1] |= edge[1:]
            dil[1:] |= edge[:-1]
            dil[:, :-1] |= edge[:, 1:]
            dil[:, 1:] |= edge[:, :-1]
            edge = dil
        return edge

    # Alpha threshold to ignore anti-alias fringe
    alpha_thresh = 180
    canton_mask = canton_rgba[..., 3] >= alpha_thresh
    country_mask = country_rgba[..., 3] >= alpha_thresh

    country_edge = _extract_edge(country_mask, country_outline_width)
    canton_edge = _extract_edge(canton_mask, canton_outline_width)

    # Remove national border part from canton edges to avoid double stroke
    canton_edge &= ~country_edge

    def _edge_to_rgba(edge_mask: np.ndarray) -> np.ndarray:
        out = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 4), dtype=np.uint8)
        out[edge_mask, 0:3] = 0  # black
        out[edge_mask, 3] = 255
        return out

    canton_outline = _edge_to_rgba(canton_edge)
    country_outline = _edge_to_rgba(country_edge)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")
    ax.imshow(glaciers_fill, extent=extent, origin="upper", zorder=1)
    ax.imshow(canton_outline, extent=extent, origin="upper", zorder=2)
    ax.imshow(country_outline, extent=extent, origin="upper", zorder=3)
    ax.imshow(lakes_fill, extent=extent, origin="upper", zorder=4)

    # Cities
    if city_labels is None:
        city_labels = {
            "Genève": (2483000, 1118000),
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

    # Canton labels
    canton_labels = {
        "VS": (2620000, 1125000),  # Valais
        "GR": (2775000, 1190000),  # Graubünden
    }
    for label, (cx, cy) in canton_labels.items():
        if extent[0] <= cx <= extent[1] and extent[2] <= cy <= extent[3]:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=city_fontsize + 2, color="gray", weight="bold", style="italic", zorder=12)

    # Glacier abbreviation colors
    rect_colors = {"AH": "#1f77b4", "CJ": "#2ca02c", "HS": "#ff7f0e"}

    # Field site markers
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
                # Base auto offset
                if field_site_label_offset is None:
                    base_dx = (extent[1] - extent[0]) * 0.014
                    base_dy = (extent[3] - extent[2]) * 0.014
                else:
                    base_dx, base_dy = field_site_label_offset
                for code, (fx, fy) in field_site_markers.items():
                    if not (extent[0] <= fx <= extent[1] and extent[2] <= fy <= extent[3]):
                        continue
                    # Per-site override
                    if field_site_label_offsets and code in field_site_label_offsets:
                        dx, dy = field_site_label_offsets[code]
                    else:
                        dx, dy = base_dx, base_dy
                    label_color = rect_colors.get(code, field_site_label_color)
                    # Place label and get bbox
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
                    # Force draw to update text position
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

    if cut_to_country_outline:
        fig.canvas.draw()
        arr = np.array(fig.canvas.renderer.buffer_rgba())
        # Resize mask to match arr shape
        from skimage.transform import resize
        mask = country_rgba[..., 3] > 0
        mask_resized = resize(mask.astype(float), arr.shape[:2], order=0, preserve_range=True) > 0.5
        arr[..., 3][~mask_resized] = 0
        ax.clear()
        ax.imshow(arr, extent=extent, origin="upper")
        ax.set_axis_off()

    # Format
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Scale bar
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

    return fig, ax

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

abbr = {"Alphubel": "AH", "Chessjen": "CJ", "Hohsaas": "HS"}
rect_colors = {"AH": "#1f77b4", "CJ": "#2ca02c", "HS": "#ff7f0e"}

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
    wspace: float = 0.12
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
        label_color = rect_colors.get(code, "k")
        ax.text(0.02, 0.02, code, transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=base_fontsize, weight="bold",
                color=label_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=label_color, alpha=0.7))

        # Upper left panel label (a/b/c, black, white box)
        panel_labels = panel_labels
        ax.text(0.02, 0.98, panel_labels.get(code, ""), transform=ax.transAxes,
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

def _add_corner_label(ax, text, fontsize=16, box=None):  # default +4
    if box is None:
        box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)
    ax.text(0.01, 0.985, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, weight="bold",
            bbox=box)

def compose_icetemp_and_vertical_profiles(
    left,                      # Figure | str/Path (image) | np.ndarray (H,W,4)
    right,                     # Figure | str/Path (image) | np.ndarray (H,W,4)
    *,
    figsize=(14, 5.6),
    dpi: int = 300,
    width_ratios: Union[str, tuple[float, float]] = "auto",  # NEW: "auto" uses image aspect
    wspace: float = 0.05,
    labels: tuple[str, str] = ("a", "b"),
    label_fontsize: int = 16,
    label_box: dict | None = None,
    render_dpi: int | None = 300,
    trim_borders: bool = True,
    trim_tolerance: int = 6,
    savefig_path: Union[str, Path, None] = None,
):
    """
    Compose a 2‑panel figure:
      - left  panel: ice temperature profile (plot_icetemp_profile) -> annotated 'A'
      - right panel: multiple vertical profiles (plot_multiple_temperature_profiles) -> annotated 'B'

    Arguments 'left' and 'right' can be Figure, image path, or RGBA ndarray.
    Panels are rendered with aspect='auto' to guarantee equal visual height.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    from PIL import Image

    def _to_rgba(obj):
        if isinstance(obj, Figure):
            return _figure_to_array(obj, dpi=render_dpi)
        if isinstance(obj, (str, Path)):
            return np.array(Image.open(obj).convert("RGBA"))
        if isinstance(obj, np.ndarray):
            if obj.ndim == 3 and obj.shape[2] in (3, 4):
                return obj if obj.shape[2] == 4 else np.dstack([obj, 255*np.ones(obj.shape[:2], np.uint8)])
        raise TypeError(f"Unsupported type for panel: {type(obj)}")

    def _trim_rgba_border(arr: np.ndarray, tol: int = 5) -> np.ndarray:
        """
        Trim uniform (near‑white) borders even for fully opaque RGBA canvases.
        tol = max channel distance from 255 to still count as border.
        """
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

    if trim_borders:
        arr_left = _trim_rgba_border(arr_left, tol=trim_tolerance)
        arr_right = _trim_rgba_border(arr_right, tol=trim_tolerance)

    # Compute widths from native aspect so both panels have same height without stretch
    if isinstance(width_ratios, str) and width_ratios.lower() == "auto":
        wl = max(1e-3, arr_left.shape[1] / arr_left.shape[0])
        wr = max(1e-3, arr_right.shape[1] / arr_right.shape[0])
        width_ratios = (wl, wr)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=width_ratios, wspace=wspace)

    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(arr_left, aspect='equal')   # preserve proportions
    axA.set_axis_off()

    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(arr_right, aspect='equal')  # preserve proportions
    axB.set_axis_off()

    # Labels
    if label_box is None:
        label_box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)
    _add_corner_label(axA, labels[0], fontsize=label_fontsize, box=label_box)
    _add_corner_label(axB, labels[1], fontsize=label_fontsize, box=label_box)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if savefig_path:
        savefig_path = Path(savefig_path)
        savefig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)

    return fig, {"A": axA, "B": axB}