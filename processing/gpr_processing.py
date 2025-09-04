import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import warnings
import re
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from shapely.geometry import Point
from scipy.interpolate import griddata
from shapely.ops import unary_union
from io import BytesIO

def read_thickness_txt(path):
    """
    Read one whitespace-separated TXT file with columns:
    profile xbed ybed zsurf zbed thick
    Returns a DataFrame with standardized columns:
    ['profile','x','y','zsurf','zbed','thickness']
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#", header=0)
    rename_map = {
        'profile':'profile', 'xbed':'x', 'ybed':'y',
        'zsurf':'zsurf', 'zbed':'zbed', 'thick':'thickness'
    }
    # normalize case
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=rename_map)
    needed = ['x','y','thickness']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
    # keep optional columns if present
    keep = ['profile','x','y','zsurf','zbed','thickness']
    df = df[[c for c in keep if c in df.columns]].copy()
    df['source'] = path
    return df

def load_points_from_txt(paths, epsg=2056, drop_duplicates=True, aggregate_duplicates='mean'):
    """
    Load multiple TXT files and return a GeoDataFrame of points in EPSG:epsg.
    Optionally aggregate duplicate XY by mean/median/last on thickness.
    """
    frames = [read_thickness_txt(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['x', 'y', 'thickness'])

    if drop_duplicates:
        if aggregate_duplicates in ('mean', 'median'):
            # Use string agg to avoid FutureWarning
            df = (df.groupby(['x', 'y'], as_index=False)
                    .agg(thickness=( 'thickness', aggregate_duplicates )))
        elif aggregate_duplicates == 'last':
            df = df.sort_index()  # keep input order
            df = df.drop_duplicates(subset=['x', 'y'], keep='last')
        else:
            df = df.drop_duplicates(subset=['x', 'y'])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['x'].values, df['y'].values)],
        crs=f"EPSG:{epsg}"
    )
    return gdf

def interpolate_to_grid(points_gdf, value_col='thickness', pixel_size=20.0, method='linear', polygon_mask: gpd.GeoDataFrame|None=None, padding=0.0):
    """
    Interpolate scattered points to a regular grid using scipy.griddata.
    method: 'linear' | 'cubic' | 'nearest'
    polygon_mask: optional glacier outline (same CRS) to mask outside to NaN.
    Returns grid, transform, crs
    """
    crs = points_gdf.crs
    X = np.array([p.x for p in points_gdf.geometry])
    Y = np.array([p.y for p in points_gdf.geometry])
    Z = points_gdf[value_col].values

    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    xmin -= padding; ymin -= padding; xmax += padding; ymax += padding
    width = int(np.ceil((xmax - xmin) / pixel_size)) + 1
    height = int(np.ceil((ymax - ymin) / pixel_size)) + 1

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmin + pixel_size*(width-1), width),
        np.linspace(ymax, ymax - pixel_size*(height-1), height)
    )

    grid = griddata(points=(X, Y), values=Z, xi=(grid_x, grid_y), method=method)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    if polygon_mask is not None and not polygon_mask.empty:
        mask = geometry_mask(                     # use imported function
            [geom for geom in polygon_mask.geometry],
            out_shape=grid.shape,
            transform=transform,
            invert=True
        )
        grid = np.where(mask, grid, np.nan)

    return grid, transform, crs

def save_geotiff(path, grid, transform, crs_epsg, nodata=np.nan):
    height, width = grid.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'height': height,
        'width': width,
        'transform': transform,
        'crs': f"EPSG:{crs_epsg}",
        'nodata': nodata
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(grid.astype('float32'), 1)

def make_coverage_polygon(points_gdf, method='alpha', alpha=None, buffer_m=0.0):
    """
    Build a polygon representing the area covered by the GPR points.
    - method: 'alpha' (concave hull, needs alphashape) or 'convex'
    - alpha: None => optimize automatically; number => pass to alphashape
    - buffer_m: optional buffer (meters) to slightly expand the area
    Returns a GeoDataFrame with one polygon in the same CRS as points_gdf.
    """
    if points_gdf.empty:
        raise ValueError("points_gdf is empty")

    geom = points_gdf.geometry
    poly = None

    if method == 'alpha' and _alphashape is not None and len(geom) >= 4:
        coords = np.array([[p.x, p.y] for p in geom])
        try:
            a = alpha if alpha is not None else _alphashape.optimizealpha(coords)
            poly = _alphashape.alphashape(coords, a)
            if poly.is_empty:
                poly = None
            # keep largest part if multi
            if poly and poly.geom_type == 'MultiPolygon':
                poly = max(list(poly.geoms), key=lambda g: g.area)
        except Exception as e:
            warnings.warn(f"alphashape failed ({e}); falling back to convex hull")
            poly = None

    if poly is None:
        poly = unary_union(geom).convex_hull

    if buffer_m and buffer_m != 0:
        poly = poly.buffer(buffer_m)

    return gpd.GeoDataFrame({'name': ['coverage']}, geometry=[poly], crs=points_gdf.crs)

def bbox_from_gdf(gdf: gpd.GeoDataFrame, buffer_m: float = 0.0):
    """
    Return (xmin, ymin, xmax, ymax) from any GeoDataFrame, optionally buffered in meters.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    if buffer_m and buffer_m != 0:
        xmin -= buffer_m; ymin -= buffer_m; xmax += buffer_m; ymax += buffer_m
    return float(xmin), float(ymin), float(xmax), float(ymax)

def download_swisstopo_orthophoto(
    bbox, out_tif, crs_epsg=2056, pixel_size=1.0,
    layer="ch.swisstopo.swissimage", fmt="image/jpeg",
    max_px=8000, timeout=60
):
    """
    Download a swisstopo orthophoto via WMS and save as GeoTIFF.
    - bbox: (xmin, ymin, xmax, ymax) in EPSG:crs_epsg
    - pixel_size: meters per pixel (width/height derived from bbox)
    - layer: e.g., 'ch.swisstopo.swissimage'
    Returns (out_tif, transform, crs)
    """
    import requests
    from PIL import Image

    xmin, ymin, xmax, ymax = map(float, bbox)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    width = int(np.clip(np.ceil(dx / pixel_size), 1, max_px))
    height = int(np.clip(np.ceil(dy / pixel_size), 1, max_px))

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": f"EPSG:{crs_epsg}",
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": fmt,
        "TRANSPARENT": "FALSE",
        "DPI": "96",
    }
    url = "https://wms.geo.admin.ch/"
    resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "asses_swiss_gl_therm_regimes/0.1"})
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    arr = np.asarray(img)  # H, W, 3

    xres = dx / arr.shape[1]
    yres = dy / arr.shape[0]
    transform = from_origin(xmin, ymax, xres, yres)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
        "crs": f"EPSG:{crs_epsg}",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)

    return out_tif, transform, rasterio.crs.CRS.from_epsg(crs_epsg)

def load_borehole_positions(
    path: str,
    epsg: int = 2056,
    keep_names: list[str] | None = None,
    case_insensitive: bool = True
):
    """
    Read a semicolon CSV with columns Name;Date;X;Y (Swiss-style decimal commas).
    Always returns ONLY the most recent measured coordinates per Name.
    Optional filtering by keep_names.

    Returns:
        - boreholes_gdf: GeoDataFrame with ['name','date','x','y','geometry'] in EPSG:epsg
                         (latest measured per name only)
        - unmeasured_df: DataFrame of names that have no measured coordinates
                         (limited to keep_names when provided). Also includes
                         rows with status='not_found' for requested names that
                         do not exist in the CSV.
    """
    # Read as strings to clean manually
    df = pd.read_csv(path, sep=';', engine='python', dtype=str, encoding='utf-8', skip_blank_lines=True)
    # Normalize columns (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}
    for need in ('name', 'date', 'x', 'y'):
        if need not in cols:
            raise ValueError(f"Missing column '{need}' in {path}")
    df = df[[cols['name'], cols['date'], cols['x'], cols['y']]].rename(
        columns={cols['name']: 'name', cols['date']: 'date', cols['x']: 'x', cols['y']: 'y'}
    )
    # Drop comment/footnote rows and empty names
    df['name'] = df['name'].astype(str).str.strip()
    df = df[df['name'].notna() & (df['name'] != '') & ~df['name'].str.startswith('*')]

    # Parse date (dd.mm.yy -> 20yy)
    df['date_parsed'] = pd.to_datetime(df['date'].astype(str).str.strip(),
                                       format='%d.%m.%y', dayfirst=True, errors='coerce')

    df['x_num'] = df['x'].apply(_to_float)
    df['y_num'] = df['y'].apply(_to_float)
    df['measured'] = df['x_num'].notna() & df['y_num'].notna()

    # Optional name filtering (case-insensitive)
    wanted = None
    if keep_names:
        keep = [str(n).strip() for n in keep_names if str(n).strip()]
        if case_insensitive:
            df['_key'] = df['name'].astype(str).str.upper()
            wanted = {n.upper() for n in keep}
        else:
            df['_key'] = df['name'].astype(str)
            wanted = set(keep)
        df = df[df['_key'].isin(wanted)]
    else:
        df['_key'] = df['name']

    # Latest measured per Name
    measured = df[df['measured']].copy()
    if not measured.empty:
        measured = (measured
                    .sort_values(['name', 'date_parsed', 'date'])
                    .groupby('name', as_index=False, sort=False)
                    .tail(1))
    latest_measured_names = set(measured['name']) if not measured.empty else set()

    # Names with no measured coords at all -> keep their latest row for info
    unmeasured = (df[~df['name'].isin(latest_measured_names)]
                    .sort_values(['name', 'date_parsed', 'date'])
                    .groupby('name', as_index=False, sort=False)
                    .tail(1)
                    .drop(columns=['x_num', 'y_num', 'measured']))

    # Add requested names not present at all
    if keep_names and wanted is not None:
        have_keys = set(df['_key'].unique())
        missing_keys = sorted(list(wanted - have_keys))
        if missing_keys:
            backmap = {(n.upper() if case_insensitive else n): n for n in keep}
            extra = pd.DataFrame({'name': [backmap[k] for k in missing_keys],
                                  'date': np.nan, 'x': np.nan, 'y': np.nan,
                                  'status': 'not_found'})
            unmeasured = (pd.concat([unmeasured, extra], ignore_index=True)
                            .sort_values(['name', 'date'], na_position='last'))

    # Clean helper column
    df = df.drop(columns=['_key'])

    # Build GeoDataFrame
    if measured.empty:
        boreholes_gdf = gpd.GeoDataFrame(columns=['name','date','x','y','geometry'], crs=f"EPSG:{epsg}")
    else:
        boreholes_gdf = gpd.GeoDataFrame(
            measured.rename(columns={'x_num':'x', 'y_num':'y'})[['name','date','x','y','date_parsed']],
            geometry=[Point(xy) for xy in zip(measured['x_num'], measured['y_num'])],
            crs=f"EPSG:{epsg}"
        ).drop(columns=['date_parsed'])

        # Enforce no duplicate columns
        boreholes_gdf = boreholes_gdf.loc[:, ~boreholes_gdf.columns.duplicated()]

    return boreholes_gdf, unmeasured

def _to_float(val: str):
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == '' or s.lower().startswith('not measured'):
        return np.nan
    # Remove spaces, apostrophes, narrow spaces
    s = s.replace('\u00A0', '').replace('\u202F', '').replace("'", '').replace(' ', '')
    # Keep only digits, sign, separators
    s = re.sub(r'[^0-9,.\-]', '', s)
    if s.count(',') and s.count('.'):
        # Decide which is decimal: the rightmost separator
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return np.nan

def _order_along_track(x: np.ndarray, y: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Return indices that sort points along the main line.
    - 'pca': project onto 1st principal component and sort by projection
    - 'x'|'y': sort by X or Y
    """
    if method == "x":
        return np.argsort(x)
    if method == "y":
        return np.argsort(y)
    # PCA-based ordering (robust for tilted lines)
    coords = np.c_[x, y].astype(float)
    ctr = coords - coords.mean(axis=0, keepdims=True)
    # 2D SVD -> first right-singular vector gives principal direction
    _, _, vh = np.linalg.svd(ctr, full_matrices=False)
    d = ctr @ vh[0]          # projection on first PC
    return np.argsort(d)

def extract_profile_table(df_or_paths, profile_id, *, order_method="pca"):
    """
    Build a profile table for one 'profile' id from the original TXT content.
    Input can be:
      - DataFrame from read_thickness_txt (or concat of several),
      - path or list of paths to TXT files.
    Returns a DataFrame with columns:
      ['profile','x','y','zsurf','zbed','thickness','distance']
    """
    # Load if paths were provided
    if isinstance(df_or_paths, (str, bytes)):
        df = read_thickness_txt(df_or_paths)
    elif isinstance(df_or_paths, list):
        frames = [read_thickness_txt(p) for p in df_or_paths]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = df_or_paths.copy()

    if 'profile' not in df.columns:
        raise ValueError("Input must contain a 'profile' column.")
    prof = df[df['profile'] == profile_id].copy()
    if prof.empty:
        raise ValueError(f"No rows found for profile={profile_id}")

    # Ensure required columns exist
    if 'zsurf' not in prof.columns and 'zbed' not in prof.columns and 'thickness' not in prof.columns:
        raise ValueError("Need at least zsurf+thickness or zbed to build a profile.")

    # Compute missing elevations if needed
    if 'zsurf' not in prof.columns and 'zbed' in prof.columns and 'thickness' in prof.columns:
        prof['zsurf'] = prof['zbed'] + prof['thickness']
    if 'zbed' not in prof.columns:
        if 'zsurf' in prof.columns and 'thickness' in prof.columns:
            prof['zbed'] = prof['zsurf'] - prof['thickness']
        else:
            raise ValueError("Cannot compute 'zbed'; provide zsurf and thickness or zbed directly.")
    if 'thickness' not in prof.columns:
        prof['thickness'] = prof['zsurf'] - prof['zbed']

    # Drop rows without geometry/elevations
    prof = prof.dropna(subset=['x','y','zsurf','zbed','thickness']).copy()

    # Order points along-track
    idx = _order_along_track(prof['x'].to_numpy(), prof['y'].to_numpy(), method=order_method)
    prof = prof.iloc[idx].reset_index(drop=True)

    # Cumulative distance in meters (planimetric)
    dx = np.diff(prof['x'].to_numpy(), prepend=prof['x'].iloc[0])
    dy = np.diff(prof['y'].to_numpy(), prepend=prof['y'].iloc[0])
    prof['distance'] = np.cumsum(np.hypot(dx, dy))
    return prof[['profile','x','y','zsurf','zbed','thickness','distance']]
