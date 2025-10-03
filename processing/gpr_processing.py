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
from scipy.interpolate import Rbf
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

def read_thickness_csv(path):
    """
    Read one CSV file with GPR interpretation data.

    Expected columns (case-insensitive):
      - Profile, X, Y
      - Depth      -> ice thickness (thickness)
      - Elevation  -> bedrock elevation (zbed)
      - Surface    -> surface elevation (zsurf) [optional]

    Returns a DataFrame with standardized columns:
      ['profile','x','y','zsurf','zbed','thickness']
    """
    df = pd.read_csv(path, engine="python")

    # Normalize column names (case-insensitive)
    df.columns = [c.strip().lower() for c in df.columns]

    # Check for required columns
    required_cols = ['profile', 'x', 'y', 'depth', 'elevation']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # Standardize names
    df = df.rename(columns={
        'profile': 'profile',
        'x': 'x',
        'y': 'y',
        'depth': 'thickness',  # Depth is ice thickness
        'elevation': 'zbed'    # Elevation is bedrock elevation
    })

    # Surface elevation: prefer explicit 'surface' column, else compute zbed + thickness
    if 'surface' in df.columns:
        df['zsurf'] = pd.to_numeric(df['surface'], errors='coerce')
    else:
        df['zsurf'] = pd.to_numeric(df['zbed'], errors='coerce') + pd.to_numeric(df['thickness'], errors='coerce')

    # Ensure numeric types
    for c in ['x', 'y', 'zsurf', 'zbed', 'thickness']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep only needed columns
    keep_cols = ['profile', 'x', 'y', 'zsurf', 'zbed', 'thickness']
    df = df[keep_cols].copy()

    # Add source file info
    df['source'] = path

    return df

def load_points_from_txt(paths, epsg=2056, drop_duplicates=True, aggregate_duplicates='mean', return_type='gdf'):
    """
    Load multiple TXT files and return a GeoDataFrame or DataFrame of points in EPSG:epsg.
    Preserves ALL columns from input files (x, y, profile, thickness, zsurf, zbed, etc.)
    """
    if isinstance(paths, (str, bytes)):
        paths = [paths]
        
    frames = [read_thickness_txt(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['x', 'y', 'thickness'])

    # Preserve ALL columns during aggregation
    if drop_duplicates:
        if aggregate_duplicates in ('mean', 'median'):
            # Identify numeric columns for aggregation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {}
            
            for col in df.columns:
                if col in ['x', 'y']:
                    agg_dict[col] = 'first'  # Keep first coordinate
                elif col == 'profile':
                    agg_dict[col] = 'first'  # Keep first profile id
                elif col in numeric_cols:
                    agg_dict[col] = aggregate_duplicates  # Aggregate numeric data
                else:
                    agg_dict[col] = 'first'  # Keep first value for other columns
            
            df = (df.sort_values('profile')
                    .groupby(['x', 'y'], as_index=False)
                    .agg(agg_dict))
        elif aggregate_duplicates == 'last':
            df = df.sort_index()
            df = df.drop_duplicates(subset=['x', 'y'], keep='last')
        else:
            df = df.drop_duplicates(subset=['x', 'y'])

    if return_type == 'df':
        return df
    
    # Return GeoDataFrame - preserve all columns, just add geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['x'].values, df['y'].values)],
        crs=f"EPSG:{epsg}"
    )
    return gdf

def load_points_from_csv(paths, epsg=2056, source_epsg=None, drop_duplicates=True, aggregate_duplicates='mean', return_type='gdf'):
    """
    Load multiple CSV files with GPR interpretation data.
    Preserves ALL columns from input files (x, y, profile, thickness, elevation data, etc.)
    """
    if isinstance(paths, (str, bytes)):
        paths = [paths]
        
    frames = [read_thickness_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    
    # Clean profile column: remove "LINE" prefix and keep only the number
    if 'profile' in df.columns:
        df['profile'] = df['profile'].astype(str).str.replace(r'^LINE', '', regex=True, case=False)
        try:
            df['profile'] = pd.to_numeric(df['profile'])
        except (ValueError, TypeError):
            pass
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['x', 'y', 'thickness'])
    
    # Handle coordinate transformation if needed
    if source_epsg and source_epsg != epsg:
        temp_gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df['x'].values, df['y'].values)],
            crs=f"EPSG:{source_epsg}"
        )
        temp_gdf = temp_gdf.to_crs(f"EPSG:{epsg}")
        df['x'] = temp_gdf.geometry.x
        df['y'] = temp_gdf.geometry.y
    
    # Handle duplicates - preserve ALL columns
    if drop_duplicates:
        if aggregate_duplicates in ('mean', 'median'):
            df['x_round'] = df['x'].round(6)
            df['y_round'] = df['y'].round(6)
            
            # Build aggregation dictionary for all columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {}
            
            for col in df.columns:
                if col in ['x_round', 'y_round']:
                    continue  # Skip these temporary columns
                elif col in ['x', 'y']:
                    agg_dict[col] = 'first'
                elif col == 'profile':
                    agg_dict[col] = 'first'
                elif col in numeric_cols:
                    agg_dict[col] = aggregate_duplicates
                else:
                    agg_dict[col] = 'first'
            
            agg_data = (df.sort_values('profile')
                           .groupby(['x_round', 'y_round'], as_index=False)
                           .agg(agg_dict))
            df = agg_data.drop(columns=['x_round', 'y_round'])
            
        elif aggregate_duplicates == 'last':
            df['x_round'] = df['x'].round(6)
            df['y_round'] = df['y'].round(6)
            df = df.sort_index()
            df = df.drop_duplicates(subset=['x_round', 'y_round'], keep='last')
            df = df.drop(columns=['x_round', 'y_round'])
        else:
            df['x_round'] = df['x'].round(6)
            df['y_round'] = df['y'].round(6)
            df = df.drop_duplicates(subset=['x_round', 'y_round'])
            df = df.drop(columns=['x_round', 'y_round'])
    
    if return_type == 'df':
        return df
    
    # Return GeoDataFrame - preserve all columns, just add geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['x'].values, df['y'].values)],
        crs=f"EPSG:{epsg}"
    )
    return gdf
    
    # Return GeoDataFrame (default behavior)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['x'].values, df['y'].values)],
        crs=f"EPSG:{epsg}"
    )
    return gdf


def interpolate_thickness_to_grid(points_gdf, value_col='thickness', pixel_size=20.0, rbf_function='linear', polygon_mask: gpd.GeoDataFrame|None=None, padding=0.0, max_rbf_points=15000):
    """
    Interpolate scattered ice thickness points to a regular grid.
    Uses RBF for small datasets, switches to scipy.griddata for large ones.
    
    Args:
        max_rbf_points: Switch to griddata if more points than this (default: 15000)
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

    # Choose interpolation method based on data size
    n_points = len(X)
    print(f"Interpolating {n_points} points...")
    
    if n_points <= max_rbf_points:
        print("Using RBF interpolation...")
        rbf = Rbf(X, Y, Z, function=rbf_function)
        grid = rbf(grid_x, grid_y)
    else:
        print("Using scipy.griddata (faster for large datasets)...")
        points = np.column_stack([X, Y])
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Use linear interpolation (fast and stable)
        grid_flat = griddata(points, Z, grid_points, method='linear', fill_value=np.nan)
        grid = grid_flat.reshape(grid_x.shape)
        
        # Optional: fill NaN holes with nearest neighbor
        if np.any(np.isnan(grid_flat)):
            print("Filling holes with nearest neighbor...")
            grid_flat_filled = griddata(points, Z, grid_points, method='nearest')
            grid_filled = grid_flat_filled.reshape(grid_x.shape)
            grid = np.where(np.isnan(grid), grid_filled, grid)

    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    if polygon_mask is not None and not polygon_mask.empty:
        mask = geometry_mask(
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
    Read the new borehole CSV (comma-separated) with columns like:
      number,name,date,X,Y,borehole depth (m),chaing/logger,chain length [m]
    - Ignores empty/comment rows (starting with '*')
    - Parses mixed date formats (dd/mm/yyyy, dd.mm.yyyy, etc.)
    - Returns only the most recent entry per name
    - Adds extra columns when available: depth_m, chain_logger, chain_length_m
    """
    # Read as strings; keep rows for our own cleaning
    df = pd.read_csv(path, sep=',', engine='python', dtype=str, skip_blank_lines=True)
    # Drop fully empty rows
    df = df.dropna(how='all')

    # Normalize headers
    orig_cols = list(df.columns)
    lower = {c.lower().strip(): c for c in df.columns}

    def find_col(*keys, default=None):
        for k in keys:
            if k in lower:
                return lower[k]
        # fuzzy search
        for lc, oc in lower.items():
            if any(all(t in lc for t in k.split()) for k in keys):
                return oc
        return default

    col_name = find_col('name')
    col_date = find_col('date')
    col_x = find_col('x')
    col_y = find_col('y')
    col_depth = find_col('borehole depth (m)', 'borehole depth', 'depth (m)', 'depth')
    col_chain_logger = find_col('chain/logger', 'chaing/logger', 'logger', 'chain id')
    col_chain_len = find_col('chain length [m]', 'chain length', 'length [m]')

    needed = [col_name, col_date, col_x, col_y]
    if any(c is None for c in needed):
        raise ValueError(f"Missing required columns in {path}. Found columns: {orig_cols}")

    df = df[[c for c in [col_name, col_date, col_x, col_y, col_depth, col_chain_logger, col_chain_len] if c in df.columns]].copy()
    df = df.rename(columns={
        col_name: 'name', col_date: 'date', col_x: 'x', col_y: 'y',
        **({col_depth: 'depth_m'} if col_depth else {}),
        **({col_chain_logger: 'chain_logger'} if col_chain_logger else {}),
        **({col_chain_len: 'chain_length_m'} if col_chain_len else {})
    })

    # Clean names and drop comment/asterisk rows
    df['name'] = df['name'].astype(str).str.strip()
    df = df[df['name'].notna() & (df['name'] != '') & ~df['name'].str.startswith('*')]

    # Parse dates (robust: dd/mm/yyyy, dd.mm.yyyy, etc.)
    df['date_parsed'] = pd.to_datetime(df['date'].astype(str).str.strip().str.replace('\\.', '/', regex=True),
                                       dayfirst=True, errors='coerce')

    # Numeric conversions
    df['x_num'] = df['x'].apply(_to_float)
    df['y_num'] = df['y'].apply(_to_float)
    if 'depth_m' in df.columns:
        df['depth_m'] = df['depth_m'].apply(_to_float)
    if 'chain_length_m' in df.columns:
        df['chain_length_m'] = df['chain_length_m'].apply(_to_float)

    df['measured'] = df['x_num'].notna() & df['y_num'].notna()

    # Optional name filter
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

    # Keep latest measured entry per name
    measured = df[df['measured']].copy()
    if not measured.empty:
        measured = (measured
                    .sort_values(['name', 'date_parsed', 'date'])
                    .groupby('name', as_index=False, sort=False)
                    .tail(1))

    latest_names = set(measured['name']) if not measured.empty else set()

    # Unmeasured or missing requested names (for info)
    unmeasured = (df[~df['name'].isin(latest_names)]
                    .sort_values(['name', 'date_parsed', 'date'])
                    .groupby('name', as_index=False, sort=False)
                    .tail(1)
                    .drop(columns=['x_num', 'y_num', 'measured'], errors='ignore'))

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

    # Build GeoDataFrame for measured
    if measured.empty:
        boreholes_gdf = gpd.GeoDataFrame(columns=['name','date','x','y','geometry'], crs=f"EPSG:{epsg}")
    else:
        cols_keep = ['name', 'date', 'x', 'y']
        for extra in ('depth_m', 'chain_logger', 'chain_length_m'):
            if extra in measured.columns:
                cols_keep.append(extra)
        boreholes_gdf = gpd.GeoDataFrame(
            measured.rename(columns={'x_num':'x', 'y_num':'y'})[cols_keep].assign(
                x=measured['x_num'].values, y=measured['y_num'].values
            ),
            geometry=[Point(xy) for xy in zip(measured['x_num'], measured['y_num'])],
            crs=f"EPSG:{epsg}"
        )
        # Ensure consistency
        boreholes_gdf['x'] = boreholes_gdf.geometry.x
        boreholes_gdf['y'] = boreholes_gdf.geometry.y
        boreholes_gdf = boreholes_gdf.loc[:, ~boreholes_gdf.columns.duplicated()]

    # Cleanup helper col
    for c in ('_key',):
        if c in df.columns:
            df = df.drop(columns=[c])

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
    Build a profile table for one 'profile' id. Now a convenience wrapper around load_points_from_*.
    """
    # Determine if input is TXT or CSV based on file extension or DataFrame columns
    if isinstance(df_or_paths, (str, bytes)):
        path = df_or_paths
        if path.lower().endswith('.csv'):
            df = load_points_from_csv([path], return_type='df')
        else:
            df = load_points_from_txt([path], return_type='df')
    elif isinstance(df_or_paths, list):
        # Assume all same type, check first file
        if df_or_paths and df_or_paths[0].lower().endswith('.csv'):
            df = load_points_from_csv(df_or_paths, return_type='df')
        else:
            df = load_points_from_txt(df_or_paths, return_type='df')
    else:
        df = df_or_paths.copy()

    if 'profile' not in df.columns:
        raise ValueError("Input must contain a 'profile' column.")
    prof = df[df['profile'] == profile_id].copy()
    if prof.empty:
        raise ValueError(f"No rows found for profile={profile_id}")

    # Ensure required columns exist (same logic as before)
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
