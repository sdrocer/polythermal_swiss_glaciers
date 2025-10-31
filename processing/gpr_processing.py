import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import warnings
import re
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint, GeometryCollection
from scipy.interpolate import griddata, Rbf
from shapely.ops import unary_union
from io import BytesIO
import os
import glob

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

def load_points_from_shps_dir(shp_dir, *, epsg=2056, source_epsg=None, drop_duplicates=True, aggregate_duplicates='mean', return_type='gdf'):
    """
    Load all .shp files from a directory. Each shapefile is treated as one profile (profile id
    extracted from filename with regex r'profil[-_]?0*(\d+)'). Extracts all vertices from
    LineString / MultiLineString / Point geometries into point rows. Returns a GeoDataFrame
    (or DataFrame if return_type=='df') in target EPSG.

    Output columns include at least: profile, x, y, source, plus any attribute fields present.
    """
    shp_files = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))
    rows = []
    file_crs = None

    for fp in shp_files:
        try:
            g = gpd.read_file(fp)
        except Exception as e:
            warnings.warn(f"failed to read '{fp}': {e}")
            continue

        # If user provided source_epsg and file has no CRS, set it
        if (g.crs is None or g.crs == {}) and source_epsg:
            try:
                g = g.set_crs(f"EPSG:{source_epsg}", allow_override=True)
            except Exception:
                pass

        # capture file CRS (first encountered)
        if g.crs is not None and file_crs is None:
            try:
                file_crs = int(g.crs.to_epsg())
            except Exception:
                file_crs = None

        # profile id from filename (e.g. profil-001)
        m = re.search(r'profil[-_]?0*([0-9]+)', os.path.basename(fp), flags=re.IGNORECASE)
        file_profile = int(m.group(1)) if m else None

        for _, feat in g.iterrows():
            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue

            # extract attributes (except geometry)
            attrs = feat.drop(labels=['geometry'], errors='ignore').to_dict()

            # prefer 'profile' attribute in file if present, otherwise use filename-derived id
            profile_val = attrs.get('profile', file_profile)
            try:
                if pd.notna(profile_val):
                    profile_val = int(profile_val)
            except Exception:
                # keep as-is if not convertible
                pass

            # iterator to yield coordinate tuples from geometry
            def iter_coords(ggeom):
                if isinstance(ggeom, Point):
                    yield (ggeom.x, ggeom.y)
                elif isinstance(ggeom, LineString):
                    for c in ggeom.coords:
                        yield (float(c[0]), float(c[1]))
                elif isinstance(ggeom, MultiLineString):
                    for part in ggeom.geoms:
                        for c in part.coords:
                            yield (float(c[0]), float(c[1]))
                elif isinstance(ggeom, MultiPoint):
                    for part in ggeom.geoms:
                        yield (float(part.x), float(part.y))
                elif isinstance(ggeom, GeometryCollection):
                    for part in ggeom.geoms:
                        for c in iter_coords(part):
                            yield c
                else:
                    # fallback: try coords attribute
                    try:
                        for c in getattr(ggeom, 'coords', []):
                            yield (float(c[0]), float(c[1]))
                    except Exception:
                        return

            for x, y in iter_coords(geom):
                row = dict(attrs)  # copy attributes
                row.update({
                    'x': float(x),
                    'y': float(y),
                    'profile': profile_val,
                    'source': fp
                })
                rows.append(row)

    if not rows:
        empty_gdf = gpd.GeoDataFrame(columns=['profile', 'x', 'y', 'source'], geometry=[], crs=f"EPSG:{epsg}")
        if return_type == 'df':
            return pd.DataFrame(columns=['profile', 'x', 'y', 'source'])
        return empty_gdf

    df = pd.DataFrame(rows)

    # Determine input CRS: prefer explicit source_epsg, else file_crs (from files)
    in_crs = source_epsg if source_epsg else file_crs

    # reproject coordinates if needed
    if in_crs and int(in_crs) != int(epsg):
        tmp = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['x'], df['y'])], crs=f"EPSG:{in_crs}")
        tmp = tmp.to_crs(f"EPSG:{epsg}")
        df['x'] = tmp.geometry.x
        df['y'] = tmp.geometry.y

    # Clean numeric and drop invalid
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])

    # Optional duplicate handling / aggregation (consistent with CSV/TXT loaders)
    if drop_duplicates:
        df['x_round'] = df['x'].round(6)
        df['y_round'] = df['y'].round(6)
        if aggregate_duplicates in ('mean', 'median'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg = {}
            for c in df.columns:
                if c in ('x_round', 'y_round'):
                    continue
                if c in ('x', 'y'):
                    agg[c] = 'first'
                elif c == 'profile':
                    agg[c] = 'first'
                elif c in numeric_cols:
                    agg[c] = aggregate_duplicates
                else:
                    agg[c] = 'first'
            df = df.sort_values('profile', na_position='last').groupby(['x_round', 'y_round'], as_index=False).agg(agg).drop(columns=['x_round', 'y_round'])
        elif aggregate_duplicates == 'last':
            df = df.sort_index()
            df = df.drop_duplicates(subset=['x_round', 'y_round'], keep='last').drop(columns=['x_round', 'y_round'])
        else:
            df = df.drop_duplicates(subset=['x_round', 'y_round']).drop(columns=['x_round', 'y_round'])

    if return_type == 'df':
        return df

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['x'], df['y'])], crs=f"EPSG:{epsg}")
    # ensure columns order similar to example (profile,x,y,...,source,geometry) if desired
    cols = [c for c in ['profile', 'x', 'y'] if c in gdf.columns] + [c for c in gdf.columns if c not in ('profile', 'x', 'y', 'geometry')]
    cols = cols + ['geometry']
    gdf = gdf.loc[:, [c for c in cols if c in gdf.columns]]
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

def square_bbox_from_points(p1, p2, buffer_m: float = 0.0):
    """
    Build an axis-aligned square bbox (xmin, ymin, xmax, ymax) from two points (x, y).
    The square is centered between p1 and p2 and its side is max(|dx|, |dy|), optionally
    expanded by buffer_m on all sides.
    """
    if p1 is None or p2 is None:
        raise ValueError("p1 and p2 must be (x, y) tuples.")
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half = 0.5 * max(abs(x2 - x1), abs(y2 - y1))
    half = half + float(buffer_m)

    if half <= 0:
        raise ValueError("Points are identical and buffer_m <= 0; square would have zero area.")

    return (cx - half, cy - half, cx + half, cy + half)

def make_square_bbox(bbox, buffer_m: float = 0.0, pixel_size: float | None = None):
    """
    Return a square bbox (xmin, ymin, xmax, ymax) centered on the input bbox.
    - buffer_m: extra margin added on all sides (meters).
    - pixel_size: if given, snap side length to an integer multiple of pixel_size
                  to ensure WMS WIDTH == HEIGHT exactly.
    """
    xmin, ymin, xmax, ymax = map(float, bbox)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    w, h = (xmax - xmin), (ymax - ymin)
    side = max(w, h) + 2.0 * float(buffer_m)
    if pixel_size and pixel_size > 0:
        side = float(np.ceil(side / float(pixel_size)) * float(pixel_size))
    half = 0.5 * side
    return (cx - half, cy - half, cx + half, cy + half)

def square_bbox_from_gdf(gdf: gpd.GeoDataFrame, buffer_m: float = 0.0, pixel_size: float | None = None):
    """
    Square bbox around a GeoDataFrame’s extent, with optional buffer and pixel snapping.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    return make_square_bbox((xmin, ymin, xmax, ymax), buffer_m=buffer_m, pixel_size=pixel_size)

def download_swisstopo_orthophoto_from_points(
    p1, p2, out_tif, *,
    crs_epsg: int = 2056,
    pixel_size: float = 1.0,
    layer: str = "ch.swisstopo.swissimage",
    fmt: str = "image/jpeg",
    max_px: int = 8000,
    timeout: int = 60,
    buffer_m: float = 0.0
):
    """
    Convenience wrapper: provide two (x,y) points, download a square orthophoto
    covering the square in between them.
    Coordinates must be in EPSG:crs_epsg (default LV95: 2056).
    """
    bbox = square_bbox_from_points(p1, p2, buffer_m=buffer_m)
    return download_swisstopo_orthophoto(
        bbox, out_tif,
        crs_epsg=crs_epsg, pixel_size=pixel_size,
        layer=layer, fmt=fmt, max_px=max_px, timeout=timeout
    )

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

def load_borehole_positions(path, epsg=2056, keep_names=None, case_insensitive=True):
    """
    Read borehole CSV and return (GeoDataFrame, missing_list).
    - auto-detects delimiter
    - strips BOM/whitespace from headers and values
    - locates name/X/Y columns case-insensitively
    - returns GeoDataFrame in EPSG:epsg and list of requested names not found
    """
    path = os.path.normpath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"borehole CSV not found: {path}")

    # sniff delimiter
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        sample = fh.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t'])
            delim = dialect.delimiter
        except Exception:
            delim = ','

    df = pd.read_csv(path, sep=delim, engine='python', dtype=str, skip_blank_lines=True)
    # clean column names and string values
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # find essential columns case-insensitively
    cols_lower = {c.lower(): c for c in df.columns}
    def find_col(names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    name_col = find_col(['name', 'station', 'site'])
    x_col = find_col(['x', 'east', 'easting', 'lon', 'longitude'])
    y_col = find_col(['y', 'north', 'northing', 'lat', 'latitude'])
    if name_col is None or x_col is None or y_col is None:
        raise ValueError(f"could not find name/X/Y columns in {path}. found columns: {df.columns.tolist()}")

    # convert coordinates
    df[x_col] = pd.to_numeric(df[x_col].str.replace("'", "").str.replace(",", ""), errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col].str.replace("'", "").str.replace(",", ""), errors='coerce')
    df = df.dropna(subset=[x_col, y_col]).copy()

    # geometry and CRS
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[x_col].astype(float), df[y_col].astype(float))], crs=f"EPSG:{epsg}")

    # standardize name column to 'name'
    if name_col != 'name':
        gdf = gdf.rename(columns={name_col: 'name'})

    # filter by keep_names if given
    missing = []
    if keep_names:
        req = keep_names
        if case_insensitive:
            present_mask = gdf['name'].str.lower().isin([r.lower() for r in req])
            present_names = gdf.loc[present_mask, 'name'].unique().tolist()
            missing = [r for r in req if r.lower() not in [p.lower() for p in present_names]]
            gdf = gdf.loc[present_mask].copy()
        else:
            present_mask = gdf['name'].isin(req)
            present_names = gdf.loc[present_mask, 'name'].unique().tolist()
            missing = [r for r in req if r not in present_names]
            gdf = gdf.loc[present_mask].copy()

    # keep useful columns, ensure 'name' exists
    cols_keep = [c for c in ['number', 'name', 'date', x_col, y_col, 'borehole depth (m)', 'chaing/logger', 'chain length [m]'] if c in gdf.columns]
    # always keep geometry
    gdf = gdf.loc[:, [c for c in cols_keep if c in gdf.columns] + ['geometry']]

    # reset index and return
    gdf = gdf.reset_index(drop=True)
    return gdf, missing
    
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
