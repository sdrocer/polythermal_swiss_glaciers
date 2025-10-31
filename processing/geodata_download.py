import time
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Sequence
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

SWISS_CRS = "EPSG:2056"
SWISSTOPO_WMS_URL = "https://wms.geo.admin.ch/"

# ------------------------------------------------------------------
# WMS layer listing
# ------------------------------------------------------------------
def list_swisstopo_wms_layers(
    *,
    url: str = SWISSTOPO_WMS_URL,
    timeout: int = 60,
    include_hidden: bool = False,
    as_dataframe: bool = True
):
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetCapabilities",
        "VERSION": "1.3.0"
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        raise RuntimeError("Failed to parse WMS GetCapabilities (response not XML).")

    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"wms": ns_uri}
    else:
        ns = {"wms": ""}

    def _iter_layers(elem):
        for child in elem.findall("wms:Layer", ns):
            name_el = child.find("wms:Name", ns)
            title_el = child.find("wms:Title", ns)
            if name_el is not None and title_el is not None:
                yield child
            yield from _iter_layers(child)

    top_layers = root.findall("wms:Capability/wms:Layer", ns)
    layers = []
    for tl in top_layers:
        for lay in _iter_layers(tl):
            name_el = lay.find("wms:Name", ns)
            title_el = lay.find("wms:Title", ns)
            if name_el is None or title_el is None:
                continue
            name = (name_el.text or "").strip()
            if (not include_hidden) and name == "":
                continue
            title = (title_el.text or "").strip()
            abstract_el = lay.find("wms:Abstract", ns)
            abstract = (abstract_el.text or "").strip() if abstract_el is not None else ""
            queryable = lay.attrib.get("queryable", "0") == "1"
            opaque = lay.attrib.get("opaque", "0") == "1"
            layers.append({
                "name": name,
                "title": title,
                "abstract": abstract,
                "queryable": queryable,
                "opaque": opaque
            })

    if not layers:
        cols = ["name", "title", "abstract", "queryable", "opaque"]
        return pd.DataFrame(columns=cols) if as_dataframe else []

    if as_dataframe:
        df = pd.DataFrame(layers)
        df = df[df["name"] != ""].drop_duplicates(subset="name")
        df = df.sort_values("name").reset_index(drop=True)
        return df
    return layers

def list_all_swisstopo_layers(limit: int | None = 20):
    """Convenience printer (WMS only now)."""
    wms_df = list_swisstopo_wms_layers()
    print(f"WMS layers total: {len(wms_df)}")
    if limit:
        print("First WMS layers:", ", ".join(wms_df.name.head(limit)))
    print("Use list_swisstopo_wms_layers() for full DataFrame.")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _load_vector(data: Union[str, Path, gpd.GeoDataFrame], target_crs=SWISS_CRS) -> gpd.GeoDataFrame:
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
    else:
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(f"Vector file not found: {data}")
        gdf = gpd.read_file(data)
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS defined.")
    if target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def _points_to_gdf(points: Sequence[tuple], crs: str) -> gpd.GeoDataFrame:
    geoms = [Point(p[0], p[1]) for p in points]
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=crs)

# ------------------------------------------------------------------
# WMS download
# ------------------------------------------------------------------
def download_swisstopo_wms(
    layer: str,
    bbox: tuple[float, float, float, float],
    *,
    pixel_size: float = 200.0,
    crs_epsg: int = 2056,
    img_format: str = "image/png",
    transparent: bool = True,
    timeout: int = 60,
    retry: int = 2,
    sleep_s: float = 1.0
):
    minx, miny, maxx, maxy = map(float, bbox)
    width = max(1, int(round((maxx - minx) / pixel_size)))
    height = max(1, int(round((maxy - miny) / pixel_size)))

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": f"EPSG:{crs_epsg}",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "FORMAT": img_format,
        "WIDTH": width,
        "HEIGHT": height,
        "DPI": 96
    }
    if img_format == "image/png":
        params["TRANSPARENT"] = "TRUE" if transparent else "FALSE"

    headers = {"User-Agent": "asses_swiss_gl_therm_regimes/0.1"}
    for attempt in range(retry + 1):
        try:
            r = requests.get(SWISSTOPO_WMS_URL, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            img_bytes = r.content
            break
        except Exception as e:
            if attempt < retry:
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"WMS failed after {retry+1} attempts: {e}") from e

    im = Image.open(io.BytesIO(img_bytes))
    arr = np.array(im)
    meta = {
        "layer": layer,
        "width": width,
        "height": height,
        "crs": f"EPSG:{crs_epsg}",
        "pixel_size": pixel_size,
        "bbox": bbox
    }
    return arr, (minx, maxx, miny, maxy), meta

    SWISSBOUNDARIES3D_BASE = "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d"

def download_swissboundaries3d(
    *,
    date: str | None = None,
    out_dir: str | Path = "data/swissboundaries3d",
    force: bool = False,
    verbose: bool = True
) -> dict:
    """
    Download & extract swissBOUNDARIES3D (latest or specific date) and
    return dict with keys: 'canton', 'country', 'raw_dir'.

    date: 'YYYYMMDD' (dataset release) or None -> auto latest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # Determine latest date
    if date is None:
        if verbose: print("[swissboundaries3d] Detecting latest release date...")
        idx = session.get(SWISSBOUNDARIES3D_BASE, timeout=60)
        idx.raise_for_status()
        dates = re.findall(r'href="(20\d{6})/', idx.text)
        if not dates:
            raise RuntimeError("Could not parse available dates.")
        date = sorted(set(dates))[-1]
        if verbose: print(f"[swissboundaries3d] Latest date: {date}")

    # Zip filename pattern (LV95 / EPSG:2056)
    zip_name = f"ch.swisstopo.swissboundaries3d_{date}_2056_5728.shp.zip"
    url = f"{SWISSBOUNDARIES3D_BASE}/{date}/{zip_name}"
    local_zip = out_dir / zip_name

    if local_zip.exists() and not force:
        if verbose: print(f"[swissboundaries3d] Using cached {local_zip.name}")
        data = local_zip.read_bytes()
    else:
        if verbose: print(f"[swissboundaries3d] Downloading {url}")
        r = session.get(url, timeout=120)
        r.raise_for_status()
        data = r.content
        local_zip.write_bytes(data)
        if verbose: print(f"[swissboundaries3d] Saved {local_zip}")

    extract_dir = out_dir / f"{date}"
    if extract_dir.exists() and not force:
        if verbose: print(f"[swissboundaries3d] Using existing extracted dir {extract_dir}")
    else:
        if verbose: print(f"[swissboundaries3d] Extracting to {extract_dir}")
        if extract_dir.exists():
            for p in extract_dir.iterdir():
                p.unlink()
        else:
            extract_dir.mkdir()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(extract_dir)

    # Heuristics to locate canton & country layers
    # Canton polygons typically contain 'KANTONSGEBIET' or 'KT'; country polygon 'LANDESGEBIET' or 'LAND'.
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise RuntimeError("No shapefiles found after extraction.")

    canton_file = None
    country_file = None
    for shp in shp_files:
        name_u = shp.stem.upper()
        if canton_file is None and ("KANTON" in name_u or re.search(r"\bKT\b", name_u)):
            canton_file = shp
        if country_file is None and ("LAND" in name_u and "GEMEINDE" not in name_u and "KANTON" not in name_u):
            country_file = shp

    if canton_file is None:
        raise RuntimeError("Could not find canton shapefile.")
    if country_file is None:
        # fallback: dissolve cantons for country
        if verbose: print("[swissboundaries3d] Country shapefile not found; will dissolve canton layer.")
    if verbose:
        print(f"[swissboundaries3d] Canton file:  {canton_file.name}")
        if country_file:
            print(f"[swissboundaries3d] Country file: {country_file.name}")

    canton_gdf = gpd.read_file(canton_file).to_crs(SWISS_CRS)
    if country_file:
        country_gdf = gpd.read_file(country_file).to_crs(SWISS_CRS)
    else:
        country_gdf = gpd.GeoDataFrame(geometry=[canton_gdf.unary_union], crs=canton_gdf.crs)

    return {
        "date": date,
        "canton": canton_gdf,
        "country": country_gdf,
        "raw_dir": extract_dir
    }