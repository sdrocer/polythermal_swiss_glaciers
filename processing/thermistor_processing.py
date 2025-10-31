import pandas as pd
import numpy as np
import re
from pathlib import Path

# Keep only processing/science imports here
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # if needed elsewhere
from scipy.interpolate import Rbf
from pykrige.ok import OrdinaryKriging

import os
os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/polythermal_swiss_glaciers/')
from calibration.thermistor_calibration import *
import calibration.thermistor_chains_icebath_references

"""
    Processing utilities for thermistor data.

    Code written by: Janosch Beer
"""

class ThermistorData:
    """
        Class to read data from a thermistor as a pandas dataframe.

        Can read data from:
            - geoprecision thermistor chains (FlexGate 2.0 output)
            - NTC thermistors
    """
    def __init__(self, file_path, delimiter, measurement_depth=None):
        if isinstance(file_path, list):
            self.file_paths = file_path
            self.file_path = file_path[0]
        else:
            self.file_paths = [file_path]
            self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth

    def calculate_ntc_offsets(self):
        """
        Calculates and returns the 0-degree offsets and stable indices for Black and White probes.
        Returns (black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white)
        """
        df = self.get_ntc_data()
        black_probe_offset, stable_indices_black = calculate_zero_degree_offset(df['Black Probe Temperature'])
        white_probe_offset, stable_indices_white = calculate_zero_degree_offset(df['White Probe Temperature'])
        return (black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white)
    
    def get_chain_data(self, start_time=None, end_time=None, snapshot_day=None):
        """
        Returns chain data for a given time range or a single snapshot day.
        If snapshot_day is provided, start_time and end_time are ignored.
        snapshot_day accepts formats like '20250808', '2025-08-08', '08.08.2025'.
        """
        data_lines = []
        columns = None
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Detect header line
                if line.startswith('NO') and 'TIME' in line:
                    columns = line.split(self.delimiter)
                    columns = [col.split(':')[0] if col.startswith('#') else col for col in columns]
                    continue
                if not line or not line[0].isdigit():
                    continue
                if columns:
                    data_lines.append(line.split(self.delimiter))

        if not data_lines or not columns:
            return pd.DataFrame()

        df = pd.DataFrame(data_lines, columns=columns)
        df['TIME'] = pd.to_datetime(df['TIME'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # If snapshot_day is provided, filter for that day only
        if snapshot_day is not None:
            # Try parsing with known formats to avoid ambiguity and warnings
            day = None
            try:
                day = pd.to_datetime(snapshot_day, format='%d.%m.%Y', errors='coerce')
            except Exception:
                pass
            if pd.isna(day):
                try:
                    day = pd.to_datetime(snapshot_day, format='%Y-%m-%d', errors='coerce')
                except Exception:
                    pass
            if pd.isna(day):
                try:
                    day = pd.to_datetime(str(snapshot_day), format='%Y%m%d', errors='coerce')
                except Exception:
                    pass
            if pd.isna(day):
                raise ValueError(f"Could not parse snapshot_day: {snapshot_day}")
            day = day.normalize()
            start_time = day
            end_time = day + pd.Timedelta(days=1)
        else:
            # Parse start/end if not None
            if start_time is not None:
                start_time = pd.to_datetime(start_time, format='%d.%m.%Y %H:%M:%S')
            if end_time is not None:
                end_time = pd.to_datetime(end_time, format='%d.%m.%Y %H:%M:%S')

        # Filter by time range
        if start_time is not None and end_time is not None:
            df = df[(df['TIME'] >= start_time) & (df['TIME'] < end_time)]

        return df

    def get_chain_data_with_offsets(self, start_time=None, end_time=None, offsets=None, snapshot_day=None, return_daily_average=False):
        """
        Returns chain data for the given time range or a single snapshot day with offsets applied.
        Uses apply_chain_offsets from thermistor_calibration.py.
        If snapshot_day is provided, start_time and end_time are ignored.
        
        Parameters:
        -----------
        start_time : str, optional
            Start time for data filtering
        end_time : str, optional  
            End time for data filtering
        offsets : dict or pd.Series, optional
            Offsets to apply to temperature readings
        snapshot_day : str, optional
            If provided, filters for that day. Accepts formats like '20250819', '2025-08-19', '19.08.2025'.
        return_daily_average : bool, default False
            If True and snapshot_day is provided, returns daily averages for each thermistor.
            If False, returns all data points for the snapshot day.
        
        Returns:
        --------
        pd.DataFrame with chain data. If return_daily_average=True and snapshot_day is provided,
        returns single row with daily averages for each thermistor.
        """
        df = self.get_chain_data(start_time=start_time, end_time=end_time, snapshot_day=snapshot_day)
        if offsets is not None and not df.empty:
            df = apply_chain_offsets(df, offsets)
        
        # If snapshot_day is provided and daily average is requested, compute averages
        if snapshot_day is not None and return_daily_average and not df.empty:
            # Parse snapshot_day to get the date for the TIME column
            day = None
            try:
                day = pd.to_datetime(snapshot_day, format='%d.%m.%Y', errors='coerce')
            except Exception:
                pass
            if pd.isna(day):
                try:
                    day = pd.to_datetime(snapshot_day, format='%Y-%m-%d', errors='coerce')
                except Exception:
                    pass
            if pd.isna(day):
                try:
                    day = pd.to_datetime(str(snapshot_day), format='%Y%m%d', errors='coerce')
                except Exception:
                    pass
            if pd.isna(day):
                raise ValueError(f"Could not parse snapshot_day: {snapshot_day}")
            
            day = day.normalize()
            
            # Calculate daily averages for all numeric columns
            exclude_cols = ['NO', 'TIME']
            numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            
            avg_data = {}
            avg_data['NO'] = [1]  # Single measurement number
            avg_data['TIME'] = [day + pd.Timedelta(hours=12)]  # Set to noon of that day
            
            for col in numeric_cols:
                avg_data[col] = [df[col].mean()]
            
            return pd.DataFrame(avg_data)
    
        return df

    def get_ntc_data(self):
            """
            Read NTC CSVs in both legacy (5-row metadata, no header) and new (with header) formats.
            Ensures:
            - Columns: Measurement, TIME, Black Probe Temperature, White Probe Temperature
            - TIME parsed to datetime
            - Temperatures as float, sentinel -42.004 -> NaN
            """
            # Detect header presence by peeking first line
            try:
                with open(self.file_path, 'r', encoding='latin1') as f:
                    first_line = f.readline().strip()
            except Exception:
                first_line = ""

            has_header = first_line.startswith("Measurement,") or first_line.startswith("Measurement;")

            if has_header:
                df = pd.read_csv(
                    self.file_path, sep=self.delimiter, header=0, encoding='latin1'
                )
            else:
                df = pd.read_csv(
                    self.file_path, sep=self.delimiter, header=None, skiprows=5,
                    names=['Measurement', 'TIME', 'Black Probe Temperature', 'White Probe Temperature'],
                    encoding='latin1'
                )

            # Normalize TIME
            df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')

            # Normalize temperature columns to float
            for col in ['Black Probe Temperature', 'White Probe Temperature']:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x))).astype(float)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            # Replace error sentinel
            df['Black Probe Temperature'] = df['Black Probe Temperature'].replace(-42.004, np.nan)
            df['White Probe Temperature'] = df['White Probe Temperature'].replace(-42.004, np.nan)

            return df

    def get_ntc_data_with_offsets(self, logger_id, offsets_df, snapshot_day=None, aggregate=None):
        """
        Get NTC data with offset correction and optional aggregation.
        Auto-detects both legacy and new CSV formats (delegates to get_ntc_data()).
        """
        import pandas as pd
        import numpy as np

        # Read raw data using the robust reader
        df = self.get_ntc_data()

        # Find offsets (robust match as string)
        off = offsets_df.copy()
        off['Logger'] = off['Logger'].astype(str)
        row = off[off['Logger'] == str(logger_id)]
        if row.empty:
            print(f"Warning: Logger ID {logger_id} not found in offsets DataFrame; using zero offsets.")
            black_offset = 0.0
            white_offset = 0.0
        else:
            black_offset = float(row['Black Probe Offset'].iloc[0])
            white_offset = float(row['White Probe Offset'].iloc[0])

        # Apply offsets (subtract)
        df['Black Probe Temperature'] = df['Black Probe Temperature'] - black_offset
        df['White Probe Temperature'] = df['White Probe Temperature'] - white_offset

        # Helper to parse a day
        def _parse_day(val):
            if val is None:
                return None
            for fmt in ('%d.%m.%Y', '%Y-%m-%d', '%Y%m%d'):
                d = pd.to_datetime(val, format=fmt, errors='coerce')
                if not pd.isna(d):
                    return d.normalize()
            d = pd.to_datetime(val, errors='coerce')
            return None if pd.isna(d) else d.normalize()

        # No aggregation requested
        if aggregate is None:
            if snapshot_day is None:
                return df
            day = _parse_day(snapshot_day)
            if day is None:
                raise ValueError(f"Could not parse snapshot_day: {snapshot_day}")
            start, end = day, day + pd.Timedelta(days=1)
            day_df = df[(df['TIME'] >= start) & (df['TIME'] < end)]
            if day_df.empty:
                raise ValueError(f"No data found for day {snapshot_day}")
            return day_df

        # Aggregation
        agg = str(aggregate).lower()
        if agg in {'all', 'overall'}:
            subset = df
            label_time = (df['TIME'].min() + (df['TIME'].max() - df['TIME'].min()) / 2) if not df.empty else pd.NaT
        else:
            day = _parse_day(snapshot_day)
            if day is None:
                raise ValueError(f"snapshot_day is required for aggregate='{aggregate}' and could not be parsed.")
            if agg in {'daily', 'day'}:
                start, end = day, day + pd.Timedelta(days=1)
                subset = df[(df['TIME'] >= start) & (df['TIME'] < end)]
                label_time = start + pd.Timedelta(hours=12)
            elif agg in {'monthly', 'month'}:
                subset = df[(df['TIME'].dt.year == day.year) & (df['TIME'].dt.month == day.month)]
                label_time = pd.Timestamp(day.year, day.month, 15)
            elif agg in {'annual', 'year'}:
                subset = df[df['TIME'].dt.year == day.year]
                label_time = pd.Timestamp(day.year, 7, 1)
            else:
                raise ValueError("aggregate must be one of: daily, monthly, annual, all")

        if subset.empty:
            raise ValueError(f"No data found for selection (aggregate='{aggregate}', snapshot_day={snapshot_day}).")

        avg_black = subset['Black Probe Temperature'].mean()
        avg_white = subset['White Probe Temperature'].mean()

        return pd.DataFrame({
            'Measurement': [1],
            'TIME': [label_time],
            'Black Probe Temperature': [avg_black],
            'White Probe Temperature': [avg_white]
        })

    def get_multiple_ntc_data(self):
        """
        Reads NTC data for all file paths in self.file_paths using get_ntc_data() for
        robust handling of both legacy and new formats.
        Returns a list of DataFrames, one per borehole.
        """
        ntc_data_list = []
        for fp in self.file_paths:
            t = ThermistorData(fp, self.delimiter, self.measurement_depth)
            df = t.get_ntc_data()
            ntc_data_list.append(df)
        return ntc_data_list

# Other helpful processing functions can be added here

def load_chain_offsets_csv(csv_path, index_col="chain"):
    """
    Load GeoPrecision chain offsets from the CSV written earlier.
    Returns dict: { 'A55201': {'#1': offset, '#2': offset, ...}, ... }
    NaNs are dropped.
    """
    df = pd.read_csv(csv_path)
    if index_col in df.columns:
        df = df.set_index(index_col)
    # build nested dict and drop NaNs
    out = {}
    for chain, row in df.to_dict(orient="index").items():
        out[chain] = {k: float(v) for k, v in row.items() if pd.notna(v)}
    return out

def combine_tynitag_data(depths_df, snapshot_df, offsets_df=None):
    """
    Combine tynitag depth, snapshot, and offset dataframes into one dataframe
    with corrected temperatures and matching depths.
    If offsets_df is provided, subtract offsets. Otherwise, use raw temperatures.
    """
    import pandas as pd

    # Get measurement time and find closest date in depths_df
    measurement_time = pd.to_datetime(snapshot_df['TIME'].iloc[0])
    depths_df['date_dt'] = pd.to_datetime(depths_df['date'], format='%d.%m.%y', errors='coerce')
    closest_idx = (depths_df['date_dt'] - measurement_time).abs().idxmin()
    depth_row = depths_df.loc[closest_idx]

    # Determine offsets
    if offsets_df is not None:
        black_offset = float(offsets_df['Black Probe Offset'].iloc[0])
        white_offset = float(offsets_df['White Probe Offset'].iloc[0])
    else:
        black_offset = 0.0
        white_offset = 0.0

    black_temp_corr = snapshot_df['Black Probe Temperature'].iloc[0] - black_offset
    white_temp_corr = snapshot_df['White Probe Temperature'].iloc[0] - white_offset

    # Combine into one DataFrame
    combined_df = pd.DataFrame({
        'date': [depth_row['date']],
        'TIME': [snapshot_df['TIME'].iloc[0]],
        'Black Probe Temperature (corrected)': [black_temp_corr],
        'White Probe Temperature (corrected)': [white_temp_corr],
        'depth black probe [m]': [depth_row['depth black probe [m]']],
        'depth white probe [m]': [depth_row['depth white probe [m]']]
    })

    # Clean up helper column
    depths_df.drop(columns='date_dt', inplace=True)

    return combined_df

def ntc_daily_snapshot(df, day_str):
    """
    Return a 1-row snapshot with daily mean temps for the given day (e.g., '20250808').
    Columns: TIME, Black Probe Temperature, White Probe Temperature
    """
    df = df.copy()
    df['TIME'] = pd.to_datetime(df['TIME'])

    # Parse day string robustly
    day = pd.to_datetime(day_str, format='%Y%m%d', errors='coerce')
    if pd.isna(day):
        day = pd.to_datetime(day_str)  # fallback

    start = day.normalize()
    end = start + pd.Timedelta(days=1)

    day_df = df[(df['TIME'] >= start) & (df['TIME'] < end)]
    if day_df.empty:
        raise ValueError(f"No data for day {day_str}")

    avg_black = day_df['Black Probe Temperature'].mean()
    avg_white = day_df['White Probe Temperature'].mean()

    # Create a single-row snapshot (TIME set to noon for display)
    return pd.DataFrame({
        'TIME': [start + pd.Timedelta(hours=12)],
        'Black Probe Temperature': [avg_black],
        'White Probe Temperature': [avg_white]
    })

def read_thermistor_depths(depth_file, when=None):
    """
    Read a thermistor depth file and return a dictionary of current measurement depths (m).

    Supported formats:
    1) GeoPrecision chain list (rows '#1', '#2', ... with a 'depth' column).
       Returns: {'#1': 5.3, '#2': 6.8, ...}
    2) New TinyTag format (columns per probe):
         date,cummulative surface melt (m ice),borehole depth [m],
         depth black probe [m],depth white probe [m],...
       Returns: {'white probe': 7.81, 'black probe': 12.81}
       The row selected is the one closest to 'when' (if provided) or the last valid row.

    Parameters
    ----------
    depth_file : str
    when : str | pandas.Timestamp | None
        Optional date to select the closest entry. Accepts 'YYYYMMDD', 'YYYY-MM-DD', 'DD.MM.YYYY', etc.
    """
    import pandas as pd
    import numpy as np

    def _to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(',', '.')
        # keep digits, sign, decimal point
        m = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        return float(m[0]) if m else np.nan

    # Load as-is
    df = pd.read_csv(depth_file, sep=',', header=0)

    # Normalize column names
    colmap = {str(c).strip().lower(): c for c in df.columns}

    def _col(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        # loose match
        for k, v in colmap.items():
            if any(n in k for n in names):
                return v
        return None

    # Detect new TinyTag format (columns per probe)
    c_white = _col('depth white probe', 'white probe depth')
    c_black = _col('depth black probe', 'black probe depth')
    c_date  = _col('date')
    if c_white or c_black:
        df2 = df.copy()

        # Parse date if present
        if c_date:
            # make '.' also work
            dates = pd.to_datetime(df2[c_date].astype(str).str.replace('.', '/', regex=False),
                                   dayfirst=True, errors='coerce')
            df2['_date'] = dates
        else:
            df2['_date'] = pd.NaT

        # Choose row: closest to 'when' if provided, else last row with at least one depth
        if when is not None and c_date:
            when_dt = pd.to_datetime(str(when), dayfirst=True, errors='coerce')
            if pd.isna(when_dt):
                # try compact yyyymmdd
                when_dt = pd.to_datetime(str(when), format='%Y%m%d', errors='coerce')
            if not pd.isna(when_dt) and df2['_date'].notna().any():
                idx = (df2['_date'] - when_dt).abs().idxmin()
                row = df2.loc[idx]
            else:
                row = df2.iloc[-1]
        else:
            # last row with any valid depth, otherwise last row
            mask_valid = pd.Series(False, index=df2.index)
            for c in [c_white, c_black]:
                if c:
                    mask_valid |= df2[c].apply(_to_float).notna()
            if mask_valid.any():
                row = df2.loc[mask_valid].iloc[-1]
            else:
                row = df2.iloc[-1]

        depths = {}
        if c_white:
            val = _to_float(row[c_white])
            if np.isfinite(val):
                depths['white probe'] = float(val)
        if c_black:
            val = _to_float(row[c_black])
            if np.isfinite(val):
                depths['black probe'] = float(val)

        if not depths:
            raise ValueError(f"No valid probe depths found in {depth_file}")
        return depths

    # Fallback: legacy chain table with rows '#1', '#2', ... and a depth column
    # Identify thermistor rows
    first_col = df.columns[0]
    thermistor_mask = (
        df[first_col].astype(str).str.startswith('#') |
        df[first_col].astype(str).str.contains('probe', case=False, na=False)
    )

    # Find a numeric depth column by scanning from the right
    depth_col = None
    for col in reversed(df.columns):
        if col == first_col:
            continue
        series = df.loc[thermistor_mask, col].dropna()
        if series.empty:
            continue
        try:
            test = _to_float(series.iloc[0])
            if np.isfinite(test):
                depth_col = col
                break
        except Exception:
            continue
    if depth_col is None:
        raise ValueError("No data column found with actual depth values")

    depths = {}
    for _, row in df.loc[thermistor_mask].iterrows():
        key = row[first_col]
        val = _to_float(row[depth_col])
        depths[key] = float(val) if np.isfinite(val) else None

    return depths

def interpolate_temperature_weighted_rbf(
    profile_distances,
    z_surf,
    z_bed,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    n_elev=300,
    depth_weight=1.5,
    rbf_function='linear'
):
    """
    Interpolate englacial temperature using a weighted RBF in (distance, elevation) space.
    depth_weight: scales the importance of elevation vs. horizontal distance.
    Returns: grid_temp (n_elev, n_x), grid_elev (n_elev,)
    """
    import numpy as np
    from scipy.interpolate import Rbf

    # Clean borehole coordinates (replace commas with dots)
    borehole_coords_df = borehole_coords_df.copy()
    borehole_coords_df['x'] = borehole_coords_df['x'].astype(str).str.replace(',', '.').astype(float)
    borehole_coords_df['y'] = borehole_coords_df['y'].astype(str).str.replace(',', '.').astype(float)

    d = profile_distances
    z_s = z_surf
    z_b = z_bed

    # Collect borehole positions and sensor depths/temps for interpolation
    interp_points = []
    interp_temps = []
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = None
        if 'x' in borehole_coords_df and 'y' in borehole_coords_df:
            profile_xy = np.column_stack([borehole_coords_df['x'], borehole_coords_df['y']])
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            profile_dist = d[profile_dist_idx]
            surface_elev = z_s[profile_dist_idx]
        else:
            profile_dist = bh_x
            surface_elev = np.interp(profile_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    interp_points.append([profile_dist, therm_elev])
                    interp_temps.append(temps[probe])

    interp_points = np.array(interp_points)
    interp_temps = np.array(interp_temps)

    # Weighted coordinates: scale elevation
    interp_points_weighted = interp_points.copy()
    interp_points_weighted[:, 1] *= depth_weight

    # Create grid for heatmap
    grid_x = d
    grid_y = np.linspace(z_b.min(), z_s.max(), n_elev)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_xx_weighted = grid_xx
    grid_yy_weighted = grid_yy * depth_weight

    # RBF interpolation
    rbf = Rbf(interp_points_weighted[:, 0], interp_points_weighted[:, 1], interp_temps, function=rbf_function)
    grid_temp = rbf(grid_xx_weighted, grid_yy_weighted)

    # Mask grid_temp outside glacier body
    for i, x in enumerate(grid_x):
        bed = np.interp(x, d, z_b)
        surf = np.interp(x, d, z_s)
        for j, y in enumerate(grid_y):
            if not (bed < y < surf):
                grid_temp[j, i] = np.nan

    return grid_temp, grid_y

def interpolate_temperature_rbf(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    n_elev=200,
    rbf_function='linear'
):
    """
    Interpolate englacial temperature using RBF in (distance, elevation) space.
    Returns: grid_temp (n_elev, n_x), grid_x (distance), grid_y (elevation)
    """

    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

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
            profile_dist = bh_x
            surface_elev = np.interp(profile_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    interp_points.append([profile_dist, therm_elev])
                    interp_temps.append(temps[probe])

    interp_points = np.array(interp_points)
    interp_temps = np.array(interp_temps)
    grid_x = d
    grid_y = np.linspace(z_b.min(), z_s.max(), n_elev)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    rbf = Rbf(interp_points[:, 0], interp_points[:, 1], interp_temps, function=rbf_function)
    grid_temp = rbf(grid_xx, grid_yy)

    # Mask grid_temp outside glacier body
    for i, x in enumerate(grid_x):
        bed = np.interp(x, d, z_b)
        surf = np.interp(x, d, z_s)
        for j, y in enumerate(grid_y):
            if not (bed < y < surf):
                grid_temp[j, i] = np.nan

    return grid_temp, grid_x, grid_y

def interpolate_temperature_kriging(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    n_elev=200,
    variogram_model='linear'
):
    """
    Interpolate englacial temperature using Ordinary Kriging in (distance, elevation) space.
    Returns: grid_temp (n_elev, n_x), grid_x (distance), grid_y (elevation)
    """
    import numpy as np

    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

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
            profile_dist = bh_x
            surface_elev = np.interp(profile_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    interp_points.append([profile_dist, therm_elev])
                    interp_temps.append(temps[probe])

    interp_points = np.array(interp_points)
    interp_temps = np.array(interp_temps)
    grid_x = d
    grid_y = np.linspace(z_b.min(), z_s.max(), n_elev)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Kriging interpolation
    OK = OrdinaryKriging(
        interp_points[:, 0], interp_points[:, 1], interp_temps,
        variogram_model=variogram_model, verbose=False, enable_plotting=False
    )
    grid_temp, _ = OK.execute('grid', grid_x, grid_y)

    # Mask grid_temp outside glacier body
    for i, x in enumerate(grid_x):
        bed = np.interp(x, d, z_b)
        surf = np.interp(x, d, z_s)
        for j, y in enumerate(grid_y):
            if not (bed < y < surf):
                grid_temp[j, i] = np.nan

    return grid_temp, grid_x, grid_y

def interpolate_glacier_temperature_field_2d(
    profile_df,
    borehole_coords_df, 
    temp_data_dict,
    depth_dict,
    n_depth=200,
    n_elev=300,
    depth_weight=2.5,  # Weight depth more heavily than horizontal distance
    rbf_function='multiquadric'  # RBF function type
):
    """
    Interpolates glacier temperature field using 2D RBF interpolation
    that respects both horizontal and vertical temperature gradients.
    """
    import numpy as np
    from scipy.interpolate import Rbf

    # Get profile geometry
    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy() 
    z_b = profile_df['zbed'].to_numpy()
    
    # Collect all measurement points in 2D: (distance, elevation, temperature)
    points_2d = []
    temps_2d = []
    
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))

        # Find borehole position along profile
        if 'x' in profile_df and 'y' in profile_df:
            profile_xy = np.column_stack([profile_df['x'], profile_df['y']])
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            idx = np.argmin(dists)
            bh_distance = d[idx]
            surf_elev = z_s[idx]
        else:
            bh_distance = bh_x
            surf_elev = np.interp(bh_distance, d, z_s)
        
        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            
            for probe, depth in depths.items():
                if probe in temps and np.isfinite(temps[probe]):
                    elev = surf_elev - depth
                    # Store as (distance, weighted_elevation)
                    points_2d.append([bh_distance, elev * depth_weight])
                    temps_2d.append(temps[probe])
    
    if len(points_2d) < 3:
        raise ValueError("Need at least 3 temperature measurements for 2D interpolation")
    
    points_2d = np.array(points_2d)
    temps_2d = np.array(temps_2d)
    
    # Create regular elevation grid
    elev_min = float(np.nanmin(z_b)) - 1.0
    elev_max = float(np.nanmax(z_s)) + 1.0
    elev_grid = np.linspace(elev_min, elev_max, n_elev)
    
    # Create grid points for interpolation
    profile_x = d
    grid_xx, grid_yy = np.meshgrid(profile_x, elev_grid)
    
    # RBF interpolation
    rbf = Rbf(
        points_2d[:, 0], points_2d[:, 1], temps_2d, 
        function=rbf_function,  # 'linear', 'cubic', 'quintic', 'thin_plate', 'multiquadric', 'inverse'
        smooth=0.1  # Add some smoothing to avoid overfitting
    )
    
    # Interpolate on the grid (with weighted elevation)
    grid_temp_elev = rbf(grid_xx, grid_yy * depth_weight)
    
    # Mask outside glacier body
    for i, x in enumerate(profile_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = elev_grid > surf_at_x
        below = elev_grid < bed_at_x
        grid_temp_elev[above, i] = np.nan
        grid_temp_elev[below, i] = np.nan
    
    return grid_temp_elev, elev_grid, profile_x

def interpolate_borehole_profile_1d(depths, temps, depth_grid):
    """
    Interpolate a borehole temperature profile onto a regular depth grid,
    filling all gaps between thermistor positions.
    - depths: array-like, depths of sensors (in meters, positive downward)
    - temps: array-like, temperatures at those depths (same length as depths)
    - depth_grid: array-like, depths at which to interpolate (must cover the range of depths)
    
    Returns: interp_temps (np.ndarray), interpolated temperatures at depth_grid
    """
    import numpy as np

    depths = np.asarray(depths, dtype=float)
    temps = np.asarray(temps, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)

    # Sort by depth in case input is unordered
    order = np.argsort(depths)
    depths = depths[order]
    temps = temps[order]

    # Fill all gaps by linear interpolation, clamp outside to nearest measured value
    interp_temps = np.interp(depth_grid, depths, temps, left=temps[0], right=temps[-1])
    return interp_temps

def interpolate_between_boreholes_stratified(profile_x, bh_x, bh_temp_profiles, method='idw'):
    """
    Interpolate horizontally between several 1D-interpolated borehole temperature profiles
    at each depth (stratified interpolation).

    Parameters
    ----------
    profile_x : 1D array-like
        Distances along the profile where interpolation is desired (e.g., profile grid).
    bh_x : 1D array-like
        Distances along the profile of each borehole (same order as bh_temp_profiles).
    bh_temp_profiles : 2D array-like, shape (n_boreholes, n_depth)
        Each row is the interpolated temperature profile for a borehole (on the same depth grid).
    method : str, default 'idw'
        Interpolation method. Only 'idw' (inverse distance weighting) is implemented.

    Returns
    -------
    grid_temp : 2D np.ndarray, shape (n_depth, len(profile_x))
        Interpolated temperature at each (profile_x, depth).
    """
    import numpy as np

    bh_x = np.asarray(bh_x)
    bh_temp_profiles = np.asarray(bh_temp_profiles)
    n_bh, n_depth = bh_temp_profiles.shape
    n_x = len(profile_x)
    grid_temp = np.full((n_depth, n_x), np.nan)

    for j in range(n_depth):
        vals = bh_temp_profiles[:, j]
        for i, x in enumerate(profile_x):
            dists = np.abs(bh_x - x)
            dists[dists == 0] = 1e-6  # avoid division by zero
            weights = 1 / dists
            weights /= weights.sum()
            grid_temp[j, i] = np.sum(weights * vals)
    return grid_temp

def create_temp_depth_dicts(thermistor_data_dict, depth_data_dict):
    """
    Create temperature and depth dictionaries from thermistor data objects and depth data.
    
    Parameters:
    -----------
    thermistor_data_dict : dict
        Dictionary with borehole names as keys and thermistor data DataFrames as values.
        e.g., {'CH1G': CH1G_data, 'CH2G': CH2G_data, 'CH5TT': CH5TT_data}
        
    depth_data_dict : dict
        Dictionary with borehole names as keys and depth dictionaries as values.
        e.g., {'CH1G': CH1G_depths, 'CH2G': CH2G_depths, 'CH5TT': CH5TT_depths}
    
    Returns:
    --------
    tuple: (temp_data_dict, depth_dict)
        - temp_data_dict: Dictionary with borehole names as keys and pd.Series of temperatures as values
        - depth_dict: Dictionary with borehole names as keys and depth dictionaries as values
    """
    temp_data_dict = {}
    depth_dict = {}
    
    for borehole, data in thermistor_data_dict.items():
        if data is None or data.empty:
            print(f"Warning: No data for {borehole}, skipping...")
            continue
            
        # Get corresponding depth data
        depths = depth_data_dict.get(borehole, {})
        if not depths:
            print(f"Warning: No depth data for {borehole}, skipping...")
            continue
            
        # Check if it's GeoPrecision chain data (has columns starting with '#')
        chain_columns = [col for col in data.columns if col.startswith('#')]
        
        if chain_columns:
            # GeoPrecision chain data
            temps = {}
            for col in chain_columns:
                if col in depths and depths[col] is not None:
                    temps[col] = data[col].iloc[0]  # Get the single daily average value
            temp_data_dict[borehole] = pd.Series(temps)
            
        elif 'White Probe Temperature' in data.columns and 'Black Probe Temperature' in data.columns:
            # Tynitag NTC data
            white_temp = data['White Probe Temperature'].iloc[0]
            black_temp = data['Black Probe Temperature'].iloc[0]
            
            temp_data_dict[borehole] = pd.Series({
                'white probe': white_temp,
                'black probe': black_temp
            })
        else:
            print(f"Warning: Unknown data format for {borehole}, skipping...")
            continue
            
        # Store depth data (only for boreholes with valid temperature data)
        depth_dict[borehole] = depths
        
        # Print summary
        print(f"{borehole}: {len(temp_data_dict[borehole])} thermistors")
        for sensor, temp in temp_data_dict[borehole].items():
            depth = depths.get(sensor, 'N/A')
            print(f"  {sensor}: {temp:.3f}°C at {depth} m")
    
    return temp_data_dict, depth_dict

def splice_timeseries(dfs, time_col="TIME", out_path=None, file_stem=None):
    # Normalize TIME
    normed = []
    for i, df in enumerate(dfs):
        dfx = df.copy()
        dfx[time_col] = pd.to_datetime(dfx[time_col])
        dfx["__chunk__"] = i  # to prefer later chunks on duplicates
        normed.append(dfx)

    # Concatenate and resolve overlaps (keep last chunk on duplicate TIME)
    out = pd.concat(normed, ignore_index=True)
    out = out.sort_values([time_col, "__chunk__"])
    out = out.drop_duplicates(subset=[time_col], keep="last").drop(columns="__chunk__")

    # Re-number Measurement
    if "Measurement" in out.columns:
        out = out.sort_values(time_col).reset_index(drop=True)
        out["Measurement"] = range(1, len(out) + 1)

    # Optional continuity check (hourly expected)
    gaps = out[time_col].sort_values().diff().dropna()
    big_gaps = gaps[gaps > pd.Timedelta(hours=1)]
    if not big_gaps.empty:
        print(f"Warning: found {len(big_gaps)} gaps > 1 hour")

    # Save
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if file_stem is None:
            file_stem = "spliced"
        out.to_csv(out_path.with_name(f"{file_stem}.csv"), index=False)
    return out

def derive_cts_line(temp_grid, x_coords, z_grid, surface_elev, bed_elev,
                     gamma_K_per_MPa=0.0742, cts_tol=0.05, min_span_frac=0.15):
    """
    Compute CTS using pressure-adjusted melting point from thermistor_processing.melting_point_at_pressure.
    Assumes z_grid are elevations (m a.s.l.). Overburden thickness h = surface_elev - z.
    """
    nz, nx = temp_grid.shape
    cts_z, cts_T, cts_Tm = [], [], []
    for j in range(nx):
        zsurf = surface_elev[j]
        zbed = bed_elev[j]
        thick = zsurf - zbed
        if thick <= 0 or np.isnan(thick):
            cts_z.append(np.nan); cts_T.append(np.nan); cts_Tm.append(np.nan)
            continue

        # Overburden thickness above each grid level (positive downward)
        h_over = zsurf - z_grid  # array
        # Compute pressure melting point (function already accounts for physics)
        Tm = melting_point_at_pressure(h_over)  # returns deg C array

        Tcol = temp_grid[:, j]
        # Limit to inside ice
        mask = (z_grid <= zsurf) & (z_grid >= zbed)
        diff = np.where(mask, np.abs(Tcol - Tm), np.nan)

        if np.all(np.isnan(diff)):
            cts_z.append(np.nan); cts_T.append(np.nan); cts_Tm.append(np.nan)
            continue

        k = np.nanargmin(diff)
        if np.isnan(diff[k]) or diff[k] > cts_tol:
            cts_z.append(np.nan); cts_T.append(np.nan); cts_Tm.append(np.nan)
        else:
            cts_z.append(z_grid[k]); cts_T.append(Tcol[k]); cts_Tm.append(Tm[k])

    cts_z = np.asarray(cts_z)
    valid = ~np.isnan(cts_z)
    if valid.sum() < max(3, int(min_span_frac * nx)):
        return {}
    return {"x": np.asarray(x_coords), "z": cts_z, "T": np.asarray(cts_T), "Tm": np.asarray(cts_Tm)}

# Pressure-dependent melting point calculations

# simple implementation (SI units)
rho_ice = 917.0        # kg m^-3
g = 9.81               # m s^-2
gamma = 0.0742         # K per MPa  (use 0.098 for air-saturated water if appropriate)

def pressure_from_overburden(h, rho=rho_ice, g=g):
    # h: ice thickness above point [m]
    return rho * g * h   # Pa

def melting_point_at_pressure(h, T0 = 0.0, gamma_K_per_MPa = gamma):
    p_Pa = pressure_from_overburden(h)
    p_MPa = p_Pa / 1e6
    Tm = T0 - gamma_K_per_MPa * p_MPa
    return Tm  # degrees C (if T0 in degC)

def interpolate_segment(
    prof_seg, borehole_coords_df, temp_data_dict, depth_dict,
    n_depth, n_elev
):
    """
    Wrapper around interpolate_glacier_temperature_field_2d.
    Returns: T_seg (n_elev x n_x), elevs_seg (n_elev,), xnodes_seg (n_x,)
    """
    T_seg, elevs_seg, xnodes_seg = interpolate_glacier_temperature_field_2d(
        prof_seg,
        borehole_coords_df,
        temp_data_dict,
        depth_dict,
        n_depth=n_depth,
        n_elev=n_elev
    )
    return T_seg, elevs_seg, xnodes_seg

def compute_cts_columnwise(
    xnodes_seg, elevs_seg, T_masked, zs_on_x, zb_on_x,
    adjust_cts_for_pressure, cts_tol
):
    """
    Column-wise zero-crossing CTS (single line).
    Returns (x_valid, z_valid) or (None, None) if insufficient points.
    """
    XX, YY = np.meshgrid(xnodes_seg, elevs_seg)
    if adjust_cts_for_pressure:
        h_over = (zs_on_x[None, :] - YY)
        Tm = melting_point_at_pressure(h_over)
        diff_field = T_masked - Tm
    else:
        diff_field = T_masked
    diff_field = np.ma.masked_where(T_masked.mask, diff_field)
    z_cts = np.full(xnodes_seg.shape, np.nan)
    for j in range(diff_field.shape[1]):
        col = diff_field[:, j]
        if col.mask.all():
            continue
        vals = col.filled(np.nan)
        good = np.isfinite(vals)
        if good.sum() < 2:
            continue
        zc = elevs_seg[good]
        vc = vals[good]
        signs = np.sign(vc)
        signs[signs == 0] = 1e-6
        sc_idx = np.where(signs[:-1] * signs[1:] < 0)[0]
        chosen_z = np.nan
        if sc_idx.size:
            bed_z = zb_on_x[j]
            best_gap = np.inf
            for k in sc_idx:
                v1, v2 = vc[k], vc[k+1]
                z1, z2 = zc[k], zc[k+1]
                z_cross = z1 + (0 - v1) * (z2 - z1)/(v2 - v1) if (v2 - v1) != 0 else (z1+z2)/2
                gap = z_cross - bed_z
                if gap >= 0 and gap < best_gap:
                    best_gap = gap
                    chosen_z = z_cross
        else:
            near_zero = np.where(np.abs(vc) <= cts_tol)[0]
            if near_zero.size:
                bed_z = zb_on_x[j]
                gaps = zc[near_zero] - bed_z
                gaps[gaps < 0] = np.inf
                idx = np.argmin(gaps)
                if np.isfinite(gaps[idx]):
                    chosen_z = zc[near_zero[idx]]
        if np.isfinite(chosen_z):
            z_cts[j] = chosen_z
    valid = np.isfinite(z_cts)
    if valid.sum() < 3:
        return None, None
    return xnodes_seg[valid], z_cts[valid]

# ---------- TinyTag thermal metrics (final) ----------

def prepare_tynitag_timeseries(
    thermistor: "ThermistorData",
    logger_id,
    offsets_df,
    start_time,
    end_time,
    *,
    resample="1D",
    smooth_window="7D",
    depth_file=None,
    when_for_depths=None
):
    """
    Prepare TinyTag NTC timeseries:
    - offset correction (per logger_id)
    - time slicing [start_time, end_time)
    - daily resampling and rolling smoothing
    - rename to match depth keys ('white probe', 'black probe')
    Returns: (df_resampled_smoothed, depths_dict)
    """
    # Read and offset-correct
    df = thermistor.get_ntc_data_with_offsets(logger_id=logger_id, offsets_df=offsets_df)
    if df is None or df.empty:
        return pd.DataFrame(), {}

    # Time handling
    df = df.copy()
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df.dropna(subset=["TIME"])
    t0 = pd.to_datetime(start_time, dayfirst=True, errors="coerce")
    t1 = pd.to_datetime(end_time, dayfirst=True, errors="coerce")
    if pd.notna(t0):
        df = df[df["TIME"] >= t0]
    if pd.notna(t1):
        df = df[df["TIME"] < t1]

    if df.empty:
        return pd.DataFrame(), {}

    # Keep only probe temps and rename
    keep = {}
    if "White Probe Temperature" in df.columns:
        keep["White Probe Temperature"] = "white probe"
    if "Black Probe Temperature" in df.columns:
        keep["Black Probe Temperature"] = "black probe"
    if not keep:
        return pd.DataFrame(), {}

    df = df[["TIME"] + list(keep.keys())].rename(columns=keep).set_index("TIME").sort_index()

    # Resample and smooth
    df_res = df.resample(resample).mean()
    if smooth_window:
        # time-based rolling window (e.g., "7D")
        df_res = df_res.rolling(smooth_window, min_periods=1, center=True).mean()

    # Depths near the analysis period end (or provided 'when')
    when = when_for_depths if when_for_depths is not None else t1
    depths = read_thermistor_depths(depth_file, when=when) if depth_file else {}

    return df_res, depths


# --- Helpers for ZAA computation ---

def amplitudes_by_depth(df_res: pd.DataFrame, depths: dict):
    """
    Compute annual amplitude (0.5 * (max - min)) for each probe present in df_res and depths.
    Returns depths_arr, amps_arr aligned by increasing depth.
    """
    pairs = []
    for probe, depth in depths.items():
        if probe in df_res.columns and np.isfinite(depth):
            series = df_res[probe].dropna()
            if not series.empty:
                amp = 0.5 * (series.max() - series.min())
                pairs.append((float(depth), float(amp)))

    if not pairs:
        return np.array([]), np.array([])

    pairs.sort(key=lambda x: x[0])  # sort by depth
    d = np.array([p[0] for p in pairs], dtype=float)
    a = np.array([p[1] for p in pairs], dtype=float)
    return d, a

def zaa_depth(depths: np.ndarray, amplitudes: np.ndarray, threshold: float = 0.2, extrapolate_if_monotonic: bool = False):
    """
    Return depth where amplitude falls below 'threshold' by linear interpolation.
    Optionally extrapolate if amplitudes decrease monotonically but do not cross threshold.
    """
    if depths is None or amplitudes is None or len(depths) == 0 or len(amplitudes) == 0:
        return np.nan

    d = np.asarray(depths, dtype=float)
    a = np.asarray(amplitudes, dtype=float)

    # Ensure sorted by depth
    order = np.argsort(d)
    d = d[order]
    a = a[order]

    # If all amplitudes are NaN or all above threshold -> no ZAA
    if np.all(~np.isfinite(a)) or (np.nanmin(a) > threshold):
        # Optional extrapolation if monotonic decreasing
        if extrapolate_if_monotonic and np.isfinite(a).sum() >= 2:
            good = np.isfinite(a)
            d2, a2 = d[good], a[good]
            if np.all(np.diff(a2) <= 0) and a2[-1] > threshold and a2[0] >= threshold:
                slope = (a2[-1] - a2[-2]) / (d2[-1] - d2[-2])
                if slope < 0:
                    z_ex = d2[-1] + (threshold - a2[-1]) / slope
                    return float(z_ex) if np.isfinite(z_ex) else np.nan
        return np.nan

    # Crossing from >= threshold to < threshold
    for i in range(len(d) - 1):
        a1, a2 = a[i], a[i + 1]
        if not np.isfinite(a1) or not np.isfinite(a2):
            continue
        crosses = (a1 >= threshold and a2 < threshold) or (a1 > threshold and a2 <= threshold)
        if crosses:
            if a2 != a1:
                frac = (threshold - a1) / (a2 - a1)
                return d[i] + frac * (d[i + 1] - d[i])
            else:
                return d[i]  # flat segment

    # If already below threshold at shallowest point
    if np.isfinite(a[0]) and a[0] < threshold:
        return d[0]

    # Optional extrapolation if monotonic decreasing but never crossed
    if extrapolate_if_monotonic and np.isfinite(a).sum() >= 2:
        good = np.isfinite(a)
        d2, a2 = d[good], a[good]
        if np.all(np.diff(a2) <= 0) and a2[-1] > threshold and a2[0] >= threshold:
            slope = (a2[-1] - a2[-2]) / (d2[-1] - d2[-2])
            if slope < 0:
                z_ex = d2[-1] + (threshold - a2[-1]) / slope
                return float(z_ex) if np.isfinite(z_ex) else np.nan

    return np.nan

def compute_tynitag_zaa(
    thermistor: "ThermistorData",
    *,
    logger_id,
    offsets_df,
    depth_file,
    start_time,
    end_time,
    resample="1D",
    smooth_window="7D",
    zaa_threshold=0.1,
    zaa_extrapolate=False
):
    """
    Compute ZAA depth for one TinyTag borehole.
    Returns dict with zaa_depth (float), depths (np.ndarray), amplitudes (np.ndarray).
    """
    df_res, depths = prepare_tynitag_timeseries(
        thermistor,
        logger_id,
        offsets_df,
        start_time,
        end_time,
        resample=resample,
        smooth_window=smooth_window,
        depth_file=depth_file,
        when_for_depths=end_time,
    )

    if df_res is None or df_res.empty or not depths:
        return {"zaa_depth": np.nan, "depths": np.array([]), "amplitudes": np.array([])}

    d, a = amplitudes_by_depth(df_res, depths)
    zaa = zaa_depth(d, a, threshold=zaa_threshold, extrapolate_if_monotonic=zaa_extrapolate)
    return {"zaa_depth": zaa, "depths": d, "amplitudes": a}

def compute_tynitag_zaa_batch(
    entries,
    offsets_df,
    *,
    start_time,
    end_time,
    resample="1D",
    smooth_window="7D",
    zaa_threshold=0.2,
    zaa_extrapolate=False
):
    """
    entries: list of dicts with keys:
      name, glacier, thermistor (ThermistorData), logger_id, depth_file
    Returns list of per-borehole dicts suitable for summarization.
    """
    out = []
    for e in entries:
        name = e.get("name")
        glacier = e.get("glacier")
        therm = e.get("thermistor")
        logger_id = e.get("logger_id")
        depth_file = e.get("depth_file")

        try:
            res = compute_tynitag_zaa(
                therm,
                logger_id=logger_id,
                offsets_df=offsets_df,
                depth_file=depth_file,
                start_time=start_time,
                end_time=end_time,
                resample=resample,
                smooth_window=smooth_window,
                zaa_threshold=zaa_threshold,
                zaa_extrapolate=zaa_extrapolate,
            )
            out.append({
                "name": name,
                "glacier": glacier,
                "logger_id": logger_id,
                "zaa_depth": res["zaa_depth"],
                "depths": res["depths"],
                "amplitudes": res["amplitudes"],
            })
        except Exception as ex:
            print(f"ZAA failed for {name} ({glacier}): {ex}")
            out.append({
                "name": name,
                "glacier": glacier,
                "logger_id": logger_id,
                "zaa_depth": np.nan,
                "depths": np.array([]),
                "amplitudes": np.array([]),
            })
    return out

def summarize_zaa_by_glacier(metrics):
    """
    metrics: list of dicts from compute_tynitag_zaa_batch
    Returns DataFrame with glacier, zaa_mean, zaa_min, zaa_max, zaa_range, n
    """
    if isinstance(metrics, list):
        df = pd.DataFrame(metrics)
    else:
        df = metrics.copy()

    df_valid = df[["glacier", "zaa_depth"]].copy()
    grouped = df_valid.groupby("glacier", dropna=False)

    summary = grouped.agg(
        zaa_mean=("zaa_depth", "mean"),
        zaa_min=("zaa_depth", "min"),
        zaa_max=("zaa_depth", "max"),
        n=("zaa_depth", "count"),
    ).reset_index()

    summary["zaa_range"] = summary["zaa_max"] - summary["zaa_min"]
    return summary.sort_values("glacier").reset_index(drop=True)