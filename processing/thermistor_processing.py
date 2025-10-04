import pandas as pd
import numpy as np
import re

# for interpolation
from scipy.ndimage import gaussian_filter1d # for smoothing
from scipy.interpolate import Rbf
from pykrige.ok import OrdinaryKriging

from pathlib import Path

# plotting
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import os
from matplotlib import dates as mdates

os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/')
from calibration.thermistor_calibration import *
import calibration.thermistor_chains_icebath_references

"""
    This script is used to process and plot thermistor data.

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

class ThermistorDataPlotter:
    """
        Class to plot thermistor data.
    """
    def __init__(self, file_path, measurement_depth=None, delimiter=','):
        if isinstance(file_path, list):
            self.file_paths = file_path
            self.file_path = file_path[0]
        else:
            self.file_paths = [file_path]
            self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth

    def format_plot(self, title, legend_loc='upper right'):
        ax = plt.gca()
        fig = plt.gcf()
        # Get figure size in inches
        fig_width, fig_height = fig.get_size_inches()
        # Use the average of width and height to scale
        scale = (fig_width + fig_height) / 2

        # Set base sizes
        base_fontsize = 22
        base_linewidth = 4

        # Scale font and line sizes
        self.fontsize = int(base_fontsize * scale / 11)  # 12 is a typical reference width
        self.linewidth = base_linewidth * scale / 11

        # Set font and line sizes
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.size'] = self.fontsize
        plt.rcParams['axes.titlesize'] = self.fontsize
        plt.rcParams['axes.labelsize'] = self.fontsize
        plt.rcParams['xtick.labelsize'] = self.fontsize
        plt.rcParams['ytick.labelsize'] = self.fontsize
        plt.rcParams['legend.fontsize'] = self.fontsize
        plt.rcParams['lines.linewidth'] = self.linewidth

        # Enforce axis label font size
        ax.set_xlabel(ax.get_xlabel(), fontsize=self.fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.fontsize)
        ax.set_title(title if title else '', fontsize=self.fontsize)
        # Enforce tick label font size
        ax.tick_params(axis='both', labelsize=self.fontsize)
        # Enforce line width for all lines
        for line in ax.get_lines():
            line.set_linewidth(self.linewidth)
        # Legend settings
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) == 1:
            pass  # Do not show legend if only one item
        elif len(labels) > 4:
            ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                    loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=self.fontsize)
        else:
            ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                    loc=legend_loc, ncol=1, fontsize=self.fontsize)
        # Grid and layout
        plt.xticks(rotation=45, fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.grid()
        plt.tight_layout()

    def plot_full_geoprecision_chain(self, start_time, end_time, offsets, savepath, title=None, depth_file=None):
        import cmcrameri.cm as cmc
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data_with_offsets(start_time, end_time, offsets)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # Load depths if provided (use the same robust parser as elsewhere)
        depths = {}
        if depth_file is not None:
            try:
                depths = read_thermistor_depths(depth_file)  # returns dict like {'#1': 5.3, ...}
            except Exception as e:
                print(f"Warning: failed to read depths from {depth_file}: {e}")
                depths = {}

        plt.figure(figsize=(12, 8),dpi=250)
        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']
        plot_columns = [col for col in data.columns if col not in exclude_cols]
        n_cols = len(plot_columns)
        colors = cmc.batlow(np.linspace(1, 0, n_cols))

        for i, column in enumerate(plot_columns):
            depth_val = depths.get(column, None)
            # Label with depth if available
            label = f"{depth_val:.1f} m" if isinstance(depth_val, (int, float)) and depth_val is not None else ""
            plt.plot(data['TIME'], pd.to_numeric(data[column], errors='coerce'), label=label, color=colors[i])

        # Use the format_plot method for consistent styling
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.axhline(y=0, color='k', linestyle='--')
        self.format_plot(title, legend_loc='lower left')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.savefig(savepath)

    def plot_temperature_profile(self, snapshot_time, offsets, depth_file, savepath, title=None):
        """
        Plot a single chain's temperature profile as the DAILY MEAN for snapshot_time.
        Depths are loaded via read_thermistor_depths for consistency.
        """
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        day_df = thermistor.get_chain_data_with_offsets(offsets=offsets, snapshot_day=snapshot_time)
        if day_df is None or day_df.empty:
            raise ValueError("No chain data available for the requested day.")

        # Parse depths consistently
        try:
            depths_dict = read_thermistor_depths(depth_file)  # e.g., {'#1': 5.3, ...}
        except Exception as e:
            raise ValueError(f"Failed to read depths from {depth_file}: {e}")

        # Compute daily mean per thermistor using the depth keys
        exclude_cols = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}
        depths_list, mean_temps = [], []
        for sensor, depth in depths_dict.items():
            if depth is None:
                continue
            if sensor in day_df.columns and sensor not in exclude_cols:
                series = pd.to_numeric(day_df[sensor], errors='coerce')
                m = series.mean(skipna=True)
                if np.isfinite(m):
                    depths_list.append(float(depth))
                    mean_temps.append(float(m))

        if len(depths_list) == 0:
            raise ValueError("No matching thermistor columns found in data for the provided depth file.")

        # Sort by depth
        depths_arr, temps_arr = zip(*sorted(zip(depths_list, mean_temps)))

        plt.figure(figsize=(2.5, 4), dpi=250)
        plt.plot(temps_arr, depths_arr, 'o-', color='k', label='Daily mean')
        plt.gca().invert_yaxis()
        plt.xlabel('Ice Temperature [°C]')
        plt.ylabel('Depth [m]')

        # Title day formatting
        day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
        if pd.isna(day):
            day = pd.to_datetime(str(snapshot_time), format='%Y%m%d', errors='coerce')
        day_str = day.strftime('%Y-%m-%d') if not pd.isna(day) else str(snapshot_time)
        plot_title = f"{title if title else 'Temperature Profile'} — {day_str}"
        plt.title(plot_title)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.tight_layout()
        self.format_plot(plot_title)
        plt.savefig(savepath)

    def plot_multiple_temperature_profiles(self, snapshot_time, offsets_list, depth_files, labels=None, savepath=None, title=None, ntc_data_list=None):
        """
        Plot temperature profiles for multiple GeoPrecision chains and optional TT/NTC boreholes.

        Depth loading logic:
        - For chains: read_thermistor_depths(depth_files[:n_profiles])
        - For NTC:   read_thermistor_depths(depth_files[n_profiles:n_profiles+n_ntc])
                     matching ntc_data_list order.

        ntc_data_list: list of 1-row DataFrames per borehole with columns:
            ['Measurement','TIME','Black Probe Temperature','White Probe Temperature'].
        """
        # Parse provided snapshot_time to a day (EU dates supported)
        day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
        if pd.isna(day):
            day = pd.to_datetime(str(snapshot_time), format='%Y%m%d', errors='coerce')
        if pd.isna(day):
            raise ValueError(f"Could not parse snapshot_time: {snapshot_time}")
        day = day.normalize()

        plt.figure(figsize=(3, 4), dpi=250)
        n_profiles = len(self.file_paths)
        n_ntc = len(ntc_data_list) if ntc_data_list is not None else 0
        total_series = n_profiles + n_ntc

        # Colors for all series (one color per borehole profile line)
        all_colors = cmc.batlow(np.linspace(0, 1, max(total_series, 1)))

        # Default labels if not provided
        if labels is None:
            labels = [f'Chain {i+1}' for i in range(n_profiles)]
            labels += [f'NTC {j+1}' for j in range(n_ntc)]

        # Safety on depth_files length
        if len(depth_files) < total_series:
            print(f"Warning: depth_files has {len(depth_files)} entries, but {total_series} needed (chains + NTC).")

        # Plot chain profiles as daily means
        exclude_cols = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}
        for i, (fp, offs, dfile, label) in enumerate(zip(self.file_paths, offsets_list, depth_files[:n_profiles], labels[:n_profiles])):
            # Read depths consistently
            try:
                depths_dict = read_thermistor_depths(dfile)  # {'#1': 5.3, ...}
            except Exception as e:
                print(f"Warning: failed to read depths for chain {fp}: {e}")
                continue

            # Read chain data for the snapshot day and apply offsets
            thermistor = ThermistorData(fp, self.delimiter, self.measurement_depth)
            day_df = thermistor.get_chain_data_with_offsets(offsets=offs, snapshot_day=snapshot_time)
            if day_df.empty:
                continue

            # Compute daily mean per thermistor using depth keys
            depths, temps = [], []
            for sensor, depth in depths_dict.items():
                if depth is None:
                    continue
                if sensor in day_df.columns and sensor not in exclude_cols:
                    series = pd.to_numeric(day_df[sensor], errors='coerce')
                    mean_temp = series.mean(skipna=True)
                    if np.isfinite(mean_temp):
                        depths.append(float(depth))
                        temps.append(float(mean_temp))

            if depths and temps:
                depths, temps = zip(*sorted(zip(depths, temps)))
                plt.plot(temps, depths, 'o-', label=label, color=all_colors[i], linewidth=3)

        # Plot TT/NTC borehole data (already averaged; use given temps and depths from depth files)
        if n_ntc > 0:
            ntc_depth_files = depth_files[n_profiles:n_profiles + n_ntc]
            for j, (ntc_df, dfile, label) in enumerate(zip(ntc_data_list, ntc_depth_files, labels[n_profiles:])):
                color_idx = n_profiles + j

                # Depths for NTC: read from depth file (expects 'white probe' and 'black probe')
                try:
                    depths_dict = read_thermistor_depths(dfile)
                except Exception as e:
                    print(f"Warning: failed to read depths for NTC ({label}): {e}")
                    continue

                # Case-insensitive lookup
                dd_lower = {str(k).lower(): v for k, v in depths_dict.items()}
                depth_white = dd_lower.get('white probe', None)
                depth_black = dd_lower.get('black probe', None)

                if ntc_df is None or ntc_df.empty:
                    continue

                # Extract temps (already averaged or single-row snapshot)
                try:
                    t_white = float(ntc_df['White Probe Temperature'].iloc[0])
                    t_black = float(ntc_df['Black Probe Temperature'].iloc[0])
                except Exception as e:
                    print(f"Warning: missing NTC temperature columns for {label}: {e}")
                    continue

                temps_ntc, depths_ntc = [], []
                if depth_white is not None and np.isfinite(t_white):
                    temps_ntc.append(t_white); depths_ntc.append(float(depth_white))
                if depth_black is not None and np.isfinite(t_black):
                    temps_ntc.append(t_black); depths_ntc.append(float(depth_black))

                if len(temps_ntc) >= 1:
                    # Sort by depth to draw a clean line between probes
                    if len(temps_ntc) > 1:
                        depths_ntc, temps_ntc = zip(*sorted(zip(depths_ntc, temps_ntc)))
                    plt.plot(temps_ntc, depths_ntc, 'o-', label=label, color=all_colors[color_idx], alpha=0.9, linewidth=2)

        plt.gca().invert_yaxis()
        plt.xlabel('Ice Temperature [°C]')
        plt.ylabel('Depth [m]')
        plot_title = f"{title if title else 'Temperature Profiles'} — {day.strftime('%Y-%m-%d')}"
        plt.title(plot_title)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.tight_layout()
        self.format_plot(plot_title)
        if labels and len(labels) > 0:
            plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc='best')
        if savepath:
            plt.savefig(savepath)

    def plot_geoprecision_thermistor(self, start_time, end_time, depths, savepath, title=None):
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data(start_time, end_time)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # adjust figure size based on legend size
        plt.figure(dpi=300)  # adjust the values as per your requirement and set dpi for resolution

        # plot the data for each depth
        for depth in depths:
            thermistor_column = f'{depth} m'
            plt.plot(data['TIME'], data[thermistor_column], label=thermistor_column)

        # format the plot
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title)
        plt.legend(title='Depth [m]', fontsize='small')
        plt.xticks(rotation=45)  # Rotate x ticks 45 degrees
        plt.grid()
        plt.tight_layout()

        # save the plot
        plt.savefig(savepath)
    
    def plot_single_ntc_borehole(self, savepath, title=None, depth_white_probe=None, depth_black_probe=None, lower_y_limit=-1):
        ntc_thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        ntc_thermistor_data = ntc_thermistor.get_ntc_data()

        # adjust figure size based on legend size
        plt.figure(dpi=300)

        # plot the data
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['White Probe Temperature'], label=str(depth_white_probe) + ' m')
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['Black Probe Temperature'], label=str(depth_black_probe) + ' m')

        # format the plot
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title)
        plt.legend()
        # plt.xticks(rotation=45)  # Rotate x ticks 45 degrees
        plt.grid()
        plt.axhline(y=0, color='k', linestyle='--')  # Add a dashed line at 0°C
        plt.ylim(lower_y_limit, 0.2)

        # modify x-axis tick format to show date and time without seconds
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.tight_layout()

        # save the plot
        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')

    def plot_multiple_ntc_boreholes(self, savepath, depths, borehole_labels, title=None, lower_y_limit=-1, legend_loc='lower right', calibrate=False, zero_deg_offsets=None):
        if self.file_paths[0] is not None:
            ntc_thermistor1 = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
            ntc_thermistor_data1 = ntc_thermistor1.get_ntc_data()
            deployment_date = ntc_thermistor_data1['TIME'].min() # set the deployment date to the first date of the first timeseries
        else:
            ntc_thermistor_data1 = pd.DataFrame()

        if len(self.file_paths) > 1 and self.file_paths[1] is not None:
            ntc_thermistor2 = ThermistorData(self.file_paths[1], self.delimiter, self.measurement_depth)
            ntc_thermistor_data2 = ntc_thermistor2.get_ntc_data()
            deployment_date = ntc_thermistor_data2['TIME'].min() # set the deployment date to the first date of the second timeseries
        else:
            ntc_thermistor_data2 = pd.DataFrame()

        # apply 0-degree offsets if calibration mode is activated
        if calibrate:
            if self.file_paths[0] is not None:
                (black_probe_offset, white_probe_offset) = zero_deg_offsets[0]
                ntc_thermistor_data1['Black Probe Temperature'] = ntc_thermistor_data1['Black Probe Temperature'] - black_probe_offset
                ntc_thermistor_data1['White Probe Temperature'] = ntc_thermistor_data1['White Probe Temperature'] - white_probe_offset
            if self.file_paths[1] is not None:
                (black_probe_offset, white_probe_offset) = zero_deg_offsets[1]
                ntc_thermistor_data2['Black Probe Temperature'] = ntc_thermistor_data2['Black Probe Temperature'] - black_probe_offset
                ntc_thermistor_data2['White Probe Temperature'] = ntc_thermistor_data2['White Probe Temperature'] - white_probe_offset

        # Create figure and axis with the specified ratio
        fig_ratio = 136.446 / 115.986
        fig_width = 10 # Set the width of the figure
        fig_height = fig_width / fig_ratio

        fig = plt.figure(figsize=(12, 7), dpi=300)

        # Define complementary color palettes for each borehole
        colors_borehole1 = plt.cm.Reds(np.linspace(0.3, 1, len(depths)))
        colors_borehole2 = plt.cm.Blues(np.linspace(0.3, 1, len(depths)))

        if self.file_paths[0] is None:
            # Plot for borehole 2 only
            plt.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['White Probe Temperature'], 
             label=f'{borehole_labels[1]} - {depths[2]} m', color=colors_borehole2[0], linewidth=4)
            plt.fill_between(ntc_thermistor_data2['TIME'], 
                     ntc_thermistor_data2['White Probe Temperature'] - 0.2, 
                     ntc_thermistor_data2['White Probe Temperature'] + 0.2, 
                     color=colors_borehole2[1], alpha=0.1)
            plt.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['Black Probe Temperature'], 
             label=f'{borehole_labels[1]} - {depths[3]} m', color=colors_borehole2[1], linewidth=4)
            plt.fill_between(ntc_thermistor_data2['TIME'], 
                     ntc_thermistor_data2['Black Probe Temperature'] - 0.2, 
                     ntc_thermistor_data2['Black Probe Temperature'] + 0.2, 
                     color=colors_borehole2[1], alpha=0.1)
        else:
            # Plot for borehole 1
            plt.plot(ntc_thermistor_data1['TIME'], ntc_thermistor_data1['White Probe Temperature'], 
             label=f'{borehole_labels[0]} - {depths[0]} m', color=colors_borehole1[0], linewidth=4)
            plt.fill_between(ntc_thermistor_data1['TIME'], 
                     ntc_thermistor_data1['White Probe Temperature'] - 0.2, 
                     ntc_thermistor_data1['White Probe Temperature'] + 0.2, 
                     color=colors_borehole1[1], alpha=0.1)
            plt.plot(ntc_thermistor_data1['TIME'], ntc_thermistor_data1['Black Probe Temperature'], 
             label=f'{borehole_labels[0]} - {depths[1]} m', color=colors_borehole1[1], linewidth=4)
            plt.fill_between(ntc_thermistor_data1['TIME'], 
                     ntc_thermistor_data1['Black Probe Temperature'] - 0.2, 
                     ntc_thermistor_data1['Black Probe Temperature'] + 0.2, 
                     color=colors_borehole1[1], alpha=0.1)
            
            # Plot for borehole 2 with dotted lines
            plt.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['White Probe Temperature'], 
             label=f'{borehole_labels[1]} - {depths[2]} m', color=colors_borehole2[0], linewidth=4)
            plt.fill_between(ntc_thermistor_data2['TIME'], 
                     ntc_thermistor_data2['White Probe Temperature'] - 0.2, 
                     ntc_thermistor_data2['White Probe Temperature'] + 0.2, 
                     color=colors_borehole2[1], alpha=0.1)
            plt.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['Black Probe Temperature'], 
             label=f'{borehole_labels[1]} - {depths[3]} m', color=colors_borehole2[1], linewidth=4)
            plt.fill_between(ntc_thermistor_data2['TIME'], 
                     ntc_thermistor_data2['Black Probe Temperature'] - 0.2, 
                     ntc_thermistor_data2['Black Probe Temperature'] + 0.2, 
                     color=colors_borehole2[1], alpha=0.1)

        # format the plot specific to the data
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.ylim(lower_y_limit, 0.4)
        plt.axhline(y=0, color='k', linestyle='--')  # Add a dashed line at 0°C
        plt.axvline(deployment_date, color='gray', linestyle='solid', linewidth=4)
        plt.text(deployment_date, plt.ylim()[0], 'Deployment', color='gray', fontsize=22, verticalalignment='top', horizontalalignment='right', rotation=45)
        ax = plt.gca()
        # Option A: auto ticks, target 5–8 labels
        locator = mdates.AutoDateLocator(minticks=8, maxticks=11)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        # Option B (fixed): uncomment to show every 2 months
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        # format the plot
        self.format_plot(title, legend_loc)

        # save the plot
        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')

    def plot_ntc_icebath_calibration(self, thermistor_chain_data, savepath, y_limits=(-1,1), title=None, legend_loc='lower right'):
        ntc_thermistor = ThermistorData(self.file_path, self.delimiter)
        ntc_thermistor_data = ntc_thermistor.get_ntc_data()

        # Use the updated method to get offsets and stable indices
        black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white = ntc_thermistor.calculate_ntc_offsets()

        # Get the time of the stable period
        stable_period_black_start = ntc_thermistor_data['TIME'].iloc[stable_indices_black[0]]
        stable_period_black_end = ntc_thermistor_data['TIME'].iloc[stable_indices_black[-1]]
        stable_period_white_start = ntc_thermistor_data['TIME'].iloc[stable_indices_white[0]]
        stable_period_white_end = ntc_thermistor_data['TIME'].iloc[stable_indices_white[-1]]

        plt.figure(figsize=(12, 8.5), dpi=300)

        # plot the ntc data
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['Black Probe Temperature'], linewidth=4, alpha=0.6, label='Black Probe (Gemini)')
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['White Probe Temperature'], linewidth=4, alpha=0.6, label='White Probe (Gemini)')

        # plot the reference geoprecision thermistor chain data
        plt.plot(thermistor_chain_data['TIME'], thermistor_chain_data['#20'], linewidth=4, alpha=0.6, label='Thermistor chain (Geoprecision)', color='black')

        # Mark the stable period for Black Probe
        plt.axvline(stable_period_black_start, color='tab:blue', linestyle='-', linewidth=2)
        plt.axvline(stable_period_black_end, color='tab:blue', linestyle='-', linewidth=2)
        plt.text(stable_period_black_start, y_limits[0], 'Stable Start', color='tab:blue', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)
        plt.text(stable_period_black_end, y_limits[0], 'Stable End', color='tab:blue', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)

        # Mark the stable period for White Probe
        plt.axvline(stable_period_white_start, color='tab:orange', linestyle='-', linewidth=2)
        plt.axvline(stable_period_white_end, color='tab:orange', linestyle='-', linewidth=2)
        plt.text(stable_period_white_start, y_limits[0], 'Stable Start', color='tab:orange', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)
        plt.text(stable_period_white_end, y_limits[0], 'Stable End', color='tab:orange', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)

        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title, fontsize=22)
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(y_limits)
        plt.xlim(ntc_thermistor_data['TIME'].min(), ntc_thermistor_data['TIME'].max())

        self.format_plot(title)

        # Draw a horizontal line at the 0°C offset for the Black and White probe
        plt.axhline(y=black_probe_offset, color='blue', linestyle='dotted', linewidth=2, label=f'Black Probe 0°C Offset: {black_probe_offset:.2f}°C')
        plt.axhline(y=white_probe_offset, color='orange', linestyle='dotted', linewidth=2, label=f'White Probe 0°C Offset: {white_probe_offset:.2f}°C')

        plt.legend(fontsize=22, frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc='upper center', bbox_to_anchor=(0.45, -0.4), ncol=2)
        plt.tight_layout()

        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')

        # Return offsets and stable indices
        return (black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white)


# Other helpful processing functions can be added here

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