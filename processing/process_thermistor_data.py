import pandas as pd
import numpy as np
import re

# plotting
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import os
from matplotlib import dates as mdates

os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/')
from calibration.thermistor_calibration import *
import calibration.thermistor_chains_0deg_references

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
    
    def get_chain_data(self, start_time, end_time):
        data_lines = []
        columns = None
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Detect header line
                if line.startswith('NO') and 'TIME' in line:
                    columns = line.split(self.delimiter)
                    # Simplify columns here
                    columns = [col.split(':')[0] if col.startswith('#') else col for col in columns]
                    continue
                # Skip metadata and empty lines
                if not line or not line[0].isdigit():
                    continue
                # Only collect data if header is set
                if columns:
                    data_lines.append(line.split(self.delimiter))

        # If no data found, return empty DataFrame
        if not data_lines or not columns:
            return pd.DataFrame()

        df = pd.DataFrame(data_lines, columns=columns)
        df['TIME'] = pd.to_datetime(df['TIME'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter by time range
        start_time = pd.to_datetime(start_time, format='%d.%m.%Y %H:%M:%S')
        end_time = pd.to_datetime(end_time, format='%d.%m.%Y %H:%M:%S')
        df = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]

        return df

    def get_chain_data_with_offsets(self, start_time, end_time, offsets):
        """
        Returns chain data for the given time range with offsets applied.
        Uses apply_chain_offsets from thermistor_calibration.py.
        """
        df = self.get_chain_data(start_time, end_time)
        df = apply_chain_offsets(df, offsets)
        return df

    def get_ntc_data(self):
        # Read the CSV file with the correct encoding and skip the first 5 rows
        df = pd.read_csv(self.file_path, sep=self.delimiter, header=None, skiprows=5, 
                    names=['Measurement', 'TIME', 'Black Probe Temperature', 'White Probe Temperature'], 
                    encoding='latin1')
        
        # Convert the TIME column to datetime format
        df['TIME'] = pd.to_datetime(df['TIME'])
        
        # Remove the special character (�C) from the temperature columns and convert to float
        df['Black Probe Temperature'] = df['Black Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        df['White Probe Temperature'] = df['White Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        
        # Replace -42.004 with NaN values
        df['Black Probe Temperature'] = df['Black Probe Temperature'].replace(-42.004, np.nan)
        df['White Probe Temperature'] = df['White Probe Temperature'].replace(-42.004, np.nan)

        return df

    def get_multiple_ntc_data(self):
        """
        Reads NTC data for all file paths in self.file_paths.
        Returns a list of DataFrames, one per borehole.
        """
        ntc_data_list = []
        for fp in self.file_paths:
            df = pd.read_csv(fp, sep=self.delimiter, header=None, skiprows=5, 
                                names=['Measurement', 'TIME', 'Black Probe Temperature', 'White Probe Temperature'], 
                                encoding='latin1')
            df['TIME'] = pd.to_datetime(df['TIME'])
            df['Black Probe Temperature'] = df['Black Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x))).astype(float)
            df['White Probe Temperature'] = df['White Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x))).astype(float)
            df['Black Probe Temperature'] = df['Black Probe Temperature'].replace(-42.004, np.nan)
            df['White Probe Temperature'] = df['White Probe Temperature'].replace(-42.004, np.nan)
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
        self.fontsize = int(base_fontsize * scale / 12)  # 12 is a typical reference width
        self.linewidth = base_linewidth * scale / 12

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

        # Load depths if provided
        depths = {}
        if depth_file is not None:
            df_depths = pd.read_csv(depth_file, sep=';', header=0)
            df_depths = df_depths[df_depths.iloc[:,0].astype(str).str.startswith('#')]
            last_col = df_depths.columns[5]
            for _, row in df_depths.iterrows():
                thermistor_name = row.iloc[0]
                try:
                    depth = float(str(row[last_col]).replace(',', '.'))
                except:
                    depth = None
                depths[thermistor_name] = depth

        plt.figure(figsize=(12, 8),dpi=250)
        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']
        plot_columns = [col for col in data.columns if col not in exclude_cols]
        n_cols = len(plot_columns)
        colors = cmc.batlow(np.linspace(1, 0, n_cols))

        for i, column in enumerate(plot_columns):
            depth = depths.get(column, None)
            label = f"{depth:.1f} m" if depth is not None else ""
            plt.plot(data['TIME'], data[column], label=label, color=colors[i])

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

        snapshot_time accepts:
          - '20250808'
          - '2025-08-08'
          - '08.08.2025'
          - '08.08.2025 13:00:00' (time ignored)
        """
        # Parse day robustly (EU dates supported)
        day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
        if pd.isna(day):
            day = pd.to_datetime(str(snapshot_time), format='%Y%m%d', errors='coerce')
        if pd.isna(day):
            raise ValueError(f"Could not parse snapshot_time: {snapshot_time}")
        day = day.normalize()

        # Build start/end strings for the whole day
        start_str = day.strftime('%d.%m.%Y 00:00:00')
        end_str = day.strftime('%d.%m.%Y 23:59:59')

        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data_with_offsets(start_str, end_str, offsets)

        # Load depths
        df_depths = pd.read_csv(depth_file, sep=';', header=0)
        df_depths = df_depths[df_depths.iloc[:,0].astype(str).str.startswith('#')]
        depth_col_candidates = [col for col in df_depths.columns if 'depth' in col.lower() and '[m]' in col.lower()]
        depth_col = depth_col_candidates[0] if depth_col_candidates else df_depths.columns[-1]

        depths = []
        mean_temps = []
        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']

        if not data.empty:
            for _, row in df_depths.iterrows():
                thermistor_name = row.iloc[0]
                try:
                    depth = float(str(row[depth_col]).replace(',', '.'))
                except Exception:
                    continue
                if thermistor_name in data.columns and thermistor_name not in exclude_cols:
                    series = pd.to_numeric(data[thermistor_name], errors='coerce')
                    m = series.mean(skipna=True)
                    if not np.isnan(m):
                        depths.append(depth)
                        mean_temps.append(m)

        if depths and mean_temps:
            depths, mean_temps = zip(*sorted(zip(depths, mean_temps)))

        plt.figure(figsize=(2.5,4), dpi=250)
        plt.plot(mean_temps, depths, 'o-', color='k', label='Daily mean')
        plt.gca().invert_yaxis()
        plt.xlabel('Ice Temperature [°C]')
        plt.ylabel('Depth [m]')
        plot_title = f"{title if title else 'Temperature Profile'} — {day.strftime('%Y-%m-%d')}"
        plt.title(plot_title)
        plt.tight_layout()
        self.format_plot(plot_title)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.savefig(savepath)

    def plot_multiple_temperature_profiles(self, snapshot_time, offsets_list, depth_files, labels=None, savepath=None, title=None, ntc_data_list=None):
        """
        Plot temperature profiles for all chains and optional TT/NTC boreholes as DAILY MEANS.

        snapshot_time: day selector for averaging. Accepts formats like:
            - '20250808'
            - '2025-08-08'
            - '08.08.2025'
            - '08.08.2025 13:00:00' (time is ignored; date is used)
        For TT/NTC boreholes, pass combined 1-row DataFrames (already averaged/corrected)
        via ntc_data_list.
        """
        # Parse provided snapshot_time to a day (EU dates supported)
        day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
        if pd.isna(day):
            # Try compact format like 20250808
            day = pd.to_datetime(str(snapshot_time), format='%Y%m%d', errors='coerce')
        if pd.isna(day):
            raise ValueError(f"Could not parse snapshot_time: {snapshot_time}")
        day = day.normalize()

        # Build start/end strings for chain reader (expects '%d.%m.%Y %H:%M:%S')
        start_str = day.strftime('%d.%m.%Y 00:00:00')
        end_str = day.strftime('%d.%m.%Y 23:59:59')

        plt.figure(figsize=(2.5, 4), dpi=250)
        n_profiles = len(self.file_paths)
        n_ntc = len(ntc_data_list) if ntc_data_list is not None else 0
        total_profiles = n_profiles + n_ntc

        # Colors for all series (chains + optional TT/NTC)
        all_colors = cmc.batlow(np.linspace(0, 1, total_profiles if total_profiles > 0 else 1))
        if labels is None:
            labels = [f'Profile {i+1}' for i in range(n_profiles)]

        # Plot chain profiles as daily means
        for i, (fp, offs, dfile, label) in enumerate(zip(self.file_paths, offsets_list, depth_files[:n_profiles], labels)):
            # Read depths
            df_depths = pd.read_csv(dfile, sep=';', header=0)
            df_depths = df_depths[df_depths.iloc[:, 0].astype(str).str.match(r'#\d+')]
            depth_col_candidates = [col for col in df_depths.columns if 'depth' in col.lower() and '[m]' in col.lower()]
            depth_col = depth_col_candidates[0] if depth_col_candidates else df_depths.columns[-1]

            # Read chain data for the whole day and apply offsets
            thermistor = ThermistorData(fp, self.delimiter, self.measurement_depth)
            day_df = thermistor.get_chain_data_with_offsets(start_str, end_str, offs)
            if day_df.empty:
                continue

            exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']
            # Compute daily mean per thermistor
            depths, temps = [], []
            for _, row in df_depths.iterrows():
                thermistor_name = row.iloc[0]
                try:
                    depth = float(str(row[depth_col]).replace(',', '.'))
                except Exception:
                    continue
                if thermistor_name in day_df.columns and thermistor_name not in exclude_cols:
                    series = pd.to_numeric(day_df[thermistor_name], errors='coerce')
                    mean_temp = series.mean(skipna=True)
                    if not np.isnan(mean_temp):
                        depths.append(depth)
                        temps.append(mean_temp)

            if depths and temps:
                depths, temps = zip(*sorted(zip(depths, temps)))
                plt.plot(temps, depths, 'o-', label=label, color=all_colors[i], linewidth=3)

        # Plot TT/NTC borehole data (already averaged and corrected)
        if ntc_data_list is not None and len(ntc_data_list) > 0:
            for j, (ntc_df, label) in enumerate(zip(ntc_data_list, labels[n_profiles:])):
                color_idx = n_profiles + j
                tynitag_temps = [
                    ntc_df['Black Probe Temperature (corrected)'].iloc[0],
                    ntc_df['White Probe Temperature (corrected)'].iloc[0]
                ]
                tynitag_depths = [
                    ntc_df['depth black probe [m]'].iloc[0],
                    ntc_df['depth white probe [m]'].iloc[0]
                ]
                plt.plot(tynitag_temps, tynitag_depths, 'o-', label=label, color=all_colors[color_idx], alpha=0.85)

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
        fig.patch.set_facecolor('#f3f2f2ff')  # Set the background color of the entire figure

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
        plt.axvline(deployment_date, color='gray', linestyle='solid', linewidth=4)
        plt.text(deployment_date, plt.ylim()[0], 'Deployment', color='gray', fontsize=22, verticalalignment='top', horizontalalignment='right', rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Set x-ticks to be at the start of each month

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

def read_tynitag_depth_file(depth_file):
    """
    Reads a Tynitag depth file and returns a DataFrame with columns:
    ['date', 'depth white probe [m]', 'depth black probe [m]']
    """
    df = pd.read_csv(depth_file, sep=';', header=6)
    # Replace comma with dot and convert to float for depth columns
    df['depth black probe [m]'] = df['depth black probe [m]'].astype(str).str.replace(',', '.').astype(float)
    df['depth white probe [m]'] = df['depth white probe [m]'].astype(str).str.replace(',', '.').astype(float)
    # Only keep relevant columns
    return df[['date', 'depth black probe [m]', 'depth white probe [m]']]

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