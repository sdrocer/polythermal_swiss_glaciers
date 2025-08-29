import pandas as pd
import numpy as np
import re

# plotting
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import os
from matplotlib import dates as mdates

os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/')
from calibration.thermistor_calibration import calculate_zero_degree_offset
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
    def __init__(self, file_path, delimiter, measurement_depth = None):
        self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth
    
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

    def get_ntc_data(self):
        # Read the CSV file with the correct encoding and skip the first 5 rows
        self.data = pd.read_csv(self.file_path, sep=self.delimiter, header=None, skiprows=5, 
                    names=['Measurement', 'TIME', 'Black Probe Temperature', 'White Probe Temperature'], 
                    encoding='latin1')
        
        # Convert the TIME column to datetime format
        self.data['TIME'] = pd.to_datetime(self.data['TIME'])
        
        # Remove the special character (�C) from the temperature columns and convert to float
        self.data['Black Probe Temperature'] = self.data['Black Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        self.data['White Probe Temperature'] = self.data['White Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        
        # Replace -42.004 with NaN values
        self.data['Black Probe Temperature'] = self.data['Black Probe Temperature'].replace(-42.004, np.nan)
        self.data['White Probe Temperature'] = self.data['White Probe Temperature'].replace(-42.004, np.nan)

        return self.data

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

    def plot_full_geoprecision_chain(self, start_time, end_time, savepath, title=None, depth_file=None):
        import cmcrameri.cm as cmc
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data(start_time, end_time)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # Load depths if provided
        depths = {}
        if depth_file is not None:
            df_depths = pd.read_csv(depth_file, sep=';', skiprows=1)
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

    def plot_stabilized_temperature_profile(self, start_time, end_time, depth_file, savepath, title=None):
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data(start_time, end_time)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # Load depths
        df_depths = pd.read_csv(depth_file, sep=';', skiprows=1)
        df_depths = df_depths[df_depths.iloc[:,0].astype(str).str.startswith('#')]
        last_col = df_depths.columns[5]
        depths = []
        stabilized_temps = []

        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']
        for _, row in df_depths.iterrows():
            thermistor_name = row.iloc[0]
            try:
                depth = float(str(row[last_col]).replace(',', '.'))
            except:
                continue
            if thermistor_name in data.columns and thermistor_name not in exclude_cols:
                temp_series = data[thermistor_name]
                offset, stable_indices = calculate_zero_degree_offset(temp_series)
                # Convert stable_indices to positions if needed
                positions = [temp_series.index.get_loc(idx) for idx in stable_indices if idx in temp_series.index]
                if positions:
                    stabilized_mean = temp_series.iloc[positions].mean()
                    if not np.isnan(stabilized_mean):
                        depths.append(depth)
                        stabilized_temps.append(stabilized_mean)

        if depths and stabilized_temps:
            depths, stabilized_temps = zip(*sorted(zip(depths, stabilized_temps)))

        plt.figure(figsize=(3,4),dpi=250)
        plt.plot(stabilized_temps, depths, 'o-', color='k', label='Stabilized Temperature')
        plt.gca().invert_yaxis()
        plt.xlabel('Ice Temperature [°C]')
        plt.ylabel('Depth [m]')
        plt.title(title if title else 'Temperature Profile')
        plt.tight_layout()
        # Use the format_plot method for consistent styling
        self.format_plot(title)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.savefig(savepath)

    def plot_temperature_heatmap(self, start_time, end_time, depth_file, savepath, title=None, cts_threshold=0.05):
        """
        Plot a temperature profile heatmap (depth vs. time, color = temperature) and mark the CTS (cold-temperate transition surface).
        CTS is defined as the depth where temperature is closest to 0°C at each time step.
        """
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data(start_time, end_time)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # Load depths
        df_depths = pd.read_csv(depth_file, sep=';', skiprows=1)
        df_depths = df_depths[df_depths.iloc[:,0].astype(str).str.startswith('#')]
        last_col = df_depths.columns[5]
        depths = []
        temp_columns = []
        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']

        for _, row in df_depths.iterrows():
            thermistor_name = row.iloc[0]
            try:
                depth = float(str(row[last_col]).replace(',', '.'))
            except:
                continue
            if thermistor_name in data.columns and thermistor_name not in exclude_cols:
                depths.append(depth)
                temp_columns.append(thermistor_name)

        # Sort depths and corresponding columns
        depths, temp_columns = zip(*sorted(zip(depths, temp_columns)))
        depths = np.array(depths)
        temp_matrix = data[list(temp_columns)].to_numpy()  # shape: (n_times, n_depths)
        times = data['TIME'].to_numpy()

        # Interpolate temperature profile to a finer depth grid
        fine_depths = np.linspace(depths.min(), depths.max(), 200)
        temp_interp = np.empty((len(times), len(fine_depths)))
        for i, temp_row in enumerate(temp_matrix):
            temp_interp[i, :] = np.interp(fine_depths, depths, temp_row)

        # Find CTS for each time step (depth where temp is closest to 0°C)
        cts_depths = []
        for temp_row in temp_interp:
            idx = np.argmin(np.abs(temp_row))
            # Only mark CTS if temperature is within threshold of 0°C
            if np.abs(temp_row[idx]) < cts_threshold:
                cts_depths.append(fine_depths[idx])
            else:
                cts_depths.append(np.nan)

        # Plot heatmap
        plt.figure(figsize=(8, 4), dpi=250)
        extent = [mdates.date2num(times[0]), mdates.date2num(times[-1]), fine_depths[-1], fine_depths[0]]
        plt.imshow(
            temp_interp.T,
            aspect='auto',
            cmap=cmc.batlow,
            extent=extent,
            vmin=np.nanmin(temp_interp),
            vmax=np.nanmax(temp_interp),
            label='Temperature [°C]'  # This label will not show in legend, but can be used for reference
        )

        plt.xlabel('Time', fontsize=self.fontsize)
        plt.ylabel('Depth [m]', fontsize=self.fontsize)
        plt.title(title if title else 'Temperature Profile Heatmap', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize, rotation=45)
        plt.yticks(fontsize=self.fontsize)
        plt.grid()

        # Mark CTS
        plt.plot(times, cts_depths, color='k', linewidth=self.linewidth, label='CTS (0°C)')
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > 0:
            plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                   loc='upper right', ncol=1, fontsize=self.fontsize)

        # Set x-axis format to month and day only
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # set format using format_plot
        self.format_plot(title)

        cbar = plt.colorbar()
        cbar.set_label('Temperature [°C]', fontsize=self.fontsize)
        cbar.ax.tick_params(labelsize=self.fontsize-2)

        plt.tight_layout()
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
    
    def plot_multiple_geoprisition_chains(self, start_time, end_time, depths, savepath, title=None):
        thermistor1 = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        thermistor2 = ThermistorData(self.file_paths[1], self.delimiter, self.measurement_depth)
        data1 = thermistor1.get_chain_data(start_time, end_time)
        data2 = thermistor2.get_chain_data(start_time, end_time)
        data1['TIME'] = pd.to_datetime(data1['TIME'])
        data2['TIME'] = pd.to_datetime(data2['TIME'])

        # adjust figure size based on legend size
        plt.figure(dpi=300)  # adjust the values as per your requirement and set dpi for resolution

        # plot the data for each depth
        thermistor_column1 = f'{depths[0]} m'
        thermistor_column2 = f'{depths[1]} m'
        plt.plot(data1['TIME'], data1[thermistor_column1], label=f'Disturbed borehole')
        plt.plot(data2['TIME'], data2[thermistor_column2], label=f'Pre-drilled borehole')

        # format the plot
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title)
        plt.legend(fontsize='small')
        plt.xticks(rotation=45)  # Rotate x ticks 45 degrees
        plt.grid()
        plt.tight_layout()

        # modify x-axis tick format to show date and time without seconds
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

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

        # Calculate the 0-degree offset for the Black and White probes
        black_probe_offset, stable_indices_black = calculate_zero_degree_offset(ntc_thermistor_data['Black Probe Temperature'])
        white_probe_offset, stable_indices_white = calculate_zero_degree_offset(ntc_thermistor_data['White Probe Temperature'])

        # Get the time of the stable period
        stable_period_black_start = ntc_thermistor_data['TIME'].iloc[stable_indices_black[0]]
        stable_period_black_end = ntc_thermistor_data['TIME'].iloc[stable_indices_black[-1]]
        stable_period_white_start = ntc_thermistor_data['TIME'].iloc[stable_indices_white[0]]
        stable_period_white_end = ntc_thermistor_data['TIME'].iloc[stable_indices_white[-1]]

        # adjust figure size based on legend size
        plt.figure(figsize=(12, 8.5), dpi=300)

        # plot the ntc data
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['Black Probe Temperature'], linewidth=4, alpha=0.6, label='Black Probe (Gemini)')
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['White Probe Temperature'], linewidth=4, alpha=0.6, label='White Probe (Gemini)')

        # plot the reference geoprecision thermistor chain data
        plt.plot(thermistor_chain_data['TIME'], thermistor_chain_data['10.0 m'], linewidth=4, alpha=0.6, label='Thermistor chain (Geoprecision)', color='black')

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

        # format the plot
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

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # save the plot
        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')

        zero_degree_offsets = (black_probe_offset, white_probe_offset)

        return zero_degree_offsets