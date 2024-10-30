import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

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
        with open(self.file_path, 'r') as file:
            columns = None
            for line in file:
                if columns is None:
                    if line.startswith('NO{0}TIME{0}'.format(self.delimiter)):
                        columns = line.strip().split(self.delimiter)
                elif line[0].isdigit():
                    if self.delimiter == ';':
                        line = line.replace(',', '.')
                    data_lines.append(line.strip().split(self.delimiter))

        self.data = pd.DataFrame(data_lines, columns=columns)
        self.data['TIME'] = pd.to_datetime(self.data['TIME'], format='%d.%m.%Y %H:%M:%S') + pd.DateOffset(hours=1)
        for col in self.data.columns[2:]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Convert input times to datetime
        start_time = pd.to_datetime(start_time, format='%d.%m.%Y %H:%M:%S')
        end_time = pd.to_datetime(end_time, format='%d.%m.%Y %H:%M:%S')

        # Filter the DataFrame based on the time range
        self.data = self.data[(self.data['TIME'] >= start_time) & (self.data['TIME'] <= end_time)]

        # Modify the column names to include depth information
        num_depth_columns = self.measurement_depth * 2
        num_thermistor_columns = len(self.data.columns) - num_depth_columns - 4  # Subtract 4 for 'NO', 'TIME', 'TEMP LOGGER' and 'TEMP BATTERY' columns

        self.data.columns = (
            self.data.columns[:2].tolist()  # 'NO' and 'TIME' columns
            + [f"# {i}" for i in range(1, num_thermistor_columns + 1)]  # 'Thermistor' columns
            + [f"{i*0.5:.1f} m" for i in range(1, num_depth_columns + 1)]  # Depth columns
            + ['TEMP LOGGER', 'TEMP BATTERY']  # 'TEMP LOGGER' and 'TEMP BATTERY' columns
        )

        return self.data

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
        # Change the default font to Arial
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.xlabel('Time', fontsize=22)
        plt.ylabel('Ice temperature [°C]', fontsize=22)
        # plt.title(title, fontsize=22)
        plt.legend(fontsize=22, frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc=legend_loc)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=3)
        plt.xticks(rotation=45, fontsize=22)  # Rotate x ticks 45 degrees
        plt.yticks(fontsize=22)
        plt.grid()
        plt.tight_layout()

    def plot_full_geopresition_chain(self, start_time, end_time, savepath, title=None):
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data(start_time, end_time)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # adjust figure size based on legend size
        plt.figure(dpi=300)  # adjust the values as per your requirement and set dpi for resolution

        # plot the data
        for column in data.columns:
            if column not in ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY']:
                print(column)
                plt.plot(data['TIME'], data[column], label=column)

        # format the plot
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Depth [m]', ncol=2, fontsize='small')  # Set ncol=2 for two columns
        plt.xticks(rotation=45)  # Rotate x ticks 45 degrees
        plt.grid()
        plt.tight_layout()

        # modify x-axis tick format to show date and time without seconds
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        # save the plot
        plt.savefig(savepath)

    def plot_geopresition_thermistor(self, start_time, end_time, depths, savepath, title=None):
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

    def plot_multiple_ntc_boreholes(self, savepath, depths, borehole_labels, title=None, lower_y_limit=-1, legend_loc='lower right'):
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

        # Create figure and axis with the specified ratio
        fig_ratio = 136.446 / 115.986
        fig_width = 10 # Set the width of the figure
        fig_height = fig_width / fig_ratio

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
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
        plt.ylim(lower_y_limit, 0.4)
        plt.axvline(deployment_date, color='gray', linestyle='solid', linewidth=4)
        plt.text(deployment_date, plt.ylim()[0], 'Deployment', color='gray', fontsize=22, verticalalignment='top', horizontalalignment='right', rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Set x-ticks to be equally spaced by 1 day
 
        # format the plot
        self.format_plot(title, legend_loc)

        # save the plot
        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')