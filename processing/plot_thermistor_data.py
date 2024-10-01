import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import dates as mdates
from read_thermistor_data import ThermistorData as td

"""
    This script is used to plot the data from a thermistor data file.
"""

class PlotThermistorData:
    def __init__(self, file_path, measurement_depth = None, delimiter = ','):
        if isinstance(file_path, list):
            self.file_paths = file_path
            self.file_path = file_path[0]
        else:
            self.file_paths = [file_path]
            self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth

    def plot_full_chain(self, start_time, end_time, savepath, title=None):
        thermistor = td(self.file_path, self.measurement_depth, self.delimiter)
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

    def plot_specific_thermistors(self, start_time, end_time, depths, savepath, title=None):
        thermistor = td(self.file_path, self.measurement_depth, self.delimiter)
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
    
    def plot_multiple_thermistor_chains(self, start_time, end_time, depths, savepath, title=None):
        thermistor1 = td(self.file_path, self.measurement_depth, self.delimiter)
        thermistor2 = td(self.file_paths[1], self.measurement_depth, self.delimiter)
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
    
    def plot_ntc_data(self, savepath, title=None, depth_white_probe=None, depth_black_probe=None, lower_y_limit=-1):
        ntc_thermistor = td(self.file_path, self.measurement_depth, self.delimiter)
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