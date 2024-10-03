import pandas as pd
import matplotlib.pyplot as plt

# import self-written modules
from process_thermistor_data import ThermistorData
from process_thermistor_data import PlotThermistorData

"""
    Python script to read & plot different types of thermistor data.
"""

# set the path to the data
cal_path  = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/NTC/NTC_calibration_data/'
temp_path = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/NTC/NTC_temperature_data/'

# set the path for the output
output_path = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/products/figures/'

## Plot 0 degree ice bath calibration results ##
# -------------------------------------------- #

# read the data
ntc_thermistor = ThermistorData(f'{data_path}/#1_ice_bath_0deg_offset.csv', ',')

# get the data
ntc_thermistor_data = ntc_thermistor.get_ntc_data()

# plot the data
plotter = PlotThermistorData(data_path)
plotter.plot_ntc_data(filename = '#8_ice_bath_0deg_offset.csv',savepath=data_path, title='0deg offset ice bath - Logger: #8')

## Plot Measurement period 1 08/2024 - 10/2024 ##
# -------------------------------------------- #

# create a PlotThermistorData object per borehole
tortin_bh3   = PlotThermistorData(temp_path + 'tortin_BH3_20240807_20240930.csv', delimiter=',')
tortin_bh4   = PlotThermistorData(temp_path + 'tortin_BH4_20240807_20240930.csv', delimiter=',')
hohlaub_bh5  = PlotThermistorData(temp_path + 'hohlaub_BH5_20240808_20240929.csv', delimiter=',')
hohlaub_bh6  = PlotThermistorData(temp_path + 'hohlaub_BH6_20240808_20240929.csv', delimiter=',')
chessjen_bh7 = PlotThermistorData(temp_path + 'chessjen_BH7_20240809_20240929.csv', delimiter=',')
chessjen_bh8 = PlotThermistorData(temp_path + 'chessjen_BH8_20240809_20240929.csv', delimiter=',')

# create a PlotThermistorData object per glacier
tortin_dirs   = [temp_path + 'tortin_BH3_20240807_20240930.csv', temp_path + 'tortin_BH4_20240807_20240930.csv']
hohlaub_dirs  = [temp_path + 'hohlaub_BH5_20240808_20240929.csv', temp_path + 'hohlaub_BH6_20240808_20240929.csv']
chessjen_dirs = [temp_path + 'chessjen_BH7_20240809_20240929.csv', temp_path + 'chessjen_BH8_20240809_20240929.csv']

tortin   = PlotThermistorData(tortin_dirs, delimiter=',')
hohlaub  = PlotThermistorData(hohlaub_dirs, delimiter=',')
chessjen = PlotThermistorData(chessjen_dirs, delimiter=',')

# plot the data per borehole
tortin_bh3.plot_ntc_data(savepath=output_path, title='Tortin BH3 08/2024-09/2024', depth_white_probe=4.2, depth_black_probe=9.2)
tortin_bh4.plot_ntc_data(savepath=output_path, title='Tortin BH4 08/2024-09/2024', depth_white_probe=8.3, depth_black_probe=13.3)
hohlaub_bh5.plot_ntc_data(savepath=output_path, title='Hohlaub BH5 08/2024-09/2024', depth_white_probe=10.6, depth_black_probe=15.6)
hohlaub_bh6.plot_ntc_data(savepath=output_path, title='Hohlaub BH6 08/2024-09/2024', depth_white_probe=10.4, depth_black_probe=15.4)
chessjen_bh7.plot_ntc_data(savepath=output_path, title='Chessjen BH7 09/2024-09/2024', depth_white_probe=6.8, depth_black_probe=11.8, lower_y_limit=-1.2)
chessjen_bh8.plot_ntc_data(savepath=output_path, title='Chessjen BH8 09/2024-09/2024', depth_white_probe=8.5, depth_black_probe=13.5, lower_y_limit=-1.2)

# plot the data per glacier
tortin.plot_multiple_ntc_boreholes(savepath=output_path, title='Glacier de Tortin 08/2024-09/2024', depths=[4.2,9.2,8.3,13.3], borehole_labels=['BH3','BH4'], lower_y_limit=-0.5)
hohlaub.plot_multiple_ntc_boreholes(savepath=output_path, title='Hohlaubgletscher 08/2024-09/2024', depths=[10.6,15.6,10.4,15.4], borehole_labels=['BH5','BH6'], lower_y_limit=-1.2)
chessjen.plot_multiple_ntc_boreholes(savepath=output_path, title='Chessjengletscher 08/2024-09/2024', depths=[6.8,11.8,8.5,13.5], borehole_labels=['BH7','BH8'], lower_y_limit=-1.2, legend_loc='upper right')