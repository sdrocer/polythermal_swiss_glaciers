import pandas as pd
import matplotlib.pyplot as plt

# import self-written modules
from process_thermistor_data import ThermistorData
from process_thermistor_data import ThermistorDataPlotter

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
plotter = ThermistorDataPlotter(ntc_thermistor_data)
plotter.plot_ntc_data(filename='#8_ice_bath_0deg_offset.csv', savepath=output_path, title='0deg offset ice bath - Logger: #8')

## Plot Measurement period 1 08/2024 - 10/2024 ##
# -------------------------------------------- #

# create a ThermistorDataPlotter object per glacier
tortin_dirs   = [temp_path + 'tortin_BH3_20240807_20240930.csv', temp_path + 'tortin_BH4_20240807_20240930.csv']
hohlaub_dirs  = [temp_path + 'hohlaub_BH5_20240808_20240929.csv', temp_path + 'hohlaub_BH6_20240808_20240929.csv']
chessjen_dirs = [temp_path + 'chessjen_BH7_20240809_20240929.csv', temp_path + 'chessjen_BH8_20240809_20240929.csv']

tortin   = ThermistorDataPlotter(tortin_dirs, delimiter=',')
hohlaub  = ThermistorDataPlotter(hohlaub_dirs, delimiter=',')
chessjen = ThermistorDataPlotter(chessjen_dirs, delimiter=',')

# plot the data per glacier
tortin.plot_multiple_ntc_boreholes(savepath=output_path, title='Glacier de Tortin 08/2024-09/2024', depths=[4.2,9.2,8.3,13.3], borehole_labels=['BH3','BH4'], lower_y_limit=-0.5)
hohlaub.plot_multiple_ntc_boreholes(savepath=output_path, title='Hohlaubgletscher 08/2024-09/2024', depths=[10.6,15.6,10.4,15.4], borehole_labels=['BH5','BH6'], lower_y_limit=-1.2)
chessjen.plot_multiple_ntc_boreholes(savepath=output_path, title='Chessjengletscher 08/2024-09/2024', depths=[6.8,11.8,8.5,13.5], borehole_labels=['BH7','BH8'], lower_y_limit=-1.2, legend_loc='upper right')