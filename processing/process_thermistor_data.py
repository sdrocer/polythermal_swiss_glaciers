import pandas as pd
import matplotlib.pyplot as plt

# import self-written modules
from read_thermistor_data import ThermistorData
from plot_thermistor_data import PlotThermistorData

"""
    Python script to read & plot different types of thermistor data.
"""

# set the path to the data
data_path = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork/Survey_I_2024-08/NTC_calibration_and_tests/'

# read the data
ntc_thermistor = ThermistorData(f'{data_path}/#1_ice_bath_0deg_offset.csv', ',')

# get the data
ntc_thermistor_data = ntc_thermistor.get_ntc_data()

# plot the data
plotter = PlotThermistorData(data_path)
plotter.plot_ntc_data(filename = '#8_ice_bath_0deg_offset.csv',savepath=data_path, title='0deg offset ice bath - Logger: #8')
