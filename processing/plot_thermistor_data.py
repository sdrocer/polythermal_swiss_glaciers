import pandas as pd
import matplotlib.pyplot as plt

# import self-written modules
os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/')
from processing.process_thermistor_data import ThermistorData
from processing.process_thermistor_data import ThermistorDataPlotter
from calibration.thermistor_chains_0deg_references import *

"""
    Python script to plot different types of thermistor data.

    Code written by: Janosch Beer
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

# set the paths to the data
sex_rouge_dirs = [temp_path + 'BH1_sex_rouge_20240806_20241019.csv', temp_path + 'BH2_sex_rouge_20240806_20241019.csv']
tortin_dirs    = [temp_path + 'BH3_tortin_20240807_20240930.csv'   , temp_path + 'BH4_tortin_20240807_20240930.csv']
hohlaub_dirs   = [temp_path + 'BH5_hohlaub_20240808_20240929.csv'  , temp_path + 'BH6_hohlaub_20240808_20240929.csv']
chessjen_dirs  = [temp_path + 'BH7_chessjen_20240809_20240929.csv' , temp_path + 'BH8_chessjen_20240809_20240929.csv']
alphubel_dirs  = [None                                             , temp_path + 'BH10_alphubel_20240821_20241021.csv']  # no data for BH9
corvatsch_dirs = [None                                             , temp_path + 'BH12_corvatsch_20240828_20241020.csv'] # no data for BH11

# create a ThermistorDataPlotter object per glacier
sex_rouge = ThermistorDataPlotter(sex_rouge_dirs, delimiter=',')
tortin    = ThermistorDataPlotter(tortin_dirs   , delimiter=',')
hohlaub   = ThermistorDataPlotter(hohlaub_dirs  , delimiter=',')
chessjen  = ThermistorDataPlotter(chessjen_dirs , delimiter=',')
alphubel  = ThermistorDataPlotter(alphubel_dirs , delimiter=',')
corvatsch = ThermistorDataPlotter(corvatsch_dirs, delimiter=',')

# plot the data per glacier
sex_rouge.plot_multiple_ntc_boreholes(savepath=output_path, title='Glacier du Sex Rouge 08/2024-10/2024', depths=[10.0,15.25,10.0,14.45], borehole_labels=['BH1','BH2'], lower_y_limit=-1.5)
tortin.plot_multiple_ntc_boreholes(savepath=output_path   , title='Glacier de Tortin 08/2024-09/2024'   , depths=[4.2,9.2,8.3,13.3]     , borehole_labels=['BH3','BH4'], lower_y_limit=-1.5)
hohlaub.plot_multiple_ntc_boreholes(savepath=output_path  , title='Hohlaubgletscher 08/2024-09/2024'    , depths=[10.6,15.6,10.4,15.4]  , borehole_labels=['BH5','BH6'], lower_y_limit=-1.5)
chessjen.plot_multiple_ntc_boreholes(savepath=output_path , title='Chessjengletscher 08/2024-09/2024'   , depths=[6.8,11.8,8.5,13.5]    , borehole_labels=['BH7','BH8'], lower_y_limit=-1.5, legend_loc='upper right')
alphubel.plot_multiple_ntc_boreholes(savepath=output_path , title='Alphubel South 08/2024-10/2024'      , depths=[8.0,13.0,9.0,14.0]    , borehole_labels=['BH9','BH10'], lower_y_limit=-3.5)
corvatsch.plot_multiple_ntc_boreholes(savepath=output_path, title='Corvatsch 08/2024-10/2024'           , depths=[2.0,7.0,4.3,9.3]      , borehole_labels=['BH11','BH12'], lower_y_limit=-3.5)

## Plot calibration data ##
# ----------------------- #

# set the paths to the calibration data
logger_1_dir = cal_path + 'raw_files/#1_ice_bath_0deg_offset_second_trial.csv'
logger_2_dir = cal_path + 'raw_files/#2_ice_bath_0deg_offset_second_trial.csv'
logger_3_dir = cal_path + 'raw_files/#3_ice_bath_0deg_offset_second_trial.csv'
logger_4_dir = cal_path + 'raw_files/#4_ice_bath_0deg_offset_second_trial.csv'
logger_5_dir = cal_path + 'raw_files/#5_ice_bath_0deg_offset_second_trial.csv'
logger_6_dir = cal_path + 'raw_files/#6_ice_bath_0deg_offset_second_trial.csv'
logger_7_dir = cal_path + 'raw_files/#7_ice_bath_0deg_offset_second_trial.csv'
logger_8_dir = cal_path + 'raw_files/#8_ice_bath_0deg_offset_second_trial.csv'
logger_9_dir = cal_path + 'raw_files/#9_ice_bath_0deg_offset.csv'
logger_10_dir = cal_path + 'raw_files/#10_ice_bath_0deg_offset.csv'
logger_11_dir = cal_path + 'raw_files/#11_ice_bath_0deg_offset.csv'
logger_12_dir = cal_path + 'raw_files/#12_ice_bath_0deg_offset.csv'
logger_13_dir = cal_path + 'raw_files/#13_ice_bath_0deg_offset.csv'
logger_14_dir = cal_path + 'raw_files/#14_ice_bath_0deg_offset.csv'
logger_15_dir = cal_path + 'raw_files/#15_ice_bath_0deg_offset_black_probe_missing.csv'
logger_16_dir = cal_path + 'raw_files/#16_ice_bath_0deg_offset.csv'

# create a ThermistorDataPlotter object per logger
logger_1 = ThermistorDataPlotter(logger_1_dir, delimiter=',')
logger_2 = ThermistorDataPlotter(logger_2_dir, delimiter=',')
logger_3 = ThermistorDataPlotter(logger_3_dir, delimiter=',')
logger_4 = ThermistorDataPlotter(logger_4_dir, delimiter=',')
logger_5 = ThermistorDataPlotter(logger_5_dir, delimiter=',')
logger_6 = ThermistorDataPlotter(logger_6_dir, delimiter=',')
logger_7 = ThermistorDataPlotter(logger_7_dir, delimiter=',')
logger_8 = ThermistorDataPlotter(logger_8_dir, delimiter=',')
logger_9 = ThermistorDataPlotter(logger_9_dir, delimiter=',')
logger_10 = ThermistorDataPlotter(logger_10_dir, delimiter=',')
logger_11 = ThermistorDataPlotter(logger_11_dir, delimiter=',')
logger_12 = ThermistorDataPlotter(logger_12_dir, delimiter=',')
logger_13 = ThermistorDataPlotter(logger_13_dir, delimiter=',')
logger_14 = ThermistorDataPlotter(logger_14_dir, delimiter=',')
logger_15 = ThermistorDataPlotter(logger_15_dir, delimiter=',')
logger_16 = ThermistorDataPlotter(logger_16_dir, delimiter=',')

# plot the data per logger
logger_1.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #1 - 0deg offset in ice bath')
logger_2.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #2 - 0deg offset in ice bath')
logger_3.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #3 - 0deg offset in ice bath')
logger_4.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #4 - 0deg offset in ice bath', y_limits=[-7,7])
logger_5.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #5 - 0deg offset in ice bath')
logger_6.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #6 - 0deg offset in ice bath')
logger_7.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #7 - 0deg offset in ice bath', y_limits=[-7,7])
logger_8.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #8 - 0deg offset in ice bath')
logger_9.plot_ntc_icebath_calibration(data_10m_chain_3rd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #9 - 0deg offset in ice bath')
logger_10.plot_ntc_icebath_calibration(data_10m_chain_3rd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #10 - 0deg offset in ice bath')
logger_11.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #11 - 0deg offset in ice bath')
logger_12.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #12 - 0deg offset in ice bath')
logger_13.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #13 - 0deg offset in ice bath')
logger_14.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #14 - 0deg offset in ice bath')
logger_15.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #15 - 0deg offset in ice bath, no black probe')
logger_16.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #16 - 0deg offset in ice bath')