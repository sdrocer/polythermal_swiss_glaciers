import os

# import thermistor chain processing functions
os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/processing')
from thermistor_processing import ThermistorData

# create a ThermistorData object
path_10m_chain = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/thermistor_chains/tinus_chains/calibration_runs/A5389E_20240822133758.csv'
path_20m_chain = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/thermistor_chains/tinus_chains/calibration_runs/A53964_20240822134409.csv'
path_10m_chain_11_16 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/thermistor_chains/tinus_chains/calibration_runs/A5389E_ice_bath_logger_11_16.csv'
path_20m_chain_11_16 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/thermistor_chains/tinus_chains/calibration_runs/A53964_ice_bath_logger_11_16.csv'


chain_10m = ThermistorData(path_10m_chain, ',', 10)
chain_20m = ThermistorData(path_20m_chain, ',', 20)
chain_10m_11_16 = ThermistorData(path_10m_chain_11_16, ',', 10)
chain_20m_11_16 = ThermistorData(path_20m_chain_11_16, ',', 20)

# read the data for loggers 1 till 8 (2nd ice bath)
start_time_2nd_ice_bath = '05.08.2024 15:00:00'
end_time_2nd_ice_bath   = '05.08.2024 16:25:00'

data_10m_chain_2nd_ice_bath = chain_10m.get_chain_data(start_time_2nd_ice_bath, end_time_2nd_ice_bath)

# read the data for loggers 9 till 10 (3rd ice bath)
start_time_3rd_ice_bath = '20.08.2024 10:17:00'
end_time_3rd_ice_bath   = '20.08.2024 13:07:00'

data_10m_chain_3rd_ice_bath = chain_10m.get_chain_data(start_time_3rd_ice_bath, end_time_3rd_ice_bath)
data_20m_chain_3rd_ice_bath = chain_20m.get_chain_data(start_time_3rd_ice_bath, end_time_3rd_ice_bath)

# read the data for loggers 11 till 16 (4th ice bath)
start_time_4th_ice_bath = '27.08.2024 11:55:00'
end_time_4th_ice_bath   = '27.08.2024 13:16:00'

data_10m_chain_4th_ice_bath = chain_10m_11_16.get_chain_data(start_time_4th_ice_bath, end_time_4th_ice_bath)
data_20m_chain_4th_ice_bath = chain_20m_11_16.get_chain_data(start_time_4th_ice_bath, end_time_4th_ice_bath)

## Reliability experiments ##
#---------------------------#

# create a ThermistorData object for reliability experiments

# Experiment 1: 30.01.2025
path_10m_chain_reliability_exp1 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment1_20250130/Thermistor_chains/A5389E_20250130181214.csv'
path_20m_chain_reliability_exp1 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment1_20250130/Thermistor_chains/A53964_20250130180705.csv'

# create a ThermistorData object
chain_10m_reliability_exp1 = ThermistorData(path_10m_chain_reliability_exp1, ',', 10)
chain_20m_reliability_exp1 = ThermistorData(path_20m_chain_reliability_exp1, ',', 20)

# read the data for loggers 13 till 16 (reliability experiment)
start_time_reliability_exp1 = '30.01.2025 15:15:00'
end_time_reliability_exp1   = '30.01.2025 18:15:00'

data_10m_chain_reliability_exp1 = chain_10m_reliability_exp1.get_chain_data(start_time_reliability_exp1, end_time_reliability_exp1)
data_20m_chain_reliability_exp1 = chain_20m_reliability_exp1.get_chain_data(start_time_reliability_exp1, end_time_reliability_exp1)

# --------------------------#
# Experiment 2: 04.03.2025
path_10m_chain_reliability_exp2 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment2_20250304/Thermistor_chains/A5389E_20250304121907.csv'
path_20m_chain_reliability_exp2 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment2_20250304/Thermistor_chains/A53964_20250304121405.csv'

# create a ThermistorData object
chain_10m_reliability_exp2 = ThermistorData(path_10m_chain_reliability_exp2, ',', 10)
chain_20m_reliability_exp2 = ThermistorData(path_20m_chain_reliability_exp2, ',', 20)

# read the data for loggers 13 till 16 (reliability experiment)
start_time_reliability_exp2 = '04.03.2025 09:30:00'
end_time_reliability_exp2   = '04.03.2025 12:30:00'

data_10m_chain_reliability_exp2 = chain_10m_reliability_exp2.get_chain_data(start_time_reliability_exp2, end_time_reliability_exp2)
data_20m_chain_reliability_exp2 = chain_20m_reliability_exp2.get_chain_data(start_time_reliability_exp2, end_time_reliability_exp2)

# --------------------------#
# Experiment 3: 03.06.2025
path_10m_chain_reliability_exp3 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment3_20250603/Thermistor_chains/A5389E_20250603153411.csv'
path_20m_chain_reliability_exp3 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments/Experiment3_20250603/Thermistor_chains/A53964_20250603153704.csv'

# create a ThermistorData object
chain_10m_reliability_exp3 = ThermistorData(path_10m_chain_reliability_exp3, ',', 10)
chain_20m_reliability_exp3 = ThermistorData(path_20m_chain_reliability_exp3, ',', 20)

# read the data for loggers 13 till 16 (reliability experiment)
start_time_reliability_exp3 = '03.06.2025 13:15:00'
end_time_reliability_exp3   = '03.06.2025 15:30:00'

data_10m_chain_reliability_exp3 = chain_10m_reliability_exp3.get_chain_data(start_time_reliability_exp3, end_time_reliability_exp3)
data_20m_chain_reliability_exp3 = chain_20m_reliability_exp3.get_chain_data(start_time_reliability_exp3, end_time_reliability_exp3)