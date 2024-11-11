import os

# import thermistor chain processing functions
os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/asses_swiss_gl_therm_regimes/processing')
from process_thermistor_data import ThermistorData

# create a ThermistorData object
path_10m_chain = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/thermistor_chains/calibration_runs/A5389E_20240822133758.csv'
path_20m_chain = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/thermistor_chains/calibration_runs/A53964_20240822134409.csv'
path_10m_chain_11_16 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/thermistor_chains/calibration_runs/A5389E_ice_bath_logger_11_16.csv'
path_20m_chain_11_16 = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers/thermistor_chains/calibration_runs/A53964_ice_bath_logger_11_16.csv'

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

# # compute the 0-degree offset of the thermistor chains parallel to NTC loggers 1 till 8
# offset_10m_2nd_ice_bath =  data_10m_2nd_ice_bath['10.0 m'].mean() # 10m thermistor chain offset
# sample_size_10m_2nd_ice_bath = data_10m_2nd_ice_bath['10.0 m'].count() # sample size

# # compute the 0-degree offset of the thermistor chains parallel to NTC loggers 9 till 10
# offset_10m_3rd_ice_bath =  data_10m_3rd_ice_bath['10.0 m'].mean() # 10m thermistor chain offset
# offset_20m_3rd_ice_bath =  data_20m_3rd_ice_bath['20.0 m'].mean() # 20m thermistor chain offset
# sample_size_10m_3rd_ice_bath = data_10m_3rd_ice_bath['10.0 m'].count() # sample size
# sample_size_20m_3rd_ice_bath = data_20m_3rd_ice_bath['20.0 m'].count() # sample size

# # compute the 0-degree offset of the thermistor chains parallel to NTC loggers 11 till 16
# offset_10m_4th_ice_bath =  data_10m_4th_ice_bath['10.0 m'].mean() # 10m thermistor chain offset
# offset_20m_4th_ice_bath =  data_20m_4th_ice_bath['20.0 m'].mean() # 20m thermistor chain offset
# sample_size_10m_4th_ice_bath = data_10m_4th_ice_bath['10.0 m'].count() # sample size
# sample_size_20m_4th_ice_bath = data_20m_4th_ice_bath['20.0 m'].count() # sample size