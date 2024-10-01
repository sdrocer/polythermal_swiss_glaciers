import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/code/process_thermistor_data')
from read_thermistor_data import ThermistorData as td
from plot_thermistor_data import PlotThermistorData as ptd

# set the paths of the thermistor data
path_10m_thermistor = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/ice_block_experiment/new_data/A5389E_20240612162616.csv'
path_20m_thermistor = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/ice_block_experiment/new_data/A53964_20240612162658.csv'

# set path to save the plots
savepath  = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/ice_block_experiment/products/'
savepath1 = savepath + 'disturbed_borehole_experiment1.png'
savepath2 = savepath + 'pre_drilled_borehole_experiment1.png'
savepath3 = savepath + 'disturbed_borehole_experiment2.png'
savepath4 = savepath + 'pre_drilled_borehole_experiment2.png'
savepath_combined1 = savepath + 'experiment1.png'
savepath_combined2 = savepath + 'experiment2.png'

# set plotting parameters
plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2.5, 'figure.figsize': (10, 6)})

# create an instance of the PlotThermistorData class
thermistor10m = ptd(path_10m_thermistor)
thermistor20m = ptd(path_20m_thermistor)

thermistors = ptd([path_10m_thermistor, path_20m_thermistor])

# plot the data for each thermistor chain separately
thermistor10m.plot_specific_thermistors(3, [10.0], savepath1, 'Disturbed borehole (Experiment 1)')   # Experiment 1
thermistor20m.plot_specific_thermistors(1, [20.0], savepath2, 'Pre-drilled borehole (Experiment 1)') # Experiment 1
thermistor10m.plot_specific_thermistors(5, [10.0], savepath3, 'Disturbed borehole (Experiment 2)')   # Experiment 2
thermistor20m.plot_specific_thermistors(3, [20.0], savepath4, 'Pre-drilled borehole (Experiment 2)') # Experiment 2

# plot the data for both thermistor chains
thermistors.plot_multiple_thermistor_chains([3,1], [10.0, 20.0], savepath_combined1, 'Ice block experiment (Experiment 1: pre-cooled thermistors)') # Experiment 1
thermistors.plot_multiple_thermistor_chains([5,3], [10.0, 20.0], savepath_combined2, 'Ice block experiment (Experiment 2: warm thermistors)')       # Experiment 2