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


## Precious plot_tynitag_thermistor_data.py

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
cal_path  = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/calibration_data/NTC_reliability_experiments'
temp_path = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/THERMAP_2024_2025/icetemperature_data/NTC_tynitag/temperature_data/2024_2025/'

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
sex_rouge_dirs = [temp_path + 'BH1_20240930_20250724.csv', temp_path + 'BH2_20240930_20250724.csv']
tortin_dirs    = [temp_path + 'BH3_20240939_20250723.csv'   , temp_path + 'BH4_20240930_20250723.csv']
hohlaub_dirs   = [temp_path + 'BH5_hohlaub_20240808_20240929.csv'  , temp_path + 'BH6_hohlaub_20240808_20240929.csv']
chessjen_dirs  = [temp_path + 'BH7_20240929_20250808.csv' , temp_path + 'BH8_20240929_20250808.csv']
alphubel_dirs  = [None                                             , temp_path + 'BH10_20241021_20250805.csv']  # no data for BH9
corvatsch_dirs = [None                                             , temp_path + 'BH12_corvatsch_20240828_20241020.csv'] # no data for BH11

# create a ThermistorDataPlotter object per glacier
sex_rouge = ThermistorDataPlotter(sex_rouge_dirs, delimiter=',')
tortin    = ThermistorDataPlotter(tortin_dirs   , delimiter=',')
hohlaub   = ThermistorDataPlotter(hohlaub_dirs  , delimiter=',')
chessjen  = ThermistorDataPlotter(chessjen_dirs , delimiter=',')
alphubel  = ThermistorDataPlotter(alphubel_dirs , delimiter=',')
corvatsch = ThermistorDataPlotter(corvatsch_dirs, delimiter=',')

# plot the data per glacier
sex_rouge.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/', title='Glacier du Sex Rouge 09/2024-07/2025', depths=[10.0,15.25,10.0,14.45], borehole_labels=['BH1','BH2'], lower_y_limit=-1.5)
tortin.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/'   , title='Glacier de Tortin 09/2024-07/2025'   , depths=[4.2,9.2,8.3,13.3]     , borehole_labels=['BH3','BH4'], lower_y_limit=-4.0)
hohlaub.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/'  , title='Hohlaubgletscher 08/2024-09/2024'    , depths=[10.6,15.6,10.4,15.4]  , borehole_labels=['BH5','BH6'], lower_y_limit=-1.5)
chessjen.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/' , title='Chessjengletscher 09/2024-08/2025'   , depths=[6.8,11.8,8.5,13.5]    , borehole_labels=['BH7','BH8'], lower_y_limit=-2.5, legend_loc='lower right')
alphubel.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/' , title='Alphubel South 10/2024-08/2025'      , depths=[8.0,13.0,9.0,14.0]    , borehole_labels=['BH9','BH10'], lower_y_limit=-3.5)
corvatsch.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/', title='Corvatsch 08/2024-10/2024'           , depths=[2.0,7.0,4.3,9.3]      , borehole_labels=['BH11','BH12'], lower_y_limit=-3.5)

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

# plot the data per logger and return the 0 degree offsets
zero_deg_offsets_logger1  = logger_1.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #1 - 0deg offset in ice bath')
zero_deg_offsets_logger2  = logger_2.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #2 - 0deg offset in ice bath')
zero_deg_offsets_logger3  = logger_3.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #3 - 0deg offset in ice bath')
zero_deg_offsets_logger4  = logger_4.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #4 - 0deg offset in ice bath', y_limits=[-7,7])
zero_deg_offsets_logger5  = logger_5.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #5 - 0deg offset in ice bath')
zero_deg_offsets_logger6  = logger_6.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #6 - 0deg offset in ice bath')
zero_deg_offsets_logger7  = logger_7.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #7 - 0deg offset in ice bath', y_limits=[-7,7])
zero_deg_offsets_logger8  = logger_8.plot_ntc_icebath_calibration(data_10m_chain_2nd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #8 - 0deg offset in ice bath')
zero_deg_offsets_logger9  = logger_9.plot_ntc_icebath_calibration(data_10m_chain_3rd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #9 - 0deg offset in ice bath')
zero_deg_offsets_logger10 = logger_10.plot_ntc_icebath_calibration(data_10m_chain_3rd_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #10 - 0deg offset in ice bath')
zero_deg_offsets_logger11 = logger_11.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #11 - 0deg offset in ice bath')
zero_deg_offsets_logger12 = logger_12.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #12 - 0deg offset in ice bath')
zero_deg_offsets_logger13 = logger_13.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #13 - 0deg offset in ice bath')
zero_deg_offsets_logger14 = logger_14.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #14 - 0deg offset in ice bath')
zero_deg_offsets_logger15 = logger_15.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #15 - 0deg offset in ice bath, no black probe')
zero_deg_offsets_logger16 = logger_16.plot_ntc_icebath_calibration(data_10m_chain_4th_ice_bath, savepath=output_path + 'thermistor_calibration/', title='Logger #16 - 0deg offset in ice bath')

# plot the data per glacier and apply 0 degree offset
sex_rouge.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/', title='Glacier du Sex Rouge 08/2024-10/2024 calibrated', depths=[10.0,15.25,10.0,14.45], borehole_labels=['BH1','BH2'], lower_y_limit=-1.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger1,zero_deg_offsets_logger2])
tortin.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/'   , title='Glacier de Tortin 08/2024-09/2024 calibrated'   , depths=[4.2,9.2,8.3,13.3]     , borehole_labels=['BH3','BH4'], lower_y_limit=-1.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger3,zero_deg_offsets_logger4])
hohlaub.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/'  , title='Hohlaubgletscher 08/2024-09/2024 calibrated'    , depths=[10.6,15.6,10.4,15.4]  , borehole_labels=['BH5','BH6'], lower_y_limit=-1.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger5,zero_deg_offsets_logger6])
chessjen.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/' , title='Chessjengletscher 08/2024-09/2024 calibrated'   , depths=[6.8,11.8,8.5,13.5]    , borehole_labels=['BH7','BH8'], lower_y_limit=-2.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger7,zero_deg_offsets_logger8])
alphubel.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/' , title='Alphubel South 08/2024-10/2024 calibrated'      , depths=[8.0,13.0,9.0,14.0]    , borehole_labels=['BH9','BH10'], lower_y_limit=-3.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger9,zero_deg_offsets_logger10])
corvatsch.plot_multiple_ntc_boreholes(savepath=output_path + 'icetemp_results/', title='Corvatsch 08/2024-10/2024 calibrated'           , depths=[2.0,7.0,4.3,9.3]      , borehole_labels=['BH11','BH12'], lower_y_limit=-3.5, calibrate=True, zero_deg_offsets=[zero_deg_offsets_logger11,zero_deg_offsets_logger12])

## Plot reliability experiments ##
# ------------------------------ #

# Experiment 1 ----------------- # 

# set the paths to the reliability experiment data
logger_13_dir_exp1 = cal_path + '/Experiment1_20250130/NTCs/13_ice_bath_rel_exp1.csv'
logger_14_dir_exp1 = cal_path + '/Experiment1_20250130/NTCs/14_ice_bath_rel_exp1.csv'
logger_15_dir_exp1 = cal_path + '/Experiment1_20250130/NTCs/15_ice_bath_rel_exp1.csv'
logger_16_dir_exp1 = cal_path + '/Experiment1_20250130/NTCs/16_ice_bath_rel_exp1.csv'

# create a ThermistorDataPlotter object per logger
logger_13_exp1 = ThermistorDataPlotter(logger_13_dir_exp1, delimiter=',')
logger_14_exp1 = ThermistorDataPlotter(logger_14_dir_exp1, delimiter=',')
logger_15_exp1 = ThermistorDataPlotter(logger_15_dir_exp1, delimiter=',')
logger_16_exp1 = ThermistorDataPlotter(logger_16_dir_exp1, delimiter=',')

# plot the data per logger and return the 0 degree offsets
zero_deg_offsets_logger13_exp1 = logger_13_exp1.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp1, savepath=output_path + 'thermistor_calibration/', title='Logger #13 - 0deg offset in ice bath - Exp1')
zero_deg_offsets_logger14_exp1 = logger_14_exp1.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp1, savepath=output_path + 'thermistor_calibration/', title='Logger #14 - 0deg offset in ice bath - Exp1')
zero_deg_offsets_logger15_exp1 = logger_15_exp1.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp1, savepath=output_path + 'thermistor_calibration/', title='Logger #15 - 0deg offset in ice bath - Exp1')
zero_deg_offsets_logger16_exp1 = logger_16_exp1.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp1, savepath=output_path + 'thermistor_calibration/', title='Logger #16 - 0deg offset in ice bath - Exp1')

# Experiment 2 ----------------- # 

# set the paths to the reliability experiment data
logger_13_dir_exp2 = cal_path + '/Experiment2_20250304/NTCs/13_ice_bath_rel_exp2.csv'
logger_14_dir_exp2 = cal_path + '/Experiment2_20250304/NTCs/14_ice_bath_rel_exp2.csv'
logger_15_dir_exp2 = cal_path + '/Experiment2_20250304/NTCs/15_ice_bath_rel_exp2.csv'
logger_16_dir_exp2 = cal_path + '/Experiment2_20250304/NTCs/16_ice_bath_rel_exp2.csv'

# create a ThermistorDataPlotter object per logger
logger_13_exp2 = ThermistorDataPlotter(logger_13_dir_exp2, delimiter=',')
logger_14_exp2 = ThermistorDataPlotter(logger_14_dir_exp2, delimiter=',')
logger_15_exp2 = ThermistorDataPlotter(logger_15_dir_exp2, delimiter=',')
logger_16_exp2 = ThermistorDataPlotter(logger_16_dir_exp2, delimiter=',')

# plot the data per logger and return the 0 degree offsets
zero_deg_offsets_logger13_exp2 = logger_13_exp2.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp2, savepath=output_path + 'thermistor_calibration/', title='Logger #13 - 0deg offset in ice bath - Exp2')
zero_deg_offsets_logger14_exp2 = logger_14_exp2.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp2, savepath=output_path + 'thermistor_calibration/', title='Logger #14 - 0deg offset in ice bath - Exp2')
zero_deg_offsets_logger15_exp2 = logger_15_exp2.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp2, savepath=output_path + 'thermistor_calibration/', title='Logger #15 - 0deg offset in ice bath - Exp2')  
zero_deg_offsets_logger16_exp2 = logger_16_exp2.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp2, savepath=output_path + 'thermistor_calibration/', title='Logger #16 - 0deg offset in ice bath - Exp2')

# Experiment 3 ----------------- #

# set the paths to the reliability experiment data
logger_13_dir_exp3 = cal_path + '/Experiment3_20250603/NTCs/13_ice_bath_rel_exp3.csv'
logger_14_dir_exp3 = cal_path + '/Experiment3_20250603/NTCs/14_ice_bath_rel_exp3.csv'
logger_15_dir_exp3 = cal_path + '/Experiment3_20250603/NTCs/15_ice_bath_rel_exp3.csv'
logger_16_dir_exp3 = cal_path + '/Experiment3_20250603/NTCs/16_ice_bath_rel_exp3.csv'

# create a ThermistorDataPlotter object per logger
logger_13_exp3 = ThermistorDataPlotter(logger_13_dir_exp3, delimiter=',')
logger_14_exp3 = ThermistorDataPlotter(logger_14_dir_exp3, delimiter=',')
logger_15_exp3 = ThermistorDataPlotter(logger_15_dir_exp3, delimiter=',')
logger_16_exp3 = ThermistorDataPlotter(logger_16_dir_exp3, delimiter=',')

# plot the data per logger and return the 0 degree offsets
zero_deg_offsets_logger13_exp3 = logger_13_exp3.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp3, savepath=output_path + 'thermistor_calibration/', title='Logger #13 - 0deg offset in ice bath - Exp3')
zero_deg_offsets_logger14_exp3 = logger_14_exp3.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp3, savepath=output_path + 'thermistor_calibration/', title='Logger #14 - 0deg offset in ice bath - Exp3') 
zero_deg_offsets_logger15_exp3 = logger_15_exp3.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp3, savepath=output_path + 'thermistor_calibration/', title='Logger #15 - 0deg offset in ice bath - Exp3')
zero_deg_offsets_logger16_exp3 = logger_16_exp3.plot_ntc_icebath_calibration(data_10m_chain_reliability_exp3, savepath=output_path + 'thermistor_calibration/', title='Logger #16 - 0deg offset in ice bath - Exp3')



def plot_icetemp_profile_piecewise(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    flip='auto',
    plot_contours=True,
    n_elev=300
):
    """
    Piecewise linear interpolation of ice temperature profile.

    Approach:
    - For each borehole build a 1D temperature-vs-depth (depth below local surface).
    - Project each borehole profile onto a single global elevation grid (so layers
      slope with the surface).
    - Linearly blend profiles horizontally between nearest boreholes.
    - Clamp extrapolation to nearest sensor values and fill fully-NaN columns
      from the nearest borehole to avoid gaps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import cmcrameri.cm as cmc

    ax = ax or plt.subplots(figsize=(10, 5), dpi=150)[1]

    # Required profile arrays
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Optionally flip profile direction to consistent left->right
    d = d0.copy()
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    # Plot surface and bed outlines
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Collect borehole locations and 1D (depth,temp) profiles (depth = below local surface)
    bh_locs = []
    bh_profiles = []  # (depths_array, temps_array, surface_elev)
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))

        # find nearest point along profile (if profile has x,y)
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if ('x' in profile_df and 'y' in profile_df) else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:, 0] - bh_x)**2 + (profile_xy[:, 1] - bh_y)**2)
            idx = np.argmin(dists)
            profile_dist = d[idx]
            surf_elev = z_s[idx]
        else:
            profile_dist = bh_x
            surf_elev = float(np.interp(profile_dist, d, z_s))

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            # Collect only sensors present in both dicts and convert to floats
            items = sorted([
                (float(depth), float(temps[probe]))
                for probe, depth in depths.items()
                if probe in temps and pd.notna(temps[probe])
            ])
            if not items:
                continue
            depths_arr = np.asarray([it[0] for it in items], dtype=float)
            temps_arr = np.asarray([it[1] for it in items], dtype=float)
            # ensure increasing depth order
            order = np.argsort(depths_arr)
            depths_arr = depths_arr[order]
            temps_arr = temps_arr[order]
            bh_locs.append(profile_dist)
            bh_profiles.append((depths_arr, temps_arr, float(surf_elev)))

            # plot borehole line and sensors
            min_elev = surf_elev - float(np.max(depths_arr))
            ax.plot([profile_dist, profile_dist], [min_elev, surf_elev], color='k', linewidth=1.5)
            for depth, temp in zip(depths_arr, temps_arr):
                ax.plot(profile_dist, surf_elev - depth, 'ko', markersize=4)
            ax.text(profile_dist, surf_elev + 6, name, color='red', fontsize=10, ha='center', va='bottom', zorder=12)

    if len(bh_locs) == 0:
        raise ValueError("No borehole profiles found in provided dicts.")

    # sort boreholes along profile
    bh_locs = np.array(bh_locs)
    sort_idx = np.argsort(bh_locs)
    bh_locs = bh_locs[sort_idx]
    bh_profiles = [bh_profiles[i] for i in sort_idx]

    # measured temperature bounds (for clipping/color scaling)
    measured_vals = np.hstack([p[1] for p in bh_profiles])
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Build global regular elevation grid (imshow requires rectangular grid)
    elev_min = float(np.nanmin(z_b)) - 1.0
    elev_max = float(np.nanmax(z_s)) + 1.0
    grid_elev = np.linspace(elev_min, elev_max, n_elev)  # ascending (bottom->top)
    grid_x = d
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_elev)  # grid_yy are elevations

    # Project each borehole's profile onto the global elevation grid.
    # depth_on_grid = surface_bh - elevation  (positive below surface)
    interp_profiles_on_elev = []
    for depths_arr, temps_arr, surf_elev in bh_profiles:
        depths_on_grid = surf_elev - grid_elev  # positive below surface
        # clamp extrapolation to nearest sensor (avoid runaway linear extrapolation)
        if depths_arr.size == 1:
            vals = np.full_like(depths_on_grid, temps_arr[0], dtype=float)
        else:
            vals = np.interp(depths_on_grid, depths_arr, temps_arr, left=temps_arr[0], right=temps_arr[-1])
        # values above the borehole surface are invalid -> NaN
        vals[grid_elev > surf_elev] = np.nan
        interp_profiles_on_elev.append(vals)
    interp_profiles_on_elev = np.array(interp_profiles_on_elev)  # shape (n_bh, n_elev)

    # Horizontal blending: for each distance column, interpolate between nearest boreholes
    grid_temp = np.full((grid_elev.size, grid_x.size), np.nan, dtype=float)
    for i, x in enumerate(grid_x):
        if x <= bh_locs[0]:
            grid_temp[:, i] = interp_profiles_on_elev[0]
        elif x >= bh_locs[-1]:
            grid_temp[:, i] = interp_profiles_on_elev[-1]
        else:
            right = np.searchsorted(bh_locs, x)
            left = right - 1
            x0, x1 = bh_locs[left], bh_locs[right]
            t0 = interp_profiles_on_elev[left]
            t1 = interp_profiles_on_elev[right]
            w = (x - x0) / (x1 - x0) if (x1 - x0) != 0 else 0.0
            blended = (1 - w) * t0 + w * t1
            both_nan = np.isnan(t0) & np.isnan(t1)
            blended[both_nan] = np.nan
            grid_temp[:, i] = blended

    # Fill entirely-NaN columns with nearest borehole profile (prevents gaps)
    for i, x in enumerate(grid_x):
        if np.all(np.isnan(grid_temp[:, i])):
            j = int(np.argmin(np.abs(bh_locs - x)))
            grid_temp[:, i] = interp_profiles_on_elev[j]

    # Mask above surface and below bed for each column
    for i, x in enumerate(grid_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = grid_elev > surf_at_x
        below = grid_elev < bed_at_x
        grid_temp[above, i] = np.nan
        grid_temp[below, i] = np.nan

    # Clip to measured bounds (avoid improbable extremes)
    grid_temp = np.where(np.isfinite(grid_temp), np.clip(grid_temp, meas_min, meas_max), np.nan)

    # Colormap: blue->...->red (preserve previous stylistic choice)
    base = cmc.vik(np.linspace(0, 0.6, 128))
    red = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([base, red])
    cmap_use = ListedColormap(colors)

    # Plot heatmap (distance x elevation)
    im = ax.imshow(
        grid_temp,
        extent=[grid_x.min(), grid_x.max(), grid_elev.min(), grid_elev.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use if cmap is None else cmap,
        vmin=meas_min if vmin is None else vmin,
        vmax=meas_max if vmax is None else vmax,
        alpha=0.85,
        zorder=0
    )

    # Contours (isotherms)
    if plot_contours:
        levels = np.linspace(meas_min, meas_max, 12)
        ax.contour(grid_xx, grid_yy, grid_temp, levels=levels, colors='k', linewidths=0.6, alpha=0.5)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    n_ticks = 6
    tick_values = np.linspace(meas_min, meas_max, n_ticks)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    plt.tight_layout()
    return ax.figure, ax

def plot_icetemp_profile(
    profile_df, borehole_coords_df, temp_data_dict, depth_dict, ax=None, title=None,
    cmap=None, vmin=None, vmax=None, flip='auto', depth_weight=3.0, n_elev=200
):
    """
    Plots glacier cross-section with borehole positions and interpolated temperature heatmap.
    Uses weighted RBF interpolation in (distance, elevation) space.
    """
    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Flip so it starts low and goes up (left->right)
    d = d0
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    # Plot surface and bed
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Plot borehole lines and sensors (unchanged)
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if 'x' in profile_df and 'y' in profile_df else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            profile_dist = d[profile_dist_idx]
            surface_elev = z_s[profile_dist_idx]
        else:
            profile_dist = bh_x  # fallback
            surface_elev = np.interp(profile_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    ax.plot(profile_dist, therm_elev, marker='o', color='k', markersize=4, zorder=11)
            min_elev = surface_elev - max(depths.values())
            max_elev = surface_elev
            ax.plot(
                [profile_dist, profile_dist],
                [min_elev, max_elev],
                color='k', linestyle='solid', alpha=1, zorder=5, linewidth=1.5
            )
            ax.text(profile_dist, surface_elev+6, name, color='red', fontsize=10, va='bottom', ha='center', zorder=12)

    # --- Use the new weighted RBF interpolation ---
    from processing.thermistor_processing import interpolate_temperature_weighted_rbf
    grid_temp, grid_y = interpolate_temperature_weighted_rbf(
        profile_distances=d,
        z_surf=z_s,
        z_bed=z_b,
        borehole_coords_df=borehole_coords_df,
        temp_data_dict=temp_data_dict,
        depth_dict=depth_dict,
        n_elev=n_elev,
        depth_weight=depth_weight,
        rbf_function='linear'
    )

    # measured bounds for clipping / color scaling
    measured_vals = []
    for name in temp_data_dict:
        measured_vals.extend(list(temp_data_dict[name].values))
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Colormap: blue->...->red (preserve previous stylistic choice)
    base = cmc.vik(np.linspace(0, 0.6, 128))
    red = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([base, red])
    cmap_use = ListedColormap(colors) if cmap is None else cmap

    im = ax.imshow(
        grid_temp,
        extent=[d.min(), d.max(), grid_y.min(), grid_y.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use,
        vmin=meas_min if vmin is None else vmin,
        vmax=meas_max if vmax is None else vmax,
        alpha=0.85,
        zorder=0
    )

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    n_ticks = 6
    tick_values = np.linspace(meas_min, meas_max, n_ticks)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    plt.tight_layout()
    return ax.figure, ax

def plot_icetemp_profile(
    profile_df, borehole_coords_df, temp_data_dict, depth_dict, ax=None, title=None,
    cmap=None, vmin=None, vmax=None, flip='auto', n_elev=200, rbf_function='linear'
):
    """
    Plots glacier cross-section with borehole positions and interpolated temperature heatmap.
    Uses RBF interpolation in (distance, elevation) space.
    Set flip=True to reverse the profile in x (distance) for visualization only.
    """
    from processing.thermistor_processing import interpolate_temperature_rbf

    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Interpolate temperature grid (always in original order)
    grid_temp, grid_x, grid_y = interpolate_temperature_rbf(
        profile_df, borehole_coords_df, temp_data_dict, depth_dict, n_elev=n_elev, rbf_function=rbf_function
    )

    # Flip so it starts low and goes up (left->right)
    d = d0
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True or flip == 'true':
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
        grid_temp = np.fliplr(grid_temp)

    # Plot surface and bed
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Plot borehole lines and sensors
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if 'x' in profile_df and 'y' in profile_df else None
        
        # flip profile_xy if profile flipped
        if flip == 'auto':
            if z_s[-1] < z_s[0] and profile_xy is not None:
                profile_xy = profile_xy[::-1]
        elif flip is True or flip == 'true':
            if profile_xy is not None:
                profile_xy = profile_xy[::-1]

        # Find nearest profile point for this borehole
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            plot_dist = d[profile_dist_idx]
            surface_elev = z_s[profile_dist_idx]
        else:
            plot_dist = bh_x
            surface_elev = np.interp(plot_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    ax.plot(plot_dist, therm_elev, marker='o', color='k', markersize=4, zorder=11)
            min_elev = surface_elev - max(depths.values())
            max_elev = surface_elev
            ax.plot(
                [plot_dist, plot_dist],
                [min_elev, max_elev],
                color='k', linestyle='solid', alpha=1, zorder=5, linewidth=1.5
            )
            ax.text(plot_dist, surface_elev+6, name, color='red', fontsize=10, va='bottom', ha='center', zorder=12)

    # measured bounds for clipping / color scaling
    measured_vals = []
    for name in temp_data_dict:
        measured_vals.extend(list(temp_data_dict[name].values))
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Colormap: blue->...->red (preserve previous stylistic choice)
    import cmcrameri.cm as cmc
    from matplotlib.colors import ListedColormap
    base = cmc.vik(np.linspace(0, 0.6, 128))
    red = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([base, red])
    cmap_use = ListedColormap(colors) if cmap is None else cmap

    im = ax.imshow(
        grid_temp,
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use,
        vmin=meas_min if vmin is None else vmin,
        vmax=meas_max if vmax is None else vmax,
        alpha=0.85,
        zorder=0
    )

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    n_ticks = 6
    tick_values = np.linspace(meas_min, meas_max, n_ticks)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    plt.tight_layout()
    return ax.figure, ax

def interpolate_temperature_bicubic(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    n_elev=200
):
    """
    Interpolate englacial temperature using bicubic interpolation in (distance, elevation) space.
    Fills outside convex hull with linear, then nearest interpolation.
    Returns: grid_temp (n_elev, n_x), grid_x (distance), grid_y (elevation)
    """
    import numpy as np
    from scipy.interpolate import griddata

    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    interp_points = []
    interp_temps = []
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if 'x' in profile_df and 'y' in profile_df else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            profile_dist = d[profile_dist_idx]
            surface_elev = z_s[profile_dist_idx]
        else:
            profile_dist = bh_x
            surface_elev = np.interp(profile_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    interp_points.append([profile_dist, therm_elev])
                    interp_temps.append(temps[probe])

    interp_points = np.array(interp_points)
    interp_temps = np.array(interp_temps)
    grid_x = d
    grid_y = np.linspace(z_b.min(), z_s.max(), n_elev)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Bicubic interpolation (griddata cubic)
    grid_temp = griddata(
        interp_points, interp_temps,
        (grid_xx, grid_yy),
        method='cubic'
    )

    # Fill NaNs (outside convex hull) with linear interpolation
    mask_nan = np.isnan(grid_temp)
    if np.any(mask_nan):
        grid_temp_lin = griddata(
            interp_points, interp_temps,
            (grid_xx, grid_yy),
            method='linear'
        )
        grid_temp[mask_nan] = grid_temp_lin[mask_nan]

    # Fill remaining NaNs with nearest-neighbor interpolation
    mask_nan = np.isnan(grid_temp)
    if np.any(mask_nan):
        grid_temp_nearest = griddata(
            interp_points, interp_temps,
            (grid_xx, grid_yy),
            method='nearest'
        )
        grid_temp[mask_nan] = grid_temp_nearest[mask_nan]

    # Mask grid_temp outside glacier body
    for i, x in enumerate(grid_x):
        bed = np.interp(x, d, z_b)
        surf = np.interp(x, d, z_s)
        for j, y in enumerate(grid_y):
            if not (bed < y < surf):
                grid_temp[j, i] = np.nan

    return grid_temp, grid_x, grid_y

def plot_icetemp_profile_piecewise(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    flip='auto',
    plot_contours=True,
    n_elev=300
):
    """
    Piecewise linear interpolation of ice temperature profile.

    Approach:
    - For each borehole build a 1D temperature-vs-depth (depth below local surface).
    - Project each borehole profile onto a single global elevation grid (so layers
      slope with the surface).
    - Linearly blend profiles horizontally between nearest boreholes.
    - Clamp extrapolation to nearest sensor values and fill fully-NaN columns
      from the nearest borehole to avoid gaps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import cmcrameri.cm as cmc

    ax = ax or plt.subplots(figsize=(10, 5), dpi=150)[1]

    # Required profile arrays
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Optionally flip profile direction to consistent left->right
    d = d0.copy()
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    # Plot surface and bed outlines
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Collect borehole locations and 1D (depth,temp) profiles (depth = below local surface)
    bh_locs = []
    bh_profiles = []  # (depths_array, temps_array, surface_elev)
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))

        # find nearest point along profile (if profile has x,y)
        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if ('x' in profile_df and 'y' in profile_df) else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:, 0] - bh_x)**2 + (profile_xy[:, 1] - bh_y)**2)
            idx = np.argmin(dists)
            profile_dist = d[idx]
            surf_elev = z_s[idx]
        else:
            profile_dist = bh_x
            surf_elev = float(np.interp(profile_dist, d, z_s))

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            # Collect only sensors present in both dicts and convert to floats
            items = sorted([
                (float(depth), float(temps[probe]))
                for probe, depth in depths.items()
                if probe in temps and pd.notna(temps[probe])
            ])
            if not items:
                continue
            depths_arr = np.asarray([it[0] for it in items], dtype=float)
            temps_arr = np.asarray([it[1] for it in items], dtype=float)
            # ensure increasing depth order
            order = np.argsort(depths_arr)
            depths_arr = depths_arr[order]
            temps_arr = temps_arr[order]
            bh_locs.append(profile_dist)
            bh_profiles.append((depths_arr, temps_arr, float(surf_elev)))

            # plot borehole line and sensors
            min_elev = surf_elev - float(np.max(depths_arr))
            ax.plot([profile_dist, profile_dist], [min_elev, surf_elev], color='k', linewidth=1.5)
            for depth, temp in zip(depths_arr, temps_arr):
                ax.plot(profile_dist, surf_elev - depth, 'ko', markersize=4)
            ax.text(profile_dist, surf_elev + 6, name, color='red', fontsize=10, ha='center', va='bottom', zorder=12)

    if len(bh_locs) == 0:
        raise ValueError("No borehole profiles found in provided dicts.")

    # sort boreholes along profile
    bh_locs = np.array(bh_locs)
    sort_idx = np.argsort(bh_locs)
    bh_locs = bh_locs[sort_idx]
    bh_profiles = [bh_profiles[i] for i in sort_idx]

    # measured temperature bounds (for clipping/color scaling)
    measured_vals = np.hstack([p[1] for p in bh_profiles])
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Build global regular elevation grid (imshow requires rectangular grid)
    elev_min = float(np.nanmin(z_b)) - 1.0
    elev_max = float(np.nanmax(z_s)) + 1.0
    grid_elev = np.linspace(elev_min, elev_max, n_elev)  # ascending (bottom->top)
    grid_x = d
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_elev)  # grid_yy are elevations

    # Project each borehole's profile onto the global elevation grid.
    # depth_on_grid = surface_bh - elevation  (positive below surface)
    interp_profiles_on_elev = []
    for depths_arr, temps_arr, surf_elev in bh_profiles:
        depths_on_grid = surf_elev - grid_elev  # positive below surface
        # clamp extrapolation to nearest sensor (avoid runaway linear extrapolation)
        if depths_arr.size == 1:
            vals = np.full_like(depths_on_grid, temps_arr[0], dtype=float)
        else:
            vals = np.interp(depths_on_grid, depths_arr, temps_arr, left=temps_arr[0], right=temps_arr[-1])
        # values above the borehole surface are invalid -> NaN
        vals[grid_elev > surf_elev] = np.nan
        interp_profiles_on_elev.append(vals)
    interp_profiles_on_elev = np.array(interp_profiles_on_elev)  # shape (n_bh, n_elev)

    # Horizontal blending: for each distance column, interpolate between nearest boreholes
    grid_temp = np.full((grid_elev.size, grid_x.size), np.nan, dtype=float)
    for i, x in enumerate(grid_x):
        if x <= bh_locs[0]:
            grid_temp[:, i] = interp_profiles_on_elev[0]
        elif x >= bh_locs[-1]:
            grid_temp[:, i] = interp_profiles_on_elev[-1]
        else:
            right = np.searchsorted(bh_locs, x)
            left = right - 1
            x0, x1 = bh_locs[left], bh_locs[right]
            t0 = interp_profiles_on_elev[left]
            t1 = interp_profiles_on_elev[right]
            w = (x - x0) / (x1 - x0) if (x1 - x0) != 0 else 0.0
            blended = (1 - w) * t0 + w * t1
            both_nan = np.isnan(t0) & np.isnan(t1)
            blended[both_nan] = np.nan
            grid_temp[:, i] = blended

    # Fill entirely-NaN columns with nearest borehole profile (prevents gaps)
    for i, x in enumerate(grid_x):
        if np.all(np.isnan(grid_temp[:, i])):
            j = int(np.argmin(np.abs(bh_locs - x)))
            grid_temp[:, i] = interp_profiles_on_elev[j]

    # Mask above surface and below bed for each column
    for i, x in enumerate(grid_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = grid_elev > surf_at_x
        below = grid_elev < bed_at_x
        grid_temp[above, i] = np.nan
        grid_temp[below, i] = np.nan

    # Clip to measured bounds (avoid improbable extremes)
    grid_temp = np.where(np.isfinite(grid_temp), np.clip(grid_temp, meas_min, meas_max), np.nan)

    # Colormap: blue->...->red (preserve previous stylistic choice)
    base = cmc.vik(np.linspace(0, 0.6, 128))
    red = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([base, red])
    cmap_use = ListedColormap(colors)

    # Plot heatmap (distance x elevation)
    im = ax.imshow(
        grid_temp,
        extent=[grid_x.min(), grid_x.max(), grid_elev.min(), grid_elev.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use if cmap is None else cmap,
        vmin=meas_min if vmin is None else vmin,
        vmax=meas_max if vmax is None else vmax,
        alpha=0.85,
        zorder=0
    )

    # Contours (isotherms)
    if plot_contours:
        levels = np.linspace(meas_min, meas_max, 12)
        ax.contour(grid_xx, grid_yy, grid_temp, levels=levels, colors='k', linewidths=0.6, alpha=0.5)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    n_ticks = 6
    tick_values = np.linspace(meas_min, meas_max, n_ticks)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc="best", ncol=1)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    plt.tight_layout()
    return ax.figure, ax

def plot_icetemp_profile_stratified(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    flip='auto',
    n_depth=200,
    n_elev=300,
    plot_contours=True
):
    """
    Stratified interpolation of englacial temperature relative to surface,
    with improved temperate layer handling (concentric isotherms).
    """

    ax = ax or plt.subplots(figsize=(10, 5), dpi=150)[1]

    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # optionally flip so left->right goes downhill->uphill consistently
    d = d0.copy()
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
    elif flip is True:
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]

    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # collect boreholes: location along profile and depth/temp arrays (depth=below local surface)
    bh_locs = []
    bh_depths_list = []
    bh_temps_list = []
    bh_surf = []

    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))

        profile_xy = np.column_stack([profile_df['x'], profile_df['y']]) if ('x' in profile_df and 'y' in profile_df) else None
        if profile_xy is not None:
            dists = np.hypot(profile_xy[:,0] - bh_x, profile_xy[:,1] - bh_y)
            idx = int(np.argmin(dists))
            loc = d[idx]
            surf_elev = float(z_s[idx])
        else:
            loc = float(bh_x)
            surf_elev = float(np.interp(loc, d, z_s))

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            items = sorted([(float(depth), float(temps[probe]))
                            for probe, depth in depths.items() if probe in temps and pd.notna(temps[probe])])
            if not items:
                continue
            depths_arr = np.array([it[0] for it in items], dtype=float)
            temps_arr = np.array([it[1] for it in items], dtype=float)
            order = np.argsort(depths_arr)
            depths_arr = depths_arr[order]
            temps_arr = temps_arr[order]

            bh_locs.append(loc)
            bh_depths_list.append(depths_arr)
            bh_temps_list.append(temps_arr)
            bh_surf.append(surf_elev)

            # plot borehole line & sensors
            min_elev = surf_elev - float(np.max(depths_arr))
            ax.plot([loc, loc], [min_elev, surf_elev], color='k', linewidth=1.2)
            for depth, temp in zip(depths_arr, temps_arr):
                ax.plot(loc, surf_elev - depth, 'ko', markersize=4)
            ax.text(loc, surf_elev + 6, name, color='red', fontsize=10, ha='center', va='bottom', zorder=12)

    if len(bh_locs) == 0:
        raise ValueError("No borehole profiles found in provided dicts.")

    # sort boreholes along profile
    bh_locs = np.array(bh_locs)
    sort_idx = np.argsort(bh_locs)
    bh_locs = bh_locs[sort_idx]
    bh_depths_list = [bh_depths_list[i] for i in sort_idx]
    bh_temps_list = [bh_temps_list[i] for i in sort_idx]
    bh_surf = np.array(bh_surf)[sort_idx]

    # --- Use the new stratified interpolation function ---
    grid_temp, grid_elev = interpolate_temperature_stratified(
        bh_locs, bh_surf, bh_depths_list, bh_temps_list,
        d, z_s, z_b, n_depth=n_depth, n_elev=n_elev
    )

    # measured bounds for clipping / color scaling
    measured_vals = np.hstack(bh_temps_list)
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # colormap
    base = cmc.vik(np.linspace(0, 0.6, 128))
    red = np.array(cmc.vik(1.0)).reshape(1, -1)
    colors = np.vstack([base, red])
    cmap_use = ListedColormap(colors) if cmap is None else cmap

    im = ax.imshow(
        grid_temp,
        extent=[d.min(), d.max(), grid_elev.min(), grid_elev.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use,
        vmin=meas_min if vmin is None else vmin,
        vmax=meas_max if vmax is None else vmax,
        alpha=0.85,
        zorder=0
    )

    # contours
    if plot_contours:
        levels = np.linspace(meas_min, meas_max, 10)
        grid_xx, grid_yy = np.meshgrid(d, grid_elev)
        ax.contour(grid_xx, grid_yy, grid_temp, levels=levels, colors='k', linewidths=0.6, alpha=0.5)

    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    tick_values = np.linspace(meas_min, meas_max, 6)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{v:.2f}" for v in tick_values])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)
    plt.tight_layout()
    return ax.figure, ax

def plot_icetemp_profile(
    profile_df, borehole_coords_df, temp_data_dict, depth_dict, ax=None, title=None,
    cmap=icetemp_cmap(), vmin=None, vmax=None, flip='auto', n_elev=200, rbf_function='linear'
):
    """
    Plots glacier cross-section with borehole positions and interpolated temperature heatmap.
    Uses RBF interpolation in (distance, elevation) space.
    Set flip=True to reverse the profile in x (distance) for visualization only.
    """
    from processing.thermistor_processing import interpolate_temperature_rbf

    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Flip logic: create a flipped copy for plotting/mapping
    d = d0
    profile_df_plot = profile_df.copy()
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
            profile_df_plot = profile_df.iloc[::-1].copy()
            profile_df_plot['distance'] = d
            profile_df_plot['zsurf'] = z_s
            profile_df_plot['zbed'] = z_b
    elif flip is True or flip == 'true':
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
        profile_df_plot = profile_df.iloc[::-1].copy()
        profile_df_plot['distance'] = d
        profile_df_plot['zsurf'] = z_s
        profile_df_plot['zbed'] = z_b

    # Interpolate temperature grid using the plotting profile
    grid_temp, grid_x, grid_y = interpolate_temperature_rbf(
        profile_df_plot, borehole_coords_df, temp_data_dict, depth_dict, n_elev=n_elev, rbf_function=rbf_function
    )

    # grid_temp, grid_x, grid_y = interpolate_temperature_kriging(
    #     profile_df_plot, borehole_coords_df, temp_data_dict, depth_dict, n_elev=n_elev, variogram_model='gaussian'
    # )

    # Enforce a minimum physical temperature gradient below the CTS
    grid_temp = enforce_minimum_temperature_gradient(grid_temp, grid_y, min_grad=0.02) 

    # Plot surface and bed
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Plot borehole lines and sensors
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df_plot['x'], profile_df_plot['y']]) if 'x' in profile_df_plot and 'y' in profile_df_plot else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            plot_dist = profile_df_plot['distance'].iloc[profile_dist_idx]
            surface_elev = profile_df_plot['zsurf'].iloc[profile_dist_idx]
        else:
            plot_dist = bh_x
            surface_elev = np.interp(plot_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    ax.plot(plot_dist, therm_elev, marker='o', color='k', markersize=4, zorder=11)
            min_elev = surface_elev - max(depths.values())
            max_elev = surface_elev
            ax.plot(
                [plot_dist, plot_dist],
                [min_elev, max_elev],
                color='k', linestyle='solid', alpha=1, zorder=5, linewidth=1.5
            )
            ax.text(plot_dist, surface_elev+6, name, color='red', fontsize=10, va='bottom', ha='center', zorder=12)

    # measured bounds for clipping / color scaling
    measured_vals = []
    for name in temp_data_dict:
        measured_vals.extend(list(temp_data_dict[name].values))
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Choose a nice step (e.g., 0.2°C)
    step = 0.1
    tick_start = np.floor(meas_min / step) * step
    tick_end = np.ceil(meas_max / step) * step

    # Ensure 0.0 is included as a level
    levels = np.arange(tick_start, tick_end + step/2, step)
    if 0.0 not in levels:
        levels = np.append(levels, 0.0)
    levels = np.unique(np.sort(levels))

    # Color map and normalization
    cmap_use = discrete_icetemp_cmap(levels)
    norm = BoundaryNorm(levels, cmap_use.N)

    norm = BoundaryNorm(levels, cmap_use.N)

    im = ax.imshow(
    grid_temp,
    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
    origin='lower',
    aspect='auto',
    cmap=cmap_use,
    norm=norm,
    alpha=0.85,
    zorder=0
    )

    # Create meshgrid for contours
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Contours
    ax.contour(
        grid_xx, grid_yy, grid_temp,
        levels=levels,
        colors='k',
        linewidths=0.6,
        alpha=0.5
    )
    # Highlight the 0°C isotherm (CTS) in red and thicker
    cts = ax.contour(
        grid_xx, grid_yy, grid_temp,
        levels=[0.0],
        colors='red',
        linewidths=2.2,
        alpha=0.95
    )

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    # If the last two ticks are both 0.0, remove one
    if len(levels) >= 2 and np.isclose(levels[-1], levels[-2]):
        levels = levels[:-1]
    cb.set_ticks(levels)
    cb.set_ticklabels([f"{v:.1f}" for v in levels])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)

    # Create a custom legend handle for the CTS
    cts_handle = Line2D([0], [0], color='red', linewidth=2.2, label='CTS (0°C isotherm)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cts_handle)
    labels.append('CTS (0°C isotherm)')
    ax.legend(handles, labels, frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc="best", ncol=1)


    plt.tight_layout()
    return ax.figure, ax

def plot_icetemp_profile_stepwise(
    profile_df,
    borehole_coords_df,
    temp_data_dict,
    depth_dict,
    ax=None,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    flip='auto',
    n_depth=200,
    n_elev=300,
    plot_contours=True
):
    """
    Plots glacier cross-section with borehole positions and interpolated temperature heatmap.
    Uses stepwise stratified interpolation: 1D along boreholes, then horizontal at constant depth below surface.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    import cmcrameri.cm as cmc

    ax = ax or plt.subplots(figsize=(10,5), dpi=150)[1]
    d0 = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()

    # Flip logic
    d = d0
    profile_df_plot = profile_df.copy()
    if flip == 'auto':
        if z_s[-1] < z_s[0]:
            z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
            profile_df_plot = profile_df.iloc[::-1].copy()
            profile_df_plot['distance'] = d
            profile_df_plot['zsurf'] = z_s
            profile_df_plot['zbed'] = z_b
    elif flip is True or flip == 'true':
        z_s = z_s[::-1]; z_b = z_b[::-1]; d = d0.max() - d0[::-1]
        profile_df_plot = profile_df.iloc[::-1].copy()
        profile_df_plot['distance'] = d
        profile_df_plot['zsurf'] = z_s
        profile_df_plot['zbed'] = z_b

    # Plot surface and bed
    ax.plot(d, z_s, color='k', linewidth=1.5, label='Surface')
    ax.plot(d, z_b, color='k', linewidth=1.5, linestyle='--', label='Bed')

    # Step 1: Interpolate each borehole profile to a common depth grid
    max_depth = np.nanmax([np.nanmax(list(depth_dict[name].values())) for name in depth_dict])
    depth_grid = np.linspace(0, max_depth, n_depth)
    bh_names, bh_locs, bh_surf, bh_temps = interpolate_borehole_profiles_to_depth_grid(
        borehole_coords_df, temp_data_dict, depth_dict, depth_grid
    )

    # Step 2: For each depth, interpolate horizontally between boreholes
    profile_x = d
    grid_temp_depth_x = horizontal_interpolation_at_each_depth(profile_x, bh_locs, bh_temps)

    # Step 3: Convert (distance, depth_below_surface) to (distance, elevation)
    z_s_interp = np.interp(profile_x, d, z_s)
    grid_elev = depth_grid_to_elevation_grid(profile_x, z_s_interp, depth_grid)

    # Step 4: Resample to regular elevation grid for plotting
    elev_min = float(np.nanmin(z_b)) - 1.0
    elev_max = float(np.nanmax(z_s)) + 1.0
    elev_grid = np.linspace(elev_min, elev_max, n_elev)
    grid_temp_elev = resample_to_regular_elevation_grid(grid_temp_depth_x, grid_elev, elev_grid)

    # Mask above surface and below bed for each column
    for i, x in enumerate(profile_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = elev_grid > surf_at_x
        below = elev_grid < bed_at_x
        grid_temp_elev[above, i] = np.nan
        grid_temp_elev[below, i] = np.nan

    # measured bounds for clipping / color scaling
    measured_vals = np.hstack(bh_temps)
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))

    # Choose a nice step (e.g., 0.2°C)
    step = 0.1
    tick_start = np.floor(meas_min / step) * step
    tick_end = np.ceil(meas_max / step) * step

    # Ensure 0.0 is included as a level
    levels = np.arange(tick_start, tick_end + step/2, step)
    if 0.0 not in levels:
        levels = np.append(levels, 0.0)
    levels = np.unique(np.sort(levels))

    # Color map and normalization (stepwise blue-to-red)
    cmap_use = discrete_icetemp_cmap(levels) if cmap is None else cmap
    norm = BoundaryNorm(levels, cmap_use.N)

    # Plot heatmap (distance x elevation)
    im = ax.imshow(
        grid_temp_elev,
        extent=[profile_x.min(), profile_x.max(), elev_grid.min(), elev_grid.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap_use,
        norm=norm,
        alpha=0.85,
        zorder=0
    )

    # Contours (isotherms)
    grid_xx, grid_yy = np.meshgrid(profile_x, elev_grid)
    if plot_contours:
        ax.contour(grid_xx, grid_yy, grid_temp_elev, levels=levels, colors='k', linewidths=0.6, alpha=0.5)
    # Highlight the 0°C isotherm (CTS) in red and thicker
    cts = ax.contour(
        grid_xx, grid_yy, grid_temp_elev,
        levels=[0.0],
        colors='red',
        linewidths=2.2,
        alpha=0.95
    )

    # Plot borehole lines and sensors (projected onto profile)
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        profile_xy = np.column_stack([profile_df_plot['x'], profile_df_plot['y']]) if 'x' in profile_df_plot and 'y' in profile_df_plot else None
        if profile_xy is not None:
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            profile_dist_idx = np.argmin(dists)
            plot_dist = profile_df_plot['distance'].iloc[profile_dist_idx]
            surface_elev = profile_df_plot['zsurf'].iloc[profile_dist_idx]
        else:
            plot_dist = bh_x
            surface_elev = np.interp(plot_dist, d, z_s)

        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            for probe, depth in depths.items():
                if probe in temps:
                    therm_elev = surface_elev - depth
                    ax.plot(plot_dist, therm_elev, marker='o', color='k', markersize=4, zorder=11)
            min_elev = surface_elev - max(depths.values())
            max_elev = surface_elev
            ax.plot(
                [plot_dist, plot_dist],
                [min_elev, max_elev],
                color='k', linestyle='solid', alpha=1, zorder=5, linewidth=1.5
            )
            ax.text(plot_dist, surface_elev+6, name, color='red', fontsize=10, va='bottom', ha='center', zorder=12)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label='Ice Temperature [°C]')
    # If the last two ticks are both 0.0, remove one
    if len(levels) >= 2 and np.isclose(levels[-1], levels[-2]):
        levels = levels[:-1]
    cb.set_ticks(levels)
    cb.set_ticklabels([f"{v:.1f}" for v in levels])

    ax.set_ylim(np.min(z_b) - 2, np.max(z_s) + 2)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    format_plot(ax=ax, title=title, x_tick_rotation=0, legend_loc='upper left', adjust_linewidths=False)

    # Create a custom legend handle for the CTS
    cts_handle = Line2D([0], [0], color='red', linewidth=2.2, label='CTS (0°C isotherm)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cts_handle)
    labels.append('CTS (0°C isotherm)')
    ax.legend(handles, labels, frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc="best", ncol=1)

    plt.tight_layout()
    return ax.figure, ax

def apply_glacial_temperature_constraints(grid_temp_elev, elev_grid, profile_x, profile_df, geothermal_gradient=0.025):
    """
    Apply realistic glacial temperature constraints:
    - Temperature increases with depth (geothermal)
    - No temperature > 0°C (pressure melting point)
    - Smooth transitions at CTS
    """
    import numpy as np
    
    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy()
    z_b = profile_df['zbed'].to_numpy()
    
    for i, x in enumerate(profile_x):
        surf_elev = np.interp(x, d, z_s)
        bed_elev = np.interp(x, d, z_b)
        
        # Get column temperatures
        col_temps = grid_temp_elev[:, i].copy()
        
        # Apply constraints from surface to bed
        for j, elev in enumerate(elev_grid):
            if bed_elev <= elev <= surf_elev:
                depth_below_surface = surf_elev - elev
                
                # Apply gentle geothermal warming
                geothermal_warming = geothermal_gradient * depth_below_surface
                
                # Ensure no temperatures > 0°C
                col_temps[j] = min(col_temps[j], 0.0)
                
                # Apply minimum warming trend (but don't force it above 0°C)
                expected_temp = -10.0 + geothermal_warming
                if col_temps[j] < expected_temp:
                    col_temps[j] = min(expected_temp, 0.0)
        
        grid_temp_elev[:, i] = col_temps
    
    return grid_temp_elev

def interpolate_glacier_temperature_field_cts_focus(
    profile_df,
    borehole_coords_df, 
    temp_data_dict,
    depth_dict,
    n_depth=400,
    n_elev=600,
    depth_weight=3.0,
    rbf_function='linear',
    geothermal_gradient=0.025
):
    """
    CTS-focused temperature interpolation with high resolution and physics constraints.
    Uses thin-plate spline for smoother CTS detection and adds bed temperature estimates.
    """
    import numpy as np
    from scipy.interpolate import Rbf
    from scipy.ndimage import gaussian_filter
    
    # Get profile geometry
    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy() 
    z_b = profile_df['zbed'].to_numpy()
    
    # Collect measurement points and add bed temperature estimates
    points_2d = []
    temps_2d = []
    
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        bh_x = float(str(bh_row['x']).replace(',', '.'))
        bh_y = float(str(bh_row['y']).replace(',', '.'))
        
        # Find borehole position along profile
        if 'x' in profile_df and 'y' in profile_df:
            profile_xy = np.column_stack([profile_df['x'], profile_df['y']])
            dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
            idx = np.argmin(dists)
            bh_distance = d[idx]
            surf_elev = z_s[idx]
            bed_elev = z_b[idx]
        else:
            bh_distance = bh_x
            surf_elev = np.interp(bh_distance, d, z_s)
            bed_elev = np.interp(bh_distance, d, z_b)
        
        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            
            # Add measured thermistor points
            for probe, depth in depths.items():
                if probe in temps and np.isfinite(temps[probe]):
                    elev = surf_elev - depth
                    points_2d.append([bh_distance, elev * depth_weight])
                    temps_2d.append(temps[probe])
            
            # Add estimated bed temperature using geothermal gradient
            ice_thickness = surf_elev - bed_elev
            estimated_bed_temp = -10.0 + (geothermal_gradient * ice_thickness)
            estimated_bed_temp = min(estimated_bed_temp, 0.0)  # Cap at 0°C
            points_2d.append([bh_distance, bed_elev * depth_weight])
            temps_2d.append(estimated_bed_temp)
    
    if len(points_2d) < 4:
        raise ValueError("Need at least 4 temperature measurements for CTS-focused interpolation")
    
    points_2d = np.array(points_2d)
    temps_2d = np.array(temps_2d)
    
    # Create high-resolution elevation grid
    elev_min = float(np.nanmin(z_b)) - 2.0
    elev_max = float(np.nanmax(z_s)) + 2.0
    elev_grid = np.linspace(elev_min, elev_max, n_elev)
    
    # Create grid points for interpolation
    profile_x = d
    grid_xx, grid_yy = np.meshgrid(profile_x, elev_grid)
    
    # High-resolution RBF interpolation with thin-plate spline for smooth CTS
    rbf = Rbf(
        points_2d[:, 0], points_2d[:, 1], temps_2d, 
        function=rbf_function,  # 'thin_plate' for smoothest CTS
        smooth=0.05  # Less smoothing for better CTS detection
    )
    
    # Interpolate on the grid
    grid_temp_elev = rbf(grid_xx, grid_yy * depth_weight)
    
    # Apply glacial temperature constraints
    grid_temp_elev = apply_glacial_temperature_constraints(
        grid_temp_elev, elev_grid, profile_x, profile_df, geothermal_gradient
    )
    
    # Mask outside glacier body
    for i, x in enumerate(profile_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = elev_grid > surf_at_x
        below = elev_grid < bed_at_x
        grid_temp_elev[above, i] = np.nan
        grid_temp_elev[below, i] = np.nan
    
    return grid_temp_elev, elev_grid, profile_x

def interpolate_glacier_temperature_field(
    profile_df,
    borehole_coords_df, 
    temp_data_dict,
    depth_dict,
    n_depth=200,
    n_elev=300
):
    """
    Interpolates glacier temperature field using stratified approach:
    1. Interpolate each borehole 1D profile to fill gaps between sensors
    2. Interpolate horizontally between boreholes at constant depth-below-surface
    3. Convert to elevation grid for plotting
    
    This respects the glacier's natural temperature structure controlled by depth.
    """
    import numpy as np
    
    # Get profile geometry
    d = profile_df['distance'].to_numpy()
    z_s = profile_df['zsurf'].to_numpy() 
    z_b = profile_df['zbed'].to_numpy()
    
    # Step 1: Create common depth grid (depth below surface)
    max_depth = np.nanmax([np.nanmax(list(depth_dict[name].values())) for name in depth_dict])
    depth_grid = np.linspace(0, max_depth, n_depth)
    
    # Step 2: Interpolate each borehole profile to fill gaps between sensors
    bh_names, bh_locs, bh_surf, bh_temp_profiles = interpolate_borehole_profiles_to_depth_grid(
        borehole_coords_df, temp_data_dict, depth_dict, depth_grid
    )
    
    # Step 3: For each depth level, interpolate horizontally between boreholes
    profile_x = d
    grid_temp_depth_x = interpolate_between_boreholes_stratified(
        profile_x, bh_locs, bh_temp_profiles
    )
    
    # Step 4: Convert from (distance, depth_below_surface) to (distance, elevation)
    z_s_interp = np.interp(profile_x, d, z_s)
    grid_elev = depth_grid_to_elevation_grid(profile_x, z_s_interp, depth_grid)
    
    # Step 5: Resample to regular elevation grid for plotting
    elev_min = np.nanmin(grid_elev)
    elev_max = np.nanmax(grid_elev)  
    elev_grid = np.linspace(elev_min, elev_max, n_elev)
    grid_temp_elev = resample_to_regular_elevation_grid(
        grid_temp_depth_x, grid_elev, elev_grid
    )
    
    # Step 6: Mask outside glacier body
    for i, x in enumerate(profile_x):
        surf_at_x = float(np.interp(x, d, z_s))
        bed_at_x = float(np.interp(x, d, z_b))
        above = elev_grid > surf_at_x
        below = elev_grid < bed_at_x
        grid_temp_elev[above, i] = np.nan
        grid_temp_elev[below, i] = np.nan
    
    return grid_temp_elev, elev_grid, profile_x

def interpolate_borehole_profiles_to_depth_grid(borehole_coords_df, temp_data_dict, depth_dict, depth_grid):
    """
    For each borehole, interpolate its temperature profile onto a common depth grid below surface.
    Returns:
        bh_names: list of borehole names
        bh_locs: array of borehole distances along profile
        bh_surf: array of surface elevations at borehole locations
        bh_temps: array of shape (n_bh, n_depth)
    """
    import numpy as np

    bh_names = []
    bh_locs = []
    bh_surf = []
    bh_temps = []
    for _, bh_row in borehole_coords_df.iterrows():
        name = bh_row['name']
        if name in temp_data_dict and name in depth_dict:
            temps = temp_data_dict[name]
            depths = depth_dict[name]
            items = sorted([(float(depth), float(temps[probe])) for probe, depth in depths.items() if probe in temps])
            if not items:
                continue
            d_arr, t_arr = zip(*items)
            d_arr = np.array(d_arr)
            t_arr = np.array(t_arr)
            t_interp = np.interp(depth_grid, d_arr, t_arr, left=t_arr[0], right=t_arr[-1])
            bh_names.append(name)
            # Find borehole location along profile and surface elevation
            bh_x = float(str(bh_row['x']).replace(',', '.'))
            bh_y = float(str(bh_row['y']).replace(',', '.'))
            profile_xy = np.column_stack([borehole_coords_df['x'], borehole_coords_df['y']]) if ('x' in borehole_coords_df and 'y' in borehole_coords_df) else None
            if profile_xy is not None:
                dists = np.sqrt((profile_xy[:,0] - bh_x)**2 + (profile_xy[:,1] - bh_y)**2)
                idx = np.argmin(dists)
                loc = bh_row['distance'] if 'distance' in bh_row else bh_x
                surf_elev = bh_row['zsurf'] if 'zsurf' in bh_row else np.nan
            else:
                loc = bh_x
                surf_elev = np.nan
            bh_locs.append(loc)
            bh_surf.append(surf_elev)
            bh_temps.append(t_interp)
    return bh_names, np.array(bh_locs), np.array(bh_surf), np.array(bh_temps)

def horizontal_interpolation_at_each_depth(profile_x, bh_x, bh_temps, method='idw'):
    """
    For each depth, interpolate horizontally between boreholes.
    profile_x: 1D array of distances along profile
    bh_x: 1D array of borehole distances along profile
    bh_temps: 2D array (n_bh, n_depth)
    Returns: grid_temp (n_depth, n_profile_x)
    """
    import numpy as np

    n_bh, n_depth = bh_temps.shape
    n_x = len(profile_x)
    grid_temp = np.full((n_depth, n_x), np.nan)
    for j in range(n_depth):
        vals = bh_temps[:, j]
        for i, x in enumerate(profile_x):
            dists = np.abs(bh_x - x)
            dists[dists == 0] = 1e-6
            weights = 1 / dists
            weights /= weights.sum()
            grid_temp[j, i] = np.sum(weights * vals)
    return grid_temp

def depth_grid_to_elevation_grid(profile_x, z_surf, depth_grid):
    """
    For each profile_x, convert depth below surface to elevation.
    Returns: grid_elev (n_depth, n_profile_x)
    """
    import numpy as np
    grid_elev = np.zeros((len(depth_grid), len(profile_x)))
    for i, x in enumerate(profile_x):
        surf = z_surf[i]
        grid_elev[:, i] = surf - depth_grid
    return grid_elev

def resample_to_regular_elevation_grid(grid_temp, grid_elev, elev_grid):
    """
    For each profile_x, resample the temperature profile onto a regular elevation grid.
    Returns: grid_temp_elev (n_elev, n_profile_x)
    """
    import numpy as np
    n_x = grid_temp.shape[1]
    n_elev = len(elev_grid)
    grid_temp_elev = np.full((n_elev, n_x), np.nan)
    for i in range(n_x):
        elev_col = grid_elev[:, i]
        temp_col = grid_temp[:, i]
        order = np.argsort(elev_col)
        grid_temp_elev[:, i] = np.interp(elev_grid, elev_col[order], temp_col[order], left=np.nan, right=np.nan)
    return grid_temp_elev

def interpolate_temperature_stratified(
    bh_locs, bh_surf, bh_depths_list, bh_temps_list,
    d, z_s, z_b,
    n_depth=200, n_elev=300
):
    """
    Interpolates temperature in a stratified (depth-below-surface) manner along a profile.
    Uses inverse distance weighting (IDW) for smooth horizontal interpolation.
    Returns: grid_temp (n_elev, n_x), grid_elev (n_elev,)
    """
    import numpy as np

    # 1. Build a uniform depth grid below surface (0 = surface)
    max_depth = max([np.max(darr) for darr in bh_depths_list])
    depth_grid = np.linspace(0.0, float(max_depth), n_depth)  # meters below local surface

    # 2. Interpolate each borehole profile onto depth_grid (clamp outside to nearest sensor value)
    bh_on_depth = []
    for depths_arr, temps_arr in zip(bh_depths_list, bh_temps_list):
        if depths_arr.size == 1:
            vals = np.full_like(depth_grid, float(temps_arr[0]), dtype=float)
        else:
            vals = np.interp(depth_grid, depths_arr, temps_arr, left=temps_arr[0], right=temps_arr[-1])
        bh_on_depth.append(vals)
    bh_on_depth = np.vstack(bh_on_depth)  # shape (n_bh, n_depth)

    # 3. Horizontally interpolate for each depth level across boreholes using IDW
    grid_x = d
    n_x = grid_x.size
    grid_temp_depth_x = np.full((n_depth, n_x), np.nan, dtype=float)
    for j, depth_val in enumerate(depth_grid):
        t_bh = bh_on_depth[:, j]
        # Inverse distance weighting (IDW)
        for ix, x in enumerate(grid_x):
            dists = np.abs(bh_locs - x)
            # Avoid division by zero
            dists[dists == 0] = 1e-6
            weights = 1 / dists
            weights /= weights.sum()
            grid_temp_depth_x[j, ix] = np.sum(weights * t_bh)

    # 4. Convert depth-grid -> global elevation grid for plotting:
    elev_min = float(np.nanmin(z_b)) - 1.0
    elev_max = float(np.nanmax(z_s)) + 1.0
    grid_elev = np.linspace(elev_min, elev_max, n_elev)  # ascending bottom->top
    grid_temp = np.full((n_elev, n_x), np.nan, dtype=float)

    # 5. For each column, compute elevation positions of depth_grid and resample onto grid_elev
    for i in range(n_x):
        surf_i = float(np.interp(grid_x[i], d, z_s))
        elev_at_depth = surf_i - depth_grid
        order = np.argsort(elev_at_depth)
        elev_sorted = elev_at_depth[order]
        temp_sorted = grid_temp_depth_x[:, i][order]
        col_vals = np.interp(grid_elev, elev_sorted, temp_sorted, left=np.nan, right=np.nan)
        bed_i = float(np.interp(grid_x[i], d, z_b))
        col_vals[grid_elev > surf_i] = np.nan
        col_vals[grid_elev < bed_i] = np.nan
        grid_temp[:, i] = col_vals
        # Fill below deepest thermistor with its value (down to bed)
        depths_arr = bh_depths_list[np.argmin(np.abs(bh_locs - grid_x[i]))]
        temps_arr = bh_temps_list[np.argmin(np.abs(bh_locs - grid_x[i]))]
        if len(depths_arr) > 0:
            deepest_depth = np.max(depths_arr)
            deepest_temp = temps_arr[np.argmax(depths_arr)]
            deepest_elev = surf_i - deepest_depth
            mask = (grid_elev < deepest_elev) & (grid_elev >= bed_i)
            grid_temp[mask, i] = deepest_temp

    # 6. Improved temperate layer: interpolate depth to 0°C isotherm and fill below with 0°C
    temp_base_depths = []
    for depths_arr, temps_arr in zip(bh_depths_list, bh_temps_list):
        # Interpolate to find depth where T=0°C
        if np.any(temps_arr == 0.0):
            idx = np.where(temps_arr == 0.0)[0][0]
            temp_base_depths.append(depths_arr[idx])
        elif np.any(temps_arr > 0.0) and np.any(temps_arr < 0.0):
            idx = np.where(temps_arr < 0.0)[0][-1]
            d0, d1 = depths_arr[idx], depths_arr[idx+1]
            t0, t1 = temps_arr[idx], temps_arr[idx+1]
            d_zero = d0 + (0.0 - t0) * (d1 - d0) / (t1 - t0)
            temp_base_depths.append(d_zero)
        else:
            temp_base_depths.append(np.max(depths_arr))  # fallback: deepest sensor
    temp_base_depths = np.array(temp_base_depths)
    temp_base_depths_interp = np.interp(grid_x, bh_locs, temp_base_depths)
    # --- Add smoothing here ---
    temp_base_depths_interp_smooth = gaussian_filter1d(temp_base_depths_interp, sigma=2)

    transition_thickness = 2.0  # meters

    for i in range(n_x):
        surf_i = float(np.interp(grid_x[i], d, z_s))
        bed_i = float(np.interp(grid_x[i], d, z_b))
        temp_base_elev = surf_i - temp_base_depths_interp_smooth[i]
        # Clamp the 0°C isotherm between bed and surface
        temp_base_elev = np.clip(temp_base_elev, bed_i, surf_i)
        for j, elev in enumerate(grid_elev):
            # If below the 0°C isotherm, assign 0°C (temperate)
            if bed_i <= elev < temp_base_elev:
                grid_temp[j, i] = 0.0
            # If within the transition zone, blend
            elif temp_base_elev <= elev < temp_base_elev + transition_thickness and elev < surf_i:
                alpha = (elev - temp_base_elev) / transition_thickness
                grid_temp[j, i] = (1 - alpha) * 0.0 + alpha * grid_temp[j, i]
            # If above the surface, keep as NaN (already set)

    # 7. Fill fully-NaN columns with nearest borehole-derived column (prevents gaps)
    for i in range(n_x):
        if np.all(np.isnan(grid_temp[:, i])):
            j_near = int(np.argmin(np.abs(bh_locs - grid_x[i])))
            surf_bh = bh_surf[j_near]
            elev_at_depth_bh = surf_bh - depth_grid
            order = np.argsort(elev_at_depth_bh)
            elev_sorted = elev_at_depth_bh[order]
            temp_sorted = bh_on_depth[j_near, :][order]
            col_vals = np.interp(grid_elev, elev_sorted, temp_sorted, left=np.nan, right=np.nan)
            bed_i = float(np.interp(grid_x[i], d, z_b))
            col_vals[grid_elev > np.interp(grid_x[i], d, z_s)] = np.nan
            col_vals[grid_elev < bed_i] = np.nan
            grid_temp[:, i] = col_vals

    # 8. Clip to measured bounds (avoid artificial extremes)
    measured_vals = np.hstack(bh_temps_list)
    meas_min = float(np.nanmin(measured_vals))
    meas_max = float(np.nanmax(measured_vals))
    grid_temp = np.where(np.isfinite(grid_temp), np.clip(grid_temp, meas_min, meas_max), np.nan)

    return grid_temp, grid_elev

def enforce_minimum_temperature_gradient(grid_temp, grid_y, min_grad=0.02):
    """
    Enforce a minimum physical temperature gradient everywhere in the ice where T < 0°C.
    This works for any stratigraphy (multiple CTS, temperate layers above cold, etc).

    Parameters
    ----------
    grid_temp : 2D np.ndarray
        Temperature grid (shape: n_elev, n_x).
    grid_y : 1D np.ndarray
        Elevation values (length: n_elev).
    min_grad : float
        Minimum allowed temperature gradient in °C per meter (default: 0.02).

    Returns
    -------
    grid_temp_new : 2D np.ndarray
        Temperature grid with enforced minimum gradient in all cold ice.
    """
    grid_temp_new = grid_temp.copy()
    n_elev, n_x = grid_temp.shape

    for i in range(n_x):
        col = grid_temp_new[:, i]
        # Traverse from bottom to top (increasing elevation)
        for j in range(n_elev-2, -1, -1):
            dz = grid_y[j+1] - grid_y[j]
            # Only enforce if both current and next are < 0°C (cold ice)
            if col[j] < 0.0 and col[j+1] < 0.0:
                min_allowed = col[j+1] - min_grad * abs(dz)
                if col[j] < min_allowed:
                    col[j] = min_allowed
            # If next is temperate, clamp to 0°C
            elif col[j+1] >= 0.0 and col[j] < 0.0:
                max_allowed = 0.0 - min_grad * abs(dz)
                if col[j] < max_allowed:
                    col[j] = max_allowed
        grid_temp_new[:, i] = col

    return grid_temp_new

def interpolate_thickness_to_grid(points_gdf, value_col='thickness', pixel_size=20.0, rbf_function='linear', polygon_mask: gpd.GeoDataFrame|None=None, padding=0.0):
    """
    Interpolate scattered ice thickness points to a regular grid using RBF.
    rbf_function: 'linear' | 'multiquadric' | 'inverse' | 'gaussian' | 'thin_plate' | 'cubic' | 'quintic'
    polygon_mask: optional glacier outline (same CRS) to mask outside to NaN.
    Returns grid, transform, crs
    """
    crs = points_gdf.crs
    X = np.array([p.x for p in points_gdf.geometry])
    Y = np.array([p.y for p in points_gdf.geometry])
    Z = points_gdf[value_col].values

    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    xmin -= padding; ymin -= padding; xmax += padding; ymax += padding
    width = int(np.ceil((xmax - xmin) / pixel_size)) + 1
    height = int(np.ceil((ymax - ymin) / pixel_size)) + 1

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmin + pixel_size*(width-1), width),
        np.linspace(ymax, ymax - pixel_size*(height-1), height)
    )

    # RBF interpolation (extrapolates smoothly)
    rbf = Rbf(X, Y, Z, function=rbf_function)
    grid = rbf(grid_x, grid_y)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    if polygon_mask is not None and not polygon_mask.empty:
        mask = geometry_mask(
            [geom for geom in polygon_mask.geometry],
            out_shape=grid.shape,
            transform=transform,
            invert=True
        )
        grid = np.where(mask, grid, np.nan)

    return grid, transform, crs