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