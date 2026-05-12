import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib import dates as mdates
import cmcrameri.cm as cmc
from scipy.ndimage import gaussian_filter  # used in heatmap smoothing

# Import processing pieces (works with or without package context)
from processing.thermistor_processing import *

# -----------------------------------------------------------------------------
# ThermistorDataPlotter and other plotting helpers
# -----------------------------------------------------------------------------

class ThermistorDataPlotter:
    """
        Class to plot thermistor data.
    """
    def __init__(self, file_path, measurement_depth=None, delimiter=','):
        if isinstance(file_path, list):
            self.file_paths = file_path
            self.file_path = file_path[0]
        else:
            self.file_paths = [file_path]
            self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth

    def format_plot(self, title, legend_loc='upper right', xtick_rotation=45, show_legend=True, base_fontsize=None):
        ax = plt.gca()
        fig = plt.gcf()
        # Use absolute, uniform fontsize for all figures (better for mosaics)
        bf = 26 if base_fontsize is None else float(base_fontsize)
        self.fontsize = int(round(bf))
        self.linewidth = max(2.0, bf / 10.0)

        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.size'] = self.fontsize
        plt.rcParams['axes.titlesize'] = self.fontsize
        plt.rcParams['axes.labelsize'] = self.fontsize
        plt.rcParams['xtick.labelsize'] = self.fontsize
        plt.rcParams['ytick.labelsize'] = self.fontsize
        plt.rcParams['legend.fontsize'] = self.fontsize
        plt.rcParams['lines.linewidth'] = self.linewidth

        ax.set_xlabel(ax.get_xlabel(), fontsize=self.fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.fontsize)
        ax.set_title(title if title else '', fontsize=self.fontsize)
        ax.tick_params(axis='both', labelsize=self.fontsize)
        for line in ax.get_lines():
            line.set_linewidth(self.linewidth)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) == 1:
                pass
            elif len(labels) > 4:
                ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                          loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=self.fontsize)
            else:
                ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                          loc=legend_loc, ncol=1, fontsize=self.fontsize)
        plt.xticks(rotation=xtick_rotation, fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.grid()
        plt.tight_layout()

    def plot_full_geoprecision_chain(self, start_time, end_time, offsets, savepath, title=None, depth_file=None):
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        data = thermistor.get_chain_data_with_offsets(start_time, end_time, offsets)
        data['TIME'] = pd.to_datetime(data['TIME'])

        # Load depths if provided (use the same robust parser as elsewhere)
        depths = {}
        if depth_file is not None:
            try:
                depths = read_thermistor_depths(depth_file)  # returns dict like {'#1': 5.3, ...}
            except Exception as e:
                print(f"Warning: failed to read depths from {depth_file}: {e}")
                depths = {}

        plt.figure(figsize=(12, 8), dpi=250)
        exclude_cols = ['NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V']
        plot_columns = [col for col in data.columns if col not in exclude_cols]
        n_cols = len(plot_columns)
        # Use cmc.grayC but skip the brightest color (index 0)
        colors = cmc.grayC(np.linspace(0.9, 0, n_cols))

        for i, column in enumerate(plot_columns):
            depth_val = depths.get(column, None)
            # Label with depth if available
            label = f"{depth_val:.1f} m" if isinstance(depth_val, (int, float)) and depth_val is not None else ""
            plt.plot(data['TIME'], pd.to_numeric(data[column], errors='coerce'), label=label, color=colors[i])

        # Use the format_plot method for consistent styling
        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.axhline(y=0, color='k', linestyle='--')
        self.format_plot(title, legend_loc='lower left')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.savefig(savepath)

    def plot_temperature_profile(self, snapshot_time, offsets, depth_file, savepath, title=None):
        """
        Plot a single chain's temperature profile as the DAILY MEAN for snapshot_time.
        Depths are loaded via read_thermistor_depths for consistency.
        """
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
        day_df = thermistor.get_chain_data_with_offsets(offsets=offsets, snapshot_day=snapshot_time)
        if day_df is None or day_df.empty:
            raise ValueError("No chain data available for the requested day.")

        # Parse depths consistently
        try:
            depths_dict = read_thermistor_depths(depth_file)  # e.g., {'#1': 5.3, ...}
        except Exception as e:
            raise ValueError(f"Failed to read depths from {depth_file}: {e}")

        # Compute daily mean per thermistor using the depth keys
        exclude_cols = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}
        depths_list, mean_temps = [], []
        for sensor, depth in depths_dict.items():
            if depth is None:
                continue
            if sensor in day_df.columns and sensor not in exclude_cols:
                series = pd.to_numeric(day_df[sensor], errors='coerce')
                m = series.mean(skipna=True)
                if np.isfinite(m):
                    depths_list.append(float(depth))
                    mean_temps.append(float(m))

        if len(depths_list) == 0:
            raise ValueError("No matching thermistor columns found in data for the provided depth file.")

        # Sort by depth
        depths_arr, temps_arr = zip(*sorted(zip(depths_list, mean_temps)))

        plt.figure(figsize=(2.5, 4), dpi=250)
        plt.plot(temps_arr, depths_arr, 'o-', color='k', label='Daily mean')
        plt.gca().invert_yaxis()
        plt.xlabel('Ice Temperature [°C]')
        plt.ylabel('Depth [m]')

        # Title day formatting
        day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
        if pd.isna(day):
            day = pd.to_datetime(str(snapshot_time), format='%Y%m%d', errors='coerce')
        day_str = day.strftime('%Y-%m-%d') if not pd.isna(day) else str(snapshot_time)
        plot_title = f"{title if title else 'Temperature Profile'} — {day_str}"
        plt.title(plot_title)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.tight_layout()
        self.format_plot(plot_title)
        plt.savefig(savepath)

    def plot_multiple_temperature_profiles(
        self, 
        snapshot_time=None,  # NOW OPTIONAL
        offsets_list=None, 
        depth_files=None, 
        figsize=(3.6, 4.2), 
        dpi=250, 
        xtick_rotation=0, 
        labels=None,
        savepath=None, 
        title=None, 
        ntc_data_list=None,
        base_fontsize: int = 14, 
        show_title: bool = False, 
        exclude_labels=None,
        xtick_step: float | None = None,
        use_full_period: bool = False,  # NEW: average over entire observation period
        start_time=None,  # NEW: optional start time for full period (GeoPrecision + non-1TT/2TT NTC)
        end_time=None,    # NEW: optional end time for full period (GeoPrecision + non-1TT/2TT NTC)
    ):
        """
        Plot temperature profiles for multiple GeoPrecision chains (daily mean at snapshot_time
        OR averaged over entire observation period) and optional TinyTag/NTC boreholes.

        **IMPORTANT**: NTC data ending in '1TT' or '2TT' (e.g., 'CJ1TT', 'SR2TT') are **always** 
        averaged over their entire measurement period, regardless of start_time/end_time filters.
        Other NTC labels (e.g., 'CJ3TT', 'CJ4TT') respect the time filters.

        Parameters
        ----------
        snapshot_time : str | datetime-like, optional
            Date of the daily mean (e.g., '20250916' or '16/09/2025').
            Ignored if use_full_period=True.
        offsets_list : list
            Per-chain offset rows/Series (same order as self.file_paths).
        depth_files : list[str]
            CSV paths with sensor depths; if ntc_data_list is provided, append its depth files at the end.
        ntc_data_list : list[pd.DataFrame] | None
            Optional TinyTag data frames containing 'White Probe Temperature' and 'Black Probe Temperature'.
            **Labels ending in '1TT' or '2TT' are ALWAYS averaged over their entire period.**
            **Other labels (e.g., '3TT', '4TT') respect start_time/end_time filters.**
        exclude_labels : list[str] | None
            List of profile labels to exclude from the plot (e.g., ['AH1G', 'CJ2TT']).
        xtick_step : float, optional
            Step size for x-axis (temperature) ticks.
        use_full_period : bool, default False
            If True, average GeoPrecision chains over the entire observation period.
            Each chain may have a different observation period.
            Ignores snapshot_time parameter.
        start_time : str, optional
            Start time filter for full period averaging (format: 'DD.MM.YYYY' or 'YYYY-MM-DD').
            **Applies to GeoPrecision chains and NTC labels NOT ending in '1TT' or '2TT'.**
            Only used if use_full_period=True.
        end_time : str, optional
            End time filter for full period averaging.
            **Applies to GeoPrecision chains and NTC labels NOT ending in '1TT' or '2TT'.**
            Only used if use_full_period=True.
        
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        # Validation
        if not use_full_period and snapshot_time is None:
            raise ValueError("Must provide either snapshot_time or set use_full_period=True")
        
        # Parse snapshot date if provided
        day = None
        if not use_full_period:
            day = pd.to_datetime(snapshot_time, dayfirst=True, errors='coerce')
            if pd.isna(day):
                day = pd.to_datetime(str(snapshot_time), format="%Y%m%d", errors='coerce')
            if pd.isna(day):
                raise ValueError(f"Could not parse snapshot_time: {snapshot_time}")
            day = day.normalize()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        n_profiles = len(self.file_paths)
        n_ntc = len(ntc_data_list) if ntc_data_list is not None else 0
        total_series = n_profiles + n_ntc

        # Labels
        if labels is None:
            labels = [f'Chain {i+1}' for i in range(n_profiles)]
            labels += [f'NTC {j+1}' for j in range(n_ntc)]

        # Normalize exclude_labels to set for fast lookup
        exclude_set = set(exclude_labels) if exclude_labels else set()

        # Colors
        color_map = build_profile_color_map(labels)
        _ncols = max(total_series, 1)
        _positions = np.linspace(0.05, 0.75, _ncols)
        fallback_colors = [cmc.romaO(float(p)) for p in _positions]

        # Common excludes
        exclude_cols = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}

        # 1) Plot GeoPrecision chains
        for i in range(n_profiles):
            label = labels[i] if i < len(labels) else f'Chain {i+1}'
            
            # Skip if in exclude list
            if label in exclude_set:
                continue
                
            fp = self.file_paths[i]
            offs = offsets_list[i] if offsets_list and i < len(offsets_list) else None
            dfile = depth_files[i] if depth_files and i < len(depth_files) else None

            if dfile is None:
                print(f"Warning: missing depth file for chain index {i}; skipping.")
                continue
            
            try:
                depths_dict = read_thermistor_depths(dfile)
            except Exception as e:
                print(f"Warning: failed to read depths for chain {fp}: {e}")
                continue

            thermistor = ThermistorData(fp, self.delimiter, self.measurement_depth)
            
            # Get data based on mode
            # GeoPrecision chains respect start_time/end_time filters
            if use_full_period:
                # Average over entire observation period (may differ per chain)
                df = thermistor.get_chain_data_with_offsets(
                    start_time=start_time,
                    end_time=end_time,
                    offsets=offs,
                    aggregate="mean"  # Returns single row with mean values
                )
            else:
                # Daily average for snapshot day
                df = thermistor.get_chain_data_with_offsets(
                    offsets=offs, 
                    snapshot_day=snapshot_time,
                    return_daily_average=True
                )
            
            if df is None or df.empty:
                print(f"Warning: no data for {label}")
                continue

            depths, temps = [], []
            for sensor, depth in depths_dict.items():
                if depth is None:
                    continue
                if sensor in df.columns and sensor not in exclude_cols:
                    series = pd.to_numeric(df[sensor], errors='coerce')
                    m = series.mean(skipna=True) if len(series) > 1 else float(series.iloc[0])
                    if np.isfinite(m):
                        depths.append(float(depth))
                        temps.append(float(m))

            if depths:
                depths, temps = zip(*sorted(zip(depths, temps)))
                color = color_map.get(label, fallback_colors[i])
                ax.plot(temps, depths, 'o-', label=label, color=color, linewidth=2.5)

        # 2) Plot TinyTag/NTC points or short profiles
        # FIXED: Only *1TT and *2TT use entire period (no time filters)
        # Other NTC labels (e.g., *3TT, *4TT) respect time filters
        if n_ntc > 0:
            ntc_depth_files = depth_files[n_profiles:n_profiles + n_ntc] if depth_files else []
            for j in range(n_ntc):
                color_idx = n_profiles + j
                label = labels[n_profiles + j] if (n_profiles + j) < len(labels) else f'NTC {j+1}'
                
                # Skip if in exclude list
                if label in exclude_set:
                    continue
                    
                ntc_df = ntc_data_list[j]
                dfile = ntc_depth_files[j] if j < len(ntc_depth_files) else None

                if dfile is None or ntc_df is None or ntc_df.empty:
                    continue
                
                try:
                    depths_dict = read_thermistor_depths(dfile)
                except Exception as e:
                    print(f"Warning: failed to read depths for NTC ({label}): {e}")
                    continue

                # Depths (robust key match)
                dd_lower = {str(k).lower(): v for k, v in depths_dict.items()}
                depth_white = dd_lower.get('white probe', dd_lower.get('white', None))
                depth_black = dd_lower.get('black probe', dd_lower.get('black', None))

                # FIXED: Check if label ends with '1TT' or '2TT' (case-insensitive)
                label_upper = label.upper()
                use_full_ntc_period = label_upper.endswith('1TT') or label_upper.endswith('2TT')
                
                # Temperatures - conditional time filtering
                try:
                    if use_full_ntc_period:
                        # Average over ALL rows (entire measurement period)
                        t_white = pd.to_numeric(ntc_df['White Probe Temperature'], errors='coerce').mean()
                        t_black = pd.to_numeric(ntc_df['Black Probe Temperature'], errors='coerce').mean()
                        
                        # Log for verification
                        n_samples = len(ntc_df)
                        time_span = "full period"
                        if 'TIME' in ntc_df.columns:
                            times = pd.to_datetime(ntc_df['TIME'], errors='coerce')
                            if times.notna().any():
                                t_start = times.min()
                                t_end = times.max()
                                time_span = f"{t_start.date()} to {t_end.date()}"
                        print(f"NTC {label} (1TT/2TT): averaging {n_samples} samples over {time_span}")
                        print(f"  White probe: {t_white:.3f}°C, Black probe: {t_black:.3f}°C")
                    else:
                        # Filter by start_time/end_time if provided (for *3TT, *4TT, etc.)
                        ntc_filtered = ntc_df.copy()
                        if 'TIME' in ntc_filtered.columns:
                            ntc_filtered['TIME'] = pd.to_datetime(ntc_filtered['TIME'], errors='coerce')
                            
                            if start_time is not None:
                                st = pd.to_datetime(start_time, dayfirst=True, errors='coerce')
                                if pd.notna(st):
                                    ntc_filtered = ntc_filtered[ntc_filtered['TIME'] >= st]
                            
                            if end_time is not None:
                                et = pd.to_datetime(end_time, dayfirst=True, errors='coerce')
                                if pd.notna(et):
                                    ntc_filtered = ntc_filtered[ntc_filtered['TIME'] <= et]
                        
                        if ntc_filtered.empty:
                            print(f"Warning: no data for {label} after time filtering")
                            continue
                        
                        t_white = pd.to_numeric(ntc_filtered['White Probe Temperature'], errors='coerce').mean()
                        t_black = pd.to_numeric(ntc_filtered['Black Probe Temperature'], errors='coerce').mean()
                        
                        # Log for verification
                        n_samples = len(ntc_filtered)
                        time_span = "filtered period"
                        if 'TIME' in ntc_filtered.columns:
                            times = ntc_filtered['TIME']
                            if times.notna().any():
                                t_start = times.min()
                                t_end = times.max()
                                time_span = f"{t_start.date()} to {t_end.date()}"
                        print(f"NTC {label} (other): averaging {n_samples} samples over {time_span}")
                        print(f"  White probe: {t_white:.3f}°C, Black probe: {t_black:.3f}°C")
                        
                except Exception as e:
                    print(f"Warning: missing NTC temperature columns for {label}: {e}")
                    continue

                temps_ntc, depths_ntc = [], []
                if depth_white is not None and np.isfinite(t_white):
                    temps_ntc.append(float(t_white))
                    depths_ntc.append(float(depth_white))
                if depth_black is not None and np.isfinite(t_black):
                    temps_ntc.append(float(t_black))
                    depths_ntc.append(float(depth_black))

                if len(temps_ntc) >= 1:
                    if len(temps_ntc) > 1:
                        depths_ntc, temps_ntc = zip(*sorted(zip(depths_ntc, temps_ntc)))
                    color = color_map.get(label, fallback_colors[color_idx])
                    ax.plot(temps_ntc, depths_ntc, 'o-', label=label, color=color, alpha=0.9, linewidth=2)

        # Axes and styling
        ax.invert_yaxis()
        ax.set_xlabel('Ice Temperature [°C]')
        ax.set_ylabel('Depth [m]')
        ax.axvline(x=0, color='k', linestyle='--')

        # Apply custom x-tick step if provided
        if xtick_step is not None:
            try:
                step = float(xtick_step)
                if step > 0:
                    xmin, xmax = ax.get_xlim()
                    x_neg = np.arange(0, xmin - step * 0.5, -step)[::-1]
                    x_pos = np.arange(0, xmax + step * 0.5, step)
                    xticks = np.unique(np.concatenate([x_neg, x_pos]))
                    xticks.sort()
                    ax.set_xticks(xticks)
            except (TypeError, ValueError) as e:
                print(f"Warning: invalid xtick_step ({xtick_step}), using automatic ticks: {e}")

        # Title/legend
        self.format_plot(None, xtick_rotation=xtick_rotation, legend_loc='best', show_legend=False, base_fontsize=base_fontsize)
        if show_title and title:
            ax.set_title(title, fontsize=max(10, int(base_fontsize)))

        # Legend
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles and legend_labels:
            ax.legend(handles, legend_labels, frameon=True, fancybox=False, edgecolor='black', 
                    framealpha=1, facecolor='white', loc='best', fontsize=max(8, int(base_fontsize)))

        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_geoprecision_thermistor(self, start_time, end_time, depths, savepath, title=None):
        thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
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
    
    def plot_single_ntc_borehole(self, savepath, title=None, depth_white_probe=None, depth_black_probe=None, lower_y_limit=-1):
        ntc_thermistor = ThermistorData(self.file_path, self.delimiter, self.measurement_depth)
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

    def plot_multiple_ntc_boreholes(
        self,
        savepath=None,
        depths=None,
        borehole_labels=None,
        title=None,
        lower_y_limit=-1,
        legend_loc='lower right',
        calibrate=False,
        zero_deg_offsets=None,
        initial_depths=None,
        include_initial_in_legend=True,
        ax=None,
        legend_outside=False,
        show_title=False,
        base_fontsize=28,
        deployment_date=None,
        show_legend=True,
        return_metadata=False,
        show_xlabel=True,
        show_xticklabels=True,
        smooth_days: float | int | None = 0,
        annotation_y=None,
        annotation_spacing: float = 0.08,
        annotation_arrow_hide_dy: float = 0.06,
        annotation_dx_pts: int = 6,
        annotation_positions=None,
        annotation_fontsize: int | None = None,
        equil_days: float = 0,
        show_depth_legend: bool = False,
        depth_legend_loc: str = "upper right",
    ):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import transforms as mtransforms
        from matplotlib import dates as mdates

        # Colors: deterministic via build_profile_color_map (batlowK left half)
        if not (borehole_labels and len(borehole_labels) >= 1):
            raise ValueError("borehole_labels must contain at least one label like ['SR1TT'].")

        cmap_dict = build_profile_color_map(borehole_labels)
        color_bh1 = cmap_dict[borehole_labels[0]]
        color_bh2 = cmap_dict[borehole_labels[1]] if len(borehole_labels) > 1 else None

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(9, 7), dpi=300)
            ax = plt.gca()
            created_fig = True
        else:
            fig = ax.figure

        if depths is None or len(depths) not in (2, 4):
            raise ValueError("depths must be 2 or 4 values: [BH1 white, BH1 black, (optional: BH2 white, BH2 black)]")
        depths = [float(d) for d in depths]

        fps = getattr(self, "file_paths", [getattr(self, "file_path", None)])
        delim = getattr(self, "delimiter", ",")
        meas  = getattr(self, "measurement_depth", None)

        ntc_thermistor_data1 = pd.DataFrame()
        ntc_thermistor_data2 = pd.DataFrame()
        if fps and fps[0]:
            t1 = ThermistorData(fps[0], delim, meas)
            ntc_thermistor_data1 = t1.get_ntc_data()
        if len(fps) > 1 and fps[1] and len(depths) == 4:
            t2 = ThermistorData(fps[1], delim, meas)
            ntc_thermistor_data2 = t2.get_ntc_data()

        def _plot_sensor_with_failure(ax, times, temps, color, lw, label=None, fail_idx=None):
            temps = np.array(temps)
            times = np.array(times)
            if fail_idx is not None and fail_idx <= 0:
                return  # sensor marked as completely failed, skip entirely
            if fail_idx is None:
                fail_idx = len(temps)
            # Plot solid up to missing data
            if fail_idx > 1:
                ax.plot(times[:fail_idx], temps[:fail_idx], linestyle='-', color=color, linewidth=lw, label=label)
            # Plot dotted line for remaining valid data after fail_idx
            if fail_idx < len(temps):
                valid_mask = ~np.isnan(temps[fail_idx:])
                if valid_mask.any():
                    ax.plot(times[fail_idx:][valid_mask], temps[fail_idx:][valid_mask], linestyle=':', color=color, linewidth=lw)
                else:
                    # If all remaining are NaN, plot a horizontal dotted line at last valid value
                    last_valid = temps[fail_idx-1] if fail_idx > 0 else np.nan
                    if np.isfinite(last_valid):
                        ax.plot(times[fail_idx:], [last_valid]*len(times[fail_idx:]), linestyle=':', color=color, linewidth=lw)

        def _normalize_fail_idx(fail_idx, original_length, current_length):
            if original_length == 0:
                return 0
            rel = fail_idx / original_length
            return int(rel * current_length)

        def _first_nan_idx(series):
            arr = np.array(series)
            nan_mask = np.isnan(arr)
            return np.argmax(nan_mask) if nan_mask.any() else len(arr)

        def _smooth_df(df, days):
            if df is None or df.empty or not days:
                return df
            d = df.copy()
            d["TIME"] = pd.to_datetime(d["TIME"])
            d = d.sort_values("TIME").drop_duplicates("TIME").set_index("TIME")
            d = d.resample("6H").mean().interpolate("time", limit_direction="both")
            win = f"{float(days)}D"
            for col in ("White Probe Temperature", "Black Probe Temperature"):
                if col in d:
                    d[col] = d[col].rolling(win, min_periods=1, center=True).mean()
            return d.reset_index()

        fail_idx_white_1 = _first_nan_idx(ntc_thermistor_data1['White Probe Temperature']) if not ntc_thermistor_data1.empty else None
        fail_idx_white_1 = _normalize_fail_idx(fail_idx_white_1, len(ntc_thermistor_data1), len(ntc_thermistor_data1))
        fail_idx_black_1 = _first_nan_idx(ntc_thermistor_data1['Black Probe Temperature']) if not ntc_thermistor_data1.empty else None
        fail_idx_black_1 = _normalize_fail_idx(fail_idx_black_1, len(ntc_thermistor_data1), len(ntc_thermistor_data1))
        fail_idx_white_2 = _first_nan_idx(ntc_thermistor_data2['White Probe Temperature']) if not ntc_thermistor_data2.empty else None
        fail_idx_white_2 = _normalize_fail_idx(fail_idx_white_2, len(ntc_thermistor_data2), len(ntc_thermistor_data2))
        fail_idx_black_2 = _first_nan_idx(ntc_thermistor_data2['Black Probe Temperature']) if not ntc_thermistor_data2.empty else None
        fail_idx_black_2 = _normalize_fail_idx(fail_idx_black_2, len(ntc_thermistor_data2), len(ntc_thermistor_data2))

        if smooth_days and float(smooth_days) > 0:
            ntc_thermistor_data1 = _smooth_df(ntc_thermistor_data1, smooth_days)
            ntc_thermistor_data2 = _smooth_df(ntc_thermistor_data2, smooth_days)

        for df_ in (ntc_thermistor_data1, ntc_thermistor_data2):
            if not df_.empty:
                df_["TIME"] = pd.to_datetime(df_["TIME"])
                df_.sort_values("TIME", inplace=True)

        _suppress_bh2_black = False  # flag to also hide from depth legend

        # Custom failure indices for Corvatsch (CV) glacier
        if borehole_labels and "CV2TT" in borehole_labels:
            fail_idx_black_2 = 0     # 9.3 m sensor failed — suppress entirely
            fail_idx_white_2 = 1250  # manually set the failure index for white probe
            _suppress_bh2_black = True

        if borehole_labels and "GT2TT" in borehole_labels:
            fail_idx_black_2 = 1300    # manually set the failure index for black probe
            fail_idx_white_2 = 1250  # manually set the failure index for white probe

        # Determine deployment date if not provided (earliest timestamp)
        if deployment_date is not None:
            deployment_date = pd.to_datetime(deployment_date, dayfirst=True, errors="coerce")
        if deployment_date is None or pd.isna(deployment_date):
            tmins = []
            if not ntc_thermistor_data1.empty:
                tmins.append(ntc_thermistor_data1["TIME"].min())
            if not ntc_thermistor_data2.empty:
                tmins.append(ntc_thermistor_data2["TIME"].min())
            tmins = [t for t in tmins if pd.notna(t)]
            deployment_date = min(tmins) if tmins else None

        # Apply optional 0°C calibration offsets
        if calibrate and zero_deg_offsets:
            try:
                if not ntc_thermistor_data1.empty and len(zero_deg_offsets) >= 1:
                    b_off, w_off = zero_deg_offsets[0]
                    ntc_thermistor_data1['Black Probe Temperature'] = pd.to_numeric(ntc_thermistor_data1['Black Probe Temperature'], errors='coerce') - float(b_off)
                    ntc_thermistor_data1['White Probe Temperature'] = pd.to_numeric(ntc_thermistor_data1['White Probe Temperature'], errors='coerce') - float(w_off)
                if not ntc_thermistor_data2.empty and len(zero_deg_offsets) >= 2:
                    b_off, w_off = zero_deg_offsets[1]
                    ntc_thermistor_data2['Black Probe Temperature'] = pd.to_numeric(ntc_thermistor_data2['Black Probe Temperature'], errors='coerce') - float(b_off)
                    ntc_thermistor_data2['White Probe Temperature'] = pd.to_numeric(ntc_thermistor_data2['White Probe Temperature'], errors='coerce') - float(w_off)
            except Exception as e:
                print(f"Warning: calibration offsets not applied ({e})")

        def _check_for_nans(df, label):
            for probe in ['White Probe Temperature', 'Black Probe Temperature']:
                if probe in df:
                    nan_count = np.isnan(df[probe]).sum()
                    if nan_count > 0:
                        print(f"NaN detected in {probe} for borehole '{label}': {nan_count} missing values.")

        if not ntc_thermistor_data1.empty:
            _check_for_nans(ntc_thermistor_data1, borehole_labels[0])
        if not ntc_thermistor_data2.empty and len(borehole_labels) > 1:
            _check_for_nans(ntc_thermistor_data2, borehole_labels[1])

        # Initial depths (for label text)
        i_w1 = i_b1 = i_w2 = i_b2 = None
        if isinstance(initial_depths, (list, tuple)):
            if len(initial_depths) == 4:
                i_w1, i_b1, i_w2, i_b2 = initial_depths
            elif len(initial_depths) == 2:
                i_w1, i_b1 = initial_depths

        # Style
        lw = 3.5
        def _alpha_for_depth(d, dmin, dmax):
            rng = max(dmax - dmin, 1e-6)
            return 0.4 + 0.6 * ((d - dmin) / rng)
        dmin, dmax = min(depths), max(depths)

        # Resolve y targets if annotation_y given (used only when explicit positions not provided)
        annoYs = [None, None, None, None]
        if annotation_y is not None:
            if isinstance(annotation_y, (list, tuple, np.ndarray)) and len(annotation_y) >= 4:
                annoYs = list(annotation_y[:4])
            elif isinstance(annotation_y, (int, float)):
                base = float(annotation_y)
                annoYs = [base, base - annotation_spacing, base - 2*annotation_spacing, base - 3*annotation_spacing]

        # Normalize explicit positions to [(x,y), ...] or None
        def _norm_pos_list(pos):
            if pos is None:
                return [None, None, None, None]
            if isinstance(pos, (list, tuple)) and len(pos) >= 4:
                out = []
                for p in pos[:4]:
                    if p is None:
                        out.append(None)
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        out.append((p[0], float(p[1])))
                    else:
                        out.append(None)
                return out
            return [None, None, None, None]
        pos_list = _norm_pos_list(annotation_positions)

        # Helpers
        def _last_xy(df, series):
            if df is None or df.empty or series not in df:
                return None, None
            x = pd.to_datetime(df["TIME"])
            y = pd.to_numeric(df[series], errors="coerce")
            m = y.notna()
            if not m.any():
                return None, None
            return x[m].iloc[-1], float(y[m].iloc[-1])

        def _y_at_time(df, series, t):
            """Interpolate series value at time t (datetime64), returns float or None."""
            if df is None or df.empty or series not in df or t is None:
                return None
            s = df[['TIME', series]].copy()
            s['TIME'] = pd.to_datetime(s['TIME'])
            s = s.dropna(subset=[series]).sort_values('TIME').drop_duplicates('TIME')
            if s.empty:
                return None
            tn = mdates.date2num(s['TIME'])
            yn = pd.to_numeric(s[series], errors='coerce').to_numpy(float)
            if not np.isfinite(yn).any():
                return None
            tnum = mdates.date2num(pd.to_datetime(t))
            tnum = np.clip(tnum, tn.min(), tn.max())
            return float(np.interp(tnum, tn, yn))

        # Draw lines (all solid)
        if not ntc_thermistor_data1.empty:
            _plot_sensor_with_failure(
                ax,
                ntc_thermistor_data1['TIME'],
                ntc_thermistor_data1['White Probe Temperature'],
                color_bh1, lw, fail_idx=fail_idx_white_1
            )
            _plot_sensor_with_failure(
                ax,
                ntc_thermistor_data1['TIME'],
                ntc_thermistor_data1['Black Probe Temperature'],
                color_bh1, lw, fail_idx=fail_idx_black_1
            )
        if not ntc_thermistor_data2.empty and len(depths) == 4:
            _plot_sensor_with_failure(
                ax,
                ntc_thermistor_data2['TIME'],
                ntc_thermistor_data2['White Probe Temperature'],
                color_bh2, lw, fail_idx=fail_idx_white_2
            )
            _plot_sensor_with_failure(
                ax,
                ntc_thermistor_data2['TIME'],
                ntc_thermistor_data2['Black Probe Temperature'],
                color_bh2, lw, fail_idx=fail_idx_black_2
            )

        def _label(depth_val, init_val):
            s = f"{float(depth_val):.1f} m"
            if include_initial_in_legend and init_val is not None and np.isfinite(init_val):
                s += f" (init {float(init_val):.1f} m)"
            return s

        def _parse_x(xlike, fallback_x):
            if xlike is None:
                return fallback_x
            try:
                return pd.to_datetime(xlike)
            except Exception:
                return fallback_x

        # pick annotation fontsize
        fs_anno = annotation_fontsize if annotation_fontsize is not None else max(8, int(base_fontsize * 0.45))

        def _annotate(df, series, color, pos, y_only, label, dx_pts):
            # End-of-series fallback anchor
            x_end, y_end = _last_xy(df, series)
            if x_end is None:
                return
            # Decide target x,y
            x_target = _parse_x(pos[0], x_end) if isinstance(pos, tuple) else x_end
            y_target = y_only if (y_only is not None and np.isfinite(y_only)) else (pos[1] if isinstance(pos, tuple) else y_end)
            y_target = float(y_target)  # <-- no clamping

            # Value of line at x_target for arrow decision
            y_line = _y_at_time(df, series, x_target)
            draw_arrow = (y_line is not None) and (abs(float(y_target) - float(y_line)) >= float(annotation_arrow_hide_dy))

            # Optional connector
            if draw_arrow:
                ax.plot(
                    [x_target, x_target],
                    [y_line, y_target],
                    linestyle='--',
                    linewidth=2.0,
                    color=color,
                    alpha=1
                )

            # Label text with x-offset in points
            trans = ax.transData + mtransforms.ScaledTranslation(dx_pts/72.0, 0, fig.dpi_scale_trans)
            ax.text(x_target, y_target, label, transform=trans, ha="left", va="center",
                    fontsize=fs_anno,
                    color=color, alpha=0.95,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
                    clip_on=True)

        # Annotations per series
        if not ntc_thermistor_data1.empty:
            _annotate(ntc_thermistor_data1, 'White Probe Temperature', color_bh1, pos_list[0], annoYs[0], _label(depths[0], i_w1), annotation_dx_pts)
            _annotate(ntc_thermistor_data1, 'Black Probe Temperature', color_bh1, pos_list[1], annoYs[1], _label(depths[1], i_b1), annotation_dx_pts)
        if not ntc_thermistor_data2.empty and len(depths) == 4:
            _annotate(ntc_thermistor_data2, 'White Probe Temperature', color_bh2, pos_list[2], annoYs[2], _label(depths[2], i_w2), annotation_dx_pts)
            _annotate(ntc_thermistor_data2, 'Black Probe Temperature', color_bh2, pos_list[3], annoYs[3], _label(depths[3], i_b2), annotation_dx_pts)

        # Axes/formatting
        ax.set_ylabel('Temperature [°C]')
        ax.set_xlabel('Time' if show_xlabel else '')
        ax.set_ylim(lower_y_limit, 0.4)
        ax.axhline(y=0, color='k', linestyle='--')

        locator = mdates.AutoDateLocator(minticks=8, maxticks=11)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis='x', labelbottom=bool(show_xticklabels))

        # Clip x-axis start to skip the equilibration phase
        if equil_days and float(equil_days) > 0 and deployment_date is not None and pd.notna(deployment_date):
            ax.set_xlim(left=deployment_date + pd.Timedelta(days=float(equil_days)))

        # Deployment marker & label on every panel
        if deployment_date is not None and pd.notna(deployment_date):
            fs = max(10, int(base_fontsize * 0.75))
            ax.axvline(deployment_date, color='gray', linestyle='solid', linewidth=2, alpha=0.9, zorder=0)
            ax.text(deployment_date, ax.get_ylim()[0], 'Deployment', color='gray',
                    fontsize=fs, va='top', ha='right', rotation=45, alpha=0.9)

        # In-panel depth legend: init -> final depth per sensor
        if show_depth_legend:
            _ha  = "right" if "right" in depth_legend_loc else "left"
            _va  = "top"   if "upper" in depth_legend_loc else "bottom"
            _ax_x = 0.98 if "right" in depth_legend_loc else 0.02
            _ax_y = 0.98 if "upper" in depth_legend_loc else 0.02
            _fs_leg = max(7, (annotation_fontsize or int(base_fontsize * 0.45)) - 1)

            _entries = []
            if i_w1 is not None or i_b1 is not None:
                _entries.append((color_bh1, borehole_labels[0],
                                 i_w1, i_b1,
                                 depths[0] if len(depths) >= 1 else None,
                                 depths[1] if len(depths) >= 2 else None))
            if len(depths) == 4 and (i_w2 is not None or i_b2 is not None):
                _entries.append((color_bh2, borehole_labels[1],
                                 i_w2, i_b2, depths[2], depths[3]))

            # Build full text for background box (invisible text sets bbox size)
            _full_lines = []
            for _clr, _lbl, _iw, _ib, _dw, _db in _entries:
                _full_lines.append(_lbl)
                if _iw is not None and _dw is not None:
                    _full_lines.append(f"  \u25cb {_iw:.1f}\u2192{_dw:.1f} m")
                if _ib is not None and _db is not None:
                    _full_lines.append(f"  \u25cf {_ib:.1f}\u2192{_db:.1f} m")
            _box_txt = ax.text(_ax_x, _ax_y, "\n".join(_full_lines),
                               transform=ax.transAxes, ha=_ha, va=_va,
                               fontsize=_fs_leg, color="none",
                               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9, lw=0.8),
                               zorder=19)

            # Draw each entry in the matching line color on top
            _sign = -1 if "upper" in depth_legend_loc else 1
            _line_frac = _fs_leg * 1.55 / (ax.get_position().height * fig.get_size_inches()[1] * 72)
            _cur_y = _ax_y
            for _ei, (_clr, _lbl, _iw, _ib, _dw, _db) in enumerate(_entries):
                _rows = [_lbl]
                if _iw is not None and _dw is not None:
                    _rows.append(f"  \u25cb {_iw:.1f}\u2192{_dw:.1f} m")
                if _ib is not None and _db is not None and _show_black(_ei):
                    _rows.append(f"  \u25cf {_ib:.1f}\u2192{_db:.1f} m")
                ax.text(_ax_x, _cur_y, "\n".join(_rows),
                        transform=ax.transAxes, ha=_ha, va=_va,
                        fontsize=_fs_leg, color=_clr, zorder=20)
                _cur_y += _sign * _line_frac * (len(_rows) + 0.4)

        # Styling
        try:
            self.format_plot(title if show_title else None, legend_loc, show_legend=False, base_fontsize=base_fontsize)
        except Exception:
            pass

        # Optional in-axes legend
        handles = []
        legend_names = []
        handles.append(Line2D([0], [0], color=color_bh1, lw=lw, linestyle='-'))
        legend_names.append(borehole_labels[0])
        if not ntc_thermistor_data2.empty and len(borehole_labels) > 1 and len(depths) == 4:
            handles.append(Line2D([0], [0], color=color_bh2, lw=lw, linestyle='-'))
            legend_names.append(borehole_labels[1])
        if show_legend and handles:
            if legend_outside:
                ax.legend(handles, legend_names, frameon=True, fancybox=False, edgecolor='black', framealpha=1,
                        facecolor='white', loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
            else:
                ax.legend(handles, legend_names, frameon=True, fancybox=False, edgecolor='black', framealpha=1,
                        facecolor='white', loc=legend_loc, ncol=1)

        # Save if requested and we created a standalone figure
        out_path = None
        if savepath and created_fig:
            fname = (title or "NTC_timeseries").replace(' ', '_').replace('/', '').replace('-', '_')
            out_path = Path(savepath) / f"{fname}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return fig, (str(out_path) if out_path else None)

    def plot_ntc_icebath_calibration(self, thermistor_chain_data, savepath, y_limits=(-1,1), title=None, legend_loc='lower right'):
        ntc_thermistor = ThermistorData(self.file_path, self.delimiter)
        ntc_thermistor_data = ntc_thermistor.get_ntc_data()

        # Use the updated method to get offsets and stable indices
        black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white = ntc_thermistor.calculate_ntc_offsets()

        # Get the time of the stable period
        stable_period_black_start = ntc_thermistor_data['TIME'].iloc[stable_indices_black[0]]
        stable_period_black_end = ntc_thermistor_data['TIME'].iloc[stable_indices_black[-1]]
        stable_period_white_start = ntc_thermistor_data['TIME'].iloc[stable_indices_white[0]]
        stable_period_white_end = ntc_thermistor_data['TIME'].iloc[stable_indices_white[-1]]

        plt.figure(figsize=(12, 8.5), dpi=300)

        # plot the ntc data
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['Black Probe Temperature'], linewidth=4, alpha=0.6, label='Black Probe (Gemini)')
        plt.plot(ntc_thermistor_data['TIME'], ntc_thermistor_data['White Probe Temperature'], linewidth=4, alpha=0.6, label='White Probe (Gemini)')

        # plot the reference geoprecision thermistor chain data
        plt.plot(thermistor_chain_data['TIME'], thermistor_chain_data['#20'], linewidth=4, alpha=0.6, label='Thermistor chain (Geoprecision)', color='black')

        # Mark the stable period for Black Probe
        plt.axvline(stable_period_black_start, color='tab:blue', linestyle='-', linewidth=2)
        plt.axvline(stable_period_black_end, color='tab:blue', linestyle='-', linewidth=2)
        plt.text(stable_period_black_start, y_limits[0], 'Stable Start', color='tab:blue', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)
        plt.text(stable_period_black_end, y_limits[0], 'Stable End', color='tab:blue', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)

        # Mark the stable period for White Probe
        plt.axvline(stable_period_white_start, color='tab:orange', linestyle='-', linewidth=2)
        plt.axvline(stable_period_white_end, color='tab:orange', linestyle='-', linewidth=2)
        plt.text(stable_period_white_start, y_limits[0], 'Stable Start', color='tab:orange', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)
        plt.text(stable_period_white_end, y_limits[0], 'Stable End', color='tab:orange', fontsize=14, verticalalignment='bottom', horizontalalignment='right', rotation=45)

        plt.xlabel('Time')
        plt.ylabel('Temperature [°C]')
        plt.title(title, fontsize=22)
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(y_limits)
        plt.xlim(ntc_thermistor_data['TIME'].min(), ntc_thermistor_data['TIME'].max())

        self.format_plot(title)

        # Draw a horizontal line at the 0°C offset for the Black and White probe
        plt.axhline(y=black_probe_offset, color='blue', linestyle='dotted', linewidth=2, label=f'Black Probe 0°C Offset: {black_probe_offset:.2f}°C')
        plt.axhline(y=white_probe_offset, color='orange', linestyle='dotted', linewidth=2, label=f'White Probe 0°C Offset: {white_probe_offset:.2f}°C')

        plt.legend(fontsize=22, frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white', loc='upper center', bbox_to_anchor=(0.45, -0.4), ncol=2)
        plt.tight_layout()

        title_with_underscores = title.replace(' ', '_').replace('/', '').replace('-', '_')
        plt.savefig(savepath + title_with_underscores + '.png')

        # Return offsets and stable indices
        return (black_probe_offset, stable_indices_black, white_probe_offset, stable_indices_white)
    
    def plot_ntc_statistics(
        self,
        savepath=None,
        depths=None,
        borehole_labels=None,
        title=None,
        calibrate=False,
        zero_deg_offsets=None,
        figsize=(12, 8),
        dpi=300,
        base_fontsize=14,
        show_boxplot=True,
        show_table=True,
        seasonal_months=None,  # e.g., [(6,7,8), (12,1,2)] for summer/winter
        equilibration_days=4  # <-- NEW: exclude first N days
    ):
        """
        Create statistical summary of NTC temperature data with table and boxplot.
        
        Parameters
        ----------
        savepath : str, optional
            Path to save the figure
        depths : list
            Sensor depths [BH1_white, BH1_black] or [BH1_white, BH1_black, BH2_white, BH2_black]
        borehole_labels : list[str]
            Labels like ['SR1TT', 'SR2TT']
        title : str, optional
            Plot title
        calibrate : bool
            Apply calibration offsets
        zero_deg_offsets : list, optional
            Calibration offsets [(black_off, white_off), ...]
        figsize : tuple
            Figure size
        dpi : int
            Figure DPI
        base_fontsize : int
            Base font size
        show_boxplot : bool
            Show boxplot of temperature distributions
        show_table : bool
            Show statistical summary table
        seasonal_months : list[tuple], optional
            List of month tuples for seasonal analysis, e.g., [(6,7,8), (12,1,2)]
            for summer (JJA) and winter (DJF)
        equilibration_days : int
            Number of days to skip at start (default 4)
        
        Returns
        -------
        fig, stats_df
            Figure and DataFrame with statistics
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Setup
        if depths is None or len(depths) not in (2, 4):
            raise ValueError("depths must be 2 or 4 values")
        depths = [float(d) for d in depths]
        
        if not borehole_labels or len(borehole_labels) < 1:
            raise ValueError("borehole_labels required")
        
        # Colors
        cmap_dict = build_profile_color_map(borehole_labels)
        colors = [cmap_dict[lab] for lab in borehole_labels]
        
        # Load data
        fps = getattr(self, "file_paths", [getattr(self, "file_path", None)])
        delim = getattr(self, "delimiter", ",")
        meas = getattr(self, "measurement_depth", None)
        
        ntc_data1 = pd.DataFrame()
        ntc_data2 = pd.DataFrame()
        if fps and fps[0]:
            t1 = ThermistorData(fps[0], delim, meas)
            ntc_data1 = t1.get_ntc_data()
        if len(fps) > 1 and fps[1] and len(depths) == 4:
            t2 = ThermistorData(fps[1], delim, meas)
            ntc_data2 = t2.get_ntc_data()
        
        # Apply calibration
        if calibrate and zero_deg_offsets:
            try:
                if not ntc_data1.empty and len(zero_deg_offsets) >= 1:
                    b_off, w_off = zero_deg_offsets[0]
                    ntc_data1['Black Probe Temperature'] = pd.to_numeric(
                        ntc_data1['Black Probe Temperature'], errors='coerce') - float(b_off)
                    ntc_data1['White Probe Temperature'] = pd.to_numeric(
                        ntc_data1['White Probe Temperature'], errors='coerce') - float(w_off)
                if not ntc_data2.empty and len(zero_deg_offsets) >= 2:
                    b_off, w_off = zero_deg_offsets[1]
                    ntc_data2['Black Probe Temperature'] = pd.to_numeric(
                        ntc_data2['Black Probe Temperature'], errors='coerce') - float(b_off)
                    ntc_data2['White Probe Temperature'] = pd.to_numeric(
                        ntc_data2['White Probe Temperature'], errors='coerce') - float(w_off)
            except Exception as e:
                print(f"Warning: calibration failed ({e})")
        
        # Process timestamps and skip equilibration period
        for df in (ntc_data1, ntc_data2):
            if not df.empty:
                df["TIME"] = pd.to_datetime(df["TIME"])
                df.sort_values("TIME", inplace=True)
                # Skip first N days
                if equilibration_days > 0:
                    cutoff = df["TIME"].min() + pd.Timedelta(days=equilibration_days)
                    df.drop(df[df["TIME"] < cutoff].index, inplace=True)
        
        def compute_statistics(df, label, depth_white, depth_black):
            """Compute comprehensive statistics for a borehole"""
            if df.empty:
                return None
            
            stats_list = []
            
            for probe, col, depth in [
                ('White', 'White Probe Temperature', depth_white),
                ('Black', 'Black Probe Temperature', depth_black)
            ]:
                temps = pd.to_numeric(df[col], errors='coerce').dropna()
                if temps.empty:
                    continue
                
                # Add month for seasonal analysis
                df_temp = df[[col, 'TIME']].copy()
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                df_temp['month'] = df_temp['TIME'].dt.month
                
                stats = {
                    'Borehole': label,
                    'Sensor': f"{probe} ({depth:.1f} m)",
                    'Mean [°C]': temps.mean(),
                    'Median [°C]': temps.median(),
                    'Std Dev [°C]': temps.std(),
                    'Min [°C]': temps.min(),
                    'Max [°C]': temps.max(),
                    'Range [°C]': temps.max() - temps.min(),
                    'Q25 [°C]': temps.quantile(0.25),
                    'Q75 [°C]': temps.quantile(0.75),
                    'IQR [°C]': temps.quantile(0.75) - temps.quantile(0.25),
                    'N samples': len(temps)
                }
                
                # Seasonal amplitude if requested
                if seasonal_months:
                    for i, months in enumerate(seasonal_months, 1):
                        mask = df_temp['month'].isin(months)
                        seasonal_temps = df_temp.loc[mask, col].dropna()
                        if not seasonal_temps.empty:
                            stats[f'Season{i} Mean [°C]'] = seasonal_temps.mean()
                            stats[f'Season{i} Std [°C]'] = seasonal_temps.std()
                    
                    # Annual amplitude (if we have 2 seasons)
                    if len(seasonal_months) == 2:
                        s1_key = f'Season1 Mean [°C]'
                        s2_key = f'Season2 Mean [°C]'
                        if s1_key in stats and s2_key in stats:
                            stats['Annual Amplitude [°C]'] = abs(stats[s1_key] - stats[s2_key])
                
                stats_list.append(stats)
            
            return pd.DataFrame(stats_list) if stats_list else None
        
        # Collect statistics
        all_stats = []
        
        if not ntc_data1.empty:
            stats1 = compute_statistics(ntc_data1, borehole_labels[0], depths[0], depths[1])
            if stats1 is not None:
                all_stats.append(stats1)
        
        if not ntc_data2.empty and len(depths) == 4:
            stats2 = compute_statistics(ntc_data2, borehole_labels[1], depths[2], depths[3])
            if stats2 is not None:
                all_stats.append(stats2)
        
        if not all_stats:
            raise ValueError("No valid statistics computed")
        
        stats_df = pd.concat(all_stats, ignore_index=True)
        
        # Create figure
        n_panels = sum([show_table, show_boxplot])
        if n_panels == 0:
            raise ValueError("Must show at least table or boxplot")
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        if show_table and show_boxplot:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
            ax_table = fig.add_subplot(gs[0])
            ax_box = fig.add_subplot(gs[1])
        elif show_table:
            ax_table = fig.add_subplot(111)
            ax_box = None
        else:
            ax_table = None
            ax_box = fig.add_subplot(111)
        
        # Table
        if show_table and ax_table:
            ax_table.axis('off')
            
            # Format columns for display
            display_cols = ['Borehole', 'Sensor', 'Mean [°C]', 'Std Dev [°C]', 
                        'Min [°C]', 'Max [°C]', 'Range [°C]', 'N samples']
            
            # Add seasonal columns if present
            season_cols = [c for c in stats_df.columns if 'Season' in c or 'Amplitude' in c]
            display_cols.extend(season_cols)
            
            table_data = stats_df[display_cols].copy()
            
            # Round numeric columns
            for col in table_data.columns:
                if '[°C]' in col or 'N samples' in col:
                    if 'N samples' in col:
                        table_data[col] = table_data[col].astype(int)
                    else:
                        table_data[col] = table_data[col].round(3)
            
            # Create table
            cell_text = table_data.values.tolist()
            col_labels = [col.replace(' [°C]', '\n[°C]') for col in display_cols]
            
            table = ax_table.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(base_fontsize * 0.7)
            table.scale(1, 2)
            
            # Style header
            for i in range(len(col_labels)):
                cell = table[(0, i)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            
            # Alternating row colors
            for i in range(1, len(cell_text) + 1):
                for j in range(len(col_labels)):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#E7E6E6')
                    else:
                        cell.set_facecolor('white')
            
            ax_table.set_title(title or 'Temperature Statistics', 
                            fontsize=base_fontsize * 1.2, weight='bold', pad=20)
        
        # Boxplot
        if show_boxplot and ax_box:
            box_data = []
            box_labels = []
            box_colors = []
            
            if not ntc_data1.empty:
                # FIXED: removed unpacking, use column name directly
                for probe_col, label_suffix, depth_val in [
                    ('White Probe Temperature', f"{borehole_labels[0]}\nWhite ({depths[0]:.1f}m)", depths[0]),
                    ('Black Probe Temperature', f"{borehole_labels[0]}\nBlack ({depths[1]:.1f}m)", depths[1])
                ]:
                    temps = pd.to_numeric(ntc_data1[probe_col], errors='coerce').dropna()
                    if not temps.empty:
                        box_data.append(temps.values)
                        box_labels.append(label_suffix)
                        box_colors.append(colors[0])
            
            if not ntc_data2.empty and len(depths) == 4:
                # FIXED: same pattern for BH2
                for probe_col, label_suffix, depth_val in [
                    ('White Probe Temperature', f"{borehole_labels[1]}\nWhite ({depths[2]:.1f}m)", depths[2]),
                    ('Black Probe Temperature', f"{borehole_labels[1]}\nBlack ({depths[3]:.1f}m)", depths[3])
                ]:
                    temps = pd.to_numeric(ntc_data2[probe_col], errors='coerce').dropna()
                    if not temps.empty:
                        box_data.append(temps.values)
                        box_labels.append(label_suffix)
                        box_colors.append(colors[1] if len(colors) > 1 else colors[0])
            
            bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                            notch=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', 
                                            markeredgecolor='darkred', markersize=6))
            
            # Color boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax_box.set_ylabel('Temperature [°C]', fontsize=base_fontsize)
            ax_box.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_box.grid(True, alpha=0.3, axis='y')
            ax_box.tick_params(labelsize=base_fontsize * 0.9)
            
            if not show_table:
                ax_box.set_title(title or 'Temperature Distribution', 
                            fontsize=base_fontsize * 1.2, weight='bold')
        
        fig.tight_layout()
        
        # Save
        if savepath:
            from pathlib import Path
            out_path = Path(savepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            
            # Also save CSV
            csv_path = out_path.with_suffix('.csv')
            stats_df.to_csv(csv_path, index=False)
            print(f"Statistics saved to: {csv_path}")
        
        return fig, stats_df

# -----------------------------------------------------------------------------
# Heatmap: depth vs time (uses gpr_plotting.format_plot for consistent styling)
# -----------------------------------------------------------------------------
def plot_chain_temperature_heatmap(
    thermistor: ThermistorData,
    *,
    start_time=None,
    end_time=None,
    offsets=None,
    depth_file=None,
    time_freq="6H",
    depth_step=0.25,
    smooth_time_sigma=0.0,
    smooth_depth_sigma=0.0,
    temp_step=0.5,
    cbar_min=None,          # override colorbar minimum
    vmax=0.0,
    figsize=(6, 4),
    dpi=300,
    cmap=None,
    show_colorbar=True,
    title=None,
    savepath=None,
    add_contours: bool = True,
    contour_kwargs: dict | None = None,
    show_sensor_depths: bool = False,
    sensor_depth_markers_kwargs: dict | None = None,
    bedrock_depth: float | None = None,            # NEW: bedrock depth (m)
    bedrock_hatch_kwargs: dict | None = None       # NEW: hatch style below deepest sensor
):
    def _discrete_icetemp_cmap_from_levels(levels):
        n_int = len(levels) - 1
        if n_int <= 0:
            colors = np.array(cmc.vik(1.0)).reshape(1, -1)
        else:
            if n_int > 1:
                blue = cmc.vik(np.linspace(0.0, 0.5, n_int - 1))
                red = np.array(cmc.vik(0.95)).reshape(1, -1)
                colors = np.vstack([blue, red])
            else:
                colors = np.array(cmc.vik(1.0)).reshape(1, -1)
        return ListedColormap(colors, name='icetemp_discrete')

    if depth_file is None:
        raise ValueError("depth_file is required to determine sensor depths.")

    df = thermistor.get_chain_data_with_offsets(
        start_time=start_time,
        end_time=end_time,
        offsets=offsets
    )
    if df.empty:
        raise ValueError("No data returned for specified time range.")

    depths_dict = read_thermistor_depths(depth_file)
    exclude = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}
    sensor_cols = [c for c in df.columns if c not in exclude and c in depths_dict and depths_dict[c] is not None]
    if not sensor_cols:
        raise ValueError("No matching sensor columns with valid depths found.")
    sensor_depths = {c: float(depths_dict[c]) for c in sensor_cols}
    sensor_cols = sorted(sensor_cols, key=lambda c: sensor_depths[c])
    depths_sorted = np.array([sensor_depths[c] for c in sensor_cols], float)

    df = df[['TIME'] + sensor_cols].copy()
    df['TIME'] = pd.to_datetime(df['TIME'])
    df = df.sort_values('TIME').drop_duplicates('TIME').set_index('TIME')
    df_res = df.resample(time_freq).mean().interpolate(method='time', limit_direction='both')
    temp_matrix = df_res.to_numpy(float)

    z_min, z_max = depths_sorted.min(), depths_sorted.max()
    depth_grid = np.arange(z_min, z_max + depth_step * 0.51, depth_step)
    nt, ns = temp_matrix.shape
    nz = depth_grid.size
    grid = np.full((nt, nz), np.nan)
    for i in range(nt):
        row = temp_matrix[i, :]
        m = np.isfinite(row)
        if m.sum() >= 2:
            grid[i, :] = np.interp(depth_grid, depths_sorted[m], row[m])

    if (smooth_time_sigma and smooth_time_sigma > 0) or (smooth_depth_sigma and smooth_depth_sigma > 0):
        valid = np.isfinite(grid)
        filled = grid.copy()
        for j in range(nz):
            col = filled[:, j]
            nmask = ~np.isfinite(col)
            if nmask.any() and (~nmask).any():
                col[nmask] = np.interp(np.flatnonzero(nmask), np.flatnonzero(~nmask), col[~nmask])
            filled[:, j] = col
        for i in range(nt):
            row = filled[i, :]
            nmask = ~np.isfinite(row)
            if nmask.any() and (~nmask).any():
                row[nmask] = np.interp(np.flatnonzero(nmask), np.flatnonzero(~nmask), row[~nmask])
            filled[i, :] = row
        grid = gaussian_filter(filled, sigma=(smooth_time_sigma, smooth_depth_sigma), mode='nearest')
        grid[~valid] = np.nan

    # beta_cc = 7.42e-4  # °C m^-1, pure ice/pure water (Cuffey & Paterson 2010, Eq. 9.9)
    beta_cc = 8.7e-4      # °C m^-1, air-saturated ice (Cuffey & Paterson 2010, Eq. 9.10)
    delta_cts = 0.05
    Tm_depth = -beta_cc * depth_grid[None, :]
    delta_grid = grid - Tm_depth

    delta_vals = delta_grid[np.isfinite(delta_grid)]
    if delta_vals.size == 0:
        raise ValueError("All interpolated values are NaN.")
    vmin_dt = float(cbar_min) if cbar_min is not None else np.floor(delta_vals.min() / temp_step) * temp_step

    neg_levels = np.arange(vmin_dt, -delta_cts, temp_step)
    levels = np.unique(np.r_[neg_levels, -delta_cts, delta_cts])
    levels.sort()

    if cmap is None:
        cmap = _discrete_icetemp_cmap_from_levels(levels)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    times = df_res.index.to_pydatetime()
    if len(times) < 2:
        raise ValueError("Need at least two time steps for heatmap.")
    t_num = mdates.date2num(times)
    dt = np.diff(t_num)
    t_edges = np.r_[t_num[0] - dt[0] / 2, t_num[:-1] + dt / 2, t_num[-1] + dt[-1] / 2]
    z_edges = np.r_[depth_grid[0] - depth_step / 2, depth_grid[:-1] + depth_step / 2, depth_grid[-1] + depth_step / 2]

    pcm = ax.pcolormesh(t_edges, z_edges, delta_grid.T, cmap=cmap, norm=norm, shading='auto')
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Time")

    span = times[-1] - times[0]
    span_days = span.total_seconds() / 86400.0
    target_ticks = 8
    if span_days <= 2.0:
        span_hours = span.total_seconds() / 3600.0
        step = max(1, round(span_hours / target_ticks))
        loc = mdates.HourLocator(interval=step)
        fmt = mdates.DateFormatter('%b %d\n%H:%M')
    elif span_days <= 14.0:
        span_hours = span.total_seconds() / 3600.0
        step = max(3, round(span_hours / target_ticks))
        loc = mdates.HourLocator(interval=step)
        fmt = mdates.DateFormatter('%b %d\n%H:%M')
    else:
        step = max(1, round(span_days / target_ticks))
        loc = mdates.DayLocator(interval=step)
        fmt = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(rotation=45)

    if add_contours:
        ck = {"colors": "k", "linewidths": 0.6, "alpha": 0.35}
        if contour_kwargs:
            ck.update(contour_kwargs)
        try:
            ax.contour(t_num, depth_grid, delta_grid.T, levels=levels, **ck)
        except Exception:
            pass

    # Bedrock line, hatched unknown zone, and dark-gray bedrock band
    if bedrock_depth is not None:
        try:
            bedrock_color = "0.35"
            z_min = float(depth_grid.min())
            z_max = float(depth_grid.max())
            z_range = max(z_max - z_min, depth_step)
            rock_height = 0.10 * z_range  # 20% of plotted depth range
            rock_bottom = float(bedrock_depth) + rock_height

            # Draw bedrock line
            ax.axhline(bedrock_depth, color="0.25", linewidth=2.2, linestyle="-", zorder=6)

            # Hatched zone (unknown data) between deepest sensor and bedrock
            if float(bedrock_depth) > z_max:
                hk = {
                    "facecolor": "none",
                    "edgecolor": "0.25",
                    "linewidth": 0.8,
                    "hatch": "///",
                    "alpha": 0.7,
                    "zorder": 5
                }
                if bedrock_hatch_kwargs:
                    hk.update(bedrock_hatch_kwargs)
                ax.fill_between(t_edges, z_max, bedrock_depth, **hk)

            # Dark-gray bedrock band below the bedrock line
            ax.fill_between(
                t_edges,
                bedrock_depth,
                rock_bottom,
                facecolor=bedrock_color,
                edgecolor=bedrock_color,
                linewidth=0.0,
                alpha=1.0,
                zorder=4,
            )

            # Extend y-limits to show the bedrock band (respect inverted axis)
            ax.set_ylim(rock_bottom, ax.get_ylim()[1])
        except Exception:
            pass

    # Optional sensor-depth markers (no raw values)
    if show_sensor_depths and depth_file is not None:
        mk = dict(marker="|", s=60, color="k", alpha=0.9, linewidths=1.0, zorder=8)
        if sensor_depth_markers_kwargs:
            mk.update(sensor_depth_markers_kwargs)
        if "markersize" in mk:  # map markersize -> s
            ms = mk.pop("markersize")
            mk["s"] = ms * ms
        x0, x1 = ax.get_xlim()
        x_pos = x1 + 0.012 * (x1 - x0)
        ax.scatter(np.full_like(depths_sorted, x_pos, dtype=float), depths_sorted, **mk)

    cb = None
    if show_colorbar:
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(r"ΔT = T - T$_m$(h) [°C]")
        cold_ticks = [lv for lv in levels if lv < -delta_cts]
        temp_tick = 0.0
        cb.set_ticks(cold_ticks + [temp_tick])
        cb.set_ticklabels([f"{t:.1f}" for t in cold_ticks] + [r"≈T$_{pmp}$"])

    try:
        from processing.gpr_plotting import format_plot
    except Exception:
        try:
            from .gpr_plotting import format_plot
        except Exception:
            format_plot = None

    if format_plot:
        format_plot(ax=ax, title=title, legend_loc='upper right',
                    x_tick_rotation=45, y_tick_rotation=0, cbar=cb, base_fontsize=26)
    else:
        def _fallback_format(ax, title, cb, base_fontsize=26):
            ax.set_title(title or "", fontsize=base_fontsize)
            ax.xaxis.label.set_size(base_fontsize)
            ax.yaxis.label.set_size(base_fontsize)
            ax.tick_params(axis='both', labelsize=base_fontsize)
            if cb:
                cb.ax.tick_params(labelsize=base_fontsize * 0.8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
        _fallback_format(ax, title, cb)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, {
        "time_index": df_res.index,
        "depth_grid": depth_grid,
        "grid": grid,
        "levels": levels,
        "raw_df": df_res,
        "sensor_depths": sensor_depths,
        "bedrock_depth": bedrock_depth  # NEW
    }

# -------------------------------------------------------------------------
# Mosaic helper: combine three chain heatmaps into one figure with shared cbar
# -------------------------------------------------------------------------
def mosaic_chain_heatmaps(
    meta_list,
    *,
    titles=None,               # optional per-panel titles
    show_titles=False,         # default off (no titles above panels)
    panel_tags=("a", "b", "c"),
    panel_fontsize=22,
    figure_fontsize=22,
    figsize=(14, 5),
    dpi=300,
    cmap=None,
    savepath=None,
    add_contours=True,
    contour_kwargs=None,       # e.g., {"colors": "k", "linewidths": 0.6, "alpha": 0.35}
    rasterize_heatmap=True,
    two_rows=False,            # 2xN layout with time series below each heatmap
    line_width=2.1,            # time-series line width
    line_alpha=0.9,            # time-series line alpha
    line_cmap=None,            # optional custom cmap; if None, use mono gradient
    line_color_base=None,      # base color or colormap; if None, defaults to cmc.lajolla
    ts_y_limits=None,          # tuple (ymin,ymax) or list/tuple of N tuples for per-panel limits
    ts_y_tick_steps=None,      # float or list/tuple of N floats for per-panel y-tick step
):
    """
    Create a mosaic from plot_chain_temperature_heatmap outputs.
      - Default: 1xN heatmaps with shared colorbar (N = 2 or 3).
      - If two_rows=True: 2xN layout, heatmaps on top, per-borehole time series below,
        with legends listing sensor depths.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    if meta_list is None or len(meta_list) not in (2, 3):
        raise ValueError("meta_list must be a list of two or three meta dictionaries.")
    n_panels = len(meta_list)

    # Shared levels
    level_sets = [m.get("levels") for m in meta_list if m.get("levels") is not None]
    if not level_sets:
        raise ValueError("No levels found in meta_list; cannot build shared colorbar.")
    levels = np.unique(np.concatenate(level_sets))
    levels.sort()

    # Colormap
    if cmap is None:
        def _discrete_icetemp_cmap_from_levels(levels_arr):
            n_int = len(levels_arr) - 1
            if n_int <= 0:
                colors = np.array(cmc.vik(1.0)).reshape(1, -1)
            else:
                if n_int > 1:
                    blue = cmc.vik(np.linspace(0.0, 0.5, n_int - 1))
                    red = np.array(cmc.vik(0.95)).reshape(1, -1)
                    colors = np.vstack([blue, red])
                else:
                    colors = np.array(cmc.vik(1.0)).reshape(1, -1)
            return ListedColormap(colors, name="icetemp_discrete")
        cmap = _discrete_icetemp_cmap_from_levels(levels)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    contour_opts = {"colors": "k", "linewidths": 0.6, "alpha": 0.35}
    if contour_kwargs:
        contour_opts = contour_kwargs

    # Layout
    if two_rows:
        fig, axs = plt.subplots(
            2, n_panels, figsize=figsize, dpi=dpi,
            gridspec_kw={"height_ratios": [1.0, 0.55]},
            sharex="col"
        )
        heat_axes = axs[0, :].ravel()
        ts_axes = axs[1, :].ravel()
    else:
        fig, axs = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi, sharey=False)
        heat_axes = np.array(axs).ravel()
        ts_axes = None

    # Line colors
    def _line_colors(n):
        # Prefer explicit line_cmap if given
        if line_cmap is not None:
            return line_cmap(np.linspace(1, 0, n))  # reversed

        base = line_color_base if line_color_base is not None else cmc.lajolla

        # If a Matplotlib Colormap is provided
        if isinstance(base, mcolors.Colormap):
            return base(np.linspace(0.95, 0.25, n))  # reversed

        # If a precomputed array of colors is provided (N x 3 or N x 4)
        if isinstance(base, np.ndarray):
            arr = base
            if arr.ndim == 2 and arr.shape[1] in (3, 4):
                idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
                return arr[idx]
            # fall through to single-color handling if shape is unexpected

        # Otherwise treat as a single color and build a dark->light gradient (reversed)
        base_rgb = np.array(mcolors.to_rgb(base))
        weights = np.linspace(1.0, 0.35, n)[:, None]  # reversed
        return np.clip(base_rgb * weights + (1.0 - weights), 0, 1)

    # Shared bedrock band height (10% of global depth span)
    br_band_frac = 0.10
    zmins = [float(m["depth_grid"].min()) for m in meta_list]
    zmaxs = [float(m["depth_grid"].max()) for m in meta_list]
    z_span_global = max(zmaxs) - min(zmins)
    rock_height_global = br_band_frac * max(z_span_global, 1e-6)    

    im = None
    for i, (ax, meta) in enumerate(zip(heat_axes, meta_list)):
        times = meta["time_index"]
        depth_grid = meta["depth_grid"]
        grid = meta["grid"]
        if len(times) < 2:
            raise ValueError("Each meta must have at least two time points.")

        t_num = mdates.date2num(times.to_pydatetime())
        dt = np.diff(t_num)
        t_edges = np.r_[t_num[0] - dt[0] / 2, t_num[:-1] + dt / 2, t_num[-1] + dt[-1] / 2]
        dz = np.diff(depth_grid)
        dz0 = dz[0] if len(dz) else 1.0
        dzn = dz[-1] if len(dz) else 1.0
        z_edges = np.r_[depth_grid[0] - dz0 / 2, depth_grid[:-1] + dz / 2, depth_grid[-1] + dzn / 2]

        im = ax.pcolormesh(t_edges, z_edges, grid.T, cmap=cmap, norm=norm, shading="auto")
        if rasterize_heatmap:
            im.set_rasterized(True)
        if add_contours:
            try:
                ax.contour(t_num, depth_grid, grid.T, levels=levels, **contour_opts)
            except Exception:
                pass
        ax.invert_yaxis()

        if show_titles and titles and i < len(titles):
            ax.set_title(titles[i], fontsize=figure_fontsize)

        # X ticks: day-month only (no time)
        span = times[-1] - times[0]
        span_days = span.total_seconds() / 86400.0
        target_ticks = 8
        step = max(1, round(span_days / target_ticks)) if span_days > 0 else 1
        loc = mdates.DayLocator(interval=step)
        fmt = mdates.DateFormatter("%b %d")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        center_idx = n_panels // 2
        ax.set_xlabel("Time" if (not two_rows and i == center_idx) else "",
                      fontsize=figure_fontsize if (not two_rows and i == center_idx) else None)
        ax.set_ylabel("Depth [m]" if i == 0 else "", fontsize=figure_fontsize if i == 0 else None)
        ax.tick_params(axis='both', labelsize=figure_fontsize)

        if panel_tags and i < len(panel_tags):
            ax.text(
                0.02, 0.98, f"({panel_tags[i]})",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=panel_fontsize, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=1.0),
                zorder=21
            )

        # Upper-right corner label with borehole title (e.g., AH1G, AH2G)
        if titles and i < len(titles):
            ax.text(
                0.98, 0.98, titles[i],
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=panel_fontsize * 0.85, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9),
                zorder=21
            )

        ax.grid(True, alpha=0.3)

        # --- Bedrock line, hatched gap to bedrock, and dark-gray bedrock band (match single heatmap) ---
        try:
            z_min = float(depth_grid.min())
            z_max = float(depth_grid.max())
            dz0 = dz0 if 'dz0' in locals() else (depth_grid[1] - depth_grid[0] if len(depth_grid) > 1 else 1.0)
            z_range = max(z_max - z_min, dz0)

            br = meta.get("bedrock_depth", None)
            if br is not None and np.isfinite(br):
                br = float(br)
                bedrock_color = "0.35"
                rock_height = 0.10 * z_range
                rock_bottom = br + rock_height

                # Bedrock line
                ax.axhline(br, color="0.25", linewidth=2.2, linestyle="-", zorder=6)

                # Hatched zone between deepest sensor and bedrock
                if br > z_max:
                    hk = {
                        "facecolor": "none",
                        "edgecolor": "0.25",
                        "linewidth": 0.8,
                        "hatch": "///",
                        "alpha": 0.7,
                        "zorder": 5
                    }
                    ax.fill_between(t_edges, z_max, br, **hk)

                # Dark-gray bedrock band
                ax.fill_between(
                    t_edges,
                    br,
                    rock_bottom,
                    facecolor=bedrock_color,
                    edgecolor=bedrock_color,
                    linewidth=0.0,
                    alpha=1.0,
                    zorder=4,
                )

                # Extend y-limits to show the bedrock band (respect inverted axis)
                ax.set_ylim(rock_bottom, ax.get_ylim()[1])
        except Exception:
            pass

        # Time series row
        if two_rows:
            ax_ts = ts_axes[i]
            raw_df = meta.get("raw_df")
            sensor_depths = meta.get("sensor_depths")
            if raw_df is None or sensor_depths is None:
                raise ValueError("two_rows=True requires meta to include 'raw_df' and 'sensor_depths'.")
            sensor_cols = [c for c in raw_df.columns if c in sensor_depths]
            sensor_cols = sorted(sensor_cols, key=lambda c: sensor_depths[c])
            colors = _line_colors(len(sensor_cols))
            for ccol, col in enumerate(sensor_cols):
                series = pd.to_numeric(raw_df[col], errors="coerce")
                if not np.isfinite(series).any():
                    continue
                ax_ts.plot(raw_df.index, series, color=colors[ccol],
                           linewidth=line_width, alpha=line_alpha,
                           label=f"{sensor_depths[col]:.1f} m")
            ax_ts.axhline(0, color="k", linestyle="--", linewidth=1)
            if i == 0:
                ax_ts.set_ylabel("T [°C]", fontsize=figure_fontsize)
            else:
                ax_ts.set_ylabel("")
            ax_ts.tick_params(axis='both', labelsize=figure_fontsize * 0.9)
            ax_ts.grid(True, alpha=0.3)
            # Legend outside, centered below each time-series panel
            ax_ts.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=1,
                         facecolor="white", fontsize=figure_fontsize * 0.75,
                         loc="upper center", bbox_to_anchor=(0.5, -0.65), ncol=3)

            # Optional per-panel y-limits
            if ts_y_limits is not None:
                if isinstance(ts_y_limits, (list, tuple)) and len(ts_y_limits) == n_panels:
                    yl = ts_y_limits[i]
                else:
                    yl = ts_y_limits  # apply same to all
                if yl is not None and len(yl) == 2:
                    ax_ts.set_ylim(float(yl[0]), float(yl[1]))

            ax_ts.set_xlim(t_edges[0], t_edges[-1])
            # No per-panel x-labels in two-row mode; use shared label instead
            ax_ts.set_xlabel("")
            # Rotate x-tick labels
            for lbl in ax_ts.get_xticklabels():
                lbl.set_rotation(45)

            # Optional per-panel y-tick steps; skip if empty tuple or non-numeric
            if ts_y_tick_steps not in (None, ()):
                step_val = ts_y_tick_steps[i] if isinstance(ts_y_tick_steps, (list, tuple)) and len(ts_y_tick_steps) == n_panels else ts_y_tick_steps
                try:
                    step = float(step_val)
                    if step > 0:
                        ymin, ymax = ax_ts.get_ylim()
                        y0 = np.floor(ymin / step) * step
                        y1 = np.ceil(ymax / step) * step
                        ax_ts.set_yticks(np.arange(y0, y1 + step * 0.5, step))
                except (TypeError, ValueError):
                    pass
            elif i == center_idx:
                ymin, ymax = ax_ts.get_ylim()
                yticks = np.arange(np.floor(ymin), np.ceil(ymax) + 0.5, 1.0)
                ax_ts.set_yticks(yticks)

    # Shared horizontal colorbar
    if im is not None:
        fig.tight_layout()
        ref_axes = list(ts_axes) if two_rows else list(heat_axes)
        left = min(ax.get_position().x0 for ax in ref_axes)
        right = max(ax.get_position().x1 for ax in ref_axes)
        bottom = min(ax.get_position().y0 for ax in ref_axes)
        pad = 0.37 if two_rows else 0.18   # extra space so legends don’t overlap colorbar
        height = 0.05
        cax = fig.add_axes([left, bottom - pad, right - left, height])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(r"Ice Temperature [°C]", fontsize=figure_fontsize)
        cold_ticks = [lv for lv in levels if lv < 0.0 and not np.isclose(lv, -0.1, atol=1e-2)]
        cb.set_ticks(cold_ticks + [0.0])
        cb.set_ticklabels([f"{t:.1f}" for t in cold_ticks] + [r"≈T$_{pmp}$"])
        cb.ax.tick_params(labelsize=figure_fontsize)
        ticks_txt = cb.ax.get_xticklabels()
        if ticks_txt:
            ticks_txt[-1].set_ha('left')
            ticks_txt[-1].set_x(ticks_txt[-1].get_position()[0] + 0.08)

    # Shared x-label for time (centered between panels) when two_rows is enabled
    if two_rows:
        fig.supxlabel("Time", fontsize=figure_fontsize, y=0.19)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"pdf.compression": 9}):
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, heat_axes if not two_rows else (heat_axes, ts_axes)

def plot_gp_statistics(
    meta_dict,
    savepath=None,
    title=None,
    borehole_label=None,  # optional override
    figsize=(12, 8),
    dpi=300,
    base_fontsize=14,
    show_boxplot=True,
    show_table=True,
    seasonal_months=None,  # e.g., [(6,7,8), (12,1,2)] for summer/winter
    equilibration_days=4   # exclude first N days
):
    """
    Create statistical summary of GeoPrecision temperature data with table and boxplot.
    
    Parameters
    ----------
    meta_dict : dict
        Output from plot_chain_temperature_heatmap containing:
        - 'raw_df': DataFrame with calibrated temperature data (TIME as index)
        - 'sensor_depths': dict mapping sensor columns to depths
        - 'borehole_label': borehole identifier (optional, can override)
    savepath : str, optional
        Path to save the figure
    title : str, optional
        Plot title (defaults to "{label} Temperature Statistics")
    borehole_label : str, optional
        Override borehole label from meta_dict
    figsize : tuple
        Figure size
    dpi : int
        Figure DPI
    base_fontsize : int
        Base font size
    show_boxplot : bool
        Show boxplot of temperature distributions
    show_table : bool
        Show statistical summary table
    seasonal_months : list[tuple], optional
        List of month tuples for seasonal analysis, e.g., [(6,7,8), (12,1,2)]
    equilibration_days : int
        Number of days to skip at start (default 4)
    
    Returns
    -------
    fig, stats_df
        Figure and DataFrame with statistics
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Extract from meta dictionary
    data = meta_dict['raw_df'].copy()
    depths_dict = meta_dict['sensor_depths']
    label = borehole_label or meta_dict.get('borehole_label', 'Chain')
    
    if data is None or data.empty:
        raise ValueError("No data available in meta dictionary")
    
    # Reset index to make TIME a column
    if data.index.name == 'TIME':
        data = data.reset_index()
    
    # Process timestamps and skip equilibration period
    data['TIME'] = pd.to_datetime(data['TIME'])
    data = data.sort_values('TIME')
    
    if equilibration_days > 0:
        cutoff = data['TIME'].min() + pd.Timedelta(days=equilibration_days)
        data = data[data['TIME'] >= cutoff]
    
    if data.empty:
        raise ValueError("No data remaining after equilibration period")
    
    # Identify sensor columns (all columns except TIME)
    sensor_cols = [col for col in data.columns if col != 'TIME' and col in depths_dict]
    
    if not sensor_cols:
        raise ValueError("No valid sensor columns found with depths")
    
    # Sort by depth
    sensor_cols = sorted(sensor_cols, key=lambda c: float(depths_dict[c]))
    
    # Color from color map
    color_map = build_profile_color_map([label])
    color = color_map.get(label, cmc.batlowK(0.5))
    
    def compute_statistics(data_df, sensor_columns, depths):
        """Compute comprehensive statistics for each sensor"""
        stats_list = []
        
        # Add month column for seasonal analysis
        data_df = data_df.copy()
        data_df['month'] = data_df['TIME'].dt.month
        
        for sensor in sensor_columns:
            depth = float(depths[sensor])
            temps = pd.to_numeric(data_df[sensor], errors='coerce').dropna()
            
            if temps.empty:
                continue
            
            stats = {
                'Borehole': label,
                'Sensor': sensor,
                'Depth [m]': depth,
                'Mean [°C]': temps.mean(),
                'Median [°C]': temps.median(),
                'Std Dev [°C]': temps.std(),
                'Min [°C]': temps.min(),
                'Max [°C]': temps.max(),
                'Range [°C]': temps.max() - temps.min(),
                'Q25 [°C]': temps.quantile(0.25),
                'Q75 [°C]': temps.quantile(0.75),
                'IQR [°C]': temps.quantile(0.75) - temps.quantile(0.25),
                'N samples': len(temps)
            }
            
            # Seasonal statistics
            if seasonal_months:
                for i, months in enumerate(seasonal_months, 1):
                    mask = data_df['month'].isin(months)
                    seasonal_temps = data_df.loc[mask, sensor]
                    seasonal_temps = pd.to_numeric(seasonal_temps, errors='coerce').dropna()
                    
                    if not seasonal_temps.empty:
                        stats[f'Season{i} Mean [°C]'] = seasonal_temps.mean()
                        stats[f'Season{i} Std [°C]'] = seasonal_temps.std()
                
                # Annual amplitude
                if len(seasonal_months) == 2:
                    s1_key = f'Season1 Mean [°C]'
                    s2_key = f'Season2 Mean [°C]'
                    if s1_key in stats and s2_key in stats:
                        stats['Annual Amplitude [°C]'] = abs(stats[s1_key] - stats[s2_key])
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list) if stats_list else None
    
    # Compute statistics
    stats_df = compute_statistics(data, sensor_cols, depths_dict)
    
    if stats_df is None or stats_df.empty:
        raise ValueError("No valid statistics computed")
    
    # Create figure
    n_panels = sum([show_table, show_boxplot])
    if n_panels == 0:
        raise ValueError("Must show at least table or boxplot")
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    if show_table and show_boxplot:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
        ax_table = fig.add_subplot(gs[0])
        ax_box = fig.add_subplot(gs[1])
    elif show_table:
        ax_table = fig.add_subplot(111)
        ax_box = None
    else:
        ax_table = None
        ax_box = fig.add_subplot(111)
    
    # Table
    if show_table and ax_table:
        ax_table.axis('off')
        
        # Display columns
        display_cols = ['Sensor', 'Depth [m]', 'Mean [°C]', 'Std Dev [°C]', 
                        'Min [°C]', 'Max [°C]', 'Range [°C]', 'N samples']
        
        # Add seasonal columns if present
        season_cols = [c for c in stats_df.columns if 'Season' in c or 'Amplitude' in c]
        display_cols.extend(season_cols)
        
        table_data = stats_df[display_cols].copy()
        
        # Round numeric columns
        for col in table_data.columns:
            if '[°C]' in col or '[m]' in col:
                table_data[col] = table_data[col].round(3)
            elif 'N samples' in col:
                table_data[col] = table_data[col].astype(int)
        
        # Create table
        cell_text = table_data.values.tolist()
        col_labels = [col.replace(' [°C]', '\n[°C]').replace(' [m]', '\n[m]') for col in display_cols]
        
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(base_fontsize * 0.7)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternating row colors
        for i in range(1, len(cell_text) + 1):
            for j in range(len(col_labels)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor('white')
        
        ax_table.set_title(title or f'{label} Temperature Statistics', 
                          fontsize=base_fontsize * 1.2, weight='bold', pad=20)
    
    # Boxplot
    if show_boxplot and ax_box:
        box_data = []
        box_labels = []
        
        for sensor in sensor_cols:
            temps = pd.to_numeric(data[sensor], errors='coerce').dropna()
            if not temps.empty:
                box_data.append(temps.values)
                depth = depths_dict[sensor]
                box_labels.append(f"{sensor}\n({depth:.1f}m)")
        
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                            notch=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', 
                                          markeredgecolor='darkred', markersize=6))
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax_box.set_ylabel('Temperature [°C]', fontsize=base_fontsize)
        ax_box.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_box.grid(True, alpha=0.3, axis='y')
        ax_box.tick_params(labelsize=base_fontsize * 0.9)
        
        if not show_table:
            ax_box.set_title(title or f'{label} Temperature Distribution', 
                            fontsize=base_fontsize * 1.2, weight='bold')
    
    fig.tight_layout()
    
    # Save
    if savepath:
        out_path = Path(savepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        
        # Also save CSV
        csv_path = out_path.with_suffix('.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to: {csv_path}")
    
    return fig, stats_df

def plot_ntc_temperature_heatmap(
    thermistor,
    *,
    logger_id,
    offsets_df,
    depth_file=None,
    start_time=None,
    end_time=None,
    time_freq="12H",
    depth_step=0.05,
    smooth_time_sigma=0.0,
    smooth_depth_sigma=0.0,
    temp_step=0.1,
    title=None,
    savepath=None,
    clip_to_data=True,    show_zaa=False,    zaa_threshold=0.2,
    zaa_extrapolate=False,
    zaa_kwargs=None,
):
    """
    TinyTag NTC heatmap: depth vs time using only the span between white/black probes.
    Above the shallower probe and below the deeper probe remains NaN if clip_to_data=True.

    Parameters
    ----------
    thermistor : ThermistorData
    logger_id  : str | int
        Logger number in offsets_df.
    offsets_df : pd.DataFrame
        Contains TinyTag zero-degree offsets (per logger).
    depth_file : str
        CSV with columns:
          - 'date' (dd.mm.yyyy or ISO)
          - 'depth white probe [m]' (name can vary)
          - 'depth black probe [m]' (name can vary)
    """
    # 1) Data and offsets
    df_raw = thermistor.get_ntc_data_with_offsets(logger_id=str(logger_id), offsets_df=offsets_df)
    if df_raw is None or df_raw.empty:
        raise ValueError("No TinyTag data found for the provided logger_id and offsets.")

    df_raw = df_raw.copy()
    df_raw['TIME'] = pd.to_datetime(df_raw['TIME'])
    df_raw = df_raw.sort_values('TIME')

    # Apply time window
    if start_time is not None:
        st = pd.to_datetime(start_time, dayfirst=True, errors='coerce')
        if pd.isna(st):
            st = pd.to_datetime(start_time)
        df_raw = df_raw[df_raw['TIME'] >= st]
    if end_time is not None:
        et = pd.to_datetime(end_time, dayfirst=True, errors='coerce')
        if pd.isna(et):
            et = pd.to_datetime(end_time)
        df_raw = df_raw[df_raw['TIME'] <= et]

    if df_raw.empty:
        raise ValueError("No data in the selected time range.")

    # 2) Resample to regular time grid
    df_res = (df_raw.set_index('TIME')[['Black Probe Temperature', 'White Probe Temperature']]
                    .resample(time_freq).mean()
                    .interpolate(method='time', limit_direction='both'))
    times = df_res.index
    if len(times) < 2:
        raise ValueError("Need at least two time steps for the heatmap.")

    # 3) Read probe depths table and interpolate to time grid
    if depth_file is None:
        raise ValueError("depth_file is required for probe depths.")
    depths = pd.read_csv(depth_file)
    depths = depths.dropna(how="all")  # drop empty rows like ,,,,,
    # Normalize column names
    cols = {c: c.strip().lower() for c in depths.columns}
    depths.rename(columns=cols, inplace=True)

    col_date  = "date"
    col_white = next((c for c in depths.columns if "depth white probe" in c), None)
    col_black = next((c for c in depths.columns if "depth black probe" in c), None)
    if col_white is None or col_black is None or col_date not in depths.columns:
        raise ValueError("Depth file must have 'date', 'depth white probe [m]', and 'depth black probe [m]' columns.")

    # Parse dates like 08/08/2024 (dd/mm/yyyy)
    depths["_date"] = pd.to_datetime(depths[col_date].astype(str).str.strip(),
                                     format="%d/%m/%Y", errors="coerce")
    depths = depths[~depths["_date"].isna()]

    # Coerce numeric
    depths["white"] = pd.to_numeric(depths[col_white], errors="coerce")
    depths["black"] = pd.to_numeric(depths[col_black], errors="coerce")

    # Deduplicate by date (some files may have multiple entries per day)
    d_agg = depths.groupby("_date", as_index=True)[["white", "black"]].mean().sort_index()

    # Daily series, then align to measurement times
    s_white = d_agg["white"].asfreq("D").interpolate("time").ffill().bfill()
    s_black = d_agg["black"].asfreq("D").interpolate("time").ffill().bfill()

    white_on_time = s_white.reindex(times).interpolate("time").ffill().bfill()
    black_on_time = s_black.reindex(times).interpolate("time").ffill().bfill()


    # 4) Depth grid only between min/max observed probe depths (non-extrapolating)
    zmin_obs = float(np.nanmin([white_on_time.min(), black_on_time.min()]))
    zmax_obs = float(np.nanmax([white_on_time.max(), black_on_time.max()]))
    depth_grid = np.arange(zmin_obs, zmax_obs + depth_step * 0.51, depth_step)

    # 5) Fill grid only between the two probes (no extrapolation outside)
    nt, nz = len(times), len(depth_grid)
    grid = np.full((nt, nz), np.nan, dtype=float)

    wtemps = df_res['White Probe Temperature'].to_numpy(float)
    btemps = df_res['Black Probe Temperature'].to_numpy(float)
    wdepths = white_on_time.to_numpy(float)
    bdepths = black_on_time.to_numpy(float)

    for i in range(nt):
        tw, tb = wtemps[i], btemps[i]
        dw, db = wdepths[i], bdepths[i]
        if not (np.isfinite(tw) and np.isfinite(tb) and np.isfinite(dw) and np.isfinite(db)):
            continue
        d_pair = np.array([dw, db], float)
        t_pair = np.array([tw, tb], float)
        order = np.argsort(d_pair)
        d_pair = d_pair[order]
        t_pair = t_pair[order]

        inside = (depth_grid >= d_pair[0]) & (depth_grid <= d_pair[1])
        if inside.any():
            grid[i, inside] = np.interp(depth_grid[inside], d_pair, t_pair)

    # 6) Optional smoothing (preserve NaNs outside probe span)
    if (smooth_time_sigma and smooth_time_sigma > 0) or (smooth_depth_sigma and smooth_depth_sigma > 0):
        valid = np.isfinite(grid)
        filled = grid.copy()

        # Fill gaps locally for smoothing stability
        for j in range(nz):
            col = filled[:, j]
            nmask = ~np.isfinite(col)
            if nmask.any() and (~nmask).any():
                col[nmask] = np.interp(np.flatnonzero(nmask), np.flatnonzero(~nmask), col[~nmask])
            filled[:, j] = col
        for i in range(nt):
            row = filled[i, :]
            nmask = ~np.isfinite(row)
            if nmask.any() and (~nmask).any():
                row[nmask] = np.interp(np.flatnonzero(nmask), np.flatnonzero(~nmask), row[~nmask])
            filled[i, :] = row

        filled = gaussian_filter(filled, sigma=(smooth_time_sigma, smooth_depth_sigma), mode='nearest')
        filled[~valid] = np.nan
        grid = filled

    # 7) Colormap/levels
    data_vals = grid[np.isfinite(grid)]
    if data_vals.size == 0:
        raise ValueError("All interpolated values are NaN.")
    vmin = np.floor(data_vals.min() / temp_step) * temp_step
    vmax = 0.0  # keep <= 0°C domain by default
    levels = np.arange(vmin, vmax + temp_step * 0.99, temp_step)

    base = cmc.vik
    n_bins = len(levels) - 1
    if n_bins <= 2:
        colors = base(np.linspace(0.3, 0.85, max(n_bins, 1)))
    else:
        cold = base(np.linspace(0.10, 0.55, max(1, n_bins - 1)))
        warm = base(np.linspace(0.90, 0.98, 1))
        colors = np.vstack([cold, warm])
    cmap = ListedColormap(colors, name="icetemp_tyntag")
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    # 8) Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
    t_num = mdates.date2num(times.to_pydatetime())
    dt = np.diff(t_num)
    dt0 = dt[0] if len(dt) else 1.0
    dtn = dt[-1] if len(dt) else 1.0
    t_edges = np.r_[t_num[0] - dt0 / 2, t_num[:-1] + dt / 2, t_num[-1] + dtn / 2]
    z_edges = np.r_[depth_grid[0] - depth_step / 2, depth_grid[:-1] + depth_step / 2, depth_grid[-1] + depth_step / 2]

    pcm = ax.pcolormesh(t_edges, z_edges, grid.T, cmap=cmap, norm=norm, shading='auto')

    locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()

    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Time")

    # Optional contours
    try:
        ax.contour(t_num, depth_grid, grid.T, levels=levels, colors='k', linewidths=0.3, alpha=0.35)
    except Exception:
        pass

    # ZAA overlay (dotted line)
    if show_zaa:
        try:
            zaa_res = compute_tynitag_zaa(
                thermistor,
                logger_id=str(logger_id),
                offsets_df=offsets_df,
                depth_file=depth_file,
                start_time=start_time,
                end_time=end_time,
                zaa_threshold=zaa_threshold,
                zaa_extrapolate=zaa_extrapolate,
            )
            zaa_val = zaa_res.get("zaa_depth", np.nan) if isinstance(zaa_res, dict) else np.nan
            if np.isfinite(zaa_val):
                line_kwargs = {"color": "k", "linestyle": ":", "linewidth": 2, "label": "ZAA"}
                if zaa_kwargs:
                    line_kwargs.update(zaa_kwargs)
                ax.axhline(zaa_val, **line_kwargs)
        except Exception as ex:
            print(f"ZAA overlay failed: {ex}")

    # Colorbar and styling
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Ice Temperature [°C]")
    cb.set_ticks(levels)
    cb.ax.set_yticklabels([f"{lv:.1f}" for lv in levels])

    try:
        from processing.gpr_plotting import format_plot
        format_plot(ax=ax, title=title, legend_loc='upper right', x_tick_rotation=0, y_tick_rotation=0, cbar=cb)
    except Exception:
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    # Optionally mask outside-probe regions (already NaN), keeping interface simple
    if not clip_to_data:
        # If a filled look is desired, you could fill with nearest here.
        pass

    return fig, ax, {
        "time_index": times,
        "depth_grid": depth_grid,
        "grid": grid,
        "levels": levels,
        "white_depths": white_on_time,
        "black_depths": black_on_time
    }

# Helper: sample batlowK excluding the brightest end (top 20%)
def _batlow_trunc(n: int, start: float = 0.0, end: float = 0.8):
    """Return n colors from cmc.batlowK sampled in [start, end]; default drops top 20% (brightest)."""
    start = float(np.clip(start, 0.0, 1.0))
    end = float(np.clip(end, 0.0, 1.0))
    if end <= start:
        end = min(1.0, start + 0.01)
    positions = np.linspace(start, end, max(1, int(n)))
    return cmc.batlowK(positions)

def _disperse_positions(k: int, start: float, end: float, reserved: list[float]) -> list[float]:
    """
    Greedy max‑separation sampler: add k midpoints between existing seeds
    (start/end + reserved), producing well‑spaced positions in [start, end].
    """
    seeds = sorted([start, end] + [float(p) for p in reserved if start <= float(p) <= end])
    for _ in range(max(0, k)):
        # find largest interval
        gaps = [(seeds[i], seeds[i+1], seeds[i+1]-seeds[i]) for i in range(len(seeds)-1)]
        a, b, _ = max(gaps, key=lambda g: g[2])
        seeds.append((a + b) / 2.0)
        seeds.sort()
    # return only the added k midpoints (exclude start/end and original reserved)
    out = [p for p in seeds if p not in ([start, end] + reserved)]
    # If we added more than k (shouldn't), trim; if less (degenerate), fill with linspace
    if len(out) > k:
        out = out[:k]
    if len(out) < k:
        fill = np.linspace(start, end, k+2)[1:-1].tolist()
        out = (out + fill)[:k]
    return out

def build_profile_color_map(labels):
    """
    Categorical color mapping from cmcrameri 'romaO' (cyclic, perceptually uniform).
    Sampled evenly across [0.05, 0.75] to get muted but distinct hues.
    """
    labs = [str(l) for l in (list(labels or []))]
    if not labs:
        return {}
    positions = np.linspace(0.05, 0.75, len(labs))
    return {lab: cmc.romaO(float(pos)) for lab, pos in zip(labs, positions)}