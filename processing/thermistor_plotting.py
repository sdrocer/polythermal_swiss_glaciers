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

    def plot_multiple_temperature_profiles(self, snapshot_time, offsets_list, depth_files, figsize=(3.6, 4.2), dpi=250, xtick_rotation=0, labels=None,
                                           savepath=None, title=None, ntc_data_list=None,
                                           base_fontsize: int = 14, show_title: bool = False):
        """
        Plot temperature profiles for multiple GeoPrecision chains (daily mean at snapshot_time)
        and optional TinyTag/NTC boreholes (single or averaged values). Colors are taken from
        build_profile_color_map so they match other figures.

        Parameters
        ----------
        snapshot_time : str | datetime-like
            Date of the daily mean (e.g., '20250916' or '16/09/2025').
        offsets_list : list
            Per-chain offset rows/Series (same order as self.file_paths).
        depth_files : list[str]
            CSV paths with sensor depths; if ntc_data_list is provided, append its depth files at the end.
        ntc_data_list : list[pd.DataFrame] | None
            Optional TinyTag data frames containing 'White Probe Temperature' and 'Black Probe Temperature'.
            If multiple rows, their mean is used.
        """
        # Normalize snapshot date
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

        # Colors
        color_map = build_profile_color_map(labels)
        fallback_colors = _batlow_trunc(max(total_series, 1), start=0.0, end=0.8)

        # Common excludes
        exclude_cols = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}

        # 1) Plot GeoPrecision chains
        for i in range(n_profiles):
            fp = self.file_paths[i]
            offs = offsets_list[i] if i < len(offsets_list) else None
            dfile = depth_files[i] if i < len(depth_files) else None
            label = labels[i] if i < len(labels) else f'Chain {i+1}'

            if dfile is None:
                print(f"Warning: missing depth file for chain index {i}; skipping.")
                continue
            try:
                depths_dict = read_thermistor_depths(dfile)
            except Exception as e:
                print(f"Warning: failed to read depths for chain {fp}: {e}")
                continue

            thermistor = ThermistorData(fp, self.delimiter, self.measurement_depth)
            day_df = thermistor.get_chain_data_with_offsets(offsets=offs, snapshot_day=snapshot_time)
            if day_df is None or day_df.empty:
                continue

            depths, temps = [], []
            for sensor, depth in depths_dict.items():
                if depth is None:
                    continue
                if sensor in day_df.columns and sensor not in exclude_cols:
                    series = pd.to_numeric(day_df[sensor], errors='coerce')
                    m = series.mean(skipna=True)
                    if np.isfinite(m):
                        depths.append(float(depth))
                        temps.append(float(m))

            if depths:
                depths, temps = zip(*sorted(zip(depths, temps)))
                color = color_map.get(label, fallback_colors[i])
                ax.plot(temps, depths, 'o-', label=label, color=color, linewidth=2.5)

        # 2) Plot TinyTag/NTC points or short profiles
        if n_ntc > 0:
            ntc_depth_files = depth_files[n_profiles:n_profiles + n_ntc]
            for j in range(n_ntc):
                color_idx = n_profiles + j
                ntc_df = ntc_data_list[j]
                dfile = ntc_depth_files[j] if j < len(ntc_depth_files) else None
                label = labels[n_profiles + j] if (n_profiles + j) < len(labels) else f'NTC {j+1}'

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

                # Temperatures (mean if multiple rows)
                try:
                    t_white = pd.to_numeric(ntc_df['White Probe Temperature'], errors='coerce').mean()
                    t_black = pd.to_numeric(ntc_df['Black Probe Temperature'], errors='coerce').mean()
                except Exception as e:
                    print(f"Warning: missing NTC temperature columns for {label}: {e}")
                    continue

                temps_ntc, depths_ntc = [], []
                if depth_white is not None and np.isfinite(t_white):
                    temps_ntc.append(float(t_white)); depths_ntc.append(float(depth_white))
                if depth_black is not None and np.isfinite(t_black):
                    temps_ntc.append(float(t_black)); depths_ntc.append(float(depth_black))

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

        # Title/legend via formatter (legend drawn below with sharp frame)
        self.format_plot(None, xtick_rotation=xtick_rotation, legend_loc='best', show_legend=False, base_fontsize=base_fontsize)
        if show_title and title:
            ax.set_title(title, fontsize=max(10, int(base_fontsize)))

        # Sharp-corner legend
        if labels and len(labels) > 0:
            ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, facecolor='white',
                      loc='best', fontsize=max(8, int(base_fontsize)))

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
        # annotation controls
        annotation_y=None,
        annotation_spacing: float = 0.08,
        annotation_arrow_hide_dy: float = 0.06,
        annotation_dx_pts: int = 6,
        annotation_positions=None,
        annotation_fontsize: int | None = None
    ):
        """
        Plot two TinyTag (NTC) boreholes (each with white/black probes) on one axis.
        Colors: cmcrameri 'batlowK' (left half). Labels ending with 1TT/2TT get indices 2/3.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import transforms as mtransforms
        from matplotlib import dates as mdates

        # Colors: deterministic via build_profile_color_map (batlowK left half)
        if not (borehole_labels and len(borehole_labels) >= 2):
            raise ValueError("borehole_labels must contain two labels like ['SR1TT','SR2TT'].")
        cmap_dict = build_profile_color_map(borehole_labels[:2])
        color_bh1 = cmap_dict[borehole_labels[0]]
        color_bh2 = cmap_dict[borehole_labels[1]]

        # Prepare axis/figure
        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(9, 7), dpi=300)
            ax = plt.gca()
            created_fig = True
        else:
            fig = ax.figure

        if depths is None or len(depths) != 4:
            raise ValueError("depths must be 4 values: [BH1 white, BH1 black, BH2 white, BH2 black]")
        depths = [float(d) for d in depths]

        # Load up to two TinyTag files
        fps = getattr(self, "file_paths", [getattr(self, "file_path", None)])
        delim = getattr(self, "delimiter", ",")
        meas  = getattr(self, "measurement_depth", None)

        ntc_thermistor_data1 = pd.DataFrame()
        ntc_thermistor_data2 = pd.DataFrame()
        if fps and fps[0]:
            t1 = ThermistorData(fps[0], delim, meas)
            ntc_thermistor_data1 = t1.get_ntc_data()
        if len(fps) > 1 and fps[1]:
            t2 = ThermistorData(fps[1], delim, meas)
            ntc_thermistor_data2 = t2.get_ntc_data()

        # Optional smoothing (6H resample + centered rolling mean)
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

        if smooth_days and float(smooth_days) > 0:
            ntc_thermistor_data1 = _smooth_df(ntc_thermistor_data1, smooth_days)
            ntc_thermistor_data2 = _smooth_df(ntc_thermistor_data2, smooth_days)

        # Standardize TIME dtype and sort
        for df_ in (ntc_thermistor_data1, ntc_thermistor_data2):
            if not df_.empty:
                df_["TIME"] = pd.to_datetime(df_["TIME"])
                df_.sort_values("TIME", inplace=True)

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

        # Initial depths (for label text)
        i_w1 = i_b1 = i_w2 = i_b2 = None
        if isinstance(initial_depths, (list, tuple)) and len(initial_depths) >= 4:
            i_w1, i_b1, i_w2, i_b2 = initial_depths[:4]

        # Style
        lw = 3
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
            a_w1 = _alpha_for_depth(depths[0], dmin, dmax)
            a_b1 = _alpha_for_depth(depths[1], dmin, dmax)
            ax.plot(ntc_thermistor_data1['TIME'], ntc_thermistor_data1['White Probe Temperature'],
                    linestyle='-', color=color_bh1, linewidth=lw, alpha=a_w1)
            ax.plot(ntc_thermistor_data1['TIME'], ntc_thermistor_data1['Black Probe Temperature'],
                    linestyle='-', color=color_bh1, linewidth=lw, alpha=a_b1)

        if not ntc_thermistor_data2.empty:
            a_w2 = _alpha_for_depth(depths[2], dmin, dmax)
            a_b2 = _alpha_for_depth(depths[3], dmin, dmax)
            ax.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['White Probe Temperature'],
                    linestyle='-', color=color_bh2, linewidth=lw, alpha=a_w2)
            ax.plot(ntc_thermistor_data2['TIME'], ntc_thermistor_data2['Black Probe Temperature'],
                    linestyle='-', color=color_bh2, linewidth=lw, alpha=a_b2)

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

            # Keep label inside y-lims
            y_min, y_max = ax.get_ylim()
            pad = 0.02 * (y_max - y_min)
            y_target = max(min(float(y_target), y_max - pad), y_min + pad)

            # Value of line at x_target for arrow decision
            y_line = _y_at_time(df, series, x_target)
            draw_arrow = (y_line is not None) and (abs(float(y_target) - float(y_line)) >= float(annotation_arrow_hide_dy))

            # Optional connector
            if draw_arrow:
                ax.annotate("", xy=(x_target, y_line), xytext=(x_target, y_target),
                            textcoords="data", arrowprops=dict(arrowstyle='-', color=color, lw=0.9, alpha=0.9),
                            annotation_clip=True)

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
        if not ntc_thermistor_data2.empty:
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

        # Deployment marker & label on every panel
        if deployment_date is not None and pd.notna(deployment_date):
            fs = max(10, int(base_fontsize * 0.75))
            ax.axvline(deployment_date, color='gray', linestyle='solid', linewidth=2, alpha=0.9, zorder=0)
            ax.text(deployment_date, ax.get_ylim()[0], 'Deployment', color='gray',
                    fontsize=fs, va='top', ha='right', rotation=45, alpha=0.9)

        # Styling
        try:
            self.format_plot(title if show_title else None, legend_loc, show_legend=False, base_fontsize=base_fontsize)
        except Exception:
            pass

        # Optional in-axes legend
        if show_legend and borehole_labels and len(borehole_labels) >= 2:
            handles = [
                Line2D([0], [0], color=color_bh1, lw=lw, linestyle='-'),
                Line2D([0], [0], color=color_bh2, lw=lw, linestyle='-'),
            ]
            if legend_outside:
                ax.legend(handles, borehole_labels[:2], frameon=True, fancybox=False, edgecolor='black', framealpha=1,
                          facecolor='white', loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
            else:
                ax.legend(handles, borehole_labels[:2], frameon=True, fancybox=False, edgecolor='black', framealpha=1,
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
    vmax=0.0,
    figsize=(11, 5),
    dpi=300,
    cmap=None,
    show_colorbar=True,
    title=None,
    savepath=None
):
    if depth_file is None:
        raise ValueError("depth_file is required to determine sensor depths.")

    # Data with offsets
    df = thermistor.get_chain_data_with_offsets(
        start_time=start_time,
        end_time=end_time,
        offsets=offsets
    )
    if df.empty:
        raise ValueError("No data returned for specified time range.")

    # Depth mapping
    depths_dict = read_thermistor_depths(depth_file)
    exclude = {'NO', 'TIME', 'TEMP LOGGER', 'TEMP BATTERY', 'HK-BAT:V'}
    sensor_cols = [c for c in df.columns if c not in exclude and c in depths_dict and depths_dict[c] is not None]
    if not sensor_cols:
        raise ValueError("No matching sensor columns with valid depths found.")
    sensor_depths = {c: float(depths_dict[c]) for c in sensor_cols}
    sensor_cols = sorted(sensor_cols, key=lambda c: sensor_depths[c])
    depths_sorted = np.array([sensor_depths[c] for c in sensor_cols], float)

    # Time index and temporal interpolation
    df = df[['TIME'] + sensor_cols].copy()
    df['TIME'] = pd.to_datetime(df['TIME'])
    df = df.sort_values('TIME').drop_duplicates('TIME').set_index('TIME')
    df_res = df.resample(time_freq).mean().interpolate(method='time', limit_direction='both')
    temp_matrix = df_res.to_numpy(float)

    # Vertical grid and interpolation
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

    # Optional 2D smoothing
    if (smooth_time_sigma and smooth_time_sigma > 0) or (smooth_depth_sigma and smooth_depth_sigma > 0):
        valid = np.isfinite(grid)
        filled = grid.copy()
        # nearest 1D fills along axes
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

    # Color levels and colormap (match icetemp profile style)
    data_vals = grid[np.isfinite(grid)]
    if data_vals.size == 0:
        raise ValueError("All interpolated values are NaN.")
    vmin = np.floor(data_vals.min() / temp_step) * temp_step
    vmax = 0.0 if vmax < 0 else vmax
    levels = np.arange(vmin, vmax + temp_step * 0.99, temp_step)

    if cmap is None:
        base = cmc.vik
        n_bins = len(levels) - 1
        if n_bins <= 2:
            colors = base(np.linspace(0.3, 0.85, n_bins))
        else:
            cold = base(np.linspace(0.10, 0.55, max(1, n_bins - 1)))
            warm = base(np.linspace(0.90, 0.98, 1))
            colors = np.vstack([cold, warm])
        cmap = ListedColormap(colors, name="icetemp_chain")
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    times = df_res.index.to_pydatetime()
    if len(times) < 2:
        raise ValueError("Need at least two time steps for heatmap.")
    t_num = mdates.date2num(times)
    dt = np.diff(t_num)
    t_edges = np.r_[t_num[0] - dt[0] / 2, t_num[:-1] + dt / 2, t_num[-1] + dt[-1] / 2]
    z_edges = np.r_[depth_grid[0] - depth_step / 2, depth_grid[:-1] + depth_step / 2, depth_grid[-1] + depth_step / 2]

    pcm = ax.pcolormesh(t_edges, z_edges, grid.T, cmap=cmap, norm=norm, shading='auto')
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Optional contours
    try:
        ax.contour(mdates.date2num(times), depth_grid, grid.T, levels=levels, colors='k', linewidths=0.3, alpha=0.35)
    except Exception:
        pass

    # Colorbar
    cb = None
    if show_colorbar:
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label("Ice Temperature [°C]")
        cb.set_ticks(levels)
        cb.ax.set_yticklabels([f"{lv:.1f}" for lv in levels])

    # Use shared formatter
    try:
        from processing.gpr_plotting import format_plot
    except Exception:
        try:
            from .gpr_plotting import format_plot
        except Exception:
            format_plot = None
    if format_plot:
        format_plot(ax=ax, title=title, legend_loc='upper right',
                    x_tick_rotation=45, y_tick_rotation=0, cbar=cb)
    else:
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, {
        "time_index": df_res.index,
        "depth_grid": depth_grid,
        "grid": grid,
        "levels": levels
    }

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
    Categorical color mapping from cmcrameri 'batlowK' (exclude brightest 20%).
    Fixed anchors:
      - '*1TT' -> near dark end
      - '*2TT' -> near upper end of kept range
      - '*1JG' or '*1G' -> stable mid‑dark
      - '*2JG' or '*2G' -> stable mid‑high
      - '*3JG' or '*3G' -> stable low‑mid
    Remaining labels are spaced to maximize separation within [0.0, 0.8].
    """
    labs = [str(l) for l in (list(labels or []))]
    ulabels = [l.upper() for l in labs]

    # Range (drop brightest 20%)
    start, end = 0.0, 0.8

    # Anchors
    pos_1tt = start + 0.08   # dark side
    pos_2tt = end   - 0.06   # high side (still < 0.8)
    # JG/G anchors (chosen to match the visual style you liked)
    pos_1jg = 0.34          # olive/greenish
    pos_2jg = 0.58          # ochre/mustard
    pos_3jg = 0.20          # darker green/teal

    reserved_pos = {}
    for lab, u in zip(labs, ulabels):
        if u.endswith("1TT"):
            reserved_pos[lab] = pos_1tt
        elif u.endswith("2TT"):
            reserved_pos[lab] = pos_2tt
        elif u.endswith("1JG") or u.endswith("1G"):
            reserved_pos[lab] = pos_1jg
        elif u.endswith("2JG") or u.endswith("2G"):
            reserved_pos[lab] = pos_2jg
        elif u.endswith("3JG") or u.endswith("3G"):
            reserved_pos[lab] = pos_3jg

    # Disperse remaining labels
    remaining = [lab for lab in labs if lab not in reserved_pos]
    positions_rem = _disperse_positions(len(remaining), start, end, list(reserved_pos.values()))
    for lab, pos in zip(remaining, positions_rem):
        reserved_pos[lab] = float(pos)

    # Convert to colors
    colors = {lab: cmc.batlowK(float(np.clip(pos, 0.0, 1.0))) for lab, pos in reserved_pos.items()}
    return colors