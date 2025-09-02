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