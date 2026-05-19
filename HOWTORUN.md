# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/sdrocer/polythermal_swiss_glaciers.git
cd polythermal_swiss_glaciers
```

## 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate polythermal_swiss_glaciers
```

## 3. Obtain the data

The raw borehole temperature and GPR data are not included in this repository. Download them from the sources listed in the [Data Availability](README.md#data-availability) section of the README and place them in a local directory of your choice.

## 4. Set the data path

At the top of each notebook, set the `DATA_ROOT` variable to point to your local data directory:

```python
DATA_ROOT = "/path/to/your/data"
```

## 5. Run the notebooks in order

The notebooks are numbered and should be run sequentially, as later notebooks depend on outputs from earlier ones:

| Notebook | Description |
|---|---|
| `1_study_sites_and_maps.ipynb` | Field site overview maps and GPR ice thickness maps |
| `2_instrument_calibration.ipynb` | Geoprecision chain and Tynitag NTC calibration |
| `3_temperature_profiles_and_historical_comparison.ipynb` | Temperature profiles, heatmaps, and historical comparison |
| `4_data_processing.ipynb` | Load and export full borehole timeseries |
| `5_temperature_timeseries.ipynb` | Englacial temperature timeseries (fig04) |
| `6_englacial_profiles_and_metrics.ipynb` | Interpolated 2D profiles and thermistor metrics |
| `7_firn_and_mass_balance.ipynb` | Firn change and mass balance figures |
| `8_glenglat.ipynb` | glenglat database analysis and data submission |
