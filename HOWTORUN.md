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

## 4. Configure your data paths

Copy the path configuration template and edit it to match your local directory layout:

```bash
cp src/config_template.py src/config.py
```

Then open `src/config.py` and set the four path variables:

```python
ICETEMP_ROOT = "/path/to/fieldwork_data/THERMAP_2024_2025/icetemperature_data"
GPR_ROOT     = "/path/to/fieldwork_data/THERMAP_2024_2025/gpr"
RGI_ROOT     = "/path/to/data/RGI"
GLENGLAT_DB  = "/path/to/data/ice_temperature/glenglat_database"
```

`config.py` is gitignored — your personal paths will never be committed to the repository.

## 5. Run the notebooks in order

The notebooks are numbered and should be run sequentially, as later notebooks depend on outputs from earlier ones. `10_extended_timeseries_2026.ipynb` is the exception — it's optional and standalone, not a dependency of any other notebook:

| Notebook | Description |
|---|---|
| `01_study_sites_and_maps.ipynb` | Field site overview maps and GPR ice thickness maps (fig01) |
| `02_instrument_calibration.ipynb` | Geoprecision chain and Tinytag NTC calibration |
| `03_temperature_profiles_and_historical_comparison.ipynb` | Temperature profiles, heatmaps, and historical comparison (fig04) |
| `04_data_processing.ipynb` | Load and export full borehole timeseries |
| `05_temperature_timeseries.ipynb` | Englacial temperature timeseries (fig03) |
| `06_englacial_profiles_and_metrics.ipynb` | Interpolated 2D profiles and thermistor metrics (fig05-07) |
| `07_firn_and_mass_balance.ipynb` | Firn change and mass balance figures (fig08) |
| `08_glenglat.ipynb` | glenglat database analysis and data submission |
| `09_firn_validation.ipynb` | Firn map validation against historical orthophotos (figS12-S13) |
| `10_extended_timeseries_2026.ipynb` | Optional: splices July 2026 read-outs onto the timeseries and produces working (non-paper) extended versions of fig03 and figS14-S16 |
