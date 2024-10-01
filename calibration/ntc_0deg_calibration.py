"""
This script is used to calibrate the NTC sensors at 0 degrees Celsius.
"""

import pandas as pd
import os

def clean_csv_file(file_path, output_file_path):
    """
    Clean the CSV file by removing all occurrences of '�C'.

    Parameters:
    file_path (str): The path to the input CSV file.
    output_file_path (str): The path to the output cleaned CSV file.
    """
    # Read the CSV file as text
    with open(file_path, 'r', encoding='latin1') as file:
        file_content = file.read()

    # Remove all occurrences of '�C'
    cleaned_content = file_content.replace(' °C', '')

    # Write the cleaned content to a new CSV file
    with open(output_file_path, 'w', encoding='latin1') as file:
        file.write(cleaned_content)

def cut_rising_temperatures(file_path, output_file_path, threshold=0.5):
    """
    Cut all values from the CSV where the temperatures are significantly rising again.

    Parameters:
    file_path (str): The path to the input CSV file.
    output_file_path (str): The path to the output cleaned CSV file.
    threshold (float): The threshold for detecting significant temperature rise.
    """
    # Read the CSV file
    data = pd.read_csv(file_path, skiprows=5)  # Skip the first 5 rows of metadata

    # Clean the data
    data.columns = ['Index', 'Time', 'Black_Probe_Temperature', 'White_Probe_Temperature']
    data = data.drop(columns=['Index'])
    data['Black_Probe_Temperature'] = pd.to_numeric(data['Black_Probe_Temperature'], errors='coerce')
    data['White_Probe_Temperature'] = pd.to_numeric(data['White_Probe_Temperature'], errors='coerce')

    # Identify the lowest point in the data series
    black_probe_lowest_index = data['Black_Probe_Temperature'].idxmin()
    white_probe_lowest_index = data['White_Probe_Temperature'].idxmin()

    # Start checking for rising temperatures past the lowest point
    start_index = max(black_probe_lowest_index, white_probe_lowest_index) + 1

    # Detect rising temperatures
    for i in range(start_index, len(data)):
        if (data['Black_Probe_Temperature'].iloc[i] - data['Black_Probe_Temperature'].iloc[i-1] > threshold or
            data['White_Probe_Temperature'].iloc[i] - data['White_Probe_Temperature'].iloc[i-1] > threshold):
            data = data.iloc[:i]
            break

    # Write the cleaned content to a new CSV file
    data.to_csv(output_file_path, index=False)

def calculate_0_degree_offset(file_path, window_size=10):
    """
    Calculate the 0-degree offset for NTC sensors from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    window_size (int): The size of the rolling window to identify the stable region.

    Returns:
    tuple: A tuple containing the 0-degree offset for the black probe and the white probe.
    """
    # Read the CSV file
    data = pd.read_csv(file_path, skiprows=5, index_col=False)  # Skip the first 5 rows of metadata

    # Clean the data
    data.columns = ['Time', 'Black_Probe_Temperature', 'White_Probe_Temperature']
    data['Black_Probe_Temperature'] = pd.to_numeric(data['Black_Probe_Temperature'], errors='coerce')
    data['White_Probe_Temperature'] = pd.to_numeric(data['White_Probe_Temperature'], errors='coerce')

    # Identify the region closest to 0 degrees
    data['Black_Probe_Abs'] = data['Black_Probe_Temperature'].abs()
    data['White_Probe_Abs'] = data['White_Probe_Temperature'].abs()

    # Find the index where the temperature is closest to 0 degrees
    black_probe_closest_to_zero_index = data['Black_Probe_Abs'].idxmin()
    white_probe_closest_to_zero_index = data['White_Probe_Abs'].idxmin()

    # Define the stable region around the closest to 0 degrees index
    black_probe_stable_region_start = max(0, black_probe_closest_to_zero_index - window_size // 2)
    black_probe_stable_region_end = black_probe_stable_region_start + window_size

    white_probe_stable_region_start = max(0, white_probe_closest_to_zero_index - window_size // 2)
    white_probe_stable_region_end = white_probe_stable_region_start + window_size

    # Ensure the stable region is within the bounds of the data
    black_probe_stable_region_end = min(black_probe_stable_region_end, len(data))
    white_probe_stable_region_end = min(white_probe_stable_region_end, len(data))

    # Compute the offset
    black_probe_offset = data['Black_Probe_Temperature'][black_probe_stable_region_start:black_probe_stable_region_end].mean()
    white_probe_offset = data['White_Probe_Temperature'][white_probe_stable_region_start:white_probe_stable_region_end].mean()

    return black_probe_offset, white_probe_offset

# Define the paths
directory = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/data/fieldwork_data/Polythermal_Glaciers_Survey_202408/NTC/NTC_calibration_data/'
file_path = directory + '#16_ice_bath_0deg_offset.csv'
cleaned_file_path = directory + '/cleaned_files/' + file_path.split('/')[-1]

# Clean the CSV file from '�C'
clean_csv_file(file_path, cleaned_file_path)

# Cut rising temperatures from the cleaned file in order to exclude the phase out of the ice bath
cut_file_path = directory + '/cut_files/' + file_path.split('/')[-1]
cut_rising_temperatures(cleaned_file_path, cut_file_path)

# Compute the 0-degree offset for the NTC sensors
sorted_files = sorted(os.listdir(cut_files_dir)) # Sort the files in the directory
# Loop over all files in the directory
for filename in sorted_files:
    if filename.startswith("#"):
        # Compute the file path
        file_path = os.path.join(cut_files_dir, filename)
        
        # Compute the 0-degree offset
        black_probe_offset, white_probe_offset = calculate_0_degree_offset(file_path)
        
        # Save the offsets to a text file in the NTC_calibration_data folder
        with open(os.path.join(directory, 'computed_offsets.txt'), 'a') as file:
            file.write(f'Logger: {filename[1:3]}\n')
            file.write(f'White Probe Offset: {white_probe_offset}\n')
            file.write(f'Black Probe Offset: {black_probe_offset}\n\n')