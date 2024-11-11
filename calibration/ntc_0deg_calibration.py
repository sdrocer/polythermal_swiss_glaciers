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

def calculate_zero_degree_offset(data, threshold=0.1, window=10):
    """
    Calculate the 0-degree offset for a given probe temperature data series.
    
    Parameters:
    - data: pd.DataFrame, the dataset containing the temperature data.
    - threshold: float, the threshold for considering the temperature as stable.
    - window: int, the window size for the rolling mean and standard deviation.
    
    Returns:
    - float, the mean temperature during the stable period.
    """
    # Filter the data to include only values below 10 degrees
    data_below_10 = data[data < 10]
    
    # Calculate the rolling mean and standard deviation to identify the stable period
    rolling_mean = data_below_10.rolling(window=window).mean()
    rolling_std = data_below_10.rolling(window=window).std()
    
    # Identify the stable period where the standard deviation is below the threshold
    stable_period = rolling_std < threshold
    
    # Filter the data to include only the stable period
    stable_data = data_below_10[stable_period]
    
    # Calculate the mean temperature during the stable period
    zero_degree_offset = stable_data.median()
    
    return zero_degree_offset, stable_data.index