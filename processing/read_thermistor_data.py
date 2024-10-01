import pandas as pd
import numpy as np
import re

"""
    Python script to read data from a thermistor as a pandas dataframe.

    Can read data from:
        - geoprecision thermistor chains (FlexGate 2.0 output)
        - NTC thermistors
"""

class ThermistorData:
    def __init__(self, file_path, delimiter, measurement_depth = None):
        self.file_path = file_path
        self.delimiter = delimiter
        self.measurement_depth = measurement_depth
    
    def get_chain_data(self, start_time, end_time):
        data_lines = []
        with open(self.file_path, 'r') as file:
            columns = None
            for line in file:
                if columns is None:
                    if line.startswith('NO{0}TIME{0}'.format(self.delimiter)):
                        columns = line.strip().split(self.delimiter)
                elif line[0].isdigit():
                    if self.delimiter == ';':
                        line = line.replace(',', '.')
                    data_lines.append(line.strip().split(self.delimiter))

        self.data = pd.DataFrame(data_lines, columns=columns)
        self.data['TIME'] = pd.to_datetime(self.data['TIME'], format='%d.%m.%Y %H:%M:%S') + pd.DateOffset(hours=1)
        for col in self.data.columns[2:]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Convert input times to datetime
        start_time = pd.to_datetime(start_time, format='%d.%m.%Y %H:%M:%S')
        end_time = pd.to_datetime(end_time, format='%d.%m.%Y %H:%M:%S')

        # Filter the DataFrame based on the time range
        self.data = self.data[(self.data['TIME'] >= start_time) & (self.data['TIME'] <= end_time)]

        # Modify the column names to include depth information
        num_depth_columns = self.measurement_depth * 2
        num_thermistor_columns = len(self.data.columns) - num_depth_columns - 4  # Subtract 4 for 'NO', 'TIME', 'TEMP LOGGER' and 'TEMP BATTERY' columns

        self.data.columns = (
            self.data.columns[:2].tolist()  # 'NO' and 'TIME' columns
            + [f"# {i}" for i in range(1, num_thermistor_columns + 1)]  # 'Thermistor' columns
            + [f"{i*0.5:.1f} m" for i in range(1, num_depth_columns + 1)]  # Depth columns
            + ['TEMP LOGGER', 'TEMP BATTERY']  # 'TEMP LOGGER' and 'TEMP BATTERY' columns
        )

        return self.data

    def get_ntc_data(self):
        # Read the CSV file with the correct encoding and skip the first 5 rows
        self.data = pd.read_csv(self.file_path, sep=self.delimiter, header=None, skiprows=5, 
                                names=['Measurement', 'TIME', 'Black Probe Temperature', 'White Probe Temperature'], 
                                encoding='latin1')
        
        # Convert the TIME column to datetime format
        self.data['TIME'] = pd.to_datetime(self.data['TIME'])
        
        # Remove the special character (�C) from the temperature columns and convert to float
        self.data['Black Probe Temperature'] = self.data['Black Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        self.data['White Probe Temperature'] = self.data['White Probe Temperature'].apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        
        return self.data