import pandas as pd
import re

def inspect_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!\n")
        print("Basic information about the dataset:")
        print(data.info())
        print("\nDescriptive statistics of the dataset:")
        print(data.describe())

        data = clean_data(data)
        
        data['DurationMinutes'] = data['Duration'].apply(convert_duration_to_minutes)
        print("\nConverted Duration column (DurationMinutes):")
        print(data[['Duration', 'DurationMinutes']].head())
        
        return data
    except Exception as e:
        print(f"Failed to load the dataset: {e}")
        return None

def clean_data(data):
    """ Clean the data: handle invalid characters, missing values, and duplicates """

    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    data['MovieName'] = data['MovieName'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x) if isinstance(x, str) else x)
    
    data.dropna(subset=['MovieName', 'ReleaseYear'], inplace=True)
    
    data.drop_duplicates(inplace=True)
    
    data['ReleaseYear'] = pd.to_numeric(data['ReleaseYear'], errors='coerce')
    data.dropna(subset=['ReleaseYear'], inplace=True)
    
    return data

def convert_duration_to_minutes(duration_str):
    """Convert duration to minutes"""
    try:
        duration_str = duration_str.lower().replace(' ', '')
        if 'h' in duration_str:
            time_parts = duration_str.split('h')
            hours = int(time_parts[0].strip())
            minutes = int(time_parts[1].replace('min', '').strip()) if len(time_parts) > 1 else 0
        else:
            hours = 0
            minutes = int(duration_str.replace('min', '').strip())
        return hours * 60 + minutes
    except ValueError as e:
        print(f"Failed to parse duration: {duration_str}, Error: {e}")
        return None
