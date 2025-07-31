import requests
import pandas as pd
from io import StringIO


params = {
    'station': 'KSYR',  # Station ID for Syracuse Airport
    'data': 'tmpf,dwpf,drct,sknt,mslp,gust',  # Temperature and dew point data
    'year1': '2024', 'month1': '7', 'day1': '22', 'hour1':'22',  # Start date
    'year2': '2024', 'month2': '7', 'day2': '22', 'hour2':'23', # End date
    'tz': 'Etc/UTC',  # Timezone
    'format': 'csv',  # Format of the returned data
    'latlon': True,
}
 
# API endpoint
url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

# Make the GET request
response = requests.get(url, params=params)

# Check the status code of the response
if response.status_code == 200:
    # Parse the CSV data
     # Define column names based on the provided data
    
    # Use StringIO to read the CSV data
    data = StringIO(response.text)
    
    # Parse the CSV data
    df_observed = pd.read_csv(data,comment='#')
    
    # Replace 'M' with NaN (missing values)
    df_observed.replace('M', pd.NA, inplace=True)
    
    print(df_observed.columns)
    
    # Function to clean a DataFrame column
    def clean_column(column):
        return pd.to_numeric(column, errors='coerce')

    # Apply the cleaning function to specific columns
    df_observed['station'] = clean_column(df_observed['station'])
    df_observed['valid'] = clean_column(df_observed['valid'])
    df_observed['lon'] = clean_column(df_observed['lon'])
    df_observed['lat'] = clean_column(df_observed['lat'])
    df_observed['tmpf'] = clean_column(df_observed['tmpf'])
    df_observed['dwpf'] = clean_column(df_observed['dwpf'])
    df_observed['drct'] = clean_column(df_observed['drct'])
    df_observed['sknt'] = clean_column(df_observed['sknt'])
    df_observed['mslp'] = clean_column(df_observed['mslp'])
    df_observed['gust'] = clean_column(df_observed['gust'])

    # Function to clean a DataFrame column
    def clean_column(column):
        return pd.to_numeric(column, errors='coerce')

    # Apply cleaning only to numeric columns
    numeric_cols = ['lon', 'lat', 'tmpf', 'dwpf', 'drct', 'sknt', 'mslp', 'gust']
    for col in numeric_cols:
        df_observed[col] = clean_column(df_observed[col])

    # Optional: convert 'valid' to datetime
    df_observed['valid'] = pd.to_datetime(df_observed['valid'], errors='coerce')

    # Drop rows with missing values in key columns
    df_observed = df_observed.dropna().reset_index(drop=True)

    # Preview cleaned temperature values
    print(df_observed['tmpf'].values)

else:
    print(f"Error: {response.status_code}")
