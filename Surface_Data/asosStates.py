import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from netCDF4 import Dataset
from wrf import getvar, ll_to_xy, extract_times, ALL_TIMES
import xarray as xr
import numpy as np
import glob

# Define which states you want
states = ['CA','OR']
start_time = datetime(2023, 1, 7, 14,50)
end_time = datetime(2023, 1, 7, 15,00)
domain = 2
# Get ASOS metadata
all_stations = []

for state in states:
    url = f'https://mesonet.agron.iastate.edu/geojson/network.py?network={state}_ASOS'
    response = requests.get(url)
    if response.status_code == 200:
        stations = response.json()
        for feature in stations['features']:
            props = feature['properties']
            lon, lat = feature['geometry']['coordinates']
            all_stations.append({
                'station_id': props['sid'],
                'name': props.get('sname', ''),
                'state': props['state'],
                'lat': lat,
                'lon': lon,
                'elevation': props.get('elevation'),
                'network': props['network']
            })
    else:
        print(f"Failed to fetch {state}_ASOS")

# Combine into DataFrame
df_stations = pd.DataFrame(all_stations)

all_data = []

def clean_column(column):
    return pd.to_numeric(column, errors='coerce')

def fetch_asos_data(station, start, end):
    params = {
        'station': station,
        'data': 'tmpf,dwpf,sknt,drct,mslp,gust,p01i',
        'year1': start.year, 'month1': start.month, 'day1': start.day, 'hour1': start.hour, 'minute1': start.minute,
        'year2': end.year, 'month2': end.month, 'day2': end.day, 'hour2': end.hour, 'minute2': end.minute,
        'tz': 'Etc/UTC',
        'format': 'csv',
        'latlon': True,
    }
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_csv(data, comment='#')

        # Apply cleaning only to numeric columns
        numeric_cols = ['lon', 'lat', 'tmpf', 'dwpf', 'drct', 'sknt', 'mslp', 'gust']
        for col in numeric_cols:
            df[col] = clean_column(df[col])
        return df

    else:
        print(f"Failed for {station}")
        return None

for station in df_stations['station_id']:
    obs_df = fetch_asos_data(station, start_time, end_time)
    if obs_df is not None:
        all_data.append(obs_df)

# Combine all station data
if all_data:
    df_all = pd.concat(all_data).reset_index(drop=True)
    print(df_all.head())
    df_all = df_all.merge(df_stations[['station_id', 'elevation']],how='left',left_on='station',right_on='station_id')
    # Save to CSV
    df_all.to_csv('asos_data_selected_states.csv', index=False)
    print("Data saved to asos_data_selected_states.csv")
else:
    print("No data returned.")

#######################
# WRF Time

path = f"/data2/white/wrf/WRFV4.5.2/run/"
wrf_files = sorted(glob.glob( path + f"wrfout_d0{domain}_*"))
wrfin = [Dataset(f) for f in wrf_files]

latitudes = getvar(wrfin[0], 'lat', timeidx=0)
longitudes = getvar(wrfin[0], 'lon', timeidx=0)
lat_min, lat_max = latitudes.min().item(), latitudes.max().item()
lon_min, lon_max = longitudes.min().item(), longitudes.max().item()

#print(f"Latitude range: {lat_min:.4f} to {lat_max:.4f}")
#print(f"Longitude range: {lon_min:.4f} to {lon_max:.4f}")

df_all = df_all[
    (df_all['lat'] >= lat_min) & (df_all['lat'] <= lat_max) &
    (df_all['lon'] >= lon_min) & (df_all['lon'] <= lon_max)
].reset_index(drop=True)

wrf_times_raw = extract_times(wrfin, timeidx=None)
wrf_times = [pd.to_datetime(t) for t in wrf_times_raw]

#print(wrf_times)

def get_nearest_wrf_value(row, wrfin, wrf_times, varname='T2'):
    # Get lat/lon from ASOS record
    lat, lon = row['lat'], row['lon']
    obs_time = pd.to_datetime(row['valid'])

    # Find closest WRF time
    time_diffs = [abs(wrf_time - obs_time) for wrf_time in wrf_times]
    closest_idx = np.argmin(time_diffs)
    wrf_time = wrf_times[closest_idx]
        
    # Convert lat/lon to x,y index
    x_y = ll_to_xy(wrfin, lat, lon)
    x,y = x_y[0], x_y[1]
    
    # Get WRF variable at that time
    var = getvar(wrfin, varname, timeidx=closest_idx)
    
    # Last check to make sure the WRF data exists and the ASOS station is not outside domain
    if not (0 <= x < var.shape[1] and 0 <= y < var.shape[0]):
            print(f"Out of bounds for station {row['station']}: x={x}, y={y}, shape={var.shape}")
            return np.nan
    # Extract value at x, y
    return float(var[y, x])  # note: y, x order in numpy

def get_nearest_hourly_precip(row, wrfin, wrf_times):
    """
    Returns the hourly precipitation (in mm) from WRF model at the ASOS station's location and closest model time.
    Assumes model outputs are at hourly or sub-hourly intervals.
    """
    lat, lon = row['lat'], row['lon']
    obs_time = pd.to_datetime(row['valid'])

    # Find closest WRF time
    time_diffs = [abs(wrf_time - obs_time) for wrf_time in wrf_times]
    closest_idx = np.argmin(time_diffs)
    
    # Find index closest to one hour before
    one_hour_before = obs_time - pd.Timedelta(hours=1)
    diffs_prev = [abs(t - one_hour_before) for t in wrf_times]
    idx_prev = int(np.argmin(diffs_prev))
    
    actual_diff = (wrf_times[closest_idx] - wrf_times[idx_prev]).total_seconds()
    if abs(actual_diff - 3600) > 600:  # warn if difference is >10 minutes off
        print(f"WARNING: WRF time gap = {actual_diff/60:.1f} minutes for {row['station']}")

    # Ensure order (in case times are reversed)
    if idx_prev >= closest_idx:
        return np.nan  # Cannot compute hourly accumulation
   
    print(f"{row['station']} obs_time={obs_time} → wrf_now={wrf_times[closest_idx]}, wrf_prev={wrf_times[idx_prev]}")
    x, y = ll_to_xy(wrfin, lat, lon)
    
    # Get total accumulated precipitation fields at t and t-1
    rainc_now = getvar(wrfin, 'RAINC', timeidx=closest_idx)
    rainnc_now = getvar(wrfin, 'RAINNC', timeidx=closest_idx)
    rainc_prev = getvar(wrfin, 'RAINC', timeidx=idx_prev)
    rainnc_prev = getvar(wrfin, 'RAINNC', timeidx=idx_prev)

    total_now = rainc_now + rainnc_now
    total_prev = rainc_prev + rainnc_prev

    # Safety check: is point in domain?
    if not (0 <= x < total_now.shape[1] and 0 <= y < total_now.shape[0]):
        print(f"Out of bounds for station {row['station']}: x={x}, y={y}, shape={total_now.shape}")
        return np.nan

    # Compute hourly precip
    hourly_precip = max(total_now[y, x] - total_prev[y, x], 0) / 25.4

    return float(hourly_precip)

df_all['T2_WRF'] = df_all.apply(lambda row: get_nearest_wrf_value(row, wrfin, wrf_times, varname='T2'),axis=1)
df_all['P01_WRF'] = df_all.apply(lambda row: get_nearest_hourly_precip(row, wrfin, wrf_times),axis=1)


start_str = start_time.strftime("%Y%m%d_%H%M")
end_str   = end_time.strftime("%Y%m%d_%H%M")

df_all.to_csv(f"asos_data_D{domain}{start_str}_to_{end_str}.csv", index=False)
print("Data saved")



