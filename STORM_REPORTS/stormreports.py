import pandas as pd
from pytz import timezone

'''
Read in the storm reports retrieved from NWS Storm Event Database
'''

# Load the CSV file
path = '/data2/white/DATA/MET390/STORM_REPORTS/'
#df = pd.read_csv(path + "storm_reports_20230107_20230111.csv")  # add delimiter='\t' if needed
df = pd.read_csv(path + "combined_storm_reports.csv")

# Select only the columns you care about
subset = df[['BEGIN_LAT', 'BEGIN_LON', 'EVENT_TYPE']]
# Drop rows with missing lat/lon (optional but recommended)
subset_clean = subset.dropna(subset=['BEGIN_LAT', 'BEGIN_LON'])

# Format BEGIN_TIME as string with leading zeros (e.g., 200 → 02:00, 1300 → 13:00)
df['BEGIN_TIME'] = df['BEGIN_TIME'].apply(lambda x: f"{int(x):04d}")
# Create a combined datetime string
df['BEGIN_DATETIME_STR'] = df['BEGIN_DATE'].astype(str) + ' ' + df['BEGIN_TIME'].str[:2] + ':' + df['BEGIN_TIME'].str[2:] + ':00'

# Convert to pandas datetime
df['BEGIN_DATETIME'] = pd.to_datetime(df['BEGIN_DATETIME_STR'])

# Localize to Pacific Time, then convert to UTC
pacific_tz = timezone('US/Pacific')
df['BEGIN_DATETIME_UTC'] = df['BEGIN_DATETIME'].apply(lambda x: pacific_tz.localize(x).astimezone(timezone('UTC')))

# Check result
print(df[['BEGIN_DATETIME', 'BEGIN_DATETIME_UTC']])

# Now filter using full timestamps!
start1 = pd.Timestamp('2023-01-07 00:00')
end1 = pd.Timestamp('2023-01-08 23:59')

start2 = pd.Timestamp('2023-01-09 00:00')
end2 = pd.Timestamp('2023-01-11 23:59')

jan7_8 = df[(df['BEGIN_DATETIME'] >= start1) & (df['BEGIN_DATETIME'] <= end1)]
jan9_11 = df[(df['BEGIN_DATETIME'] >= start2) & (df['BEGIN_DATETIME'] <= end2)]

# Select desired columns
jan7_8_subset = jan7_8[['BEGIN_DATETIME', 'BEGIN_LAT', 'BEGIN_LON', 'EVENT_TYPE']]
jan9_11_subset = jan9_11[['BEGIN_DATETIME', 'BEGIN_LAT', 'BEGIN_LON', 'EVENT_TYPE']]

# Print
print("Jan 7-8 Reports: ", len(jan7_8_subset))
print(jan7_8_subset)

print("\nJan 9-11 Reports: ", len(jan9_11_subset))
print(jan9_11_subset)

# Most common event types
common_types1 = jan7_8_subset['EVENT_TYPE'].value_counts()
common_types2 = jan9_11_subset['EVENT_TYPE'].value_counts()

print("\nMost common event types (Jan 7-8):")
print(common_types1)

print("\nMost common event types (Jan 9-11):")
print(common_types2)
# Print or export
print(subset_clean)

# Optional: Save to a new CSV
#subset_clean.to_csv('lat_lon_eventtype.csv', index=False)
