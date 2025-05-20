import wrffuncs
from datetime import datetime, timedelta


# User input for file
wrf_date_time = datetime(1997,1,12,1,55,00)
domain = 2
numtimeidx = 4
file_interval = 20
SIMULATION = "NORMAL"
height = 850

wrf_date_time = wrffuncs.round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)

timeidx, pattern = wrffuncs.get_timeidx_and_wrf_file(wrf_date_time, file_interval, numtimeidx,domain)
print("Using this file: ", pattern)

print("Pattern: ", pattern)
print("Timeidx: ", timeidx)
