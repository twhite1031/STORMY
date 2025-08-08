'''
The National Weather Service (NWS) typically launches two soundings
a day (0Z and 12Z) to gather upper air observations to aid in forecasting
and input into numerical weather models. These soundings are stored by 
Iowa State in .csv files. We being by importing necessary packages.
'''
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT
from datetime import datetime
import pandas as pd
import STORMY

'''
After importing, we must define which NWS station(s) you would like to use for the soundings and the 
time range (start and end) you'd like to grab. 
'''

stations = ['KBUF']
start_time, end_time = datetime(2022, 11, 18, 23,55), datetime(2022, 11, 19, 00, 30)
path_out = r"C:\Users\thoma\Documents"
STORMY.download_NWS_SOUNDING(start_time=start_time,end_time=end_time,stations=stations,path_out=path_out)



