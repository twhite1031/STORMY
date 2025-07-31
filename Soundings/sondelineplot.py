import wrf
from wrf import to_np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os, sys, glob
import math
from scipy.signal import medfilt
import pandas as pd
import concurrent.futures
from datetime import datetime
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
# ---- User input for file ----

# Format in YEAR, MONTH, DAY, HOUR, MINUTE, DOMAIN
fileinfo = [2023, 10, 14, 16, 20, 3]

# To keep domain constant for WRF comparison
domain = fileinfo[5]
domain_constant = True
var = "tc"

# Boolean to save figure
savefig = False

# Lattitude and Longitude of the desired area
lat_lon = [34.98, -106.04]

# ---- End User input ----

# Directory where sonde data is stored
indir = "/data2/white/Downloads/NEBP/sondedata/"

filelist = sorted(glob.glob(f'{indir}/SUO25*fix.txt'))
nfiles = len(filelist)
col_names = ['time', 'pressure', 'temperature', 'relative_humidity',
             'speed', 'direction', 'long', 'lat', 'alt', 'geopot', 'dewpoint']

# Column 1: time (UTC)
# Column 2: pressure (hPa)
# Column 3: temperature (C)
# Column 4: relative humidity (%)
# Column 5: wind speed (m/s)
# Column 6: wind direction (degrees)
# Column 10: geopotential (m)
# Column 12: dewpoint (C)

for i in range(0,1):
# Read the data in
    df = pd.read_csv(filelist[i],
                 encoding='latin', header=18,
                 delim_whitespace=True,
                 usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                 names=col_names)

    print("i:", i)

    # Replace '-' with np.NaN
    df = df.replace('-', np.NaN)

    # Drop any rows with all NaN values for T, Td, winds
    df = df.dropna(subset=('pressure', 'temperature', 'relative_humidity',
                           'direction','speed', 'dewpoint', 'geopot','long','lat', 'alt'), how='any'
                   ).reset_index(drop=True)

    # Read in the specific variables
    time = df['time'].values
    p = df['pressure'].values*units.hPa
    T = df['temperature'].values*units.degC
    RH = df['relative_humidity'].values*units('%')
    wind_speed = df['speed'].values
    wind_dir = df['direction'].values
    Td = df['dewpoint'].values*units.degC
    geopot = df['geopot'].values
    long = df['long'].values
    lat = df['lat'].values
    alt = df['alt'].values

    # Convert from objects to floats
    wind_speed2 = np.zeros(len(wind_speed))
    wind_dir2 = np.zeros(len(wind_dir))
    p2 = df['pressure'].values
    p3 = np.zeros(len(p))
    T2 = df['temperature'].values
    Td2 = df['dewpoint'].values
    T3 = np.zeros(len(T))
    Td3 = np.zeros(len(Td))
    g3 = np.zeros(len(geopot))
    alt3 = np.zeros(len(alt))
    time3 = np.zeros(len(time))
    # Convert variables to floats for mathematical operations
    for j in range(0, len(wind_speed)):
        wind_speed2[j] = float(wind_speed[j])*1.944
        wind_dir2[j] = float(wind_dir[j])
        p3[j] = float(p2[j])
        T3[j] = float(T2[j])
        Td3[j] = float(Td2[j])
        g3[j] = float(geopot[j])
        alt3[j] = float(alt[j])
    # Compute the u and v components of wind
    wind_speed22 = wind_speed2[:]*units("knots")
    wind_dir22 = wind_dir2[:]*units.degrees
    u, v = mpcalc.wind_components(wind_speed22, wind_dir22)


    # Find times where geopotential decreases
    # and identify what time it recovers
    decrease_height = np.zeros((len(g3),2))
    for k in range(0,len(g3)-6):
        decrease_height[k,1] = g3[k]
        if g3[k+5]-g3[k] <= 0:
            decrease_height[k,0] = 1
            g3[k] = np.nan
            alt3[k] = np.nan
            u[k] = np.nan
            v[k] = np.nan
            T3[k] = np.nan
            Td3[k] = np.nan
            p3[k] = np.nan
            time3[k] = np.nan

    print(len(T3),len(Td3),len(p3))

    # Figure out where the balloons altitude is greater than it was
    # prior to drop*
    where_recovery = np.where(decrease_height[:] == 1)
    if where_recovery[0].size != 0:
        height_prior_to_loss = g3[where_recovery[0][0]-1]
        if height_prior_to_loss > np.nanmax(g3):
            next_height = np.where(g3 > height_prior_to_loss)
            u2 = u[0:where_recovery[0][0]-1]
            u3 = u[next_height[0][0]:]
            v2 = v[0:where_recovery[0][0]-1]
            v3 = v[next_height[0][0]:]
            g4 = g3[0:where_recovery[0][0]-1]
            g5 = g3[next_height[0][0]:]
            T4 = T3[0:where_recovery[0][0]-1]
            T5 = T3[next_height[0][0]:]
            Td4 = Td3[0:where_recovery[0][0]-1]
            Td5 = Td3[next_height[0][0]:]
            p4 = p3[0:where_recovery[0][0]-1]
            p5 = p3[next_height[0][0]:]
            alt4 = alt3[0:where_recovery[0][0]-1]
            alt5 = alt3[next_height[0][0]:]
            time4 = time[0:where_recovery[0][0]-1]
            time5 = time[next_height[0][0]:]
            # Bring the variables back together as one profile
            u = np.concatenate([np.array(u2),np.array(u3)])
            v = np.concatenate([np.array(v2),np.array(v3)])
            geopot = np.concatenate([np.array(g4),np.array(g5)])
            p = np.concatenate([np.array(p4),np.array(p5)])
            T = np.concatenate([np.array(T4),np.array(T5)])
            Td = np.concatenate([np.array(Td4),np.array(Td5)])
            alt = np.concatenate([np.array(alt4), np.array(alt5)])
            time = np.concatenate([np.array(time4),np.array(time5)])
        else:
            p = p3
            T = T3
            Td = Td3
            geopot = g3
            u = u
            v = v
            alt = alt3
            time = time
    else:
        p = p3
        T = T3
        Td = Td3
        geopot = g3
        v = v
        u = u
        alt = alt3
        time = time

    print(len(p),len(Td),len(T))
    

    # Make time array datetime objects
    
    # Smooth pressure
    smooth_pres = medfilt(p, 7) * units.hPa
    p = np.array(smooth_pres)

    # Remove p < 1 mb
    where_lt_1 = np.where(p < 1)
    p[where_lt_1] = np.nan

    # Find where the pressure reaches a minimum and only analyze those points
    minmin = np.nanmin(p)
    minmin = np.where(p == minmin)
    minmin = minmin[0][0]
    p = p[0:minmin,]
    T = T[0:minmin,]
    Td = Td[0:minmin,]
    u = u[0:minmin,]
    v = v[0:minmin,]
    geopot = geopot[0:minmin,]
    long = long[0:minmin,]
    lat = lat[0:minmin,]
    alt = alt[0:minmin,]
    print(len(p),len(Td),len(T))
    print(np.nanmin(p))

# Open the first NetCDF file to use as a reference for height and to make sure we have the right lat and lon
    path = f"/data2/white/WRF_OUTPUTS/NEBP/ECLIP_RUN_1/"
    pattern = f"wrfout_d0{fileinfo[5]}_{fileinfo[0]}-{fileinfo[1]}-{fileinfo[2]}_{fileinfo[3]}:{fileinfo[4]}:00"
    path2 = f"/data2/white/WRF_OUTPUTS/NEBP/NON_ECLIP_RUN_1/"

    with Dataset(path+pattern) as wrfin:
        start_x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])
        hght1 = wrf.getvar(wrfin, "z",timeidx=0)
        wrfz = hght1[:,start_x_y[1],start_x_y[0]]
        lat1 = wrf.getvar(wrfin, "lat", timeidx=0)
        lon1 = wrf.getvar(wrfin, "lon", timeidx=0)
        print("The lattitude WRF is seeing: ", to_np(lat1[start_x_y[1],start_x_y[0]]))
        print("The longittude WRF is seeing: ", to_np(lon1[start_x_y[1],start_x_y[0]]))

    plottingdata = np.array([])
    plottingdata2 = np.array([])
    wrfheight = np.array([])
    # Function to determine the WRF file to use
    def determine_wrffile(minute, domain):
        domain_index = [0,20,5,1]
        if minute < 20:
            wrfmin = "00"
            timeidx = int(minute/domain_index[domain])
            return timeidx, wrfmin
        elif minute >= 40:
            wrfmin= "40"
            timeidx = int((minute - 40)/domain_index[domain])
            return timeidx, wrfmin
        else:
            wrfmin = "20"
            timeidx = int((minute-20)/domain_index[domain])
            return timeidx, wrfmin

    # Function to read in WRF variables for skewT
    
    def get_vars(var, filelist,idx,timeidx, x_y):
        with Dataset(filelist) as wrfin:
                datavalue = wrf.getvar(wrfin,var,timeidx=timeidx)
                wrfheightvalue = wrf.getvar(wrfin,"z",timeidx=timeidx)[idx,x_y[1],x_y[0]] 
                
                if(to_np(datavalue).ndim == 2):
                    datavalue = datavalue[x_y[1],x_y[0]]

                else:
                    print("Dimension of data: " , to_np(datavalue).ndim)
                    #datavalue = datavalue[0]
                    print("Height index: ", idx)
                    print("Height value: ", wrfheightvalue)
                    datavalue = datavalue[idx,x_y[1],x_y[0]]
                    print("Data value gathered in function: ", datavalue) 
        return datavalue, wrfheightvalue
    
    indices = []
    filelist = []
    filelist2 = []
    x_y_list = []
    wrftimeidx = []
    for idx,value in enumerate(to_np(wrfz)):
        
        # Get the index where the WRF value closest matches the observed value
        index = (np.abs(value - np.array(alt))).argmin()
        indices.append(int(index))
        print("The WRF Height value: " + str(value) + "\nThe nearest observed height: " + str(alt[index]))

        # Find the lat, lon, and time based on the height of the observed value
        matchlong = long[indices]
        matchlat = lat[indices]
        matchtime = time[indices]
        matchu = u[indices]
        matchv = v[indices]
        matchalt = alt[indices]
        matchtemp = T[indices]
        print("The array of indices to match radiosonde with WRF heights: ", indices)

        # Find the best gridbox of the lat and long
        with Dataset(path+pattern) as wrfin:
            x_y = wrf.ll_to_xy(wrfin, matchlat, matchlong)
            x_y_list.append(x_y)
        
        # Test to make sure we have the right values
        matchtime = datetime.strptime(matchtime[idx], "%H:%M:%S")
        hour, minute  = matchtime.hour, matchtime.minute
        print("Hour that value was seen: ", hour)
        print("Minute that value was seen: ", minute)

        
        timeidx, wrfmin = determine_wrffile(minute, domain)
        wrftimeidx.append(timeidx)
        print("Closest WRF timeidx: ", timeidx)
        print("Closest WRF 'minute' or file: ", wrfmin)


        wrfpattern = f"wrfout_d0{domain}_2023-10-14_{hour}:{wrfmin}:00"
        filelist += glob.glob(path+wrfpattern)
        filelist2 += glob.glob(path2+wrfpattern)
        print(filelist)
    counter = np.arange(0, len(filelist))
    if domain_constant != True:
        try:
            with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
                results = executor.map(get_vars,np.full_like(filelist, var),sorted(filelist),np.arange(0,len(filelist)),wrftimeidx,x_y_list)
                results2 = executor.map(get_vars,np.full_like(filelist2, var),sorted(filelist2),np.arange(0,len(filelist)),wrftimeidx,x_y_list)
        except IndexError:
            domain = domain-1
            with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
                results = executor.map(get_vars,np.full_like(filelist, var),sorted(filelist),np.arange(0,59,1),wrftimeidx,x_y_list)
                results2 = executor.map(get_vars,np.full_like(filelist2, var),sorted(filelist2),np.arange(0,len(filelist)),wrftimeidx,x_y_list)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
                results = executor.map(get_vars,np.full_like(filelist, var),sorted(filelist),counter,wrftimeidx,x_y_list)
                results2 = executor.map(get_vars,np.full_like(filelist2, var),sorted(filelist2),np.arange(0,len(filelist)),wrftimeidx,x_y_list)

        # Unpack and collect the results
        for file_data in results:
            datavalue, wrfheightvalue  = file_data  
            if(to_np(datavalue).ndim >= 1):
                plottingdata  = np.append(plottingdata, to_np(datavalue)[-1])
                wrfheight = np.append(wrfheight, to_np(wrfheightvalue)[-1])
            else:
                plottingdata = np.append(plottingdata, to_np(datavalue))
                wrfheight = np.append(wrfheight, to_np(wrfheightvalue))
            print('Unpacked datavalue: ', to_np(datavalue))

        for file_data in results2:
            datavalue, wrfheightvalue  = file_data
            if(to_np(datavalue).ndim >= 1):
                plottingdata2  = np.append(plottingdata2, to_np(datavalue)[-1])
            else:
                plottingdata2 = np.append(plottingdata2, to_np(datavalue))
            print('Unpacked datavalue: ', to_np(datavalue))
    
    #matchedsondedata = mpcalc.wind_speed(matchu, matchv).to(units.meter / units.second)
    matchedsondedata = matchtemp
    #matchedsondedata = matchedsondedata.magnitude
    #print("geopt = ", geopt)
    #print("othergeopt = ", othergeopt)
    #print("hght = ", wrfhght)
# ---- Calculate Index of Agreement -----
    def ioa(predictions, targets):
    # Mean of targets
        mean_observed = np.mean(targets)

    # Calculate the numerator and denominator
        numerator = np.sum((predictions - targets) ** 2)
        denominator = np.sum((np.abs(predictions - mean_observed) + np.abs(targets - mean_observed)) ** 2)

    # Index of Agreement
        index_of_agreement = 1 - (numerator / denominator)

        return index_of_agreement

    ioa_val_eclip = ioa(plottingdata, matchedsondedata)
    ioa_val_noeclip = ioa(plottingdata2, matchedsondedata)
    ioa_val_sim = ioa(plottingdata, plottingdata2)

# --- Calculate RMSE ----
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    rmse_val_eclip = rmse(plottingdata, matchedsondedata)
    rmse_val_noeclip = rmse(plottingdata2, matchedsondedata)

# ---- Calculate Systematic RMSE (RMSEs) ----
    def rmses(predictions, targets):
        model = LinearRegression().fit(predictions.reshape(-1,1),targets)
        intercept = model.intercept_
        slope = model.coef_[0]
        prediction_star = np.array([])
        prediction_star = intercept + (slope * targets)
        return np.sqrt(((prediction_star - targets) ** 2).mean())

    rmses_val_eclip = rmses(plottingdata, matchedsondedata)
    rmses_val_noeclip = rmses(plottingdata2, matchedsondedata)

# --- Calculate un-systematic RMSE ----
    def rmseu(predictions, targets):
        model = LinearRegression().fit(predictions.reshape(-1,1),targets)
        intercept = model.intercept_
        slope = model.coef_[0]
        prediction_star = np.array([])
        prediction_star = intercept + (slope * targets)
        return np.sqrt(((predictions - prediction_star) ** 2).mean())

    rmseu_val_eclip = rmseu(plottingdata, matchedsondedata)
    rmseu_val_noeclip = rmseu(plottingdata2, matchedsondedata)

# --- Calculate mean bias ---
    def meanbias(predictions, targets):
        return ((predictions - targets).mean())

    meanbias_val_eclip = meanbias(plottingdata, matchedsondedata)
    meanbias_val_noeclip = meanbias(plottingdata2, matchedsondedata)

# --- Calculate fractional bias ---
    def fracbias(predictions, targets):
        return (2 * (predictions.mean() - targets.mean()) / ((predictions.mean() + targets.mean())))

    fracbias_val_eclip = fracbias(plottingdata, matchedsondedata)
    fracbias_val_noeclip = fracbias(plottingdata2, matchedsondedata)

#Create table
    data = [
        ['Index of Agreement', ioa_val_eclip, ioa_val_noeclip, ioa_val_sim], \
        ['RMSE', rmse_val_eclip, rmse_val_noeclip ], \
        ['RMSEs', rmses_val_eclip, rmses_val_noeclip], \
        ['RMSEu' , rmseu_val_eclip, rmseu_val_noeclip], \
        ['Mean Bias', meanbias_val_eclip, meanbias_val_noeclip], \
        ['Fractional Bias', fracbias_val_eclip, fracbias_val_noeclip]]

# Make a table
    table = tabulate(data,headers = ['Statistic','Sonde vs. WRF Eclipse', 'Sonde vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse'])
    print(table)


# Create a DataFrame
    df = pd.DataFrame(data)

# Customize the export settings
    custom_header = ['Statistic','Sonde vs. WRF Eclipse', 'Sonde vs. WRF No Eclipse', 'WRF Eclipse vs WRF No Eclipse']

#df.to_excel('output.xlsx', index=True, na_rep='N/A', header=custom_header, index_label='Statistic')

# Create a hodograph
#ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1)
#h = Hodograph(ax_hod, component_range=80.)
#h.add_grid(increment=20)
#h.plot_colormapped(u, v, hght)

    print("Length of WRF Data: ", len(plottingdata))
    print("Length of Sonde Data: ", len(matchu))

    # Plot the Skew-Ts
    fig = plt.figure(figsize=(12, 16))

    # Plot data, assign color of line and width
    plt.plot(plottingdata,wrfheight,color='k',linewidth=2,label='WRF Eclipse')
    plt.plot(matchedsondedata,matchalt, color='red',linewidth=2,label='Radiosonde')
    plt.plot(plottingdata2, wrfheight, color='blue', linewidth= 2,label='WRF No Eclipse')

    # Here we can adjust the sizing and spacing of the ticks
    plt.yticks(fontsize=18)
    #time_difference = end_time - start_time
    #min_difference = time_difference.total_seconds() / new_lufft_freq
    #plt.xticks(ticks=np.arange(0,min_difference,10),labels=times[0:-1:10],fontsize=10,rotation=45)

    # Here we plot the time of annularity in relation to our dataset
    #plt.axvline(x=120,linestyle='--', color='gray')

    # Here we give our axes labels
    plt.xlabel('Time (UTC)',fontsize=24,fontweight='bold')
    plt.ylabel('',fontsize=24,fontweight='bold')
    plt.legend()

    # This line makes the y axis start at the bottom, rather than slightly above the bottom
    plt.ylim(ymin=0)

# Here we assign a fitting title
    #plt.title(f"{var} from {fmt_start_time} to {fmt_end_time} {resampled} ",fontsize=25,fontweight='bold')

    
    # Save the plot
    #plt.savefig('sonde_' + launch_time +'_'+ date + 'new.png',dpi=300)
    plt.show()

