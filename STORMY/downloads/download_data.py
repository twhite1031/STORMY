# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------------------------------------------
'''
Description: Download Meteorological Data Straight from Source
- GOES 16-19
- WSR88D LVL II Data
- NSSL LMA h5 Files
- ASOS by State

In Progress:
- MRMS
- ERA5
Author: Thomas White
E-mail: thomaswhite675@gmail.com
Created date: July 31, 2025
Modification date: 
'''
#-----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import s3fs
from datetime import *
import requests
import os
import subprocess
import nexradaws
import pandas as pd
import gzip
import shutil
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cdsapi

from io import StringIO
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

#-----------------------------------------------------------------------------------------------------------------------------------

# Taken from GOES github
def download_file(URL, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):

    '''

    Save data in file.

    Parameters
    ----------
    URL : str
        Link of file.

    name_file : str 
        Name of output file.

    path_out : str, optional, default ''
        Path of folder where file will be saved.

    retries : int, optional, default 10
        Defines the retries number to connect to server.
        See: https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry

    backoff: int, optional, default 0.2
        A backoff factor to apply between attempts after the second try.
        See: https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry


    size_format: str, optional, default 'Decimal'
        Defines how is print the size of file.
        Options are:
            'Decimal' : divide file size (in bytes) by (1000*1000) 
            'Binary' : divide file size (in bytes) by (1024*1024)

    show_download_progress : boolean, optional, default True
        Parameter to enable and disable the visualization of download progress.

    overwrite_file : boolean, optional, default False
        Parameter to overwrite or keep a file already downloaded.
        If overwrite_file=False the downloaded file is keep.
        If overwrite_file=True the downloaded file is overwrite (the file is
        downloaded again).

    '''

    StartTime = datetime.now()

    retries_config = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])

    session = requests.Session()
    session.mount('http://', HTTPAdapter(max_retries=retries_config))
    session.mount('https://', HTTPAdapter(max_retries=retries_config))
    req = session.get(URL, stream=True)
    #req = requests.get(URL, stream=True)
    total_size = int(req.headers['content-length'])
    size = 0
    if size_format == 'Binary':
        dsize = 1024*1024
    else:
        dsize = 1000*1000


    make_download = True

    if os.path.isfile(path_out+name_file)==True:
        if os.path.getsize(path_out+name_file)==total_size:
            if overwrite_file==False:
                print('  {} already exists.'.format(name_file))
                make_download = False
            else:
                print('  {} will be overwritten.'.format(name_file))
                make_download = True
        else:
            make_download = True


    if make_download == True:
        with open(path_out+name_file,'wb') as output_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    rec_size = output_file.write(chunk)
                    size = rec_size + size
                    if show_download_progress==True:
                        print('  {} {:3.0f}% {:.1f}MB {}'.format(name_file,100.0*size/total_size, size/dsize, '{}m{}s'.format(round((datetime.now()-StartTime).seconds/60.0),(datetime.now()-StartTime).seconds%60) if (datetime.now()-StartTime).seconds>60 else '{}s'.format((datetime.now()-StartTime).seconds) ), end="\r") #, flush=True)
                        #print('\t{}\t{:3.0f}%\t{:.2f} min'.format(name_file,100.0*size/total_size, (datetime.now()-StartTime).seconds/60.0), end="\r") #, flush=True)
                        if size == total_size:
                            #print('\n')
                            print('  {} {:3.0f}% {:.1f}MB {}'.format(name_file,100.0*size/total_size, size/dsize, '{}m{}s'.format(round((datetime.now()-StartTime).seconds/60.0),(datetime.now()-StartTime).seconds%60) if (datetime.now()-StartTime).seconds>60 else '{}s'.format((datetime.now()-StartTime).seconds) ))

    #print('\b')

#-----------------------------------------------------------------------------------------------------------------------------------

# Taken from GOES github
def download_GOES(Satellite, Product, DateTimeIni=None, DateTimeFin=None, domain=None, channel=None, rename_fmt=False, path_out='', retries=10, backoff=10, size_format='Decimal', show_download_progress=True, overwrite_file=False):

    '''

    Download data of GOES-16, GOES-17, GOES-18 and GOES-19 from Amazon server.
    This function is based on the code of
    blaylockbk https://gist.github.com/blaylockbk/d60f4fce15a7f0475f975fc57da9104d


    Parameters
    ----------
    Satellite : str
        Indicates serie of GOES, the options are 'goes16', 'goes17', 'goes18' and 'goes19'


    Product : str
        Indicates the instrument and level of product. The products
        can be list using: GOES.show_products()


    DateTimeIni : str
        String that indicates the initial datetime. Its structure
        must be yyyymmdd-HHMMSS
        Example:
            DateTimeIni='20180520-183000'


    DateTimeFin : str
        String that indicates the final datetime. Its structure
        must be yyyymmdd-HHMMSS
        Example:
            DateTimeFin='20180520-183000'


    domain : str
        This parameter just is necessary with Mesoescale products.
        The options are:
            M1 : Mesoscale 1
            M2 : Mesoscale 2


    channel : list
        This parameter just is necessary with ABI-L1b-Rad and ABI-L2-CMIP products.
        List indicates the channel or channels that will be download.
        The channels can be mentioned individually as elements of the list
        or as a sequence of channels separated by a hyphen ('-').
        Example:
            channel = ['02','08','09','10','11','13']
            channel = ['02','08-11','13']


    rename_fmt : boolean or str, optional, default False
        Is an optional parameter and its default value is rename_fmt=False which
        indicates that the file name is kept. If would you like that the file name
        just keep the start time of scan you have to define the format of datetime.
        See the next link to know about datetime format:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
        Example:
            rename_fmt = '%Y%m%d%H%M%S'
            rename_fmt = '%Y%m%d%H%M'
            rename_fmt = '%Y%j%H%M'


    path_out : str, optional, default ''
        Optional string that indicates the folder where data will be download.
        The default value is folder where python was open.


    retries: int, optional, default 10
        Defines the retries number to connect to server.
        See: https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry

    backoff: int, optional, default 10
        A backoff factor to apply between attempts after the second try.
        See: https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry


    size_format: str, optional, default 'Decimal'
        It defines how is print the size of file.
        Options are:
            'Decimal' : divide file size (in bytes) by (1000*1000) 
            'Binary' : divide file size (in bytes) by (1024*1024)

    show_download_progress : boolean, optional, default True
        Parameter to enable and disable the visualization of download progress.

    overwrite_file : boolean, optional, default False
        Parameter to overwrite or keep a file already downloaded.
        If overwrite_file=False the downloaded file is keep.
        If overwrite_file=True the downloaded file is overwrite (the file is
        downloaded again).


    Return
    ------
    Download_files : list
        List with the downloaded files (path+filename).

    '''

    # ---------- Satellite -------------------
    try:
        assert Satellite == 'goes16' or Satellite == 'goes17' or Satellite == 'goes18' or Satellite == 'goes19'
    except AssertionError:
        print('\nSatellite should be goes16, goes17, goes18 or goes19\n')
        return
    else:
        if Satellite == 'goes16':
            Sat = 'G16'
        elif Satellite == 'goes17':
            Sat = 'G17'
        elif Satellite == 'goes18':
            Sat = 'G18'
        elif Satellite == 'goes19':
            Sat = 'G19'

    # ---------- Product and Domain -------------------
    if Product[-1] == 'M':
        try:
            assert domain == 'M1' or domain == 'M2'
        except AssertionError:
            print("\nProduct domain is mesoscale so you need define domain='M1' or domain='M2'\n")
            return
        else:
            if domain == 'M1':
                Product2 = Product+'1'
            elif domain == 'M2':
                Product2 = Product+'2'
    else:
        Product2 = Product

    # ---------- DateTimeIni -------------------
    try:
        assert DateTimeIni != None
    except AssertionError:
        print('\nYou must define initial DateTimeIni\n')
        return
    else:
        DateTimeIni = datetime.strptime(DateTimeIni, '%Y%m%d-%H%M%S')

    # ---------- DateTimeFin -------------------
    if DateTimeFin == None :
        DateTimeFin = DateTimeIni
    else:
        DateTimeFin = datetime.strptime(DateTimeFin, '%Y%m%d-%H%M%S')

    # ---------- channel -------------------

    if Product[:-1] in ['ABI-L1b-Rad','ABI-L2-CMIP']:

        try:
            assert channel != None
        except AssertionError:
            print('\nYou must define channel or channels\n')
            return
        else:

            try:
                assert isinstance(channel, list) == True
            except AssertionError:
                print('\nChannel must be a list\n')
                return
            else:
                ChannelList = []
                for item in channel:

                    try:
                        assert isinstance(item, str) == True
                    except AssertionError:
                        print('\nEach elements of channel must have string format\n')
                        return
                    else:

                        try:
                            assert len(item) == 2 or len(item) == 5
                        except AssertionError:
                            print('\nElement of channel must be string with two or five characters\n')
                            return
                        else:
                            if len(item) == 2 :
                                ChannelList.append(item)
                            elif len(item) == 5 :
                                ChIni, ChEnd = item.split('-')
                                for Chn in range(int(ChIni),int(ChEnd)+1):
                                    ChannelList.append('{:02d}'.format(Chn))

                #if download_info == 'minimal' or download_info == 'full':
                #    print('channel list: {}'.format(ChannelList))


    #"""
    Downloaded_files = []

    if show_download_progress == True:
        print('Files:')

    # ---------- Loop -------------------
    DateTimeIniLoop = DateTimeIni.replace(minute=0)
    DateTimeFinLoop = DateTimeFin.replace(minute=0)+timedelta(minutes=60)
    while DateTimeIniLoop < DateTimeFinLoop :

        DateTimeFolder = DateTimeIniLoop.strftime('%Y/%j/%H/')

        server = 's3://noaa-'+Satellite+'/'+Product+'/'
        fs = s3fs.S3FileSystem(anon=True)
        ListFiles = np.array(fs.ls(server+DateTimeFolder))

        for line in ListFiles:
            if Product[:-1] in ['ABI-L1b-Rad','ABI-L2-CMIP']:
                NameFile = line.split('/')[-1]
                ChannelFile = NameFile.split('_')[1][-2:]
                DateTimeFile = datetime.strptime(NameFile[NameFile.find('_s')+2:NameFile.find('_e')-1], '%Y%j%H%M%S')

                if Product2 in NameFile    and    ChannelFile in ChannelList    and    DateTimeIni <= DateTimeFile <= DateTimeFin:

                    if rename_fmt == False:
                        NameOut = NameFile
                    else:
                        NameOut = NameFile[:NameFile.find('_s')+2] + DateTimeFile.strftime(rename_fmt) + '.nc'

                    #print(ChannelFile, DateTimeFile, NameOut)
                    download_file('https://noaa-'+Satellite+'.s3.amazonaws.com'+line[len('noaa-'+Satellite):], NameOut, path_out, retries=retries, backoff=backoff, size_format=size_format, show_download_progress=show_download_progress, overwrite_file=overwrite_file)
                    Downloaded_files.append(path_out+NameOut)

            else:
                NameFile = line.split('/')[-1]
                DateTimeFile = datetime.strptime(NameFile[NameFile.find('_s')+2:NameFile.find('_e')-1], '%Y%j%H%M%S')

                if Product2 in NameFile    and    DateTimeIni <= DateTimeFile <= DateTimeFin:

                    if rename_fmt == False:
                        NameOut = NameFile
                    else:
                        NameOut = NameFile[:NameFile.find('_s')+2] + DateTimeFile.strftime(rename_fmt) + '.nc'

                    #print(DateTimeFile, NameOut)
                    download_file('https://noaa-'+Satellite+'.s3.amazonaws.com'+line[len('noaa-'+Satellite):], NameOut, path_out, retries=retries, backoff=backoff, size_format=size_format, show_download_progress=show_download_progress, overwrite_file=overwrite_file)
                    Downloaded_files.append(path_out+NameOut)

        DateTimeIniLoop = DateTimeIniLoop + timedelta(minutes=60)

    Downloaded_files.sort()

    return Downloaded_files

#-----------------------------------------------------------------------------------------------------------------------------------

def download_WSR88D(radar, DateTimeIni=None, DateTimeFin=None, path_out=''):
    """
    Download Level II WSR-88D radar scans from the NEXRAD AWS archive
    for a specified radar site and time range.

    Parameters:
    ----------
    radar : str
        4-letter radar station ID (e.g., 'KBUF', 'KTLX').
    DateTimeIni : datetime.datetime
        Start time for scan retrieval (required).
    DateTimeFin : datetime.datetime or None
        End time for scan retrieval (optional). If None, only DateTimeIni is used.
    path_out : str
        Output directory path to save downloaded radar files.

    Notes:
    ------
    - Uses the `nexradaws` interface to query and download data from NOAA's AWS S3 archive.
    - Only downloads files that do not already exist in the output directory.
    """
    downloaded_files = []
    if DateTimeIni is None:
        print('\nYou must define initial DateTimeIni\n')
        return
    
    if DateTimeFin is None:
        DateTimeFin = DateTimeIni
    
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(DateTimeIni, DateTimeFin, radar)
    
    if(len(scans) ==0):
        print(f"No Radar Scans found for {radar} between {DateTimeIni} and {DateTimeFin}")
        return

    print(f"\nThere are {len(scans)} scans available for {radar} between {DateTimeIni} and {DateTimeFin}\n")


    for scan in scans:
        local_path = os.path.join(path_out, scan.filename)
        downloaded_files.append(local_path)
        if os.path.exists(local_path):
            print(f"{scan.filename} already exists")
        else:
            print(f"Downloading {scan.filename}")
            conn.download(scan, path_out)

    return downloaded_files
    

def download_LMA(start, tbuffer=1800,path_out=''):
    """
    Download flash-sorted HDF5 (.h5) files from NSSL's public archive
    over a given time range.

    Parameters:
    ----------
    start : datetime.datetime
        The start time for the download window (UTC).
    tbuffer : int, optional (default=1800)
        Time buffer in seconds. Downloads files from `start` to `start + tbuffer`,
        in 10-minute increments (each file represents 10 minutes of data).
    path_out : str, optional (default='')
        Directory to save the downloaded LMA files. Will be created if it doesn't exist.

    Returns:
    -------
    downloaded_files : list of str
        Paths to the downloaded (or already existing) .h5 files.
    
    Notes:
    ------
    - Files are downloaded from:
      https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files
    - Files are skipped if they already exist locally.
    """

    base_url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files'

    os.makedirs(path_out, exist_ok=True)
    downloaded_files = []

    # Loop through each 10-min interval from start to start + tbuffer
    for i in range(int(tbuffer // 600) + 1):
        timestamp = start + timedelta(seconds=i * 600)
        filename = f"LYLOUT_{timestamp.strftime('%y%m%d_%H%M')[:-1]}000_0600.dat.flash.h5"
        local_path = os.path.join(path_out, filename)

        if not os.path.exists(local_path):
            url = f"{base_url}/{timestamp.strftime('%Y/%m/%d')}/{filename}"
            print(f"Downloading: {url}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
                downloaded_files.append(local_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            downloaded_files.append(local_path)
            print(f"{local_path} already exists.")
    print('\n')
    return downloaded_files

def download_ASOS(states=[], start_time=None, end_time=None, path_out='asos_data.csv'):
    """
    Download ASOS weather observations from IEM (Iowa Environmental Mesonet) 
    for a list of U.S. states and a specified time range.

    Parameters:
    ----------
    states : list of str
        List of 2-letter U.S. state abbreviations (e.g., ['NY', 'PA']).
    start_time : datetime.datetime
        Start time of the observation window (UTC).
    end_time : datetime.datetime
        End time of the observation window (UTC).
    path_out : str
        File path or directory to save the CSV output. If a directory is given,
        the file name is auto-generated based on states and time range.

    Returns:
    -------
    df_all : pandas.DataFrame
        A DataFrame containing the combined ASOS observations with metadata.
        Returns None if no data was retrieved or if input validation fails.

    Notes:
    ------
    - Observation data is retrieved in CSV format from:
      https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
    - Station metadata (lat/lon/elevation) is retrieved from GeoJSON via:
      https://mesonet.agron.iastate.edu/geojson/network.py
    - If the specified file already exists, it is loaded and returned directly
      without re-downloading.
    - Columns include temperature, dewpoint, wind, pressure, precipitation, 
      and cloud cover (skyc1 to skyc4).
    """
    if not states:
        print("No states specified.")
        return
    
    if start_time is None or end_time is None:
        print("Please provide both start_time and end_time as datetime objects.")
        return

    # If user passed a directory, generate a complete filepath
    if os.path.isdir(path_out):
        fname = f"asos_data_{'_'.join(states)}_{start_time.strftime('%Y%m%d%H%M')}_{end_time.strftime('%Y%m%d%H%M')}.csv"
        path_out = os.path.join(path_out, fname)

    # Check if the file already exists
    if os.path.exists(path_out):
        print(f"{path_out} already exists.")
        return path_out 

    # Step 1: Fetch station metadata
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
            print(f"Failed to fetch station list for {state}")
    
    if not all_stations:
        print("No stations found.")
        return None

    df_stations = pd.DataFrame(all_stations)

    # Step 2: Fetch observations for each station
    def clean_column(column):
        return pd.to_numeric(column, errors='coerce')

    def fetch_asos_data(station, start, end):
        params = {
            'station': station,
            'data': 'tmpf,dwpf,sknt,drct,mslp,gust,p01i,skyc1,skyc2,skyc3,skyc4',
            'year1': start.year, 'month1': start.month, 'day1': start.day,
            'hour1': start.hour, 'minute1': start.minute,
            'year2': end.year, 'month2': end.month, 'day2': end.day,
            'hour2': end.hour, 'minute2': end.minute,
            'tz': 'Etc/UTC',
            'format': 'csv',
            'latlon': True,
        }
        url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data, comment='#')
            numeric_cols = ['lon', 'lat', 'tmpf', 'dwpf', 'drct', 'sknt', 'mslp', 'gust']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = clean_column(df[col])
            return df
        else:
            print(f"Failed to fetch data for {station}")
            return None

    all_data = []
    for station in df_stations['station_id']:
        df_obs = fetch_asos_data(station, start_time, end_time)
        if df_obs is not None and not df_obs.empty:
            all_data.append(df_obs)

    if not all_data:
        print("No ASOS observations returned.")
        return

    df_all = pd.concat(all_data).reset_index(drop=True)

    # Step 3: Merge elevation
    df_all = df_all.merge(
        df_stations[['station_id', 'elevation']],
        how='left',
        left_on='station',
        right_on='station_id'
    )

    # Step 4: Save to file
    if os.path.isdir(path_out):
        fname = f"asos_data_{'_'.join(states)}_{start_time.strftime('%Y%m%d%H%M')}_{end_time.strftime('%H%M')}.csv"
        path_out = os.path.join(path_out, fname)
    df_all.to_csv(path_out, index=False)
    print(f"Data saved to {path_out}\n")

    return path_out

def download_MRMS(field, start_time, end_time, path_out='mrms'):
    """
    Download all MRMS grib2.gz files between start_time and end_time
    from NOAA's AWS S3 archive and unzip them to the specified local directory.

    Parameters:
    ----------
    field : str
        MRMS field name (e.g., 'PrecipRate', 'Reflectivity', etc.).
    start_time : datetime.datetime
        Start of the desired time window.
    end_time : datetime.datetime
        End of the desired time window.
    path_out : str, optional (default='mrms')
        Directory to save the downloaded and unzipped GRIB2 files.

    Returns:
    -------
    files_downloaded : list of str
        List of paths to the unzipped .grib2 files downloaded within the time window.
    """

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    os.makedirs(path_out, exist_ok=True)

    files_downloaded = []
    prefix = f"CONUS/{field}/{start_time.strftime('%Y%m%d')}"
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket='noaa-mrms-pds', Prefix=prefix)

    for page in page_iterator:
        for obj in page.get('Contents', []):
            try:
                key = obj['Key']
                timestamp_str = key[-24:-9]
                current_dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")

                if start_time <= current_dt <= end_time:
                    save_filename = os.path.join(path_out, f"{field}_{current_dt.strftime('%Y%m%d%H%M%S')}.grib2.gz")
                    unzip_filename = save_filename[:-3]

                    if os.path.exists(unzip_filename):
                        files_downloaded.append(unzip_filename)
                        print(f"{unzip_filename} already exists. Skipping.")
                        continue

                    print(f"Downloading {key}")
                    s3.download_file('noaa-mrms-pds', key, save_filename)

                    with gzip.open(save_filename, 'rb') as gz_file:
                        with open(unzip_filename, 'wb') as out_file:
                            shutil.copyfileobj(gz_file, out_file)

                    os.remove(save_filename)
                    print(f"Unzipped to {unzip_filename}")
                    files_downloaded.append(unzip_filename)

            except Exception as e:
                print(f"Failed to process key: {key} — {e}")
                continue

    if not files_downloaded:
        print(f"No MRMS files found between {start_time} and {end_time}")
    print('\n')
    
    return files_downloaded

def download_ERA5_SINGLE(start_time, end_time, variables, area, path_out=''):
    """
    Download ERA5 single-level hourly GRIB data for one or more variables over a specified datetime range.

    Parameters:
    ----------
    start_time : datetime.datetime
        Start time of the data request (UTC).
    end_time : datetime.datetime
        End time of the data request (UTC).
    variables : list of str
        List of ERA5 single-level variable names (e.g., ['2m_temperature', 'total_precipitation']).
    area : list of float
        Bounding box in the format [North, West, South, East].
    path_out : str, optional (default='')
        Directory to save the output GRIB files. Will be created if it does not exist.

    Returns:
    -------
    downloaded_files : list of str
        List of full file paths to the downloaded GRIB files.

    Notes:
    ------
    - Uses the CDS API (`cdsapi`) to download ERA5 single-level data from the Copernicus Climate Data Store.
    - Data is retrieved on a daily basis, broken down by hour range per day.
    - Automatically skips downloads if the output file already exists.
    - Each variable is requested independently, with one GRIB file per day per variable.
    - Output filenames follow the pattern:
      'ERA5S_<variable>_YYYYMMDD_HHMM-HHMM.grib'
    - You will need an account with an API key https://cds.climate.copernicus.eu/
    """
    
    assert isinstance(variables, list) and len(variables) > 0, "variables must be a non-empty list"
    os.makedirs(path_out, exist_ok=True)

    # Step 1: Generate day-by-day breakdown of times per date
    date_cursor = start_time.date()
    end_date = end_time.date()

    date_times = {}

    while date_cursor <= end_date:
        if date_cursor == start_time.date() and date_cursor == end_time.date():
            hour_range = range(start_time.hour, end_time.hour + 1)
        elif date_cursor == start_time.date():
            hour_range = range(start_time.hour, 24)
        elif date_cursor == end_time.date():
            hour_range = range(0, end_time.hour + 1)
        else:
            hour_range = range(0, 24)

        times = [f"{h:02d}:00" for h in hour_range]
        date_times[date_cursor] = times
        date_cursor += timedelta(days=1)

    # Step 2: Loop through each variable and day
    c = cdsapi.Client()
    downloaded_files = []

    for variable in variables:
        for date, hours in date_times.items():
            date_str = date.strftime('%Y%m%d')
            filename = f"ERA5S_{variable.replace(' ', '_')}_{date_str}_{hours[0].replace(':', '')}-{hours[-1].replace(':', '')}.grib"
            target_path = os.path.join(path_out, filename)

            if os.path.exists(target_path):
                print(f"File already exists: {target_path}")
                downloaded_files.append(target_path)
                continue

            print(f"Requesting ERA5 {variable} for {date_str} hours {hours}...")

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': variable,
                    'year': str(date.year),
                    'month': f"{date.month:02d}",
                    'day': f"{date.day:02d}",
                    'time': hours,
                    'area': area,
                },
                target_path
            )

            print(f"Downloaded: {target_path}\n")
            downloaded_files.append(target_path)

    return downloaded_files

def download_NWS_SOUNDING(start_time, end_time, stations, path_out=None):
    """
    Fetches NWS upper-air sounding data from the IEM RAOB API.

    Parameters:
    -------
    - start_time (datetime): Start time (UTC)
    - end_time (datetime): End time (UTC)
    - stations (list): List of NWS station identifiers (e.g., ['BUF', 'ALB'])
    - path_out (str, optional): Path to save CSV file. If None, does not save.

    Returns:
    -------

    - pd.DataFrame: Sounding data for given stations and time range.
    """
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"

    payload = {
        "sts": start_time.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_time.strftime("%Y-%m-%dT%H:%MZ"),
        "station": ','.join(stations),
        "format": "comma",
        "fields": "all",
    }
    # Filename and full path
    filename = f"nws_soundings_{start_time.strftime('%Y%m%d%H')}_{end_time.strftime('%Y%m%d%H')}_{'_'.join(stations)}.csv"
    full_path = os.path.join(path_out, filename)
    
    # === Skip if file already exists ===
    if os.path.exists(full_path):
        print("Sounding data already exists.")
        return full_path

    # === Fetch from the web ===
    print(f"Requesting data from {payload['sts']} to {payload['ets']} for {stations}")
    response = requests.get(base_url, params=payload)

    if response.status_code != 200 or not response.text.startswith("station,valid"):
        raise RuntimeError("Failed to retrieve data. Check station codes and time format.")

    # === Convert response to DataFrame ===
    df = pd.read_csv(StringIO(response.text))

    if df.empty:
        print("No sounding data was returned for the specified time range and stations.")
        return None

    # === Save to CSV ===
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(full_path, index=False)
    print(f"Saved sounding data to: {full_path}")
    
    return full_path

