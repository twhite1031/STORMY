import urllib.request
from datetime import datetime, timedelta

savepath = "/data2/white/DATA/PROJ_LEE/LMADATA/"

# base_url = "https://data.nssl.noaa.gov/thredds/catalog/WRDD/OKLMA/deployments/LEE/%Y/%m/%d/catalog.html?dataset=WRDD/OKLMA/deployments/LEE/%Y/%m/%d/LYLOUT_%y%m%d_%H%M%S_0600.dat.gz"
base_url = "https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/LEE/%Y/%m/%d/LYLOUT_%y%m%d_%H%M%S_0600.dat.gz"

first_time = datetime(2022,11,19,0,0,0)
last_time = datetime(2022,11,19,0,20,0)
file_time_step = timedelta(0, 600)
n_files = (last_time-first_time)/file_time_step


all_times = (first_time + file_time_step*i for i in range(int(n_files)))
filenames = [t.strftime(base_url) for t in all_times]
for fn in filenames:
    base_fn = fn.split('/')[-1]
    print("Downloading", savepath + base_fn)
    urllib.request.urlretrieve(fn, savepath + base_fn)
