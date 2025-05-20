import xarray as xr

'''
Combine ERA5 (NetCDF) files along an identical dimension
'''

path = "/data2/white/DATA/MET399/ERA5/"
ds1 = xr.open_dataset(path + "ERA520191231.nc")
ds2 = xr.open_dataset(path + "ERA520200101.nc")
combined_ds = xr.concat([ds1, ds2], dim="valid_time")

combined_ds.to_netcdf("/data2/white/DATA/MET399/ERA5/combined.nc")
