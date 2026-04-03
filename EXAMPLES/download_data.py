from STORMY import STORMY_downloader
from datetime import datetime

downloader = STORMY_downloader(data_root=r'C:\Users\thoma\Documents')

GOES_result = downloader.download_GOES(
    satellite='goes16',
    product='ABI-L2-CMIPF',
    start_time=datetime(2022, 11, 18, 13, 50),
    end_time=datetime(2022, 11, 18, 14, 10),
    channels=['13']
)

WSR88D_result = downloader.download_WSR88D(
    station="KTYX",
    start_time=datetime(2022, 11, 18, 20,00),
    end_time=datetime(2022, 11, 18, 20,10),
)


#LMA_result = downloader.download_LMA(
#    start_time=datetime(2022, 11, 18, 20, 0),
#   end_time=datetime(2022, 11, 18, 20, 30),
#    path_out=r'C:\Users\thomas.james.white\Documents\LMA_data'
#)

ASOS_result = downloader.download_ASOS(
    states=["IA"],
    start_time=datetime(2022, 11, 18, 20, 0),
    end_time=datetime(2022, 11, 18, 20, 10),
)

MRMS_result = downloader.download_MRMS(
    field='MergedReflectivityQCComposite_00.50',
    start_time=datetime(2022, 11, 18, 13, 50),
    end_time=datetime(2022, 11, 18, 14, 5),
)
ERA5_result = downloader.download_ERA5SINGLE(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    variables=['2m_temperature', 'total_precipitation'],
    area=[22, -70, 20, -60],  # North Americals
)

SOUNDING_result = downloader.download_NWSSOUNDING(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    stations=['KBUF'],
)

result = downloader.download_SENTINEL(
    bands=["B02"],
    start_time=datetime(2025, 7, 1),
    end_time=datetime(2025, 7, 31),
    bbox=[-97.2, 47.8, -96.7, 48.1],   # [W, S, E, N]
    cloud_cover_lt=100,
    max_items=10,
)