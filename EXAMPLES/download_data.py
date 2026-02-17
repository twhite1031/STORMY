import STORMY
from STORMY import STORMY_downloader
from datetime import datetime

'''
STORMY.download_GOES('goes16', 'ABI-L2-CMIPF',
                      DateTimeIni='20221118-135000', DateTimeFin='20221118-140100', 
                      channel=['13'], path_out='/data2/white/DATA/MISC/SATELLITE/')

STORMY.download_WSR88D("KTYX", DateTimeIni=datetime(2022,11,18,20,00), DateTimeFin=datetime(2022,11,18,20,10),path_out='/data2/white/MISC/WSR88D/LVL2/')

STORMY.download_LMA(datetime(2022,11,18,20,00),tbuffer=1800, path_out='/data2/white/MISC/LMA/')

STORMY.download_ASOS_STATES(
    states=["IA"],
    start_time=datetime(2022, 11, 18, 20, 0),
    end_time=datetime(2022, 11, 18, 20, 10),
    path_out='/data2/white/DATA/MISC/ASOS/'
)

STORMY.download_MRMS(
    field='MergedReflectivityQCComposite_00.50',
    start_time=datetime(2022, 11, 18, 13, 50),
    end_time=datetime(2022,11,18,14,5),
    path_out='/data2/white/DATA/MISC/MRMS/'
)

STORMY.download_ERA5_SINGLE(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    variables=['2m_temperature', 'total_precipitation'],
    area=[22, -70, 20, -60],  # North Americals
    path_out='/data2/white/DATA/MISC/ERA5/'
)


STORMY.download_NWS_SOUNDING(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    stations=['KBUF'],
    path_out='/data2/white/DATA/MISC/SOUNDINGS/'
)

'''
downloader = STORMY_downloader(data_root=r'C:\Users\thomas.james.white\Documents')
result = downloader.download_SENTINEL(path_out='',
    bands=["rendered_preview"],
    start_time=datetime(2025, 7, 1),
    end_time=datetime(2025, 7, 31),
    bbox=[-97.2, 47.8, -96.7, 48.1],   # [W, S, E, N]
    cloud_cover_lt=100,
    max_items=10,
)