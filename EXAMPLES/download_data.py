import STORMY
from datetime import datetime


STORMY.download_GOES('goes16', 'ABI-L2-CMIPF',
                      DateTimeIni='20221118-135000', DateTimeFin='20221118-140100', 
                      channel=['13'], path_out='/data2/white/DATA/MISC/SATELLITE/')

STORMY.download_WSR88D("KTYX", DateTimeIni=datetime(2022,11,18,20,00), DateTimeFin=datetime(2022,11,18,20,10),path_out='/data2/white/MISC/WSR88D/LVL2/')

STORMY.download_LMA(datetime(2022,11,18,20,00),tbuffer=1800, path_out='/data2/white/MISC/LMA/')

STORMY.download_ASOS(
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
    area=[22, -70, 20, -60],  # North America
    path_out='/data2/white/DATA/MISC/ERA5/'
)


STORMY.download_NWS_SOUNDING(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    stations=['KBUF'],
    path_out='/data2/white/DATA/MISC/SOUNDINGS/'
)

