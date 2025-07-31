import STORMY
from datetime import datetime


STORMY.download_GOES('goes16', 'ABI-L2-CMIPF',
                      DateTimeIni='20200320-203000', DateTimeFin='20200320-210100', 
                      channel=['13'], path_out='/data2/white/DATA/SATELLITE/')

STORMY.download_WSR88D("KTYX", datetime(2022,11,18,20,00), datetime(2022,11,18,20,10),path_out='/data2/white/')

STORMY.download_LMA(datetime(2022,11,18,20,00),tbuffer=1800, path_out='/data2/white/')

STORMY.download_ASOS(
    states=["IA"],
    start_time=datetime(2022, 11, 18, 20, 0),
    end_time=datetime(2022, 11, 18, 20, 10),
    path_out='/data2/white/'
)

STORMY.download_MRMS(
    field='Reflectivity_-10C_00.50',
    time=datetime(2023, 1, 7, 15, 0),
    tbuffer=100
)

STORMY.download_ERA5_SINGLE(
    start_time=datetime(2023, 1, 7, 18),
    end_time=datetime(2023, 1, 7, 19),
    variables=['2m_temperature', 'total_precipitation'],
    area=[22, -70, 20, -60],  # North America
    path_out='/data2/white/'
)
