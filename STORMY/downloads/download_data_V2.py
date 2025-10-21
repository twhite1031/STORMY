"""
STORMY: Scientific Tools for Observational and Research Meteorology

Refactored download module with improved error handling, type hints,
and consistent API.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union, Protocol
import logging
import numpy as np
import s3fs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class STORMYError(Exception):
    """Base exception for STORMY operations"""
    pass


class DataNotFoundError(STORMYError):
    """Raised when requested data is not available"""
    pass


class InvalidParameterError(STORMYError):
    """Raised when invalid parameters are provided"""
    pass

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimeRange:
    """Represents a time range with iteration support"""
    start: datetime
    end: datetime
    step: timedelta = timedelta(minutes=5)
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("start must be before or equal to end")
    
    def __iter__(self):
        current = self.start
        while current <= self.end:
            yield current
            current += self.step
    
    def __contains__(self, dt: datetime) -> bool:
        return self.start <= dt <= self.end


@dataclass
class DownloadResult:
    """Result of a download operation"""
    files: List[Path]
    success_count: int
    failure_count: int
    total_size_mb: float
    
    @property
    def success(self) -> bool:
        return self.failure_count == 0 and self.success_count > 0


# ============================================================================
# Base Classes
# ============================================================================

class DataDownloader(ABC):
    """Abstract base class for data downloaders"""
    
    def __init__(self, path_out: Union[str, Path] = '.'):
        self.path_out = Path(path_out)
        self.path_out.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def download(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> DownloadResult:
        """Download data for specified time range"""
        pass
    
    def _normalize_time_range(
        self,
        start_time: datetime,
        end_time: Optional[datetime],
        step: timedelta = timedelta(minutes=5)) -> TimeRange:
        """Normalize time inputs to TimeRange"""
        if end_time is None:
            end_time = start_time
        return TimeRange(start_time, end_time, step)


# ============================================================================
# Specific Downloaders
# ============================================================================

class GOESDownloader(DataDownloader):
    """
    Download GOES satellite data from AWS.
    
    Example:
        >>> downloader = GOESDownloader(path_out='/data/satellite')
        >>> result = downloader.download(
        ...     satellite='goes16',
        ...     product='ABI-L2-CMIPF',
        ...     start_time=datetime(2022, 11, 18, 13, 50),
        ...     end_time=datetime(2022, 11, 18, 14, 0),
        ...     channels=['13']
        ... )
        >>> print(f"Downloaded {result.success_count} files")
    """
    
    VALID_SATELLITES = {'goes16', 'goes17', 'goes18', 'goes19'}
    # Taken from GOES github
    @staticmethod
    def download_file(URL, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):

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
        output_path = Path(path_out) / name_file
        if output_path.exists():
            if output_path.stat().st_size == total_size:
                if not overwrite_file:
                    print(f"  {name_file} already exists.")
                    make_download = False
                else:
                    print(f"  {name_file} will be overwritten.")
                    make_download = True
            else:
                make_download = True

        if make_download == True:
            with open(output_path,'wb') as output_file:
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
    def download(
        self,
        satellite: str,
        product: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        channels: Optional[List[str]] = None,
        domain: Optional[str] = None,
        overwrite_file: bool = False,
        **kwargs
    ) -> DownloadResult:
        """
        Download GOES data.
        
        Args:
            satellite: Satellite ID ('goes16', 'goes17', 'goes18', 'goes19')
            product: Product ID (e.g., 'ABI-L2-CMIPF')
            start_time: Start datetime
            end_time: End datetime (defaults to start_time)
            channels: List of channels (e.g., ['02', '13'])
            domain: Domain for mesoscale ('M1' or 'M2')
            overwrite: Whether to overwrite existing files
            
        Returns:
            DownloadResult with download statistics
            
        Raises:
            InvalidParameterError: If parameters are invalid
            DataNotFoundError: If no data found
        """
        retries = kwargs.get("retries", 10)
        backoff = kwargs.get("backoff", 0.2)
        size_format = kwargs.get("size_format", "Decimal")
        show_download_progress = kwargs.get("show_download_progress", True)
        # Validate inputs
        if satellite.lower() not in self.VALID_SATELLITES:
            raise InvalidParameterError(
                f"Invalid satellite '{satellite}'. "
                f"Must be one of {self.VALID_SATELLITES}"
            )
        
        if product.endswith('M'):
            if domain not in ['M1', 'M2']:
                raise InvalidParameterError(
                    "Mesoscale product requires domain='M1' or 'M2'"
                )
            else:
                # Append domain number to product
                if domain == 'M1':
                    product2 = product + '1'
                elif domain == 'M2':
                    product2 = product + '2'
        else:
            product2 = product
        
        if channels is None and product.startswith('ABI-L'):
            raise InvalidParameterError(
                f"ABI products require channels parameter"
            )
        
        
        if product[:-1] in ['ABI-L1b-Rad', 'ABI-L2-CMIP']:
            if channels is None:
                raise InvalidParameterError("You must define 'channels' for ABI products.")

            if not isinstance(channels, list):
                raise InvalidParameterError("'channels' must be provided as a list.")

            channel_list = []
            for item in channels:
                if not isinstance(item, str):
                    raise InvalidParameterError("Each channel must be a string.")

                if len(item) not in [2, 5]:
                    raise InvalidParameterError(
                        "Each channel string must be two characters ('13') "
                        "or a five-character range ('02-06')."
                    )

                if len(item) == 2:
                    # Single channel (e.g. '13')
                    channel_list.append(item)
                else:
                    # Range (e.g. '02-06')
                    ch_ini, ch_end = item.split('-')
                    for chn in range(int(ch_ini), int(ch_end) + 1):
                        channel_list.append(f"{chn:02d}")

        else:
            # Non-ABI products don’t need channels
            channel_list = None

        
        time_range = self._normalize_time_range(start_time, end_time)
        
        # Implementation would go here
        downloaded_files = []
        success_count = 0
        failure_count = 0
        
        logger.info(
            f"Downloading {satellite} {product} from "
            f"{time_range.start} to {time_range.end}"
        )

        DateTimeIniLoop = start_time.replace(minute=0)
        DateTimeFinLoop = end_time.replace(minute=0)+timedelta(minutes=60)
        time_range_loop = self._normalize_time_range(DateTimeIniLoop, DateTimeFinLoop,timedelta(hours=1))
        for time in time_range_loop: 
            DateTimeFolder = time.strftime('%Y/%j/%H/')

            server = 's3://noaa-'+satellite+'/'+product+'/'
            fs = s3fs.S3FileSystem(anon=True)
            ListFiles = np.array(fs.ls(server+DateTimeFolder))

            for line in ListFiles:
                NameOut = line.split('/')[-1]
                output_path = self.path_out / NameOut
                if output_path.exists() and not overwrite_file:
                    logger.warning(f"File already exists — skipping: {NameOut}")
                    downloaded_files.append(output_path)
                    continue  # skip early, no print, no redundant work
                if product[:-1] in ['ABI-L1b-Rad','ABI-L2-CMIP']:
                    
                    ChannelFile = NameOut.split('_')[1][-2:]
                    DateTimeFile = datetime.strptime(NameOut[NameOut.find('_s')+2:NameOut.find('_e')-1], '%Y%j%H%M%S')

                    if product2 in NameOut    and    ChannelFile in channel_list    and    start_time <= DateTimeFile <= end_time:
                    

                        #print(ChannelFile, DateTimeFile, NameOut)
                        self.download_file('https://noaa-'+satellite+'.s3.amazonaws.com'+line[len('noaa-'+satellite):], NameOut, self.path_out, retries=retries, backoff=backoff, size_format=size_format, show_download_progress=show_download_progress, overwrite_file=overwrite_file)
                        downloaded_files.append(output_path)

                else:
                    DateTimeFile = datetime.strptime(NameOut[NameOut.find('_s')+2:NameOut.find('_e')-1], '%Y%j%H%M%S')

                    if product2 in NameOut    and    start_time <= DateTimeFile <= end_time:
                        #print(DateTimeFile, NameOut)
                        self.download_file('https://noaa-'+satellite+'.s3.amazonaws.com'+line[len('noaa-'+satellite):], NameOut, self.path_out, retries=retries, backoff=backoff, size_format=size_format, show_download_progress=show_download_progress, overwrite_file=overwrite_file)
                        downloaded_files.append(output_path)

        
        if not downloaded_files:
            raise DataNotFoundError(
                f"No {satellite} {product} data found for specified time range"
            )
        
        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        
        return DownloadResult(
            files=downloaded_files,
            success_count=success_count,
            failure_count=failure_count,
            total_size_mb=total_size
        )

class WSR88DDownloader(DataDownloader):

    @staticmethod
    def download_file(URL, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):

        StartTime = datetime.now()

        retries_config = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])

        session = requests.Session()
        session.mount('http://', HTTPAdapter(max_retries=retries_config))
        session.mount('https://', HTTPAdapter(max_retries=retries_config))
        req = session.get(URL, stream=True)
        #req = requests.get(URL, stream=True)
        total_size = int(req.headers.get('content-length', 0))
        size = 0
        if size_format == 'Binary':
            dsize = 1024*1024
        else:
            dsize = 1000*1000
        make_download = True
        output_path = Path(path_out) / name_file
        if output_path.exists():
            if output_path.stat().st_size == total_size:
                if not overwrite_file:
                    print(f"  {name_file} already exists.")
                    make_download = False
                else:
                    print(f"  {name_file} will be overwritten.")
                    make_download = True
            else:
                make_download = True

        if make_download:
            with open(output_path, "wb") as output_file:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        rec_size = output_file.write(chunk)
                        size += rec_size

                        if show_download_progress:
                            elapsed = datetime.now() - StartTime

                            # If total size known — show percentage
                            if total_size > 0:
                                pct = 100.0 * size / total_size
                                print(
                                    f"  {name_file} {pct:3.0f}% {size/dsize:.1f}MB "
                                    f"{elapsed.seconds//60}m{elapsed.seconds%60}s",
                                    end="\r",
                                )
                            else:
                                # No total size known — show size only
                                print(
                                    f"  {name_file} {size/dsize:.1f}MB "
                                    f"{elapsed.seconds//60}m{elapsed.seconds%60}s",
                                    end="\r",
                                )
                if total_size > 0:
                    print(
                        f"\n✅  {name_file} downloaded "
                        f"({size/dsize:.1f}MB, 100%)"
                    )
                else:
                    print(
                        f"\n✅  {name_file} downloaded "
                        f"({size/dsize:.1f}MB, unknown total size)"
                    )
    def download(
        self,
        station: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        overwrite_file: bool = False,
        **kwargs
    ) -> DownloadResult:

        time_range = self._normalize_time_range(start_time, end_time)
        fs = s3fs.S3FileSystem(anon=True)
        downloaded_files = []

        logger.info(f"Downloading {station} from {time_range.start} to {time_range.end}")

        # AWS bucket structure: s3://noaa-nexrad-level2/YYYY/MM/DD/STATION/
        bucket_root = "s3://unidata-nexrad-level2"

        for day in pd.date_range(time_range.start.date(), time_range.end.date()):
            day_path = f"{bucket_root}/{day.year}/{day.month:02d}/{day.day:02d}/{station.upper()}/"
            
            try:
                files = np.array(fs.ls(day_path))
            except FileNotFoundError:
                logger.warning(f"No data found for {station} on {day.date()}")
                continue

            for f in files:
                fname = Path(f).name
                # Example: KTLX20221118_135000_V06
                try:
                    ts_str = fname[len(station):len(station)+15]  # "20221118_135000"
                    ftime = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

                except Exception:
                    continue

                if start_time <= ftime <= end_time:
                    print(ftime)
                    output_path = self.path_out / fname
                    if output_path.exists() and not overwrite_file:
                        downloaded_files.append(output_path)
                        logger.warning(f"  {fname} already exists.")
                        continue
                    print("f", f)
                    url = f"https://unidata-nexrad-level2.s3.amazonaws.com/{f[len('unidata-nexrad-level2/'):]}"
                    print(fname)
                    self.download_file(url, fname, self.path_out)
                    downloaded_files.append(output_path)

        if not downloaded_files:
            raise DataNotFoundError(f"No {station} data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, len(downloaded_files), 0, total_size)

class LMADownloader(DataDownloader):
    """Download NSSL LMA HDF5 data from THREDDS server."""

    BASE_URL = "https://data.nssl.noaa.gov/thredds/fileServer/WRDD"

    @staticmethod
    def _download_file(url, name_file, path_out, overwrite_file=False):
        """Stream download with simple progress bar."""
        output_path = Path(path_out) / name_file
        start_time = datetime.now()

        # Check if file exists
        if output_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return output_path

        # Stream request
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            size = 0
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    size += len(chunk)

                    # --- progress bar ---
                    elapsed = datetime.now() - start_time
                    pct = (size / total * 100) if total > 0 else 0
                    mb = size / 1_000_000
                    eta = f"{elapsed.seconds//60}m{elapsed.seconds%60}s"
                    print(f"  {name_file} {pct:3.0f}% {mb:.1f}MB {eta}", end="\r")

        print(f"\n✅ {name_file} downloaded ({mb:.1f}MB)")
        return output_path

    def download(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:

        time_range = self._normalize_time_range(start_time, end_time)
        downloaded_files = []
        success_count = 0
        failure_count = 0

        logger.info(
            f"Downloading LMA data from {time_range.start} to {time_range.end}"
        )

        base_url = f"{self.BASE_URL}/OKLMA/deployments/flashsort_6/h5_files"

        # Snap times to nearest 10-min intervals
        start_aligned = start_time.replace(minute=(start_time.minute // 10) * 10, second=0)
        end_aligned = end_time.replace(minute=(end_time.minute // 10) * 10, second=0)

        for timestamp in pd.date_range(start_aligned, end_aligned, freq="10min"):
            filename = f"LYLOUT_{timestamp.strftime('%y%m%d_%H%M')[:-1]}000_0600.dat.flash.h5"
            url = f"{base_url}/{timestamp.strftime('%Y/%m/%d')}/{filename}"

            try:
                output_path = self._download_file(
                    url, filename, self.path_out, overwrite_file=overwrite_file
                )
                downloaded_files.append(output_path)
                success_count += 1
            except Exception as e:
                failure_count += 1
                logger.error(f"❌ Failed to download {filename}: {e}")

        if not downloaded_files:
            raise DataNotFoundError(f"No LMA data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)

class NWSSoundingDownloader(DataDownloader):
    """Download NWS Sounding data."""

    BASE_URL = "https://data.nssl.noaa.gov/thredds/fileServer/WRDD"

    @staticmethod
    def _download_file(url, name_file, path_out, overwrite_file=False):
        """Stream download with simple progress bar."""
        output_path = Path(path_out) / name_file
        start_time = datetime.now()

        # Check if file exists
        if output_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return output_path

        # Stream request
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            size = 0
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    size += len(chunk)

                    # --- progress bar ---
                    elapsed = datetime.now() - start_time
                    pct = (size / total * 100) if total > 0 else 0
                    mb = size / 1_000_000
                    eta = f"{elapsed.seconds//60}m{elapsed.seconds%60}s"
                    print(f"  {name_file} {pct:3.0f}% {mb:.1f}MB {eta}", end="\r")

        print(f"\n✅ {name_file} downloaded ({mb:.1f}MB)")
        return output_path

    def download(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:

        time_range = self._normalize_time_range(start_time, end_time)
        downloaded_files = []
        success_count = 0
        failure_count = 0

        logger.info(
            f"Downloading LMA data from {time_range.start} to {time_range.end}"
        )

        base_url = f"{self.BASE_URL}/OKLMA/deployments/flashsort_6/h5_files"

        # Snap times to nearest 10-min intervals
        start_aligned = start_time.replace(minute=(start_time.minute // 10) * 10, second=0)
        end_aligned = end_time.replace(minute=(end_time.minute // 10) * 10, second=0)

        for timestamp in pd.date_range(start_aligned, end_aligned, freq="10min"):
            filename = f"LYLOUT_{timestamp.strftime('%y%m%d_%H%M')[:-1]}000_0600.dat.flash.h5"
            url = f"{base_url}/{timestamp.strftime('%Y/%m/%d')}/{filename}"

            try:
                output_path = self._download_file(
                    url, filename, self.path_out, overwrite_file=overwrite_file
                )
                downloaded_files.append(output_path)
                success_count += 1
            except Exception as e:
                failure_count += 1
                logger.error(f"❌ Failed to download {filename}: {e}")

        if not downloaded_files:
            raise DataNotFoundError(f"No LMA data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)
# ============================================================================
# High-level API
# ============================================================================

class STORMY_downloader:
    """
    Main API for STORMY package.
    
    Example:
        >>> from STORMY import STORMY
        >>> stormy = STORMY(data_root='/data')
        >>> 
        >>> # Download GOES data
        >>> result = stormy.download_goes(
        ...     satellite='goes16',
        ...     product='ABI-L2-CMIPF',
        ...     start_time=datetime(2022, 11, 18, 13, 50),
        ...     channels=['13']
        ... )
    """
    
    def __init__(self, data_root: Union[str, Path] = '.'):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def download_GOES(self, **kwargs) -> DownloadResult:
        """Download GOES satellite data"""
        path_out = self.data_root / 'GOES_files'
        downloader = GOESDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_WSR88D(self, **kwargs) -> DownloadResult:
        """Download WSR-88D radar data"""
        path_out = self.data_root / 'WSR88D_files'
        downloader = WSR88DDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_LMA(self, **kwargs) -> DownloadResult:
        """Download Lightning Mapping Array (LMA) data"""
        path_out = self.data_root / 'LMA_files'
        downloader = LMADownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_NWS_SOUNDING(self, **kwargs) -> DownloadResult:
        """Download NWS Radiosonde data"""
        path_out = self.data_root / 'NWS_SOUNDING_files'
        downloader = NWSSoundingDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_MRMS(self, **kwargs) -> DownloadResult:
        """Download MRMS data"""
        path_out = self.data_root / 'MRMS_files'
        downloader = MRMSDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_ERA5_SINGLE(self, **kwargs) -> DownloadResult:
        """Download ERA5 single-level data"""
        path_out = self.data_root / 'ERA5_SINGLE_files'
        downloader = ERA5SingleDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_ASOS(self, **kwargs) -> DownloadResult:
        """Download ASOS data"""
        path_out = self.data_root / 'ASOS_files'
        downloader = ASOSDownloader(path_out)
        return downloader.download(**kwargs)

# ============================================================================
# Backward-compatible functions
# ============================================================================

def download_GOES(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.
    
    Deprecated: Use STORMY().download_goes() instead.
    """
    import warnings
    warnings.warn(
        "download_GOES is deprecated. Use STORMY().download_goes()",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert to new API
    downloader = GOESDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_WSR88D(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.
    
    Deprecated: Use STORMY().download_radar() instead.
    """
    import warnings
    warnings.warn(
        "download_WSR88D is deprecated. Use STORMY().download_radar()",
        DeprecationWarning,
        stacklevel=2
    )
    
    downloader = WSR88DDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files