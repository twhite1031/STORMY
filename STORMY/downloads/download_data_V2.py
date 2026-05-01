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
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import gzip
import shutil
import cdsapi
from pystac_client import Client
import planetary_computer as pc

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
    
    @property
    def num_files(self) -> int:
        return len(self.files)


# ============================================================================
# Base Classes
# ============================================================================

class DataDownloader(ABC):
    """Abstract base class for data downloaders"""
    
    def __init__(self, path_out: Union[str, Path] = '.'):
        self.path_out = Path(path_out)
        self.path_out.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
    
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

    @staticmethod
    def _size_divisor(size_format: str) -> int:
        return 1024 * 1024 if size_format == "Binary" else 1000 * 1000

    @staticmethod
    def _elapsed_label(started_at: datetime) -> str:
        elapsed_seconds = int((datetime.now() - started_at).total_seconds())
        if elapsed_seconds >= 60:
            return f"{elapsed_seconds // 60}m{elapsed_seconds % 60}s"
        return f"{elapsed_seconds}s"

    @staticmethod
    def _human_size_label(size_bytes: int, size_format: str = "Decimal") -> str:
        divisor = 1024 if size_format == "Binary" else 1000
        unit = "KiB" if size_format == "Binary" else "KB"
        if size_bytes < divisor * divisor:
            return f"{size_bytes / divisor:.1f}{unit}"
        mb_divisor = divisor * divisor
        mb_unit = "MiB" if size_format == "Binary" else "MB"
        return f"{size_bytes / mb_divisor:.1f}{mb_unit}"

    def _print_download_progress(
        self,
        label: str,
        size_bytes: int,
        total_bytes: int,
        started_at: datetime,
        *,
        size_format: str = "Decimal",
    ) -> None:
        size_label = self._human_size_label(size_bytes, size_format)
        elapsed = self._elapsed_label(started_at)
        if total_bytes > 0:
            if size_bytes >= total_bytes:
                pct = 100.0
            else:
                pct = max(1.0, min(99.0, 100.0 * size_bytes / total_bytes))
            print(f"  {label} {pct:3.0f}% {size_label} {elapsed}", end="\r")
        else:
            print(f"  {label} {size_label} {elapsed}", end="\r")

    def _stream_response_to_file(
        self,
        response,
        output_path: Path,
        *,
        label: Optional[str] = None,
        chunk_size: int = 8192,
        size_format: str = "Decimal",
        show_download_progress: bool = True,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_size = int(response.headers.get("content-length", 0))
        size = 0
        started_at = datetime.now()
        download_label = label or output_path.name

        with open(output_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                output_file.write(chunk)
                size += len(chunk)
                if show_download_progress:
                    self._print_download_progress(
                        download_label,
                        size,
                        total_size,
                        started_at,
                        size_format=size_format,
                    )

        return output_path


# ============================================================================
# Specific Downloaders
# ============================================================================

# Parts from GOES github
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

    def download_file(self, URL, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):
        retries_config = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount('http://', HTTPAdapter(max_retries=retries_config))
        session.mount('https://', HTTPAdapter(max_retries=retries_config))

        req = session.get(URL, stream=True)
        total_size = int(req.headers.get('content-length', 0))

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
            self._stream_response_to_file(
                req,
                output_path,
                label=name_file,
                chunk_size=1024,
                size_format=size_format,
                show_download_progress=show_download_progress,
            )
    # Main download method
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
        
        # Handle mesoscale products, ensure proper format
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

        # Implementation would go here
        downloaded_files = []
        success_count = 0
        failure_count = 0
        
        logger.info(
            f"Downloading {satellite} {product} from "
            f"{start_time} to {end_time}"
        )

        # Set up time loop (hourly folders)
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
                    success_count += 1
                    downloaded_files.append(output_path)
                    continue  # skip early, no print, no redundant work
                
                # Check if the we are using channels (for ABI products)
                if product[:-1] in ['ABI-L1b-Rad','ABI-L2-CMIP']:
                    
                    ChannelFile = NameOut.split('_')[1][-2:]
                    DateTimeFile = datetime.strptime(NameOut[NameOut.find('_s')+2:NameOut.find('_e')-1], '%Y%j%H%M%S')

                    try:
                        if product2 in NameOut and ChannelFile in channel_list and start_time <= DateTimeFile <= end_time:
                            url = f"https://noaa-{satellite}.s3.amazonaws.com{line[len('noaa-' + satellite):]}"
                            self.download_file(
                                url,
                                NameOut,
                                self.path_out,
                                retries=retries,
                                backoff=backoff,
                                size_format=size_format,
                                show_download_progress=show_download_progress,
                                overwrite_file=overwrite_file
                            )
                            downloaded_files.append(output_path)
                            success_count += 1

                    except Exception as e:
                        failure_count += 1
                        logger.error(f"❌ Failed to download {NameOut}: {e}")

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

    def download_file(self, URL, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):
        retries_config = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount('http://', HTTPAdapter(max_retries=retries_config))
        session.mount('https://', HTTPAdapter(max_retries=retries_config))
        req = session.get(URL, stream=True)

        total_size = int(req.headers.get('content-length', 0))
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
            self._stream_response_to_file(
                req,
                output_path,
                label=name_file,
                chunk_size=1024,
                size_format=size_format,
                show_download_progress=show_download_progress,
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
                    output_path = self.path_out / fname
                    if output_path.exists() and not overwrite_file:
                        downloaded_files.append(output_path)
                        logger.warning(f"  {fname} already exists.")
                        continue
                    url = f"https://unidata-nexrad-level2.s3.amazonaws.com/{f[len('unidata-nexrad-level2/'):]}"
                    self.download_file(url, fname, self.path_out, overwrite_file=overwrite_file)
                    downloaded_files.append(output_path)

        if not downloaded_files:
            raise DataNotFoundError(f"No {station} data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, len(downloaded_files), 0, total_size)

class LMADownloader(DataDownloader):
    """Download NSSL LMA HDF5 data from THREDDS server."""

    BASE_URL = "https://data.nssl.noaa.gov/thredds/fileServer/WRDD"

    def _download_file(self, url, name_file, path_out, overwrite_file=False):
        """Stream download with simple progress bar."""
        output_path = Path(path_out) / name_file

        # Check if file exists
        if output_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return output_path

        # Stream request
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            return self._stream_response_to_file(r, output_path, label=name_file)
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

    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"

    def _download_file(self, url, payload, name_file, path_out, overwrite_file=False):
        """Stream download with simple progress bar."""
        output_path = Path(path_out) / name_file

        # Check if file exists
        if output_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return output_path

        # Stream request
        with requests.get(url, params=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            return self._stream_response_to_file(r, output_path, label=name_file)

    def download(
        self,
        stations: list[str],
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
            f"Downloading {stations} NWS Sounding data from {time_range.start} to {time_range.end}"
        )

        payload = {
        "sts": start_time.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_time.strftime("%Y-%m-%dT%H:%MZ"),
        "station": ','.join(stations),
        "format": "comma",
        "fields": "all",
        }

        # Filename and full path
        name_file = f"nws_soundings_{start_time.strftime('%Y%m%d%H')}_{end_time.strftime('%Y%m%d%H')}_{'_'.join(stations)}.csv"
        base_url = self.base_url

        try:
            output_path = self._download_file( base_url, payload, name_file, self.path_out, overwrite_file=overwrite_file)
            downloaded_files.append(output_path)
            success_count += 1
        except Exception as e:
            failure_count += 1
            logger.error(f"❌ Failed to download {name_file}: {e}")

        if not downloaded_files:
            raise DataNotFoundError(f"No {stations} data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)

class MRMSDownloader(DataDownloader):
    """
    Hybrid downloader using boto3 for S3 listing + presigned URLs,
    and requests for robust, controllable downloads with progress.
    """

    def download_file(self, s3, bucket,key, name_file, path_out, retries=10, backoff=0.2, size_format='Decimal', show_download_progress=True, overwrite_file=False):
        gz_path = Path(path_out) / name_file

        # Check if file exists
        if gz_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return gz_path
        
        # --- Generate presigned URL for this file ---
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600,
        )

        # --- Download via requests with retries ---
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            return self._stream_response_to_file(
                r,
                gz_path,
                label=name_file,
                size_format=size_format,
                show_download_progress=show_download_progress,
            )

        return gz_path

    def download(
        self,
        field: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:
        
        logger.info(
            f"Downloading MRMS {field} from {start_time} to {end_time}"
        )
        # --- Setup AWS S3 (public access) ---
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket = 'noaa-mrms-pds'

        # --- Find files via paginator ---
        prefix = f"CONUS/{field}/{start_time.strftime('%Y%m%d')}"
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded_files = []
        success_count = 0
        failure_count = 0

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                ts_str = key[-24:-9]
                try:
                    ftime = datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
                except ValueError:
                    continue

                if not (start_time <= ftime <= end_time):
                    continue

                name_file = f"{field}_{ftime:%Y%m%d%H%M%S}.grib2.gz"
                output_path = Path(self.path_out) / name_file[:-3]  # remove .gz

                # Check if file exists
                if output_path.exists() and not overwrite_file:
                    logger.warning(f"  {name_file} already exists.")
                    downloaded_files.append(output_path)
                    continue

                # Download compressed file 
                try:
                    downloaded_gz_path = self.download_file(s3, bucket, key, name_file, self.path_out, overwrite_file=overwrite_file)
                    success_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to download {name_file}: {e}")
                    failure_count += 1
                    continue
               
                # --- Unzip ---
                with gzip.open(downloaded_gz_path, 'rb') as gz:
                    with open(output_path, 'wb') as out:
                        shutil.copyfileobj(gz, out)
                downloaded_gz_path.unlink()  # remove .gz
                
                print(f"✅ Downloaded & unzipped: {name_file}")
                downloaded_files.append(output_path)

        if not downloaded_files:
            print("⚠️ No MRMS files found in specified range.")
        else:
            print(f"\n🎯 Finished: {len(downloaded_files)} files downloaded.")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)
    
class ASOSDownloader(DataDownloader):
    """Download ASOS data."""

    base_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

    def _download_file(self, url, payload,name_file, path_out, overwrite_file=False):
        """Stream download with simple progress bar."""
        output_path = Path(path_out) /  name_file

        if output_path.exists() and not overwrite_file:
            logger.warning(f"  {name_file} already exists.")
            return output_path

        # Stream request
        with requests.get(url, params=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            return self._stream_response_to_file(r, output_path, label=name_file)

    def download(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:
        
        states = kwargs.get("states")
        stations = kwargs.get("stations")

        # Validation
        if (states and stations) or (not states and not stations):
            raise InvalidParameterError("Provide either `states` or `stations`, but not both.")
        
        id_str = '_'.join(stations) if stations else '_'.join(states)
        name_file = f"ASOS_{start_time:%Y%m%d%H}_{end_time:%Y%m%d%H}_{id_str}.csv"

        downloaded_files = []
        success_count = 0
        failure_count = 0
        if stations:
            logger.info(
                f"Downloading {stations} ASOS data from {start_time} to {end_time}"
            )
        elif states:
            logger.info(
                f"Downloading ASOS data for states {states} from {start_time} to {end_time}"
            )

        base_payload = {
            'data': 'tmpf,dwpf,sknt,drct,mslp,gust,p01i,skyc1,skyc2,skyc3,skyc4',
            'year1': start_time.year, 'month1': start_time.month, 'day1': start_time.day,
            'hour1': start_time.hour, 'minute1': start_time.minute,
            'year2': end_time.year, 'month2': end_time.month, 'day2': end_time.day,
            'hour2': end_time.hour, 'minute2': end_time.minute,
            'tz': 'Etc/UTC',
            'format': 'csv',
            'latlon': True,
        }

        if stations:
            payload = {**base_payload, 'station': stations}
        elif states:
            all_stations = []
            # Loop through each state to get station list
            for state in states:
                url = f"https://mesonet.agron.iastate.edu/geojson/network.py?network={state}_ASOS"
                response = requests.get(url)
                if response.status_code == 200:
                    data_json = response.json()
                    for feature in data_json["features"]:
                        props = feature["properties"]
                        all_stations.append(props["sid"])
                else:
                    logger.warning(f"⚠️ Failed to fetch ASOS list for {state}")

            payload = {**base_payload, "station": all_stations}

        # Download file with new paylod
        try:
            output_path = self._download_file(self.base_url, payload, name_file, self.path_out, overwrite_file=overwrite_file)
            downloaded_files.append(output_path)
            success_count += 1
        except Exception as e:
            failure_count += 1
            logger.error(f"❌ Failed to download {name_file}: {e}")

        if not downloaded_files:
            raise DataNotFoundError(f"No ASOS data found for specified time range")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)
    
class ERA5SingleDownloader(DataDownloader):
    """Download ERA5 single-level hourly GRIB data from the Copernicus CDS."""

    dataset = "reanalysis-era5-single-levels"

    def download(
        self,
        variables: list[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        area: Optional[list[float]] = None,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:
        
        if area is None:
            # Global coverage (N, W, S, E)
            area = [90, -180, -90, 180]
            logger.info("No area provided — defaulting to global domain.")

        # --- Sanity checks ---
        assert isinstance(variables, list) and variables, "variables must be a non-empty list"
        if end_time is None:
            end_time = start_time

        # Generate the full list of dates in the requested range using an hour increment
        dates = []
        date_cursor = start_time
        while date_cursor <= end_time:
            dates.append(date_cursor)
            date_cursor += timedelta(hours=1)

        # Generate unique years, months, days using sorted sets
        years = sorted({d.year for d in dates})
        months = sorted({f"{d.month:02d}" for d in dates})
        days = sorted({f"{d.day:02d}" for d in dates})

        # Handle if multiple days are requested, we need all 24 hours
        if len(days) > 1:
            hours = [f"{h:02d}:00" for h in range(24)]
        else:
            # Single day: only include the hours that actually appear
            hours = sorted({f"{d.hour:02d}:00" for d in dates})

        downloaded_files = []
        success_count = 0
        failure_count = 0
          
        logger.info(
            f"Downloading ERA5 Single {variables} from {start_time} to {end_time}"
        )

        # --- Step 2: Download day by day ---
        c = cdsapi.Client()

        filename = f"ERA5S_{start_time:%Y%m%d}_{end_time:%Y%m%d}.grib"
        target_path = Path(self.path_out) / filename

        if target_path.exists() and not overwrite_file:
            logger.info(f"File already exists: {target_path}")
            downloaded_files.append(target_path)
            return DownloadResult([target_path], 1, 0, target_path.stat().st_size / (1024**2))

        try:
            logger.info(f"Requesting ERA5 data from {start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M}...")
            request = {
                "product_type": "reanalysis",
                "data_format": kwargs.get("data_format", "grib"),  #  Check kwargs for format, otherwise default to 'grib'
                "variable": variables,
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "area": area,
            }
            
            # !!! TEMP SOLUTION TO REMOVE path_out from KWARGS !!!
            # Merge user overrides
            kwargs.pop("path_out", None)  # Remove path_out if it exists in kwargs
            request.update(kwargs)

            c.retrieve(self.dataset, request, str(target_path))
            print(f"✅ Downloaded: {target_path}")

            downloaded_files.append(target_path)
            success_count += 1

        except Exception as e:
            logger.error(f"❌ Failed to download ERA5 data: {e}")
            failure_count += 1

        # --- Final report ---
        if not downloaded_files:
            raise DataNotFoundError(f"No ERA5 single-level data found for variables={variables}")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024 ** 2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)
    
class ERA5PressureDownloader(DataDownloader):
    """Download ERA5 pressure-level hourly GRIB data from the Copernicus CDS."""

    dataset = "reanalysis-era5-pressure-levels"

    def download(
        self,
        variables: list[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        area: Optional[list[float]] = None,
        pressure_levels: Optional[list[int]] = None,
        overwrite_file: bool = False,
        **kwargs,
    ) -> DownloadResult:
        
        if pressure_levels is None:
            pressure_levels = [
                "1", "2", "3",
                "5", "7", "10",
                "20", "30", "50",
                "70", "100", "125",
                "150", "175", "200",
                "225", "250", "300",
                "350", "400", "450",
                "500", "550", "600",
                "650", "700", "750",
                "775", "800", "825",
                "850", "875", "900",
                "925", "950", "975",
                "1000"
                ]
            
        if area is None:
            # Global coverage (N, W, S, E)
            area = [90, -180, -90, 180]
            logger.info("No area provided — defaulting to global domain.")

        # --- Sanity checks ---
        assert isinstance(variables, list) and variables, "variables must be a non-empty list"
        if end_time is None:
            end_time = start_time

        # Generate the full list of dates in the requested range using an hour increment
        dates = []
        date_cursor = start_time
        while date_cursor <= end_time:
            dates.append(date_cursor)
            date_cursor += timedelta(hours=1)

        # Generate unique years, months, days using sorted sets

        years = sorted({d.year for d in dates})
        months = sorted({f"{d.month:02d}" for d in dates})
        days = sorted({f"{d.day:02d}" for d in dates})

        # Handle if multiple days are requested, we need all 24 hours
        if len(days) > 1:
            hours = [f"{h:02d}:00" for h in range(24)]
        else:
            # Single day: only include the hours that actually appear
            hours = sorted({f"{d.hour:02d}:00" for d in dates})

        # Create a filename and path for the output file
        filename = f"ERA5P_{start_time:%Y%m%d}_{end_time:%Y%m%d}.grib"
        target_path = Path(self.path_out) / filename
        
        downloaded_files = []
        success_count = 0
        failure_count = 0

        # Check if file exists
        if target_path.exists() and not overwrite_file:
            logger.info(f"File already exists: {target_path}")
            downloaded_files.append(target_path)
            return DownloadResult([target_path], 1, 0, target_path.stat().st_size / (1024**2))
        
        logger.info(
            f"Downloading ERA5 Pressure {variables} from {start_time} to {end_time}"
        )
        # --- Step 2: Download day by day ---
        c = cdsapi.Client()

        try:
            logger.info(f"Requesting ERA5 data from {start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M}...")
            request = {
                "product_type": "reanalysis",
                "format": kwargs.get("format", "grib"),  #  Check kwargs for format, otherwise default to 'grib'
                "variable": variables,
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "pressure_level": pressure_levels,
                "area": area,
            }

            # Merge user overrides
            request.update(kwargs)

            c.retrieve(self.dataset, request, str(target_path))
            logger.info(f"✅ Downloaded: {target_path}")

            downloaded_files.append(target_path)
            success_count += 1

        except Exception as e:
            logger.error(f"❌ Failed to download ERA5 data: {e}")
            failure_count += 1

        # --- Final report ---
        if not downloaded_files:
            raise DataNotFoundError(f"No ERA5 single-level data found for variables={variables}")

        total_size = sum(f.stat().st_size for f in downloaded_files) / (1024 ** 2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size)
    
class Sentinel2Downloader(DataDownloader):
    """
    Download selected Sentinel-2 L2A bands (COGs) using a STAC API.

    Default STAC endpoint: Microsoft Planetary Computer STAC API. :contentReference[oaicite:3]{index=3}
    Collection: sentinel-2-l2a. :contentReference[oaicite:4]{index=4}
    """

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection = "sentinel-2-l2a"

    def download(
        self,
        bands: list[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        *,
        bbox: Optional[list[float]] = None,        # [W, S, E, N]
        cloud_cover_lt: float = 20.0,
        max_items: int = 50,
        overwrite_file: bool = False,
        per_item_subdir: bool = True,
        timeout_s: int = 180,
        **kwargs,
    ) -> DownloadResult:
        """
        Parameters
        ----------
        bands : list[str]
            STAC asset keys to download, commonly: B02, B03, B04, B08, SCL, etc.
        bbox : list[float], optional
            [west, south, east, north] in lon/lat.
        cloud_cover_lt : float
            Cloud filter using eo:cloud_cover. (Not perfect, but useful.)
        max_items : int
            Limit number of scenes returned/downloaded.
        per_item_subdir : bool
            If True, write files under <path_out>/<item_id>/...
        """

        if end_time is None:
            end_time = start_time

        assert isinstance(bands, list) and bands, "bands must be a non-empty list"

        # If no bbox is provided, you can still search by time,
        # but Sentinel-2 will return a lot of scenes; strongly prefer bbox.
        if bbox is None:
            raise ValueError("bbox is required for practical Sentinel-2 searches (format: [W, S, E, N]).")

        # STAC search uses RFC3339 interval like "YYYY-MM-DDTHH:MM:SSZ/YYYY..."
        # We'll use inclusive-ish range with 'Z' suffix.
        time_range = f"{start_time.isoformat()}Z/{end_time.isoformat()}Z"

        # Open STAC catalog; Planetary Computer assets require signing.
        client = Client.open(self.stac_url, modifier=pc.sign_inplace)

        search = client.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
            max_items=max_items
        )

        items = list(search.get_items())
        if not items:
            raise DataNotFoundError(
                f"No Sentinel-2 items found for bbox={bbox}, time={time_range}, cloud<{cloud_cover_lt}"
            )

        downloaded_files: list[Path] = []
        success_count = 0
        failure_count = 0

        for item in items:
            # Ensure URLs are signed (modifier should have done this; this is a safe extra step).
            pc.sign_inplace(item)

            item_dir = self.path_out / item.id if per_item_subdir else self.path_out
            item_dir.mkdir(parents=True, exist_ok=True)

            for band in bands:
                if band not in item.assets:
                    # Helpful when someone uses "B8" instead of "B08", etc.
                    available = list(item.assets.keys())
                    raise KeyError(f"Band/asset '{band}' not in item assets. Example keys: {available[:25]}")

                href = item.assets[band].href

                # Most MPC Sentinel-2 assets are Cloud-Optimized GeoTIFFs; keep extension from href.
                # If href ends in ".tif", you'll get GeoTIFF.
                ext = Path(href.split("?")[0]).suffix or ".tif"
                out_path = item_dir / f"{item.id}_{band}{ext}"

                if out_path.exists() and not overwrite_file:
                    downloaded_files.append(out_path)
                    success_count += 1
                    continue

                try:
                    progress_label = f"{item.id}_{band}"
                    self._stream_download(href, out_path, timeout_s=timeout_s, label=progress_label)
                    downloaded_files.append(out_path)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    print(f"❌ Failed to download {band} for item {item.id}: {e}")

        total_size_mb = sum(p.stat().st_size for p in downloaded_files if p.exists()) / (1024**2)
        return DownloadResult(downloaded_files, success_count, failure_count, total_size_mb)

    def _stream_download(self, url: str, out_path: Path, *, timeout_s: int = 180, label: str) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Split connect/read timeouts for better behavior
        timeout = (20, timeout_s)

        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            self._stream_response_to_file(
                r,
                out_path,
                label=label,
                chunk_size=128 * 1024,
            )


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
    
    def download_NWSSOUNDING(self, **kwargs) -> DownloadResult:
        """Download NWS Radiosonde data"""
        path_out = self.data_root / 'NWS_SOUNDING_files'
        downloader = NWSSoundingDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_MRMS(self, **kwargs) -> DownloadResult:
        """Download MRMS data"""
        path_out = self.data_root / 'MRMS_files'
        downloader = MRMSDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_ASOS(self, **kwargs) -> DownloadResult:
        """Download ASOS data"""
        path_out = self.data_root / 'ASOS_files'
        downloader = ASOSDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_ERA5SINGLE(self, **kwargs) -> DownloadResult:
        """Download ERA5 single-level data"""
        path_out = self.data_root / 'ERA5_SINGLE_files'
        downloader = ERA5SingleDownloader(path_out)
        return downloader.download(**kwargs)

    def download_ERA5PRESSURE(self, **kwargs) -> DownloadResult:
        """Download ERA5 pressure-level data"""
        path_out = self.data_root / 'ERA5_PRESSURE_files'
        downloader = ERA5PressureDownloader(path_out)
        return downloader.download(**kwargs)
    
    def download_SENTINEL(self, **kwargs) -> DownloadResult:
        """Download Sentinel satellite data"""
        path_out = self.data_root / 'SENTINEL_files'
        downloader = Sentinel2Downloader(path_out)
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


def download_LMA(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_LMA() instead.
    """
    import warnings
    warnings.warn(
        "download_LMA is deprecated. Use STORMY().download_LMA()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = LMADownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_NWSSOUNDING(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_NWSSOUNDING() instead.
    """
    import warnings
    warnings.warn(
        "download_NWSSOUNDING is deprecated. Use STORMY().download_NWSSOUNDING()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = NWSSoundingDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_MRMS(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_MRMS() instead.
    """
    import warnings
    warnings.warn(
        "download_MRMS is deprecated. Use STORMY().download_MRMS()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = MRMSDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_ASOS(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_ASOS() instead.
    """
    import warnings
    warnings.warn(
        "download_ASOS is deprecated. Use STORMY().download_ASOS()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = ASOSDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_ERA5SINGLE(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_ERA5SINGLE() instead.
    """
    import warnings
    warnings.warn(
        "download_ERA5SINGLE is deprecated. Use STORMY().download_ERA5SINGLE()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = ERA5SingleDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_ERA5PRESSURE(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_ERA5PRESSURE() instead.
    """
    import warnings
    warnings.warn(
        "download_ERA5PRESSURE is deprecated. Use STORMY().download_ERA5PRESSURE()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = ERA5PressureDownloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_SENTINEL(*args, **kwargs) -> List[Path]:
    """
    Legacy function for backward compatibility.

    Deprecated: Use STORMY().download_SENTINEL() instead.
    """
    import warnings
    warnings.warn(
        "download_SENTINEL is deprecated. Use STORMY().download_SENTINEL()",
        DeprecationWarning,
        stacklevel=2
    )

    downloader = Sentinel2Downloader(kwargs.get('path_out', '.'))
    result = downloader.download(*args, **kwargs)
    return result.files


def download_ASOS_STATES(*args, **kwargs) -> List[Path]:
    """
    Backward-compatible alias for the historical ASOS states helper.
    """
    return download_ASOS(*args, **kwargs)


def download_ERA5_SINGLE(*args, **kwargs) -> List[Path]:
    """
    Backward-compatible alias for the historical ERA5 single-level helper.
    """
    return download_ERA5SINGLE(*args, **kwargs)


def download_NWS_SOUNDING(*args, **kwargs) -> List[Path]:
    """
    Backward-compatible alias for the historical NWS sounding helper.
    """
    return download_NWSSOUNDING(*args, **kwargs)
