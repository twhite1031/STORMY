import glob
import os
from datetime import datetime, timedelta

def parse_filename_datetime_obs(filepath):
    
    # Get the filename 
        filename = filepath.split('/')[-1]
    # Extract the date part (8 characters starting from the 5th character)
        date_str = filename[4:12]
    # Extract the time part (6 characters starting from the 13th character)
        time_str = filename[13:19]
    # Combine the date and time strings
        datetime_str = date_str + time_str 
        return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

def find_closest_radar_file(target_datetime, directory, radar_prefix=None):
    """Finds the file in the directory with the datetime closest to the target datetime."""
    closest_file = None
    closest_diff = None
    
    # Search all matching *_V06 files
    all_files = glob.glob(os.path.join(directory, '*_V06*'))

    # Filter by radar prefix, if provided
    if radar_prefix:
        radar_prefix_lower = radar_prefix.lower()
        filtered_files = [
            f for f in all_files 
            if os.path.basename(f).lower().startswith(radar_prefix_lower)
        ]

    else:
        filtered_files = all_files
        
    for filepath in filtered_files:
        # Extract the filename
        filename = os.path.basename(filepath)
        try:
            # Parse the datetime from the filename
            file_datetime = parse_filename_datetime_obs(filename)
            # Calculate the difference between the file's datetime and the target datetime
            diff = abs((file_datetime - target_datetime).total_seconds())
            # Update the closest file if this file is closer
            if closest_diff is None or diff < closest_diff:
                closest_file = filepath
                closest_diff = diff
        except ValueError:
            # If the filename does not match the expected format, skip it
            continue
    if closest_diff is None or closest_diff > 600:
        print("Nearest radar file is over ten minutes off or does not exist")
        return closest_file

    else:
        
        return closest_file


