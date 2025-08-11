# STORMY

Python package and scripts to manipulate and download observed and modeled meteorological data for research
<br><br>

# Version
1.1 - STORMY is updated frequently, re-install for the latest version.
<br><br>
# Environment Installation
This project relies on several Python packages, so a Conda environment is provided for convenience.
<br><br>
Refer to [this](https://www.anaconda.com/docs/getting-started/miniconda/main) link if you are unfamiliar or do not have Miniconda/Anaconda
<br><br>
There are a plethora of packages which enable use of STORMY.
A conda environment (.yml) is provided [here](https://github.com/twhite1031/STORMY/tree/main/envs)

<br><br>
Install for Linux using:
<br><br>
```
conda env create -f useSTORMY_linux_1-1.yml
```
<br><br>
Install for Windows using:
<br><br>
```
conda env create -f useSTORMY_win_1-1.yml
```
<br><br>
Activate conda environment to use scripts:
<br><br>
```
conda activate useSTORMY
```

**Note**: Due to the large number of packages, installation may take >15 minutes. Please be patient.
<br><br>
# Cloning the Repository
To have this repository on your on system, simply use git within your environment:
```
git clone https://github.com/twhite1031/STORMY.git
```

# Types of Data
As you can see by the directory stucture, there are eight types of data used:
- WRF
- WSR88D (NEXRAD LVL II RADAR)
- DOW (Doppler on Wheels)
- SURFACE (ASOS)
- STORM REPORTS (SPC & NWS)
- ERA5 (EWCMF REANALYSIS)
- MRMS
- SOUNDINGS (NWS & CUSTOM FORMATS)
- GOES
- EFM (Electric Field Meter; Work in Progress)

# Functionality
STORMY serves as a github repository as well as a custom python package. The goal of
STORMY is to streamline the data analysis proccess, allowing for all skill levels
to make meaningful plots with complex data. 

I recommened beginning in the [EXAMPLES directory](https://github.com/twhite1031/STORMY/tree/main/EXAMPLES), 
where you can learn how the data is downloaded, formatted, and how to plot it on a basic map. 

# Change log
All notable changes to this project will be documented in this [file](https://github.com/twhite1031/STORMY/blob/main/CHANGELOG.md)
<br><br>

# Contact
For questions, bugs, or collaboration, feel free to reach out by opening an issue or contacting the maintainer.
<br><br>
Email: thomaswhite675@gmail.com or thomas.james.white@und.edu



