# plotting
import matplotlib.pyplot as plt

# import spNNGP class
from spNNGP_py import spNNGP
# from predict import predict
from predict_py import predict
# summary of NNGP
from summary_NNGP_py import summary
# calc time/date
from date_time_calc import date_calcNREL, date_calcAW, calc_hourAW, calc_hourNREL, get_solar_no_cloud

# get the times
import time

# for matricies
import numpy as np
import pandas as pd

# for date calcs
from datetime import date, datetime


##################################################
# location were main nngp-downscaling file is at #
##################################################

main = ".."

########################
# Ambient Weather Data #
########################

# latitude -> y
# longitude -> x


# CID
Latitude_CID = 18.2115553
Longitude_CID = -67.1370868
file_CID = main + "/nngp-downscaling/Data Puerto Rico/Ambient Weather Data/CID.csv"
# Casa Juan, Mayagüez
Latitude_CJ = 18.200897
Longitude_CJ = -67.148253
file_CJ = main + "/nngp-downscaling/Data Puerto Rico/Ambient Weather Data/CJ.csv"
# Mayaguez Urb Paseo Los Robles , Mayagüez
Latitude_UP = 18.231220
Longitude_UP = -67.155010
file_UP = main + "/nngp-downscaling/Data Puerto Rico/Ambient Weather Data/UP.csv"
# My Weather Station, Mayagüez
Latitude_MWS = 18.234288
Longitude_MWS = -67.150602    
file_MWS = main + "/nngp-downscaling/Data Puerto Rico/Ambient Weather Data/MWS.csv"
# My Weather Station, Mayagüez (Blue logo)
Latitude_MWSB = 18.233073
Longitude_MWSB = -67.136155   
file_MWSB = main + "/nngp-downscaling/Data Puerto Rico/Ambient Weather Data/MWSB.csv"

#############
# NREL Data #
#############

# number of files
nFiles = 20

# 18.18
file_name_1 = "18_18/97_18.18_-67.17_2017.csv"
file_name_2 = "18_18/127_18.18_-67.15_2017.csv"
file_name_3 = "18_18/157_18.18_-67.13_2017.csv"
file_name_4 = "18_18/187_18.18_-67.11_2017.csv"

# 18.20
file_name_5 = "18_20/96_18.20_-67.17_2017.csv"
file_name_6 = "18_20/126_18.20_-67.15_2017.csv"
file_name_7 = "18_20/156_18.20_-67.13_2017.csv"
file_name_8 = "18_20/186_18.20_-67.11_2017.csv"

# 18.22
file_name_9 = "18_22/95_18.22_-67.17_2017.csv"
file_name_10 = "18_22/125_18.22_-67.15_2017.csv"
file_name_11 = "18_22/155_18.22_-67.13_2017.csv"
file_name_12 = "18_22/185_18.22_-67.11_2017.csv"

# 18.24
file_name_13 = "18_24/94_18.24_-67.17_2017.csv"
file_name_14 = "18_24/124_18.24_-67.15_2017.csv"
file_name_15 = "18_24/154_18.24_-67.13_2017.csv"
file_name_16 = "18_24/184_18.24_-67.11_2017.csv"

# 18.26
file_name_17 = "18_26/93_18.26_-67.17_2017.csv"
file_name_18 = "18_26/123_18.26_-67.15_2017.csv"
file_name_19 = "18_26/153_18.26_-67.13_2017.csv"
file_name_20 = "18_26/183_18.26_-67.11_2017.csv"


# location
location = main + "/nngp-downscaling/Data Puerto Rico/NREL Data/"

# files
file_1 = location + file_name_1
file_2 = location + file_name_2
file_3 = location + file_name_3
file_4 = location + file_name_4
file_5 = location + file_name_5
file_6 = location + file_name_6
file_7 = location + file_name_7
file_8 = location + file_name_8
file_9 = location + file_name_9
file_10 = location + file_name_10
file_11 = location + file_name_11
file_12 = location + file_name_12
file_13 = location + file_name_13
file_14 = location + file_name_14
file_15 = location + file_name_15
file_16 = location + file_name_16
file_17 = location + file_name_17
file_18 = location + file_name_18
file_19 = location + file_name_19
file_20 = location + file_name_20

#################################################################
# Get NREL DataFrames and get Lon Lat data for both NREL and AW #
#################################################################

# get files data (time, date, solar irradiance)
    # care about order
        # flip to match AW using "[::-1]"
site1 = pd.read_csv(file_1, skiprows=2)[::-1]
site2 = pd.read_csv(file_2, skiprows=2)[::-1]
site3 = pd.read_csv(file_3, skiprows=2)[::-1]
site4 = pd.read_csv(file_4, skiprows=2)[::-1]
site5 = pd.read_csv(file_5, skiprows=2)[::-1]
site6 = pd.read_csv(file_6, skiprows=2)[::-1]
site7 = pd.read_csv(file_7, skiprows=2)[::-1]
site8 = pd.read_csv(file_8, skiprows=2)[::-1]
site9 = pd.read_csv(file_9, skiprows=2)[::-1]          # Values for each location
site10 = pd.read_csv(file_10, skiprows=2)[::-1]
site11 = pd.read_csv(file_11, skiprows=2)[::-1]
site12 = pd.read_csv(file_12, skiprows=2)[::-1]
site13 = pd.read_csv(file_13, skiprows=2)[::-1]
site14 = pd.read_csv(file_14, skiprows=2)[::-1]
site15 = pd.read_csv(file_15, skiprows=2)[::-1]
site16 = pd.read_csv(file_16, skiprows=2)[::-1]
site17 = pd.read_csv(file_17, skiprows=2)[::-1]
site18 = pd.read_csv(file_18, skiprows=2)[::-1]
site19 = pd.read_csv(file_19, skiprows=2)[::-1]
site20 = pd.read_csv(file_20, skiprows=2)[::-1]


# file array used to get location
    # don't care about order
siteL1 = pd.read_csv(file_1, skiprows=range(2, len(site1)+3))
siteL2 = pd.read_csv(file_2, skiprows=range(2, len(site2)+3))
siteL3 = pd.read_csv(file_3, skiprows=range(2, len(site3)+3))
siteL4 = pd.read_csv(file_4, skiprows=range(2, len(site4)+3))
siteL5 = pd.read_csv(file_5, skiprows=range(2, len(site5)+3))
siteL6 = pd.read_csv(file_6, skiprows=range(2, len(site6)+3))
siteL7 = pd.read_csv(file_7, skiprows=range(2, len(site7)+3))
siteL8 = pd.read_csv(file_8, skiprows=range(2, len(site8)+3))
siteL9 = pd.read_csv(file_9, skiprows=range(2, len(site9)+3))         # locations for each site
siteL10 = pd.read_csv(file_10, skiprows=range(2, len(site10)+3))
siteL11 = pd.read_csv(file_11, skiprows=range(2, len(site11)+3))
siteL12 = pd.read_csv(file_12, skiprows=range(2, len(site12)+3))
siteL13 = pd.read_csv(file_13, skiprows=range(2, len(site13)+3))
siteL14 = pd.read_csv(file_14, skiprows=range(2, len(site14)+3))
siteL15 = pd.read_csv(file_15, skiprows=range(2, len(site15)+3))
siteL16 = pd.read_csv(file_16, skiprows=range(2, len(site16)+3))
siteL17 = pd.read_csv(file_17, skiprows=range(2, len(site17)+3))
siteL18 = pd.read_csv(file_18, skiprows=range(2, len(site18)+3))
siteL19 = pd.read_csv(file_19, skiprows=range(2, len(site19)+3))
siteL20 = pd.read_csv(file_20, skiprows=range(2, len(site20)+3))

##############################
# Get Data (Ambient Weather) #
##############################

data_CID = pd.read_csv(file_CID)
data_CJ = pd.read_csv(file_CJ)
data_UP = pd.read_csv(file_UP)
data_MWS = pd.read_csv(file_MWS)
data_MWSB = pd.read_csv(file_MWSB)

#######################################
# Get coord data for both AW and NREL #
#######################################

# long
siteL1_lon = float(siteL1["Longitude"][0])
siteL2_lon = float(siteL2["Longitude"][0])
siteL3_lon = float(siteL3["Longitude"][0])
siteL4_lon = float(siteL4["Longitude"][0])
siteL5_lon = float(siteL5["Longitude"][0])
siteL6_lon = float(siteL6["Longitude"][0])
siteL7_lon = float(siteL7["Longitude"][0])
siteL8_lon = float(siteL8["Longitude"][0])
siteL9_lon = float(siteL9["Longitude"][0])
siteL10_lon = float(siteL10["Longitude"][0])
siteL11_lon = float(siteL11["Longitude"][0])      # longitude for each site
siteL12_lon = float(siteL12["Longitude"][0])
siteL13_lon = float(siteL13["Longitude"][0])
siteL14_lon = float(siteL14["Longitude"][0])
siteL15_lon = float(siteL15["Longitude"][0])
siteL16_lon = float(siteL16["Longitude"][0])
siteL17_lon = float(siteL17["Longitude"][0])
siteL18_lon = float(siteL18["Longitude"][0])
siteL19_lon = float(siteL19["Longitude"][0])
siteL20_lon = float(siteL20["Longitude"][0])

# lat
siteL1_lat = float(siteL1["Latitude"][0])
siteL2_lat = float(siteL2["Latitude"][0])
siteL3_lat = float(siteL3["Latitude"][0])
siteL4_lat = float(siteL4["Latitude"][0])
siteL5_lat = float(siteL5["Latitude"][0])
siteL6_lat = float(siteL6["Latitude"][0])
siteL7_lat = float(siteL7["Latitude"][0])
siteL8_lat = float(siteL8["Latitude"][0])
siteL9_lat = float(siteL9["Latitude"][0])
siteL10_lat = float(siteL10["Latitude"][0])       # latitude for each site
siteL11_lat = float(siteL11["Latitude"][0])
siteL12_lat = float(siteL12["Latitude"][0])
siteL13_lat = float(siteL13["Latitude"][0])
siteL14_lat = float(siteL14["Latitude"][0])
siteL15_lat = float(siteL15["Latitude"][0])
siteL16_lat = float(siteL16["Latitude"][0])
siteL17_lat = float(siteL17["Latitude"][0])
siteL18_lat = float(siteL18["Latitude"][0])
siteL19_lat = float(siteL19["Latitude"][0])
siteL20_lat = float(siteL20["Latitude"][0])

Location_model = np.array([[siteL1_lon, siteL1_lat],
                            [siteL2_lon, siteL2_lat],
                            [siteL3_lon, siteL3_lat],
                            [siteL4_lon, siteL4_lat],
                            [siteL5_lon, siteL5_lat],
                            [siteL6_lon, siteL6_lat],
                            [siteL7_lon, siteL7_lat],
                            [siteL8_lon, siteL8_lat],
                            [siteL9_lon, siteL9_lat],
                            [siteL10_lon, siteL10_lat],     # total model of lon, lat
                            [siteL11_lon, siteL11_lat],
                            [siteL12_lon, siteL12_lat],
                            [siteL13_lon, siteL13_lat],
                            [siteL14_lon, siteL14_lat],
                            [siteL15_lon, siteL15_lat],
                            [siteL16_lon, siteL16_lat],
                            [siteL17_lon, siteL17_lat],
                            [siteL18_lon, siteL18_lat],
                            [siteL19_lon, siteL19_lat],
                            [siteL20_lon, siteL20_lat]])

# for testing out locations
Location_local = np.array([[Longitude_CID, Latitude_CID],
                            [Longitude_CJ, Latitude_CJ],
                            [Longitude_UP, Latitude_UP],            # lon, lat of AW data
                            [Longitude_MWS, Latitude_MWS],
                            [Longitude_MWSB, Latitude_MWSB]])



##################################################################
# Print to see what parts are Ambient Weather Data vs NREL data #
##################################################################

# bounds
    # x -> -67.16, -67.10
    # y -> 18.17, 18.27
# get image of background
file_img = main + "/nngp-downscaling/Data Puerto Rico/map.png"

BBox = (-67.18, -67.10, 18.17, 18.27)

# to see the different locations Ambient Weather vs NREL
print_loc = False
if(print_loc):
    img = plt.imread(file_img)
    fig, ax = plt.subplots(figsize = (8, 7))
    ax.scatter(Location_local[:, 0], Location_local[:, 1], c = "b") # Ambient Weather
    ax.scatter(Location_model[:, 0], Location_model[:, 1], c = "r") # NREL
    plt.legend(['Ambient Weather', 'NREL'])
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(img, zorder=0, extent = BBox)
    plt.show()

########################################################
# make an array of long for plotting (Ambient Weather) #
########################################################

lon = np.array([Longitude_CID, Longitude_CJ, Longitude_UP, Longitude_MWS, Longitude_MWSB])
names = np.array(["CID", "Casa Juan", "Mayaguez Urb Paseo Los Robles", "Calle La Torre", "Calle C"])

#################################################
# Y Variable Data (Ambient Weather) and (NREL) #
#################################################

# AW
SolR_CID = np.array(data_CID["Solar Radiation (W/m^2)"])
SolR_CJ = np.array(data_CJ["Solar Radiation (W/m^2)"])
SolR_UP = np.array(data_UP["Solar Radiation (W/m^2)"])
SolR_MWS = np.array(data_MWS["Solar Radiation (W/m^2)"])
SolR_MWSB = np.array(data_MWSB["Solar Radiation (W/m^2)"])

# NREL
sol_rad_1 = np.array(site1["GHI"])
sol_rad_2 = np.array(site2["GHI"])
sol_rad_3 = np.array(site3["GHI"])
sol_rad_4 = np.array(site4["GHI"])
sol_rad_5 = np.array(site5["GHI"])
sol_rad_6 = np.array(site6["GHI"])
sol_rad_7 = np.array(site7["GHI"])
sol_rad_8 = np.array(site8["GHI"])
sol_rad_9 = np.array(site9["GHI"])
sol_rad_10 = np.array(site10["GHI"])
sol_rad_11 = np.array(site11["GHI"])
sol_rad_12 = np.array(site12["GHI"])
sol_rad_13 = np.array(site13["GHI"])
sol_rad_14 = np.array(site14["GHI"])
sol_rad_15 = np.array(site15["GHI"])
sol_rad_16 = np.array(site16["GHI"])
sol_rad_17 = np.array(site17["GHI"])
sol_rad_18 = np.array(site18["GHI"])
sol_rad_19 = np.array(site19["GHI"])
sol_rad_20 = np.array(site20["GHI"])

## combind the two data sets
    # all of the solar values
    # Takes cloud coverage into account
SolR = np.array(np.concatenate([SolR_CID, SolR_CJ, SolR_UP, SolR_MWS, SolR_MWSB, # Ambient Weather 
                           sol_rad_1, sol_rad_2, sol_rad_3, sol_rad_4, sol_rad_5, sol_rad_6, # NREL
                           sol_rad_7, sol_rad_8, sol_rad_9, sol_rad_10, sol_rad_11, sol_rad_12, 
                           sol_rad_13, sol_rad_14, sol_rad_15, sol_rad_16, sol_rad_17, sol_rad_18,
                           sol_rad_19, sol_rad_20], axis = 0))

###################
# X Variable Data #
###################

####################
# Date calculation #
####################

## AW
# x variable
date_CID = data_CID["Simple Date"]
date_CJ = data_CJ["Simple Date"]
date_UP = data_UP["Simple Date"]
date_MWS = data_MWS["Simple Date"]
date_MWSB = data_MWSB["Simple Date"]

# convert to MM/DD/YY
date_numb_CID = date_calcAW(date_CID)
date_numb_CJ = date_calcAW(date_CJ)
date_numb_UP = date_calcAW(date_UP)
date_numb_MWS = date_calcAW(date_MWS)
date_numb_MWSB = date_calcAW(date_MWSB)


## NREL
# converts days into numb
date_numb_1 = date_calcNREL(np.array(site1["Month"]), np.array(site1["Day"]))
date_numb_2 = date_calcNREL(np.array(site2["Month"]), np.array(site2["Day"]))
date_numb_3 = date_calcNREL(np.array(site3["Month"]), np.array(site3["Day"]))
date_numb_4 = date_calcNREL(np.array(site4["Month"]), np.array(site4["Day"]))
date_numb_5 = date_calcNREL(np.array(site5["Month"]), np.array(site5["Day"]))
date_numb_6 = date_calcNREL(np.array(site6["Month"]), np.array(site6["Day"]))
date_numb_7 = date_calcNREL(np.array(site7["Month"]), np.array(site7["Day"]))
date_numb_8 = date_calcNREL(np.array(site8["Month"]), np.array(site8["Day"]))
date_numb_9 = date_calcNREL(np.array(site9["Month"]), np.array(site9["Day"]))
date_numb_10 = date_calcNREL(np.array(site10["Month"]), np.array(site10["Day"]))
date_numb_11 = date_calcNREL(np.array(site11["Month"]), np.array(site11["Day"]))
date_numb_12 = date_calcNREL(np.array(site12["Month"]), np.array(site12["Day"]))
date_numb_13 = date_calcNREL(np.array(site13["Month"]), np.array(site13["Day"]))
date_numb_14 = date_calcNREL(np.array(site14["Month"]), np.array(site14["Day"]))
date_numb_15 = date_calcNREL(np.array(site15["Month"]), np.array(site15["Day"]))
date_numb_16 = date_calcNREL(np.array(site16["Month"]), np.array(site16["Day"]))
date_numb_17 = date_calcNREL(np.array(site17["Month"]), np.array(site17["Day"]))
date_numb_18 = date_calcNREL(np.array(site18["Month"]), np.array(site18["Day"]))
date_numb_19 = date_calcNREL(np.array(site19["Month"]), np.array(site19["Day"]))
date_numb_20 = date_calcNREL(np.array(site20["Month"]), np.array(site20["Day"]))

date_numb_tot = np.array(np.concatenate([date_numb_CID, date_numb_CJ, date_numb_UP, date_numb_MWS, date_numb_MWSB, # Ambient Weather 
                           date_numb_1, date_numb_2, date_numb_3, date_numb_4, date_numb_5, date_numb_6, # NREL
                           date_numb_7, date_numb_8, date_numb_9, date_numb_10, date_numb_11, date_numb_12, 
                           date_numb_13, date_numb_14, date_numb_15, date_numb_16, date_numb_17, date_numb_18,
                           date_numb_19, date_numb_20], axis = 0))

date_numb_pred = np.concatenate([date_numb_CID, date_numb_CJ, date_numb_UP, date_numb_MWS, date_numb_MWSB,
                                np.zeros(len(date_numb_1) + len(date_numb_2) + len(date_numb_3) + len(date_numb_4) + len(date_numb_5) + len(date_numb_6) + len(date_numb_7) +
                                len(date_numb_8) + len(date_numb_9) + len(date_numb_10) + len(date_numb_11) + len(date_numb_12) + len(date_numb_13) + len(date_numb_14) + len(date_numb_15) +
                                len(date_numb_16) + len(date_numb_17) + len(date_numb_18) + len(date_numb_19) + len(date_numb_20))], axis = 0) # zeros for the NREL data

####################
# Time Calculation #
####################

## AW
time_CID = calc_hourAW(date_CID)
time_CJ = calc_hourAW(date_CJ)
time_UP = calc_hourAW(date_UP)
time_MWS = calc_hourAW(date_MWS)
time_MWSB = calc_hourAW(date_MWSB)

## NREL
time_1 = calc_hourNREL(np.array(site1["Hour"]), np.array(site1["Minute"]))
time_2 = calc_hourNREL(np.array(site2["Hour"]), np.array(site2["Minute"]))
time_3 = calc_hourNREL(np.array(site3["Hour"]), np.array(site3["Minute"]))
time_4 = calc_hourNREL(np.array(site4["Hour"]), np.array(site4["Minute"]))
time_5 = calc_hourNREL(np.array(site5["Hour"]), np.array(site5["Minute"]))
time_6 = calc_hourNREL(np.array(site6["Hour"]), np.array(site6["Minute"]))
time_7 = calc_hourNREL(np.array(site7["Hour"]), np.array(site7["Minute"]))
time_8 = calc_hourNREL(np.array(site8["Hour"]), np.array(site8["Minute"]))
time_9 = calc_hourNREL(np.array(site9["Hour"]), np.array(site9["Minute"]))
time_10 = calc_hourNREL(np.array(site10["Hour"]), np.array(site10["Minute"]))
time_11 = calc_hourNREL(np.array(site11["Hour"]), np.array(site11["Minute"]))
time_12 = calc_hourNREL(np.array(site12["Hour"]), np.array(site12["Minute"]))
time_13 = calc_hourNREL(np.array(site13["Hour"]), np.array(site13["Minute"]))
time_14 = calc_hourNREL(np.array(site14["Hour"]), np.array(site14["Minute"]))
time_15 = calc_hourNREL(np.array(site15["Hour"]), np.array(site15["Minute"]))
time_16 = calc_hourNREL(np.array(site16["Hour"]), np.array(site16["Minute"]))
time_17 = calc_hourNREL(np.array(site17["Hour"]), np.array(site17["Minute"]))
time_18 = calc_hourNREL(np.array(site18["Hour"]), np.array(site18["Minute"]))
time_19 = calc_hourNREL(np.array(site19["Hour"]), np.array(site19["Minute"]))
time_20 = calc_hourNREL(np.array(site20["Hour"]), np.array(site20["Minute"]))

time_tot = np.array(np.concatenate([time_CID, time_CJ, time_UP, time_MWS, time_MWSB, # Ambient Weather 
                           time_1, time_2, time_3, time_4, time_5, time_6, # NREL
                           time_7, time_8, time_9, time_10, time_11, time_12, 
                           time_13, time_14, time_15, time_16, time_17, time_18,
                           time_19, time_20], axis = 0))

##############################
# Clear Sky Solar Irradiance #
##############################

# NREL
clear_1 = np.array(site1["Clearsky GHI"])
clear_2 = np.array(site2["Clearsky GHI"])
clear_3 = np.array(site3["Clearsky GHI"])
clear_4 = np.array(site4["Clearsky GHI"])
clear_5 = np.array(site5["Clearsky GHI"])
clear_6 = np.array(site6["Clearsky GHI"])
clear_7 = np.array(site7["Clearsky GHI"])
clear_8 = np.array(site8["Clearsky GHI"])
clear_9 = np.array(site9["Clearsky GHI"])
clear_10 = np.array(site10["Clearsky GHI"])
clear_11 = np.array(site11["Clearsky GHI"])
clear_12 = np.array(site12["Clearsky GHI"])
clear_13 = np.array(site13["Clearsky GHI"])
clear_14 = np.array(site14["Clearsky GHI"])
clear_15 = np.array(site15["Clearsky GHI"])
clear_16 = np.array(site16["Clearsky GHI"])
clear_17 = np.array(site17["Clearsky GHI"])
clear_18 = np.array(site18["Clearsky GHI"])
clear_19 = np.array(site19["Clearsky GHI"])
clear_20 = np.array(site20["Clearsky GHI"])

## AW
    # take the closest NREL site and take the data
# CID -> site_7
# CJ -> site_7
# UP -> site_14
# MWS -> site_14
# MWSB -> site_15

dateH = int(date_numb_CID[0])
dateL = int(date_numb_CID[len(date_numb_CID)-1])

clear_CID = get_solar_no_cloud(time_7, date_numb_7, clear_7, time_CID, date_numb_CID, dateH, dateL)
clear_CJ = get_solar_no_cloud(time_7, date_numb_7, clear_7, time_CJ, date_numb_CJ, dateH, dateL)
clear_UP = get_solar_no_cloud(time_14, date_numb_14, clear_14, time_UP, date_numb_UP, dateH, dateL)
clear_MWS = get_solar_no_cloud(time_14, date_numb_14, clear_14, time_MWS, date_numb_MWS, dateH, dateL)
clear_MWSB = get_solar_no_cloud(time_15, date_numb_15, clear_15, time_MWSB, date_numb_MWSB, dateH, dateL)

clear_tot = np.array(np.concatenate([clear_CID, clear_CJ, clear_UP, clear_MWS, clear_MWSB, # Ambient Weather 
                           clear_1, clear_2, clear_3, clear_4, clear_5, clear_6, # NREL
                           clear_7, clear_8, clear_9, clear_10, clear_11, clear_12, 
                           clear_13, clear_14, clear_15, clear_16, clear_17, clear_18,
                           clear_19, clear_20], axis = 0))


######################################################
# corods Variable Data (Ambient Weather) and (NREL) #
######################################################

# create long for lenght of date
lon_lat_CID = np.column_stack((np.repeat(Longitude_CID, len(date_CID)),np.repeat(Latitude_CID, len(date_CID))))
lon_lat_CJ = np.column_stack((np.repeat(Longitude_CJ, len(date_CJ)),np.repeat(Latitude_CJ, len(date_CJ))))
lon_lat_UP = np.column_stack((np.repeat(Longitude_UP, len(date_UP)),np.repeat(Latitude_UP, len(date_UP))))
lon_lat_MWS = np.column_stack((np.repeat(Longitude_MWS, len(date_MWS)),np.repeat(Latitude_MWS, len(date_MWS))))
lon_lat_MWSB = np.column_stack((np.repeat(Longitude_MWSB, len(date_MWSB)),np.repeat(Latitude_MWSB, len(date_MWSB))))

# create long for lenght of sol_rad 
lon_lat_1 = np.column_stack((np.repeat(siteL1_lon, len(sol_rad_1)),np.repeat(siteL1_lat, len(sol_rad_1))))
lon_lat_2 = np.column_stack((np.repeat(siteL2_lon, len(sol_rad_2)),np.repeat(siteL2_lat, len(sol_rad_2))))
lon_lat_3 = np.column_stack((np.repeat(siteL3_lon, len(sol_rad_3)),np.repeat(siteL3_lat, len(sol_rad_3))))
lon_lat_4 = np.column_stack((np.repeat(siteL4_lon, len(sol_rad_4)),np.repeat(siteL4_lat, len(sol_rad_4))))
lon_lat_5 = np.column_stack((np.repeat(siteL5_lon, len(sol_rad_5)),np.repeat(siteL5_lat, len(sol_rad_5))))
lon_lat_6 = np.column_stack((np.repeat(siteL6_lon, len(sol_rad_6)),np.repeat(siteL6_lat, len(sol_rad_6))))
lon_lat_7 = np.column_stack((np.repeat(siteL7_lon, len(sol_rad_7)),np.repeat(siteL7_lat, len(sol_rad_7))))
lon_lat_8 = np.column_stack((np.repeat(siteL8_lon, len(sol_rad_8)),np.repeat(siteL8_lat, len(sol_rad_8))))
lon_lat_9 = np.column_stack((np.repeat(siteL9_lon, len(sol_rad_9)),np.repeat(siteL9_lat, len(sol_rad_9))))
lon_lat_10 = np.column_stack((np.repeat(siteL10_lon, len(sol_rad_10)),np.repeat(siteL10_lat, len(sol_rad_10))))
lon_lat_11 = np.column_stack((np.repeat(siteL11_lon, len(sol_rad_11)),np.repeat(siteL11_lat, len(sol_rad_11))))
lon_lat_12 = np.column_stack((np.repeat(siteL12_lon, len(sol_rad_12)),np.repeat(siteL12_lat, len(sol_rad_12))))
lon_lat_13 = np.column_stack((np.repeat(siteL13_lon, len(sol_rad_13)),np.repeat(siteL13_lat, len(sol_rad_13))))
lon_lat_14 = np.column_stack((np.repeat(siteL14_lon, len(sol_rad_14)),np.repeat(siteL14_lat, len(sol_rad_14))))
lon_lat_15 = np.column_stack((np.repeat(siteL15_lon, len(sol_rad_15)),np.repeat(siteL15_lat, len(sol_rad_15))))
lon_lat_16 = np.column_stack((np.repeat(siteL16_lon, len(sol_rad_16)),np.repeat(siteL16_lat, len(sol_rad_16))))
lon_lat_17 = np.column_stack((np.repeat(siteL17_lon, len(sol_rad_17)),np.repeat(siteL17_lat, len(sol_rad_17))))
lon_lat_18 = np.column_stack((np.repeat(siteL18_lon, len(sol_rad_18)),np.repeat(siteL18_lat, len(sol_rad_18))))
lon_lat_19 = np.column_stack((np.repeat(siteL19_lon, len(sol_rad_19)),np.repeat(siteL19_lat, len(sol_rad_19))))
lon_lat_20 = np.column_stack((np.repeat(siteL20_lon, len(sol_rad_20)),np.repeat(siteL20_lat, len(sol_rad_20))))

## combind the two data sets
# all of locations
lon_lat_total = np.array(np.concatenate([lon_lat_CID, lon_lat_CJ, lon_lat_UP, lon_lat_MWS, lon_lat_MWSB,  # Ambient Weather
                                        lon_lat_1, lon_lat_2, lon_lat_3, lon_lat_4, lon_lat_5, lon_lat_6, lon_lat_7, lon_lat_8, # NREL
                                        lon_lat_9, lon_lat_10, lon_lat_11, lon_lat_12, lon_lat_13, lon_lat_14, lon_lat_15, lon_lat_16,
                                        lon_lat_17, lon_lat_18, lon_lat_19, lon_lat_20], axis = 0))


######################################################
# Get solar irradiance without clouds and wind speed #
######################################################

## Wind speed

# convert mph to m/s -> / 2.237
conv = 2.237

# ambient weather
wind_CID = np.array(data_CID["Wind Speed (mph)"])
wind_CJ = np.array(data_CJ["Wind Speed (mph)"])
wind_UP = np.array(data_UP["Wind Speed (mph)"])
wind_MWS = np.array(data_MWS["Wind Speed (mph)"])
wind_MWSB = np.array(data_MWSB["Wind Speed (mph)"])


# convert to m/s
wind_CID = wind_CID/conv
wind_CJ = wind_CJ/conv
wind_UP = wind_UP/conv
wind_MWS = wind_MWS/conv
wind_MWSB = wind_MWSB/conv

# NREL
wind_1 = np.array(site1["Wind Speed"])
wind_2 = np.array(site2["Wind Speed"])
wind_3 = np.array(site3["Wind Speed"])
wind_4 = np.array(site4["Wind Speed"])
wind_5 = np.array(site5["Wind Speed"])
wind_6 = np.array(site6["Wind Speed"])
wind_7 = np.array(site7["Wind Speed"])
wind_8 = np.array(site8["Wind Speed"])
wind_9 = np.array(site9["Wind Speed"])
wind_10 = np.array(site10["Wind Speed"])
wind_11 = np.array(site11["Wind Speed"])
wind_12 = np.array(site12["Wind Speed"])
wind_13 = np.array(site13["Wind Speed"])
wind_14 = np.array(site14["Wind Speed"])
wind_15 = np.array(site15["Wind Speed"])
wind_16 = np.array(site16["Wind Speed"])
wind_17 = np.array(site17["Wind Speed"])
wind_18 = np.array(site18["Wind Speed"])
wind_19 = np.array(site19["Wind Speed"])
wind_20 = np.array(site20["Wind Speed"])

wind_tot = np.concatenate([wind_CID, wind_CJ, wind_UP, wind_MWS, wind_MWSB, wind_1, wind_2,
                               wind_3, wind_4, wind_5, wind_6, wind_7, wind_8, wind_9, wind_10,
                               wind_11, wind_12, wind_13, wind_14, wind_15, wind_16, wind_17,
                               wind_18, wind_19, wind_20], axis = 0)

#########################
# Get Rid of NAN values #
#########################

## x variable
# time variable
time_tot = time_tot[np.isnan(SolR) == False]
# date variables
date_numb_tot = date_numb_tot[np.isnan(SolR) == False]
# wind variables
wind_tot = wind_tot[np.isnan(SolR) == False]
# clearsky solar
clear_tot = clear_tot[np.isnan(SolR) == False]

## Predictions
date_numb_pred = date_numb_pred[np.isnan(SolR) == False]

## coords variable
# location
lon_lat_total = lon_lat_total[np.isnan(SolR) == False]

## y variable
# solar radiance
SolR = SolR[np.isnan(SolR) == False]

#########################
# Trim 0's of solar rad #
#########################

## x variable
# time variable
time_tot = time_tot[SolR != 0]
# date variables
date_numb_tot = date_numb_tot[SolR != 0]
# wind variables
wind_tot = wind_tot[SolR != 0]
# clearsky solar
clear_tot = clear_tot[SolR != 0]

## Predictions
date_numb_pred = date_numb_pred[SolR != 0]

## coords variable
# location
lon_lat_total = lon_lat_total[SolR != 0]

## y variable
# solar radiance
SolR = SolR[SolR != 0]

##################
# Get Date Range #
##################


## High

# Predictions
date_numb_pred = date_numb_pred[date_numb_tot <= dateH]

# coords
lon_lat_total = lon_lat_total[date_numb_tot <= dateH]

# Y
SolR = np.array(SolR[date_numb_tot <= dateH])

# x
time_tot = time_tot[date_numb_tot <= dateH]
wind_tot = wind_tot[date_numb_tot <= dateH]
clear_tot = clear_tot[date_numb_tot <= dateH]
date_numb_tot = date_numb_tot[date_numb_tot <= dateH]


## Low

# Predictions
date_numb_pred = date_numb_pred[date_numb_tot >= dateL]

# coords
lon_lat_total = lon_lat_total[date_numb_tot >= dateL]

# Y
SolR = np.array(SolR[date_numb_tot >= dateL])

# x
time_tot = time_tot[date_numb_tot >= dateL]
wind_tot = wind_tot[date_numb_tot >= dateL]
clear_tot = clear_tot[date_numb_tot >= dateL]
date_numb_tot = date_numb_tot[date_numb_tot >= dateL]


################
# Variables x2 #
################



# variables definied
# Longitude and Latitude pairs
n = len(time_tot)

print(n)
# number of samples
n_samples = 50
# number of days to predict per site
n_report = 50

nDaysP = 1

# Data for Gaussian Process
sigma_sq = 5
tau_sq = 1
phi = 3/0.5

######################
# Make the variables #
######################

## x
# has 1, time(hours), and date(days)
x = np.column_stack((np.ones(len(time_tot)), time_tot, date_numb_tot, clear_tot, wind_tot))

## y
# has the solar radiance
y = SolR

## coords
# longitude, latitude
coords = lon_lat_total

##################################
# Hold out (for specific day(s)) #
##################################

days_unique = np.unique(date_numb_pred)

# get random __ amount of days
rand_day = np.random.choice(days_unique, nDaysP, replace = False)

rand_day = np.array([191.])

for i in range(0, nDaysP, 1):
    if(i == 0): # if at the first one make the first part of the tests
        ## holding data (test data)
        y_test = y[date_numb_pred == rand_day[i]]
        x_test = x[date_numb_pred == rand_day[i]]
        numb_days_test = date_numb_pred[date_numb_pred == rand_day[i]]
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = coords[date_numb_pred == rand_day[i]] # coords

    else: # everwhere else append the days
        ## holding data (test data)
        y_test = np.append(y_test, y[date_numb_pred == rand_day[i]], axis = 0)
        x_test = np.append(x_test, x[date_numb_pred == rand_day[i]], axis = 0)
        numb_days_test = np.append(numb_days_test, date_numb_pred[date_numb_pred == rand_day[i]], axis = 0)
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = np.append(coords_test, coords[date_numb_pred == rand_day[i]], axis = 0) # coords
            
    ## not holding data (train data)
        # x and y are random variables
    y_train = y[date_numb_pred != rand_day[i]]
    x_train = x[date_numb_pred != rand_day[i]]
    # w_train = w[hold_in]
    # weight_train = weight[train] # weight
    coords_train = coords[date_numb_pred != rand_day[i]] # coords

# phi, sigma_sq = 5, tau_sq = 1
starting = np.array([phi, 5, 1])
# phi = 0.5, sigma_sq = 0.5, tau_sq = 0.5
tuning = np.array([0.01, 0.01, 0.01])
# phi_Unif = [3/1, 3/0.1], sigma_sq_IG = [2, 5], tau_sq_IG = [2, 1]
priors = np.array([[3/1, 3/0.01], [2, 5], [2, 1]]) 
# model
covariance_model = "exponential"

# increase/decrease tuning to make the acceptance better
# change the unif to somthing else maybe??

################
# Run Response #
################

startM = time.time()
m_r = spNNGP(x = x_train, y = y_train, coords = coords_train, starting = starting, method = "response", n_neighbors = 3,
            tuning = tuning, priors = priors, cov_model = covariance_model, n_samples = n_samples,
            n_report = n_report, n_reps = 10)
stopM = time.time()

###################
# Run Predictions #
###################

startP = time.time()
p_r = predict(m_r, x_test, coords_test, progress_rep = True, n_reps = 1000)
stopP = time.time()

#################
# Print Results #
#################


# get the y hat
y_hat = p_r
# get the average
y_hat = y_hat.mean(axis = 1)

# show y hat vs y_test
        # print("y_hat")
        # print(y_hat)
        # print("y_test")
        # print(y_test)

diff = abs(y_hat - y_test)

plt.boxplot(diff)
plt.show()

print("----------------------------------------------------------")
print(" Statistics (W/m^2)                     ")
print("----------------------------------------------------------")

# variance of y
print("Variance of y")
print(round(np.var(y_train), 2))

# mean(y.hat - y.ho)^2
eq = (y_hat - y_test)**2
mean_eq = np.average(eq)

# MSE
print("mean(y.hat - y.ho)^2")
print(round(mean_eq, 2))

####################
# Show the summary #
####################

summary(m_r)

print("----------------------------------------------------------")
print(" Times (s)                              ")
print("----------------------------------------------------------")

# times
print("Time for creating the model: " + str(round((stopM - startM), 2)) + " sec")
print("Time for predictions: " + str(round((stopP - startP), 2)) + " sec")

print("----------------------------------------------------------")
print(" Day Range                              ")
print("----------------------------------------------------------")

year = "2022"
# low
dateLstr = str(int(dateL))
dateLstr.rjust(3 + len(dateLstr), '0')
dateL_conv = datetime.strptime(year + "/" + dateLstr, "%Y/%j").strftime("%m/%d/%Y")
# high
dateHstr = str(int(dateH))
dateHstr.rjust(3 + len(dateHstr), '0')
dateH_conv = datetime.strptime(year + "/" + dateHstr, "%Y/%j").strftime("%m/%d/%Y")

print("Date Range = " + str(dateL_conv) + " to " + str(dateH_conv))

print("Chosen Date(s) for Predictions: ")

for i in range(0, nDaysP, 1):
    year = "2022"
    dayChosen = str(int(rand_day[i]))
    dayChosen.rjust(3 + len(dayChosen), '0')
    dayPredict = datetime.strptime(year + "/" + dayChosen, "%Y/%j").strftime("%m/%d/%Y")

    print(str(i + 1) + ": " + str(dayPredict))

print("Location(s): ")

for i in range(0, len(names), 1):
    print(str(i+1) + ": " + str(names[i]))

###########################
# plot each day predicted #
###########################

for i in range(0, nDaysP, 1):
    print("Day " + str(i +1))
    for j in range(0, 5, 1):

        ###########################
        # Get values for that day #
        ###########################

        # get the values of the day
        
        y_hat_day = y_hat[numb_days_test == rand_day[i]]
        y_test_day = y_test[numb_days_test == rand_day[i]]
        x_test_day = x_test[numb_days_test == rand_day[i]]
        coords_test_day = coords_test[numb_days_test == rand_day[i]]

        # get for a specific coord
            # for each spot "j"
        y_hat_day = y_hat_day[coords_test_day[:, 0] == lon[j]]
        y_test_day = y_test_day[coords_test_day[:, 0] == lon[j]]
        x_test_day = x_test_day[coords_test_day[:, 0] == lon[j]]

        plt.subplot(3, 2, j+1)
        plt.plot(x_test_day[:, 1], y_hat_day)
        plt.plot(x_test_day[:, 1], y_test_day)

        year = "2022"
        dayChosen = str(int(rand_day[i]))
        dayChosen.rjust(3 + len(dayChosen), '0')
        dayPredict = datetime.strptime(year + "/" + dayChosen, "%Y/%j").strftime("%m/%d/%Y")

        plt.legend(['Predicted', 'Observed'])
        # plt.title(("Solar Irradiance of y_hat vs y_test on: " + dayPredict + ", at location: " + str(names[j])))
        plt.title((str(names[j])) + " on: " + dayPredict)
        plt.xlabel("Time (hr)")
        plt.ylabel("Solar Irradiance (W/m^2)")
    plt.show()
    
    for j in range(0, 5, 1):

        #########################
        # Get diff for that day #
        #########################

        # get the values of the day
        
        y_hat_day = y_hat[numb_days_test == rand_day[i]]
        y_test_day = y_test[numb_days_test == rand_day[i]]
        coords_test_day = coords_test[numb_days_test == rand_day[i]]

        # get for a specific coord
            # for each spot "j"
        y_hat_day = y_hat_day[coords_test_day[:, 0] == lon[j]]
        y_test_day = y_test_day[coords_test_day[:, 0] == lon[j]]

        diff_day = np.abs(y_hat_day - y_test_day)

        plt.subplot(3, 2, j+1)
        plt.boxplot(diff_day)

        year = "2022"
        dayChosen = str(int(rand_day[i]))
        dayChosen.rjust(3 + len(dayChosen), '0')
        dayPredict = datetime.strptime(year + "/" + dayChosen, "%Y/%j").strftime("%m/%d/%Y")

        # plt.title(("Solar Irradiance of y_hat vs y_test on: " + dayPredict + ", at location: " + str(names[j])))
        plt.title((str(names[j])) + " on: " + dayPredict)
        plt.ylabel("Solar Irradiance (W/m^2)")
    plt.show()
