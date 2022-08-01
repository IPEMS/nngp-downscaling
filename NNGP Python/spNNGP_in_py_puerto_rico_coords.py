# plotting
import matplotlib.pyplot as plt

# import spNNGP class
from spNNGP_py import spNNGP
# from predict import predict
from predict_py import predict
# summary of NNGP
from summary_NNGP_py import summary

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

# for testing out locations
Location_local = np.array([[Longitude_CID, Latitude_CID],
                            [Longitude_CJ, Latitude_CJ],
                            [Longitude_UP, Latitude_UP],
                            [Longitude_MWS, Latitude_MWS],
                            [Longitude_MWSB, Latitude_MWSB]])

##############
# NREL Data #
##############

# number of files
nFiles = 20

# file names
file_name_1 = "6374214_18.26_-67.17_2020.csv"
file_name_2 = "6374215_18.24_-67.17_2020.csv"
file_name_3 = "6374216_18.22_-67.17_2020.csv"
file_name_4 = "6374217_18.20_-67.17_2020.csv"   # -67.17
file_name_5 = "6374218_18.18_-67.17_2020.csv"

file_name_6 = "6378060_18.26_-67.15_2020.csv"
file_name_7 = "6378061_18.24_-67.15_2020.csv"
file_name_8 = "6378062_18.22_-67.15_2020.csv"   # -67.15
file_name_9 = "6378063_18.20_-67.15_2020.csv"
file_name_10 = "6378064_18.18_-67.15_2020.csv"

file_name_11 = "6381900_18.26_-67.13_2020.csv"
file_name_12 = "6381901_18.24_-67.13_2020.csv"
file_name_13 = "6381902_18.22_-67.13_2020.csv"   # -67.13
file_name_14 = "6381903_18.20_-67.13_2020.csv"
file_name_15 = "6381904_18.18_-67.13_2020.csv"

file_name_16 = "6385740_18.26_-67.11_2020.csv"
file_name_17 = "6385741_18.24_-67.11_2020.csv"
file_name_18 = "6385742_18.22_-67.11_2020.csv"   # -67.11
file_name_19 = "6385743_18.20_-67.11_2020.csv"
file_name_20 = "6385744_18.18_-67.11_2020.csv"

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

# file array used to get location
file = np.array([file_name_1, file_name_2, file_name_3, file_name_4, file_name_5, file_name_6, file_name_7, 
                file_name_8, file_name_9, file_name_10, file_name_11, file_name_12, file_name_13, file_name_14, 
                file_name_15, file_name_16, file_name_17, file_name_18, file_name_19, file_name_20])

# get files data (time, date, solar irradiance)
site1 = pd.read_csv(file_1, skiprows=2)
site2 = pd.read_csv(file_2, skiprows=2)
site3 = pd.read_csv(file_3, skiprows=2)
site4 = pd.read_csv(file_4, skiprows=2)
site5 = pd.read_csv(file_5, skiprows=2)
site6 = pd.read_csv(file_6, skiprows=2)
site7 = pd.read_csv(file_7, skiprows=2)
site8 = pd.read_csv(file_8, skiprows=2)
site9 = pd.read_csv(file_9, skiprows=2)
site10 = pd.read_csv(file_10, skiprows=2)
site11 = pd.read_csv(file_11, skiprows=2)
site12 = pd.read_csv(file_12, skiprows=2)
site13 = pd.read_csv(file_13, skiprows=2)
site14 = pd.read_csv(file_14, skiprows=2)
site15 = pd.read_csv(file_15, skiprows=2)
site16 = pd.read_csv(file_16, skiprows=2)
site17 = pd.read_csv(file_17, skiprows=2)
site18 = pd.read_csv(file_18, skiprows=2)
site19 = pd.read_csv(file_19, skiprows=2)
site20 = pd.read_csv(file_20, skiprows=2)


################################
# get solar irradiance (NREL) #
################################

sol_rad_1 = site1["GHI"]
sol_rad_2 = site2["GHI"]
sol_rad_3 = site3["GHI"]
sol_rad_4 = site4["GHI"]
sol_rad_5 = site5["GHI"]
sol_rad_6 = site6["GHI"]
sol_rad_7 = site7["GHI"]
sol_rad_8 = site8["GHI"]
sol_rad_9 = site9["GHI"]
sol_rad_10 = site10["GHI"]
sol_rad_11 = site11["GHI"]
sol_rad_12 = site12["GHI"]
sol_rad_13 = site13["GHI"]
sol_rad_14 = site14["GHI"]
sol_rad_15 = site15["GHI"]
sol_rad_16 = site16["GHI"]
sol_rad_17 = site17["GHI"]
sol_rad_18 = site18["GHI"]
sol_rad_19 = site19["GHI"]
sol_rad_20 = site20["GHI"]

# later combined with Ambient weather data <--- @ Y Variable Data

#############################
# Get location Data (NREL) #
#############################

# coords from the title
    # 8:13 -> latitude
    # 14:20 -> longitude

loc_model = np.empty((nFiles, 2))

for i in range(0, nFiles, 1):
    file_temp = file[i] # first is file_1
    loc_model[i, :] = np.array([float(file_temp[14:20]), float(file_temp[8:13])]) # get long and lat

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
print_loc = True
if(print_loc):
    img = plt.imread(file_img)
    fig, ax = plt.subplots(figsize = (8, 7))
    ax.scatter(Location_local[:, 0], Location_local[:, 1], c = "b") # Ambient Weather
    ax.scatter(loc_model[:, 0], loc_model[:, 1], c = "r") # NREL
    plt.legend(['Ambient Weather', 'NREL'])
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(img, zorder=0, extent = BBox)
    plt.show()

##############################################
# make an array of long for plotting (NREL) #
##############################################
lon = np.array([Longitude_CID, Longitude_CJ, Longitude_UP, Longitude_MWS, Longitude_MWSB])
names = np.array(["CID", "Casa Juan", "Mayaguez Urb Paseo Los Robles", "Calle La Torre", "Calle C"])

##############################
# Get Data (Ambient Weather) #
##############################

data_CID = pd.read_csv(file_CID)
data_CJ = pd.read_csv(file_CJ)
data_UP = pd.read_csv(file_UP)
data_MWS = pd.read_csv(file_MWS)
data_MWSB = pd.read_csv(file_MWSB)

# needed variables
    # solar radiance
    # time, date

#################################################
# Y Variable Data (Ambient Weather) and (NREL) #
#################################################

# y variable
SolR_CID = data_CID["Solar Radiation (W/m^2)"]
SolR_CJ = data_CJ["Solar Radiation (W/m^2)"]
SolR_UP = data_UP["Solar Radiation (W/m^2)"]
SolR_MWS = data_MWS["Solar Radiation (W/m^2)"]
SolR_MWSB = data_MWSB["Solar Radiation (W/m^2)"]

## combind the two data sets
# all of the solar values
SolR = np.array(pd.concat([SolR_CID, SolR_CJ, SolR_UP, SolR_MWS, SolR_MWSB, # Ambient Weather 
                           sol_rad_1, sol_rad_2, sol_rad_3, sol_rad_4, sol_rad_5, sol_rad_6, # NREL
                           sol_rad_7, sol_rad_8, sol_rad_9, sol_rad_10, sol_rad_11, sol_rad_12, 
                           sol_rad_13, sol_rad_14, sol_rad_15, sol_rad_16, sol_rad_17, sol_rad_18,
                           sol_rad_19, sol_rad_20], axis = 0))

###################
# X Variable Data #
###################

# x variable
date_CID = data_CID["Simple Date"]
date_CJ = data_CJ["Simple Date"]
date_UP = data_UP["Simple Date"]
date_MWS = data_MWS["Simple Date"]
date_MWSB = data_MWSB["Simple Date"]

# all of the date/time variables
date_time = np.array(pd.concat([date_CID, date_CJ, date_UP, date_MWS, date_MWSB], axis = 0))

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
lon_lat_1 = np.column_stack((np.repeat(loc_model[0, 0], len(sol_rad_1)),np.repeat(loc_model[0, 1], len(sol_rad_1))))
lon_lat_2 = np.column_stack((np.repeat(loc_model[1, 0], len(sol_rad_2)),np.repeat(loc_model[1, 1], len(sol_rad_2))))
lon_lat_3 = np.column_stack((np.repeat(loc_model[2, 0], len(sol_rad_3)),np.repeat(loc_model[2, 1], len(sol_rad_3))))
lon_lat_4 = np.column_stack((np.repeat(loc_model[3, 0], len(sol_rad_4)),np.repeat(loc_model[3, 1], len(sol_rad_4))))
lon_lat_5 = np.column_stack((np.repeat(loc_model[4, 0], len(sol_rad_5)),np.repeat(loc_model[4, 1], len(sol_rad_5))))
lon_lat_6 = np.column_stack((np.repeat(loc_model[5, 0], len(sol_rad_6)),np.repeat(loc_model[5, 1], len(sol_rad_6))))
lon_lat_7 = np.column_stack((np.repeat(loc_model[6, 0], len(sol_rad_7)),np.repeat(loc_model[6, 1], len(sol_rad_7))))
lon_lat_8 = np.column_stack((np.repeat(loc_model[7, 0], len(sol_rad_8)),np.repeat(loc_model[7, 1], len(sol_rad_8))))
lon_lat_9 = np.column_stack((np.repeat(loc_model[8, 0], len(sol_rad_9)),np.repeat(loc_model[8, 1], len(sol_rad_9))))
lon_lat_10 = np.column_stack((np.repeat(loc_model[9, 0], len(sol_rad_10)),np.repeat(loc_model[9, 1], len(sol_rad_10))))
lon_lat_11 = np.column_stack((np.repeat(loc_model[10, 0], len(sol_rad_11)),np.repeat(loc_model[10, 1], len(sol_rad_11))))
lon_lat_12 = np.column_stack((np.repeat(loc_model[11, 0], len(sol_rad_12)),np.repeat(loc_model[11, 1], len(sol_rad_12))))
lon_lat_13 = np.column_stack((np.repeat(loc_model[12, 0], len(sol_rad_13)),np.repeat(loc_model[12, 1], len(sol_rad_13))))
lon_lat_14 = np.column_stack((np.repeat(loc_model[13, 0], len(sol_rad_14)),np.repeat(loc_model[13, 1], len(sol_rad_14))))
lon_lat_15 = np.column_stack((np.repeat(loc_model[14, 0], len(sol_rad_15)),np.repeat(loc_model[14, 1], len(sol_rad_15))))
lon_lat_16 = np.column_stack((np.repeat(loc_model[15, 0], len(sol_rad_16)),np.repeat(loc_model[15, 1], len(sol_rad_16))))
lon_lat_17 = np.column_stack((np.repeat(loc_model[16, 0], len(sol_rad_17)),np.repeat(loc_model[16, 1], len(sol_rad_17))))
lon_lat_18 = np.column_stack((np.repeat(loc_model[17, 0], len(sol_rad_18)),np.repeat(loc_model[17, 1], len(sol_rad_18))))
lon_lat_19 = np.column_stack((np.repeat(loc_model[18, 0], len(sol_rad_19)),np.repeat(loc_model[18, 1], len(sol_rad_19))))
lon_lat_20 = np.column_stack((np.repeat(loc_model[19, 0], len(sol_rad_20)),np.repeat(loc_model[19, 1], len(sol_rad_20))))

## combind the two data sets
# all of locations
lon_lat_total = np.array(np.concatenate([lon_lat_CID, lon_lat_CJ, lon_lat_UP, lon_lat_MWS, lon_lat_MWSB,  # Ambient Weather
                                        lon_lat_1, lon_lat_2, lon_lat_3, lon_lat_4, lon_lat_5, lon_lat_6, lon_lat_7, lon_lat_8, # NREL
                                        lon_lat_9, lon_lat_10, lon_lat_11, lon_lat_12, lon_lat_13, lon_lat_14, lon_lat_15, lon_lat_16,
                                        lon_lat_17, lon_lat_18, lon_lat_19, lon_lat_20], axis = 0))



##########################
# Get the number of days #
##########################

# Ambient Weather
    # get year, month, and date
year = np.zeros(len(date_time), dtype = int)
month = np.zeros(len(date_time), dtype = int) # size of SolR
day = np.zeros(len(date_time), dtype = int)
for i in range(0, len(date_time), 1):
    year[i]=int(date_time[i][0:4])
    month[i]=int(date_time[i][5:7])
    day[i]=int(date_time[i][8:10])

########################################################
# Convert date to ## (Used to get rand predicted date) #
########################################################

# start jan 1st, 2022
d0 = date(2022, 1, 1)

date_stored = np.array(len(date_time), dtype='datetime64[D]')
numb_of_days_conv = np.zeros(len(date_time))

numb_of_days = np.zeros(len(date_time))
for i in range(0, len(date_time), 1):
    d1 = date(year[i], month[i], day[i]) # get the date in DD/MM/YY
    numb_of_days[i] = int((d1 - d0).days) # convert to ##
    numb_of_days_conv[i] = numb_of_days[i] # convert back


# convert back to DD/MM/YY
date_conv = np.char.add(month.astype(str), '/')
date_conv = np.char.add(date_conv, day.astype(str))
date_conv = np.char.add(date_conv, '/')
date_conv = np.char.add(date_conv, year.astype(str))

# the lowest Date vs Highest Date
dateH = numb_of_days[0]
dateL = numb_of_days[len(numb_of_days)-1]

numb_of_days = np.append(numb_of_days, np.zeros(len(SolR) - len(numb_of_days)))

# NREL
    # get year, month, and date
year_N = np.concatenate([np.array(site1["Year"]), np.array(site2["Year"]), np.array(site3["Year"]), np.array(site4["Year"]), np.array(site5["Year"]), 
                        np.array(site6["Year"]), np.array(site7["Year"]), np.array(site8["Year"]), np.array(site9["Year"]), np.array(site10["Year"]),
                        np.array(site11["Year"]), np.array(site12["Year"]), np.array(site13["Year"]), np.array(site14["Year"]), np.array(site15["Year"]), 
                        np.array(site16["Year"]), np.array(site17["Year"]), np.array(site18["Year"]), np.array(site19["Year"]), np.array(site20["Year"])])

month_N = np.concatenate([np.array(site1["Month"]), np.array(site2["Month"]), np.array(site3["Month"]), np.array(site4["Month"]), np.array(site5["Month"]), 
                        np.array(site6["Month"]), np.array(site7["Month"]), np.array(site8["Month"]), np.array(site9["Month"]), np.array(site10["Month"]),
                        np.array(site11["Month"]), np.array(site12["Month"]), np.array(site13["Month"]), np.array(site14["Month"]), np.array(site15["Month"]), 
                        np.array(site16["Month"]), np.array(site17["Month"]), np.array(site18["Month"]), np.array(site19["Month"]), np.array(site20["Month"])])

day_N = np.concatenate([np.array(site1["Day"]), np.array(site2["Day"]), np.array(site3["Day"]), np.array(site4["Day"]), np.array(site5["Day"]), 
                        np.array(site6["Day"]), np.array(site7["Day"]), np.array(site8["Day"]), np.array(site9["Day"]), np.array(site10["Day"]),
                        np.array(site11["Day"]), np.array(site12["Day"]), np.array(site13["Day"]), np.array(site14["Day"]), np.array(site15["Day"]), 
                        np.array(site16["Day"]), np.array(site17["Day"]), np.array(site18["Day"]), np.array(site19["Day"]), np.array(site20["Day"])])

# start jan 1st, 2022
d0 = date(2022, 1, 1)

# Combind the dates
year_tot = np.concatenate([year, year_N])
month_tot = np.concatenate([month, month_N])
day_tot = np.concatenate([day, day_N])

#######################################################
# Get the time in hours (Ambient Weather) and (NREL) #
#######################################################

hour = np.zeros(len(date_time), dtype = float)
min_hour = np.zeros(len(date_time), dtype = float)
for i in range(0, len(date_time), 1):
    hour[i]=int(date_time[i][11:13])
    min_hour[i]=int(date_time[i][14:16])/60

# time variable
time_hour = hour + min_hour 

hour_N = np.concatenate([np.array(site1["Hour"]), np.array(site2["Hour"]), np.array(site3["Hour"]), np.array(site4["Hour"]), np.array(site5["Hour"]), 
                        np.array(site6["Hour"]), np.array(site7["Hour"]), np.array(site8["Hour"]), np.array(site9["Hour"]), np.array(site10["Hour"]),
                        np.array(site11["Hour"]), np.array(site12["Hour"]), np.array(site13["Hour"]), np.array(site14["Hour"]), np.array(site15["Hour"]), 
                        np.array(site16["Hour"]), np.array(site17["Hour"]), np.array(site18["Hour"]), np.array(site19["Hour"]), np.array(site20["Hour"])])

## combind the two data sets
time_hour = np.append(time_hour, hour_N)

########################
# Get temperature data #
########################

# ambient weather
temp_CID = data_CID["Outdoor Temperature (°F)"]
temp_CJ = data_CJ["Outdoor Temperature (°F)"]
temp_UP = data_UP["Outdoor Temperature (°F)"]
temp_MWS = data_MWS["Outdoor Temperature (°F)"]
temp_MWSB = data_MWSB["Outdoor Temperature (°F)"]

# NREL
temp_1 = site1["Temperature"]
temp_2 = site2["Temperature"]
temp_3 = site3["Temperature"]
temp_4 = site4["Temperature"]
temp_5 = site5["Temperature"]
temp_6 = site6["Temperature"]
temp_7 = site7["Temperature"]
temp_8 = site8["Temperature"]
temp_9 = site9["Temperature"]
temp_10 = site10["Temperature"]
temp_11 = site11["Temperature"]
temp_12 = site12["Temperature"]
temp_13 = site13["Temperature"]
temp_14 = site14["Temperature"]
temp_15 = site15["Temperature"]
temp_16 = site16["Temperature"]
temp_17 = site17["Temperature"]
temp_18 = site18["Temperature"]
temp_19 = site19["Temperature"]
temp_20 = site20["Temperature"]

temp_tot = np.array(pd.concat([temp_CID, temp_CJ, temp_UP, temp_MWS, temp_MWSB, temp_1, temp_2,
                               temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10,
                               temp_11, temp_12, temp_13, temp_14, temp_15, temp_16, temp_17,
                               temp_18, temp_19, temp_20], axis = 0))


#########################
# Get Rid of NAN values #
#########################

## x variable
# time variable
time_hour = time_hour[np.isnan(SolR) == False]
# date variables
year_tot = year_tot[np.isnan(SolR) == False]
month_tot = month_tot[np.isnan(SolR) == False]
day_tot = day_tot[np.isnan(SolR) == False]
# temp variables
temp_tot = temp_tot[np.isnan(SolR) == False]

## coords variable
# location
lon_lat_total = lon_lat_total[np.isnan(SolR) == False]

# for predictions
numb_of_days = numb_of_days[np.isnan(SolR) == False]

## y variable
# solar radiance
SolR = SolR[np.isnan(SolR) == False]

#########################
# Trim 0's of solar rad #
#########################

## x variable
# time variable
time_hour = time_hour[SolR != 0]
# date variables
year_tot = year_tot[SolR != 0]
month_tot = month_tot[SolR != 0]
day_tot = day_tot[SolR != 0]
# temp variables
temp_tot = temp_tot[SolR != 0]

## coords variable
# location
lon_lat_total = lon_lat_total[SolR != 0]

# for predictions
numb_of_days = numb_of_days[SolR != 0]

## y variable
# solar radiance
SolR = SolR[SolR != 0]

##############################
# Trim the data by the month #
##############################

monthH = 7
monthL = 6

time_final = np.array([])
year_final = np.array([])
month_final = np.array([])
day_final = np.array([])
lon_lat_final1 = np.array([])
SolR_final = np.array([])
temp_final = np.array([])
# for predictions
numb_of_days_final = np.array([])

for i in range(monthL, monthH + 1, 1):
    ## x variable
    # time variable
    time_final = np.append(time_final, time_hour[month_tot == i])
    # date variables
    year_final = np.append(year_final, year_tot[month_tot == i])
    month_final = np.append(month_final, month_tot[month_tot == i])
    day_final = np.append(day_final, day_tot[month_tot == i])
    # temp variables
    temp_final = np.append(temp_final, temp_tot[month_tot == i])

    ## coords variable
    # location
    lon_lat_final1 = np.append(lon_lat_final1, lon_lat_total[month_tot == i])

    ## y variable
    # solar radiance
    SolR_final = np.append(SolR_final, SolR[month_tot == i])

    # for predictions
    numb_of_days_final = np.append(numb_of_days_final, numb_of_days[month_tot == i])

############################
# Make second coords value #
############################

lon_lat_final = lon_lat_final1.reshape((len(SolR_final), 2))

################
# Variables x2 #
################


# variables definied
# Longitude and Latitude pairs
n = len(time_final)
# number of samples
n_samples = 500
# number of reports
n_report = 500
# number of days to predict per site
nDaysP = 1

# Data for Gaussian Process
sigma_sq = 741.9203
tau_sq = 3.7308
phi = 8.9911

######################
# Make the variables #
######################

## x
# has 1, time(hours), and date(days)
x = np.column_stack((np.ones(len(time_final)), time_final, year_final, month_final, day_final, temp_final))

## y
# has the solar radiance
y = SolR_final

## coords
# longitude, latitude
coords = lon_lat_final

##################################
# Hold out (for specific day(s)) #
##################################

days_unique = np.unique(numb_of_days)

# get random __ amount of days
rand_day = np.random.choice(days_unique, nDaysP, replace = False)

for i in range(0, nDaysP, 1):
    if(i == 0): # if at the first one make the first part of the tests
        ## holding data (test data)
        y_test = y[numb_of_days_final == rand_day[i]]
        x_test = x[numb_of_days_final == rand_day[i]]
        numb_days_test = numb_of_days_final[numb_of_days_final == rand_day[i]]
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = coords[numb_of_days_final == rand_day[i]] # coords

    else: # everwhere else append the days
        ## holding data (test data)
        y_test = np.append(y_test, y[numb_of_days_final == rand_day[i]], axis = 0)
        x_test = np.append(x_test, x[numb_of_days_final == rand_day[i]], axis = 0)
        numb_days_test = np.append(numb_days_test, numb_of_days_final[numb_of_days_final == rand_day[i]], axis = 0)
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = np.append(coords_test, coords[numb_of_days_final == rand_day[i]], axis = 0) # coords
            
    ## not holding data (train data)
        # x and y are random variables
    y_train = y[numb_of_days_final != rand_day[i]]
    x_train = x[numb_of_days_final != rand_day[i]]
    # w_train = w[hold_in]
    # weight_train = weight[train] # weight
    coords_train = coords[numb_of_days_final != rand_day[i]] # coords

# phi, sigma_sq = 5, tau_sq = 1
starting = np.array([phi, 5, 1])
# phi = 0.5, sigma_sq = 0.5, tau_sq = 0.5
tuning = np.array([0.5, 0.5, 0.5])
# phi_Unif = [3/1, 3/0.1], sigma_sq_IG = [2, 5], tau_sq_IG = [2, 1]
priors = np.array([[3/1, 3/0.01], [2, 5], [2, 1]]) 
# model
covariance_model = "exponential"


################
# Run Response #
################

startM = time.time()
m_r = spNNGP(x = x_train, y = y_train, coords = coords_train, starting = starting, method = "response", n_neighbors = 10,
            tuning = tuning, priors = priors, cov_model = covariance_model, n_samples = n_samples,
            n_report = n_report, n_reps = 10)
stopM = time.time()

###################
# Run Predictions #
###################

startP = time.time()
p_r = predict(m_r, x_test, coords_test)
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

####################
# Show the summary #
####################

summary(m_r)

# MSE
print("mean(y.hat - y.ho)^2")
print(round(mean_eq, 2))

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
