# plotting
import matplotlib.pyplot as plt

# import spNNGP class
from spNNGP_py import spNNGP
# from predict import predict
from predict_py import predict

# get the times
import time

# for matricies
import numpy as np
import pandas as pd

# for date calcs
from datetime import date, datetime

# CID
Latitude_CID = 18.2115553
Longitude_CID = -67.1370868
file_CID = "..Data Puerto Rico/CID.csv"
# Casa Juan, Mayagüez
Latitude_CJ = 18.200897
Longitude_CJ = -67.148253
file_CJ = "..Data Puerto Rico/CJ.csv"
# Mayaguez Urb Paseo Los Robles , Mayagüez
Latitude_UP = 18.231220
Longitude_UP = -67.155010
file_UP = "..Data Puerto Rico/UP.csv"
# My Weather Station, Mayagüez
Latitude_MWS = 18.234288
Longitude_MWS = -67.150602    
file_MWS = "..Data Puerto Rico/MWS.csv"
# My Weather Station, Mayagüez (Blue logo)
Latitude_MWSB = 18.233073
Longitude_MWSB = -67.136155   
file_MWSB = "..Data Puerto Rico/MWSB.csv"

######################################
# make an array of long for plotting #
######################################
lon = np.array([Longitude_CID, Longitude_CJ, Longitude_UP, Longitude_MWS, Longitude_MWSB])
names = np.array(["CID", "Casa Juan", "Mayaguez Urb Paseo Los Robles", "Calle La Torre", "Calle C"])

############
# Get Data #
############

data_CID = pd.read_csv(file_CID)
data_CJ = pd.read_csv(file_CJ)
data_UP = pd.read_csv(file_UP)
data_MWS = pd.read_csv(file_MWS)
data_MWSB = pd.read_csv(file_MWSB)

# needed variables
    # solar radiance
    # time, date

###################
# Y Variable Data #
###################

# y variable
SolR_CID = data_CID["Solar Radiation (W/m^2)"]
SolR_CJ = data_CJ["Solar Radiation (W/m^2)"]
SolR_UP = data_UP["Solar Radiation (W/m^2)"]
SolR_MWS = data_MWS["Solar Radiation (W/m^2)"]
SolR_MWSB = data_MWSB["Solar Radiation (W/m^2)"]

# all of the solar values
SolR = np.array(pd.concat([SolR_CID, SolR_CJ, SolR_UP, SolR_MWS, SolR_MWSB], axis = 0))

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

# temperature
temp_CID = data_CID["Outdoor Temperature (°F)"]
temp_CJ = data_CJ["Outdoor Temperature (°F)"]
temp_UP = data_UP["Outdoor Temperature (°F)"]
temp_MWS = data_MWS["Outdoor Temperature (°F)"]
temp_MWSB = data_MWSB["Outdoor Temperature (°F)"]

temp_tot = np.array(pd.concat([temp_CID, temp_CJ, temp_UP, temp_MWS, temp_MWSB], axis = 0))

UV_CID = data_CID["Ultra-Violet Radiation Index"]
UV_CJ = data_CJ["Ultra-Violet Radiation Index"]
UV_UP = data_UP["Ultra-Violet Radiation Index"]
UV_MWS = data_MWS["Ultra-Violet Radiation Index"]
UV_MWSB = data_MWSB["Ultra-Violet Radiation Index"]

UV_tot = np.array(pd.concat([UV_CID, UV_CJ, UV_UP, UV_MWS, UV_MWSB], axis = 0))

########################
# corods Variable Data #
########################

# create long for lenght of date
lon_lat_CID = np.column_stack((np.repeat(Longitude_CID, len(date_CID)),np.repeat(Latitude_CID, len(date_CID))))
lon_lat_CJ = np.column_stack((np.repeat(Longitude_CJ, len(date_CJ)),np.repeat(Latitude_CJ, len(date_CJ))))
lon_lat_UP = np.column_stack((np.repeat(Longitude_UP, len(date_UP)),np.repeat(Latitude_UP, len(date_UP))))
lon_lat_MWS = np.column_stack((np.repeat(Longitude_MWS, len(date_MWS)),np.repeat(Latitude_MWS, len(date_MWS))))
lon_lat_MWSB = np.column_stack((np.repeat(Longitude_MWSB, len(date_MWSB)),np.repeat(Latitude_MWSB, len(date_MWSB))))

# all of locations
lon_lat_total = np.array(np.concatenate([lon_lat_CID, lon_lat_CJ, lon_lat_UP, lon_lat_MWS, lon_lat_MWSB], axis = 0))

##########################
# Get the number of days #
##########################

# get year, month, and date
year = np.zeros(len(date_time), dtype = int)
month = np.zeros(len(date_time), dtype = int)
day = np.zeros(len(date_time), dtype = int)
for i in range(0, len(date_time), 1):
    year[i]=int(date_time[i][0:4])
    month[i]=int(date_time[i][5:7])
    day[i]=int(date_time[i][8:10])


# start jan 1st, 2022
d0 = date(2022, 1, 1)

date_stored = np.array(len(date_time), dtype='datetime64[D]')
numb_of_days_conv = np.zeros(len(date_time))

numb_of_days = np.zeros(len(date_time))
for i in range(0, len(date_time), 1):
    d1 = date(year[i], month[i], day[i])
    numb_of_days[i] = int((d1 - d0).days)
    numb_of_days_conv[i] = numb_of_days[i]

date_conv = np.char.add(month.astype(str), '/')
date_conv = np.char.add(date_conv, day.astype(str))
date_conv = np.char.add(date_conv, '/')
date_conv = np.char.add(date_conv, year.astype(str))


#########################
# Get the time in hours #
#########################

hour = np.zeros(len(date_time), dtype = float)
min_hour = np.zeros(len(date_time), dtype = float)
for i in range(0, len(date_time), 1):
    hour[i]=int(date_time[i][11:13])
    min_hour[i]=int(date_time[i][14:16])/60

# time variable
time_hour = hour + min_hour 

#########################
# Get Rid of NAN values #
#########################

## x variable
# time variable
time_hour = time_hour[np.isnan(SolR) == False]
# date variable
numb_of_days = numb_of_days[np.isnan(SolR) == False]
# temp variable
temp_tot = temp_tot[np.isnan(SolR) == False]
# UV variable
UV_tot = UV_tot[np.isnan(SolR) == False]

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
time_hour = time_hour[SolR != 0]
# date variable
numb_of_days = numb_of_days[SolR != 0]
# temp variable
temp_tot = temp_tot[SolR != 0]
# UV variable
UV_tot = UV_tot[SolR != 0]

## coords variable
# location
lon_lat_total = lon_lat_total[SolR != 0]

## y variable
# solar radiance
SolR = SolR[SolR != 0]

dateH = numb_of_days[0]
dateL = numb_of_days[len(numb_of_days)-1]

############################
# Make second coords value #
############################

################
# Variables x2 #
################


# variables definied
# Longitude and Latitude pairs
n = len(time_hour)
# number of samples
n_samples = 1000
# number of reports
n_report = 1000
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
x = np.column_stack((np.ones(len(time_hour)), time_hour, numb_of_days, UV_tot, temp_tot))

## y
# has the solar radiance
y = SolR

## coords
# longitude, latitude
coords = lon_lat_total

##################################
# Hold out (for specific day(s)) #
##################################

days_unique = np.unique(numb_of_days)

# get random __ amount of days
rand_day = np.random.choice(days_unique, nDaysP, replace = False)

#
for i in range(0, nDaysP, 1):
    if(i == 0): # if at the first one make the first part of the tests
        ## holding data (test data)
        y_test = y[numb_of_days == rand_day[i]]
        x_test = x[numb_of_days == rand_day[i]]
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = coords[numb_of_days == rand_day[i]] # coords

    else: # everwhere else append the days
        ## holding data (test data)
        y_test = np.append(y_test, y[numb_of_days == rand_day[i]], axis = 0)
        x_test = np.append(x_test, x[numb_of_days == rand_day[i]], axis = 0)
        # w_test = w[hold_out]
        # weight_test = weight[test] # weight
        coords_test = np.append(coords_test, coords[numb_of_days == rand_day[i]], axis = 0) # coords
            
    ## not holding data (train data)
        # x and y are random variables
    y_train = y[numb_of_days != rand_day[i]]
    x_train = x[numb_of_days != rand_day[i]]
    # w_train = w[hold_in]
    # weight_train = weight[train] # weight
    coords_train = coords[numb_of_days != rand_day[i]] # coords

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
            n_report = n_report)
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
    indx = np.where(numb_of_days_conv == rand_day[i])
    days_test = date_conv[indx]
    print(str(i+1) + ": " + str(days_test[0]))

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
        
        y_hat_day = y_hat[x_test[:, 2] == rand_day[i]]
        y_test_day = y_test[x_test[:, 2] == rand_day[i]]
        x_test_day = x_test[x_test[:, 2] == rand_day[i]]
        coords_test_day = coords_test[x_test[:, 2] == rand_day[i]]

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
