# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# import spNNGP class
from spNNGP_py import spNNGP
# from predict import predict
from predict_py import predict

# get the times
import time

# for matricies
import numpy as np
import pandas as pd

# practice data set
from sklearn import datasets

# to get a random day
import random

# for date calcs
from datetime import date

file_1 = "..Data Puerto Rico/Ambient Weather Data/CID_data_1year.csv"

data = pd.read_csv(file_1)

    # print(data)

date_time = np.array(data["Simple Date"], dtype = object)

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

# print(date_conv[10])

# date_conv = date_conv.astype(int)


hour = np.zeros(len(date_time), dtype = float)
min_hour = np.zeros(len(date_time), dtype = float)
for i in range(0, len(date_time), 1):
    hour[i]=int(date_time[i][11:13])
    min_hour[i]=int(date_time[i][14:16])/60

# time variable
time_hour = hour + min_hour 

# x parts
Temp = data["Outdoor Temperature (°F)"]
UV = data["Ultra-Violet Radiation Index"]
rain = data["Hourly Rain (in/hr)"]
# and date

# y part
y = data["Solar Radiation (W/m^2)"]

#########################
# Get Rid of NAN values #
#########################

# time variable
time_hour = time_hour[np.isnan(y) == False]

# x parts
Temp = Temp[np.isnan(y) == False]
UV = UV[np.isnan(y) == False]
rain = rain[np.isnan(y) == False]
numb_of_days = numb_of_days[np.isnan(y) == False]
# and date

y = y[np.isnan(y) == False]

##################
# Get Date Range #
##################

# max days
dayMaxH = numb_of_days[0]
dayMinL = numb_of_days[len(numb_of_days)-1]

# set the max
dayL = 181
dayH = 201     # ~ 90 days is a season
nDaysP = 1

## High

# coords
time_hour = time_hour[numb_of_days <= dayH]

# Y
y = np.array(y[numb_of_days <= dayH])

# x
Temp = Temp[numb_of_days <= dayH]
UV = UV[numb_of_days <= dayH]
rain = rain[numb_of_days <= dayH]
numb_of_days = numb_of_days[numb_of_days <= dayH]

## Low

# coords
time_hour = time_hour[numb_of_days >= dayL]

# Y
y = np.array(y[numb_of_days >= dayL])

# x
Temp = Temp[numb_of_days >= dayL]
UV = UV[numb_of_days >= dayL]
rain = rain[numb_of_days >= dayL]
numb_of_days = numb_of_days[numb_of_days >= dayL]


#########################
# Trim 0's of solar rad #
#########################

# x
numb_of_days = numb_of_days[y != 0]
Temp = Temp[y != 0]
UV = UV[y != 0]
rain = rain[y != 0]


# coords
time_hour = time_hour[y != 0]

# Y
y = np.array(y[y != 0])

############################
# Make second coords value #
############################

# time after
time_after = time_hour + 5/60 # add 5 min to get to next time step


################
# Variables x2 #
################


# variables definied
# Longitude and Latitude pairs
n = len(time_hour)
# number of samples
n_samples = 10
# number of reports
n_report = 10

# Data for Gaussian Process
sigma_sq = 741.9203
tau_sq = 3.7308
phi = 8.9911


##########
# Coords #
##########

# longitude = x, latitude = y
coords = np.column_stack((time_hour, time_after)) 

# make x side of the equation
    #
x = np.column_stack((np.ones(shape = (n, 1)), numb_of_days, Temp, rain))

    # rand_day = np.zeros(nDaysP)

##################################
# Hold out (for specific day(s)) #
##################################
    # for i in range(0, nDaysP, 1):
    #     rand_day[i] = random.randint(dayL, dayH)

rand_day = np.random.choice(numb_of_days, nDaysP, replace = False)

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

print("Date Range = " + str(dayL) + " to " + str(dayH))
print("Maximun Date Range = " + str(dayMinL) + " to " + str(dayMaxH))

print("Chosen Date(s) for Predictions: ")

for i in range(0, nDaysP, 1):
    indx = np.where(numb_of_days_conv == rand_day[i])
    days_test = date_conv[indx]
    print(str(days_test[0]) + " <---- " + str(round(rand_day[i])))

# plot each day predicted

for i in range(0, nDaysP, 1):

    ###########################
    # Get values for that day #
    ###########################

    coords_test_day = coords_test[x_test[:, 1] == rand_day[i]]
    y_hat_day = y_hat[x_test[:, 1] == rand_day[i]]
    y_test_day =  y_test[x_test[:, 1] == rand_day[i]]


    plt.plot(coords_test_day[:, 0], y_hat_day)
    plt.plot(coords_test_day[:, 0], y_test_day)
    plt.legend(['Predicted', 'Observed'])
    plt.title(("Solar Irradiance of y_hat vs y_test on " + str(rand_day[i])))
    plt.xlabel("Time (s)")
    plt.ylabel("Solar Irradiance (W/m^2)")
    plt.show()
