# calculates the number of days and time for both AW and NREL

# plotting
import matplotlib.pyplot as plt

# matrix
import numpy as np

# calculate number of days
def date_calcNREL(month, day):
    # for date calcs
    from datetime import date, datetime

    year = np.repeat(2022, len(day))

    # start jan 1st, 2022
    d0 = date(2022, 1, 1)

    numb_of_days = np.empty(len(day))

    for i in range(0, len(day), 1):
        d1 = date(year[i], month[i], day[i]) # get the date in DD/MM/YY
        numb_of_days[i] = int((d1 - d0).days) # convert to ##
    
    return numb_of_days

def date_calcAW(date_chosen):
    # for date calcs
    from datetime import date, datetime

    year = np.empty(len(date_chosen)).astype(int)
    month = np.empty(len(date_chosen)).astype(int)
    day = np.empty(len(date_chosen)).astype(int)

    for i in range(0, len(date_chosen), 1):
        year[i]=int(date_chosen[i][0:4])
        month[i]=int(date_chosen[i][5:7])
        day[i]=int(date_chosen[i][8:10])

    # start jan 1st, 2022
    d0 = date(2022, 1, 1)

    numb_of_days = np.empty(len(date_chosen))

    for i in range(0, len(date_chosen), 1):
        d1 = date(year[i], month[i], day[i]) # get the date in DD/MM/YY
        numb_of_days[i] = int((d1 - d0).days) # convert to ##
    
    return numb_of_days

def calc_hourAW(date_chosen):
    hour = np.zeros(len(date_chosen), dtype = float)
    mins = np.zeros(len(date_chosen), dtype = float)

    for i in range(0, len(date_chosen), 1):
        hour[i]=int(date_chosen[i][11:13])
        mins[i]=int(date_chosen[i][14:16])
    
    for i in range(0, len(date_chosen), 1): # make sure the timestep is 5 min
        if(mins[i] % 5 != 0 and mins[i] != 0): # if number isnt 0, 5, 10, 15...
                            # remainder  
            mins[i] = mins[i] - mins[i] % 5

    return hour + mins/60 # total time

def calc_hourNREL(hour, min):
    return hour + min/60 #total time

def get_solar_no_cloud(time_NREL, date_num_NREL, clearNREL, time_AW, date_num_AW, dateH, dateL, test = False, sol = None):
    clear_AW = np.array([])

    for j in range(dateH, dateL -1, -1):
        # get the day
        time_NREL_temp = time_NREL[date_num_NREL == j] # get time on 201
        clear_NREL_temp = clearNREL[date_num_NREL == j]
        if(test):
            sol_temp = sol[date_num_NREL == j]

        time_AW_temp = time_AW[date_num_AW == j]

        if(test):
            plt.plot(time_NREL_temp, clear_NREL_temp)
            plt.plot(time_NREL_temp, sol_temp)
            plt.show()


        k = 0 # increment for time_7
        i = 0

        index_del = np.array([]).astype(int)

        while(i < len(time_AW_temp)):
            if(time_NREL_temp[k] != time_AW_temp[i]):
                # if(test == True):
                #     print("bad at " + str(k) + ", " + str(j))
                #     print("7 = " + str(time_NREL_temp[k]) + " CID = "  + str(time_AW_temp[i]))
                index_del = np.append(index_del, k)
                k += 1 # by 1, and don't change i
            else:
                k += 1 # normally follow i
                i += 1

        # takes out deletes
        if(len(index_del) != 0): # if not empty
            clear_NREL_temp = np.delete(clear_NREL_temp, index_del)
            time_NREL_temp = np.delete(time_NREL_temp, index_del)

        # takes out last spots if not used
        while(len(time_NREL_temp) != len(time_AW_temp)):
            time_NREL_temp = np.delete(time_NREL_temp, (len(time_NREL_temp) - 1)) # delete the last element
            clear_NREL_temp = np.delete(clear_NREL_temp, (len(time_NREL_temp) - 1))
        
        clear_AW = np.append(clear_AW, clear_NREL_temp) # add the coorect data for each day

    return clear_AW