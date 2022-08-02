# graphs the sampels
# for 5000 samples
    # used to check what the best tuning value is..

# matrix
import numpy as np

# plotting
import matplotlib.pyplot as plt


tuning = np.array([.1, .05, .03, .01])
acc = np.array([10.74, 18.92, 32.02, 55.78])

plt.plot(tuning, acc)
plt.show()
