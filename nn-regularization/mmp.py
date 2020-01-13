import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots

x = np.arange(10)
y = x
plt.scatter(x, y, c='r', cmap=plt.cm.Spectral)
plt.show()





