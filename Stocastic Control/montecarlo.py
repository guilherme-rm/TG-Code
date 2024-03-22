import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


prob = []
total = 5000
a=0

for i in range(1, total):
    
    x = np.random.choice([0, 1], 50, p=[0.6, 0.4])

    if np.sum(x[:25]) >= 0.6*np.sum(x):
        a += 1 

    prob.append(a/i)


plt.plot(np.arange(1, total), prob)
plt.grid()
plt.show()