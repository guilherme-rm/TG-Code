import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb

matplotlib.use('TkAgg')

T = 150
C = 6

t = np.arange(T-1)
x = np.zeros(T, dtype=np.int32)
d = np.random.choice([0, 1, 2], len(t), p=[0.7, 0.2, 0.1])
d[0] = 0
x[0] = 6

for dt in t:
    x[dt] = max(x[dt] - d[dt], 0)

    if x[dt] <= 1:
        x[dt+1] = C - x[dt]
    else: x[dt+1] = x[dt]

t = np.append(t, (t[-1] + 1))

plt.plot(t, x, '-o', color='blue', linewidth=1)
plt.grid()
plt.show()