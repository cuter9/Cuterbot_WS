import numpy as np
import matplotlib.pyplot as plt
import time
x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.ion()
# ax = plt.gca()
fig, ax = plt.plot()
ax.set_autoscale_on(True)
line, = ax.plot(x, y)

for i in range(100):
    # plt.clf()
    line.set_ydata(y)
    ax.relim()
    ax.autoscale_view(True, True, True)
    plt.draw()
    y = y * 1.1
    time.sleep(2)
    # plt.pause(100)
