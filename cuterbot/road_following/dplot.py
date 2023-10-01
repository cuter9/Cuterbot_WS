import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
matplotlib.use("TkAgg")

x = np.linspace(0, 10 * np.pi, 100)
y = np.sin(x)

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')

for phase in np.linspace(0, 10 * np.pi, 100):
    # plt.cla()
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)
