# importing libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np



# This file is used to test the animation module and will be deleted later


# # Example 1
# from https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# plt.style.use('fivethirtyeight')
# fig = plt.figure()
# # creating a subplot
# ax1 = fig.add_subplot(1, 1, 1)
#
# def animate(i):
#     data = open('stock.txt', 'r').read()
#     # print(data)
#     lines = data.split('\n')
#     xs = []
#     ys = []
#     for line in lines[:-1]:
#         x, y = line.split(',')  # Delimiter is comma
#         xs.append(float(x))
#         ys.append(float(y))
#
#     ax1.clear()
#     ax1.plot(xs, ys)
#
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.title('Live graph with matplotlib')
#
#
# ani = animation.FuncAnimation(fig, animate, interval=1000)

# Example 2
# from: https://matplotlib.org/3.1.1/api/animation_api.html
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()