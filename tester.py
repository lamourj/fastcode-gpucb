import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt

file1 = open("reference_output.txt", 'r')
file2 = open("mu_c.txt", 'r')

lines1 = [line.replace("\n", "").split(",") for line in file1.readlines()]
data1 = []

for line in lines1:
    data1 += line

reference = np.array([float(item) for item in data1])

lines2 = [line.replace("\n", "").split(",") for line in file2.readlines()]
data2 = []

for line in lines2:
    data2 += line

mu = []

for item in data2:
    try :
        mu.append(float(item))
    except:
        pass

mu = np.array(mu)

x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)

grid = np.meshgrid(x, y)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(grid[0], grid[1], mu.reshape(grid[0].shape), alpha=0.5, color='g')
ax.plot_wireframe(grid[0], grid[1], reference.reshape(grid[0].shape), alpha=0.5, color='b')

plt.savefig('fig.png')
