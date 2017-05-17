import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt

file1 = open("reference_output.txt", 'r')
file2 = open("mu_c.txt", 'r')


def sample(x):
    # return np.sin(x[0]) + np.cos(x[1])
    return -x[0] ** 2 - x[1] ** 2


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
    try:
        mu.append(float(item))
    except:
        pass

mu = np.array(mu)
x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)

grid = np.meshgrid(x, y)
original = sample(grid)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(grid[0], grid[1], mu.reshape(grid[0].shape), alpha=0.5, color='g', label='our estimation')
#ax.plot_wireframe(grid[0], grid[1], reference.reshape(grid[0].shape), alpha=0.5, color='b', label='py reference')
ax.plot_wireframe(grid[0], grid[1], original, alpha=0.5, color='Orange', label='real function')
#ax.plot_wireframe(grid[0], grid[1], mu.reshape(grid[0].shape) - reference.reshape(grid[0].shape), alpha=0.5, color='Orange', label='diff mu-reference')
plt.legend()

plt.savefig('fig.png')

# plt.clf()
#plt.imshow((original.flatten()-mu).reshape(24,24), cmap='hot', interpolation='nearest')
#plt.colorbar()
#plt.show()

print("np.linalg.norm(reference-mu) = %lf" % np.linalg.norm(mu - reference))
print("np.linalg.norm(original-mu)=%lf, np.linalg.norm(original-reference)=%lf" % (
    np.linalg.norm(original.flatten() - mu), np.linalg.norm(original.flatten() - reference)))
