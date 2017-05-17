# coding: utf-8
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import sys


class GPUCB(object):
    def __init__(self, meshgrid, environment, beta=100.):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for 2D space
        with |x_i| < 1 constraint.
        environment: Environment class which is equipped with sample() method to
        return observed value.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution solution (i.e. larger curiosity)
        '''
        self.meshgrid = np.array(meshgrid)
        # print(self.meshgrid.shape)

        self.environment = environment
        self.beta = beta

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []

    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def max_point(self):
        maxGrididx = np.argmax(self.mu)
        x, y = self.X_grid[maxGrididx]
        maxMu = np.amax(self.mu)
        print("\nMaximal point found by python code is %lf at [%lf %lf]\n" % (maxMu, x, y))



    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])
        # gp = GaussianProcessRegressor()
        gp = GaussianProcessRegressor(random_state=0, kernel=RBF(length_scale=[1, 1], length_scale_bounds=(1e-5, 1e-5)),
                                      optimizer=None)
        gp.fit(self.X, self.T)
        #  prevMu = np.copy(self.mu)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        # print(self.mu)
        #  print(self.mu - prevMu)

    def sample(self, x, printSampled=True):
        t = self.environment.sample(x)
        if printSampled:
            print("(python) Sampled: [%lf %lf], result: %lf" % (x[0], x[1], t))
        self.X.append(x)
        self.T.append(t)

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.environment.sample(self.meshgrid), alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.T, c='r',
                   marker='o', alpha=1.0)
        plt.savefig('fig_%02d.png' % len(self.X))


if __name__ == '__main__':
    class DummyEnvironment(object):
        def sample(self, x):
            # return np.sin(x[0]) + np.cos(x[1])
            return -x[0] ** 2 - x[1] ** 2


    x = np.arange(-3, 3, 0.25)
    # x = np.arange(0, 3, 1)
    # y = np.arange(0, 3, 1)
    # print(x)
    y = np.arange(-3, 3, 0.25)
    # print(np.meshgrid(x,y))
    env = DummyEnvironment()
    agent = GPUCB(np.meshgrid(x, y), env)
    nIter = int(sys.argv[1])
    for i in range(nIter):
        agent.learn()
        # agent.plot()
    agent.max_point()
    np.savetxt("reference_output.txt", agent.mu.reshape(agent.meshgrid[0].shape), fmt='%.5f', delimiter=',')
    np.savetxt("reference_variance.txt", agent.sigma.reshape(agent.meshgrid[0].shape), fmt='%.5f', delimiter=',')
    np.savetxt("sampler_output.txt", agent.environment.sample(agent.meshgrid), fmt='%   .5f', delimiter=',')
