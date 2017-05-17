import numpy as np
import matplotlib.pyplot as plt

def flops(n, I=20, k=5, f=0):
	# n = meshgrid
	# I = total number of iterations
	# k = kernel cost
	# f = function cost
	return I*2*n**2+I*f+I*(I+1)*(2*I+1)/6.0*k+I**2*(I+1)**2/4.0 + I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0+n**2*I*(I+1)/2.0*k+n**2*I*(I+1)+n**2/2.0*(I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0)+n**2*I*(I+1)+n**2*k

data = np.loadtxt('runtimes.txt', delimiter=',', skiprows=1)
n = data[:,0]
nIterations = data[:,1][0]
seconds_per_it = data[:,2]
cycles_per_it = data[:,3]

flops_per_cycle = np.array([flops(v, I=nIterations) for v in n]) / cycles_per_it

plt.plot(n, flops_per_cycle)
plt.title("Haswell, %d" % nIterations)
plt.legend()
plt.xlabel('n')
plt.ylabel('Perf [f/c]')
plt.ylim(0,0.2)
plt.show()
