import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import ndimage


def flops(n, I=20, k=5, f=0):
    # n = meshgrid
    # I = total number of iterations
    # k = kernel cost
    # f = function cost
    return I*2*n**2+I*f+I*(I+1)*(2*I+1)/6.0*k+I**2*(I+1)**2/4.0 + I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0+n**2*I*(I+1)/2.0*k+n**2*I*(I+1)+n**2/2.0*(I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0)+n**2*I*(I+1)+n**2*k

def my_legend(axis = None):

    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    print(Nlines)

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0
    ws[N-1,:] = 0

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        #plt.figure()
        #plt.imshow(p, interpolation='nearest')
        #plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first
        best_x, best_y =  (pos / N, pos % N)
        x = xmin + (xmax-xmin) * best_x / N
        y = ymin + (ymax-ymin) * best_y / N


        axis.text(x, y, axis.lines[l].get_label(),
                  horizontalalignment='center',
                  verticalalignment='center')

def plot(taglist, title="", filename="plot.pdf"):
    for tag in taglist:
        data = np.genfromtxt(tag+'.csv', delimiter='\t', skip_header=True)
        N = data[:,0]
        I = data[:,1]
        cycles = data[:,2]

        flops_per_cycle = np.array([flops(n, I=i) for (n, i) in zip(N, I)]) / cycles
        xaxis = N if tag[-1] is 'N' else I
        labelsuffix = "(I="+str(int(I[0]))+")" if tag[-1] is 'N' else "(I="+str(int(N[0]))+")"
        plt.plot(xaxis, flops_per_cycle, label=tag[:-2]+labelsuffix)

    plt.title(title)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    my_legend()
    plt.xlabel(tag[-1], rotation=0, fontsize=12)
    plt.ylabel("Perf [f/c]", rotation=0, fontsize=12)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_label_coords(+0.07, 1.005)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

plot(["baseline_N", "baseline-comp-chol_N"], filename="baseline.pdf")
