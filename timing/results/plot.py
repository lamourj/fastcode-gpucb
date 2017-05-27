import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
from scipy import ndimage

def flops_(n, I=20, k=20, f=0):
    # n = meshgrid
    # I = total number of iterations
    # k = kernel cost
    # f = function cost
    return I*2*n**2+I*f+I*(I+1)*(2*I+1)/6.0*k+I**2*(I+1)**2/4.0 + I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0+n**2*I*(I+1)/2.0*k+n**2*I*(I+1)+n**2/2.0*(I*(I+1)*(2*I+1)/6.0+I*(I+1)/2.0)+n**2*I*(I+1)+n**2*k

def flops(n, I, k=40):
    return 64*4+64*k


def plot(taglist, title="GPUCB (single precision) on Core i7(Skylake)@3.4 GHz", filename="plot.pdf"):

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    xaxis = 0
    ymax =0.5
    xmax = 0

    for rank, tag in enumerate(taglist):
        data = np.genfromtxt(tag+'.csv', delimiter='\t', skip_header=True)
        N = data[:,0]
        I = data[:,1]
        cycles = data[:,2]

        flops_per_cycle = np.array([flops(n, I=i) for (n, i) in zip(N, I)]) / cycles

        ypos = flops_per_cycle[-1]
        xaxis = N if tag[-1] is 'N' else I
        xmax = max(xmax,np.amax(xaxis))
        ymax = max(ymax,np.amax(flops_per_cycle))

        labelsuffix = "(I="+str(int(I[0]))+")" if tag[-1] is 'N' else "(I="+str(int(N[0]))+")"
        plt.plot(xaxis, flops_per_cycle, label=tag[:-2]+labelsuffix, linewidth=2, marker='o', color=tableau20[rank*2], markeredgecolor='none')
        plt.text(xaxis[-1]+10, ypos, tag[:-2], fontsize=12, color=tableau20[rank*2])

    ymax = math.ceil(ymax)
    print("ymax: ",ymax)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    for y in range(0, ymax*10*15, 10**math.floor(math.log10(ymax))):
        plt.plot(range(0, np.amax(xaxis).astype(int)), [y] * len(range(0, np.amax(xaxis).astype(int))), "--", lw=0.5, color="black", alpha=0.3)
    print("lines plotted")

    font = {#'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 12,
            }
    plt.text(xmax+10, 4, 'scalar peak', fontsize=12, color='black')
    plt.axhline(y=4,xmax=0.67, color='black', linestyle='--')
    #plt.legend(loc=0, borderaxespad=0.5, frameon=False)

    plt.xlabel(tag[-1], rotation=0, fontdict=font)
    plt.ylabel("Perf [f/c]", rotation=0, fontdict=font)
    plt.title(title, fontdict=font, fontsize=17, weight='bold')
    plt.ylim(bottom=0, top=1.5*ymax)
    plt.xlim(right=xmax*1.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_label_coords(+0.04, 1.005)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

#plot(["baseline_I", "baseline_I"], filename="baseline.pdf")
plot(["triangle_solve_vec2_I", "triangle_solve_I"], filename="triangle_solve.png")
