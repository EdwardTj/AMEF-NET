
#relu激活函数

import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style

def xyplot(x_vals, y_vals, name):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = (5, 3.5)
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy(), label = name, linewidth=2.0, )
    # plt.grid(True,linestyle=':')
    plt.legend(loc='upper left')
    #dark_background, seaborn, ggplot
    plt.style.use("seaborn")
    # ax = plt.gca()
    # ax.spines['right'].set_color("none")
    # ax.spines['top'].set_color("none")
    # ax.spines['bottom'].set_position(("data",0))
    # ax.spines['left'].set_position(("data",0))
    # ax.spines['bottom'].set_linewidth(0.5)
    # ax.spines['left'].set_linewidth(0.5)
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # plt.ylim((0.0,1.0))
    plt.show()
def main():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = x.relu()
    xyplot(x, y, 'Relu')

if __name__ == '__main__':
    main()

