import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline


if __name__ == '__main__':
    x1 = [2.00, 4.044, 5.701, 7.467, 10.518, 13.449, 16.223, 30.063, 56.526]
    y1 = [12.38, 11.8,  10.8,  10.18,  9.06,  8.18,   7.84,   5.96,   4.54]
    x2 = [2.00, 3.938, 5.959, 7.741, 11.964, 16.130, 18.479, 39.695, 56.526]
    y2 = [7.2, 11.3, 11.2, 10.88, 11.12, 10.88, 10.54, 11.2, 10.98]
    x34 = [2.00, 56.526]
    y3 = [9.02, 9.02]
    y4 = [4.4, 4.4]

    x_min = 2.0
    x_max = 56.526
    x1_smooth = np.linspace(x_min, x_max, 200)
    y1_smooth = spline(x1, y1, x1_smooth)
    x2_smooth = np.linspace(x_min, x_max, 200)
    y2_smooth = spline(x2, y2, x2_smooth)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, y1, linewidth=2.0, label='S0, A only, 2 antennas')
    ax.plot(x2, y2, linewidth=2.0, label='S0, A <-> B, 2 antennas')
    ax.plot(x34, y4, '.--', label='S0, A only, 1 antenna', linewidth=2.0,
            color='purple')
    ax.plot(x34, y3, '.--', label='S0, A <-> B, 1 antenna', linewidth=2.0,
            color='orange')
    ax.legend()
    ax.set_xlabel('antenna switch interval')
    ax.set_ylabel('inventory rounds number')
    ax.set_title("Number of rounds per tag")
    ax.tick_params(labelsize=17)
    ax.set_ylim(2, 17)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max, 10.0))
    ax.grid()
    plt.savefig("/home/andrey/Workspace/Lab69/article-ieee-rfid2017/paper/"
                "img/round_duration_vs_antennas.eps")
    plt.savefig("/home/andrey/Workspace/Lab69/article-ieee-rfid2017/paper/"
                "img/round_duration_vs_antennas.png")
    plt.show()
