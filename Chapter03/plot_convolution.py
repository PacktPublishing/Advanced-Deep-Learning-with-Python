import matplotlib.pyplot as plt
import numpy as np


def plot_convolution(f, g):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.plot(f, color='blue', label='f')
    ax1.legend()

    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.plot(g, color='red', label='g')
    ax2.legend()

    filtered = np.convolve(f, g, "same") / sum(g)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.plot(filtered, color='green', label='f * g')
    ax3.legend()

    plt.show()


def plot_convolution_step_by_step(f, g):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.plot(f, color='blue', label='f')
    ax1.plot(np.roll(g, -10000), color='red', label='g')

    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.plot(f, color='blue', label='f')
    ax2.plot(np.roll(g, -5000), color='red', label='g')

    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.plot(f, color='blue', label='f')
    ax3.plot(g, color='red', label='g')

    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.plot(f, color='blue', label='f')
    ax4.plot(np.roll(g, 5000), color='red', label='g')

    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    ax5.plot(f, color='blue', label='f')
    ax5.plot(np.roll(g, 10000), color='red', label='g')

    plt.show()


signal = np.zeros(30000)
signal[10000:20000] = 1

kernel = np.zeros(30000)
kernel[10000:20000] = np.linspace(1, 0, 10000)

plot_convolution(signal, kernel)
plot_convolution_step_by_step(signal, kernel)
