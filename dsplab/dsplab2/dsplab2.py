import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import scipy
import winsound
from scipy import signal
def dsplab2_1():
    """
    part a:
    With referrence to the image, we can break a signal down
    to a series of scaled and time shifted impulses, and deduce the
    the output response summing the response of each of the impulses

    part b:
    Amp and phase changes but frequency remains the same
    Eigen function makes it such that the output is a scaled version
    of the input
    """
    cyclec= 5
    h = np.array([0.2,0.3,-0.5])
    n =np.arange(0,3,1)
    cycle =5
    amp = 1
    f =0.1 #* cycle
    x =amp * np.cos(f*np.pi*n)
    y = np.convolve(x, h,"full")
    plt.figure(1, figsize=(30, 20))
    plt.subplot(211)
    plt.stem(n, x, use_line_collection=True, basefmt="b", linefmt='y')
    plt.subplot(212)
    plt.stem(n, y[0:3], use_line_collection=True, basefmt="b", linefmt='r')
    plt.show()



dsplab2_1()
