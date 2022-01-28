import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound

# y[n] = 2x[n] + x[n-1] -x[n-3]
#create an inpulse input
xin=np.identity(10)
print(xin)
y1=2* xin[0,:] + xin[1,:] - xin[3,:]
print("part a) impulse response is: ",y1)

#the convolution of impulse to impulse response
h = [2,1,0,-1]
y2 = np.convolve(h,xin[0,:])
print("part b) the convlution result: ",y2)