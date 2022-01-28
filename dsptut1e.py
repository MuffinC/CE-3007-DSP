import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound

fig,ax = plt.subplots(1,1)
ohm = w = 1.8* np.pi
ax.set_title("Q1E Solution")
#ct
t = np.arange(-10,10,1/1000)
y1 = np.sin(t * ohm)
ax.plot(t,y1,'g')


#discrete
n = np.arange(-10,10,1)
y2 = np.sin(n * w )
ax.plot(n,y2,'r')
ax.grid()
fig.show()