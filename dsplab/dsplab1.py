import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound

#sound conversion
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))

def music(y,Fs,filename):
    #music play portion
    y_16bit = fnNormalizeFloatTo16Bit(y)
    y_float = fnNormalize16BitToFloat(y_16bit)
    wavfile.write(filename, Fs, y_16bit)
    wavfile.write(filename, Fs, y_float)  # wavfile write can save it as a 32 bit format float
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    os.remove(filename)

def lab3_1():
    #part a
    #y(t) = 0.1cos(2 *np.pi*F*T)

    Fs = 16000
    numSamples = 76
    F = 1000
    startt=0
    endt=startt+1/Fs*numSamples
    fai =0
    n = np.arange(startt, endt, 1.0/ Fs)  # because 1 second
    y = 0.1 *np.cos(2* np.pi *F *n + fai)
    plt.figure(1)
    plt.plot(n[0:numSamples], y[0:numSamples], 'r--o');
    plt.stem(n, y, 'g-');
    #    plt.plot(t, y,'ro');
    plt.xlabel('time in seconds');
    plt.ylabel('y(t)')
    plt.title('cosine of signal')
    plt.grid()
    plt.show()
    filename = "l1_31a.wav"
    music(y, Fs, filename)




    #part b



    #part c
lab3_1()