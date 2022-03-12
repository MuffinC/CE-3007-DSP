import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def myDTFS(x,N):
    X = np.zeros(len(x), dtype=complex)
    Omega = np.zeros(len(x))

    for k in np.arange(0,len(x)):
        tmpVal = 0.0
        Omega[k] = (2*np.pi/N)*k
        for n in np.arange(0,len(x)):
            tmpVal = tmpVal + x[n]*np.exp(-1j*(2*np.pi/N)*k*n)
        X[k] = tmpVal/N
    return (X,Omega)



def myIDTFS(X,N):
    x = np.zeros(len(X), dtype=float)
    for n in np.arange(0,len(x)):
        tmpVal = 0.0
        for k in np.arange(0,len(X)):
            tmpVal = tmpVal + X[k]*np.exp(+1j*(2*np.pi/N)*k*n)
        x[n] = np.absolute(tmpVal)
    return (x)

def myDFT(x,N):
    X = np.zeros(N, dtype=complex)
    Omega = np.zeros(N)

    for k in np.arange(0,len(x)):
        tmpVal = 0.0
        Omega[k] = (2*np.pi/N)*k
        for n in np.arange(0,len(x)):
            tmpVal = tmpVal + x[n]*np.exp(-1j*(2*np.pi/N)*k*n)
        X[k] = tmpVal
    return (X,Omega)



def myIDFT(X,N):
    x = np.zeros(len(X), dtype=float)
    for n in np.arange(0,len(x)):
        tmpVal = 0.0
        for k in np.arange(0,len(X)):
            tmpVal = tmpVal + X[k]*np.exp(+1j*(2*np.pi/N)*k*n)
        x[n] = np.absolute(tmpVal)/N
    return (x)


def plotMagPhase(x, rad,figure_name):
    f, axarr = plt.subplots(2, sharex=True)
    x = np.absolute(x)
    axarr[0].stem(np.arange(0, len(x)), x)
    axarr[0].set_ylabel('mag value')
    axarr[1].stem(np.arange(0, len(rad)), rad)
    axarr[1].set_ylabel('Phase (rad)')
    f.suptitle(figure_name, fontsize=16)
    plt.show()


def plotK(W):
    f, axarr = plt.subplots()
    phaseW = np.angle(W)
    print(phaseW)
    axarr.stem(np.arange(0, len(phaseW)), phaseW)
    plt.show()
    print("\n\n\n")


def plotDTFSDTFTMag(X):
    N = len(X)
    x = np.absolute(X)
    f, axarr = plt.subplots(figsize=(18, 2.5))
    axarr.stem(np.arange(0, N), x)
    axarr.set_ylabel('DTFS mag value')
    plt.show()
    x = [element * N for element in x]
    f, axarr = plt.subplots(figsize=(18, 2.5))
    axarr.stem(np.arange(0, N), x)
    axarr.set_ylabel('DTFT mag value')
    ticks = range(N)
    ticks = [round(element * 2 / N, 2) for element in ticks]
    #     ticks = [round(element * 2 *np.pi/N,2) for element in ticks]
    plt.xticks(np.arange(0, N), ticks)
    plt.xlabel('w*pi (rad/sample) ')
    plt.show()

    print("\n\n\n")


def myMatrixDFT(x,N):
    X = np.zeros(len(x), dtype=complex)
    Omega = np.zeros(N)
    rows, cols = (N, N)
    W = [[0 for i in range(cols)] for j in range(rows)]
    for k in range(N):
        for n in range(N):
            W[k][n] = round(np.exp(-1j * 2 * np.pi * k * n / N), 3)
    X = np.matmul(W, x)

    for i in range(len(W)):
        print("k = ", str(i))
        plotK(W[i])

    for k in np.arange(0, len(x)):
        Omega[k] = (2 * np.pi / N) * k

    return (X, Omega)


def myDFTConvolve(ipX, impulseH):
    L = len(ipX) + len(impulseH) - 1
    X1 = np.zeros(L)
    X2 = np.zeros(L)
    X1[0:len(ipX)] = ipX
    X2[0:len(impulseH)] = impulseH

    X1 = np.fft.fft(X1)
    X2 = np.fft.fft(X2)

    X3 = X1 * X2

    X3 = np.fft.ifft(X3)
    return np.round(X3, 5)

def lab3_2_1():
    x=[1,1,0,0,0,0,0,0,0,0,0,0]
    N = len(x)
    (X1, W1) = myDTFS(x,N)
    (X2, W2) = myDFT(x,N)
    Xf1 = np.fft.fft(x)
    Xang1 = np.angle(Xf1)
    print("DTFS")
    plotMagPhase(X1, Xang1,"DTFS")


    print("DFT")
    plotMagPhase(X2, Xang1,"DFT")
    arr = []
    for i in range(len(W1)):
        print("K =", str(i), "w =", W1[i])


    #Q2c part IDTFS willl return same value as IDFT
    N1=len(X1)
    N2 = len(X2)
    x1 = myIDTFS(X1,N1)
    x2 = myIDFT(X2,N2)
    print("\n")
    print(np.round(x1, 5))
    print(np.round(x2, 5))


    #part 2d
    ipX2 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ipX3 = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    N2 = len(ipX2)
    N3 = len(ipX3)
    (X2, W2) = myDTFS(ipX2, N2)
    (X3, W3) = myDTFS(ipX3, N3)

    #this is just to fast fourier transform the matrix
    Xf2 = np.fft.fft(ipX2)
    Xf3 = np.fft.fft(ipX3)

    #to find the phasor value of the ft
    Xang2 = np.angle(Xf2)
    Xang3 = np.angle(Xf3)
    print("X2")
    plotMagPhase(X2, Xang2,"ipX2")
    # In X2, Amplitude is same, with phase difference
    print("X3")
    plotMagPhase(X3, Xang3,"ipX3")
    # In X3, Phase is same, with amplitude scaled

def lab3_3_1():
    #dft forward analysis
    x = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X4, W4 = myMatrixDFT(x,len(x))
    Xf4 = np.fft.fft(x)
    Xang4 = np.angle(Xf4)
    #^ this will generate the phasor for the multiple k values

    plotMagPhase(X4, Xang4,"Q3 Analysis")
"""
Each row is 2*pi*k divided by N; for k = 1 to 5, The phase goes around k times faster compared to k=1.
for each k value so 1, 2, 3, 4, 5,
the value in which the phasors differ when they add increases by 0.5 each time

k =  1( it will go 1*0.5)
[ 0.         -0.52361148 -1.04718485]

k =  2( it will go 2*0.5)
[ 0.         -1.04718485 -2.0944078]

k =  3( it will go 3*0.5)
[ 0.         -1.57079633 -3.14159265]

k =  4 ( it will go 4*0.5)
[ 0.        -2.0944078  2.0944078]

k =  5
[ 0.         -2.61798118  1.04718485]

k = 6 is something like the peak
 
For k=7 to 11, they are conjugate of k = 5 to 1 (i.e. 1 = conjugate(11), 2 = conjugate(10) etc),
they are the flipped version from 5to 1

k =  7
[ 0.          2.61798118 -1.04718485]

k =  8
[ 0.         2.0944078 -2.0944078 ]
hence the phase is flipped
"""

def lab3_4_1():

    # q4a: Because the result of DTFT is continuous where N approaches infinity. We can't have an infinite-sized list
    #becuase it is aperiodic which means it approaches infinity
    NList = [12, 24, 48, 96]
    for N in NList:
        print("N =", str(N))
        arr = np.zeros(N)
        arr[0:7] = 1
        (X5, W5) = myDTFS(arr,len(arr))
        #         Xf5 = np.fft.fft(arr)
        plotDTFSDTFTMag(X5)
#       DTFS = DTFT/N for an aperiodic signal

def lab3_5_1():
    x = [1,1,1,0,0,0,0,0,0]
    h = [1,1,0,0]
    print("myDFTConvolve:")
    y = myDFTConvolve(x,h)
    print(y)
    print("Scipy fftconvolve:")
    print(np.round(signal.fftconvolve(x,h),5))


#lab3_2_1()
#lab3_3_1()
#lab3_4_1()
lab3_5_1()