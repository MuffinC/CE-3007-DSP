import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound

#sound conversion
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

def music(y,Fs,filename,x=0):
    #music play portion
    y_16bit = fnNormalizeFloatTo16Bit(y)
    wavfile.write(filename, Fs, y_16bit)  # wavfile write can save it as a 32 bit format float
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    os.remove(filename)

def aliasier(F,F_14,cycles=6):
    num_pts_percycle = 32
    x_1biv = np.arange(0,cycles/F,1/F/num_pts_percycle)
    y_1biv = np.cos(2*np.pi*F*x_1biv)
    y_alias = np.zeros_like(y_1biv) # create an empty array that is similar to y
    my_color = ["k" for _ in y_alias]
    for i in range(len(x_1biv)):
        if i%int(num_pts_percycle*F/F_14) == 0: #no remainder then store into the aliasing array
            y_alias[i] = np.cos(2*np.pi*F*x_1biv[i])
            my_color[i] = "r"
    plt.figure(figsize=(20,3))
    plt.vlines(x_1biv,0,y_alias,color=my_color)
    plt.scatter(x_1biv,y_alias,color=my_color)
    plt.plot(x_1biv,y_1biv)
    recon_x = [x_1biv[i] for i in range(len(my_color)) if my_color[i] == "r"]
    recon_y = [y_1biv[i] for i in range(len(my_color)) if my_color[i] == "r"]
    plt.plot(recon_x,recon_y)
    plt.show()


def lab3_1():
    #part a
    #y(t) = 0.1cos(2 *np.pi*F*T)

    Fs = 16000
    numSamples = 72
    F = 1000
    startt=0
    endt=startt+1/Fs*numSamples
    fai =0
    na1 = np.arange(startt, endt, 1.0/ Fs)  # because 1 second
    ya1 = 0.1 *np.cos(2* np.pi *F *na1 + fai)

    plt.figure(1)
    plt.stem(na1, ya1, 'g-');
    #    plt.plot(t, y,'ro');
    plt.xlabel('time in seconds');
    plt.ylabel('y(t)')
    plt.title('cosine of signal')
    plt.grid()
    plt.show()
    filename = "l1_31a.wav"
    music(ya1, Fs, filename)

    """
    #playing the multiple ranges
    endt=0.5
    for Fx in range(0,32001,2000):
        print("Now playing at F=",Fx)
        n = np.arange(startt, endt, 1.0 / Fs)
        y= 0.1 *np.cos(2* np.pi *Fx *n + fai)
        music(y, Fs, filename)

    #Explaination on what happened:
    obs: Up until 8khz it had an increasing frequency which is noted in the tone. Which then decreased until 16khz
    which then increase back up until 24khz and finally decreasing to 32khz
    Simply put, this is aliasing, which will occur when Fs<2F, since Fs=16khz.
    at F=8khz, Fs is 16khz which is when aliasing will start to occur because the sampLing rate is <2F
    this cause the illusion that the cosine wave is going slower or rather it is assumed that it is going 
    at a different frequency. The algorithm will just pick up the wave with the lowest frequencY as such from 
    8-16khz the pitch decreeases
    same thing occurs at 16khz and at 24khz as the pitch is mirrored
    """
    Fs = 16000
    F = 1000
    startt = 0
    cyclec = 6
    Points = int(Fs/F *cyclec)
    endt = startt + 1 / Fs * Points
    fai = 0
    na2 = np.arange(startt, endt, 1.0 / Fs)  # because 1 second
    ya2 = 0.1 * np.cos(2 * np.pi * F * na2 + fai)

    plt.figure(2)
    plt.subplot(311)
    plt.plot(na2, ya2)#y(T)

    plt.subplot(312)
    plt.plot(na2, ya2,'g-')
    plt.stem(na2, ya2,use_line_collection=True)#y[nT]


    na2_i= np.arange(0,len(na2),1)#put all x values into an array
    plt.subplot(313)
    plt.stem(na2_i, ya2)  # y(T)
    plt.show()


    """
    i) y(t) vs y[nt]:
    y(t) is a continuous time signal while  y[nt] is discrete time which is multiple sampled points of the signal
    Both functions take respect to time meaning that they will have the same x axis
    
    ii) y[n] vs y[nt] 
    y[n] is different as it is discrete time as such it wont share the same axis at y[nt] and the x axis is 
    in terms of samples instead of t,time.
    
    iii) To check if a signal is periodic, for y[n] there need to exist a period of N such that 
    y[n] = y[n+N]. This essentially means that after after N amount of samples the signal will repeat itself
    For this case lets let N = K, where K is an arbituary value.
    We now sub in the y[n] equation so:
    0.1 cos(2*pi*F/Fs*n) = 0.1 cos(2* pi* F/Fs* (n+K)) since we now that the cosine wave is periodice 
    every 2*pi, this would imply that 
    2*pi*F/Fs*n && 2* pi* F/Fs* (n+K) are both multiples of 2*pi
    thus we can see that as long as the result of F/Fs *n is an integer the signal will be periodic 
    Conversely, if it is not an integer it is aperiodic    
    since our question uses F=1000 and Fs =16000, as such 1/16 = 1/minK which results in minK being 16
    on the assumption that n =0. Therefore, every 16 samples the signal is periodic. 
    """
    #part c
    aliasier(17000, 16000)

#this is for part 3.2
class DTMF:
    #based on online search, unit is in hz
    freqarray = [697,770,852,941,1209,1336,1477,1633]
    tonedict = {
        '1': (freqarray[0],freqarray[4]),
        '2': (freqarray[0],freqarray[5]),
        '3': (freqarray[0],freqarray[6]),
        '4': (freqarray[1],freqarray[4]),
        '5': (freqarray[1],freqarray[5]),
        '6': (freqarray[1],freqarray[6]),
        '7': (freqarray[2],freqarray[4]),
        '8': (freqarray[2],freqarray[5]),
        '9': (freqarray[2],freqarray[6]),
        '*': (freqarray[3],freqarray[4]),
        '0': (freqarray[3],freqarray[5]),
        '#': (freqarray[3], freqarray[6]),
        'A': (freqarray[0],freqarray[7]),
        'B': (freqarray[1],freqarray[7]),
        'C': (freqarray[2], freqarray[7]),
        'D': (freqarray[3], freqarray[7]),
        'FO': (freqarray[0], freqarray[7]),
        'F': (freqarray[1], freqarray[7]),
        'I': (freqarray[2], freqarray[7]),
        'P': (freqarray[3], freqarray[7])
    }
    def GenSampledDTMF(seq,Fs,durTone):
        Amp= 0.5
        startt=0
        faiz =0

        #check if the sequence is part of the dict
        if seq in DTMF.tonedict:
            (key1,key2) = DTMF.tonedict.get(seq)
            #now to generate the t for the signal
            t = np.arange(startt, durTone,1/Fs)
            #DTMF is a summation of 2 signals
            y = (Amp *np.cos((2 *np.pi *key1 *t +faiz))) + np.cos(2 *np.pi *key2 *t +faiz)
            return t, y
        else:
            print("invalid key/sequence inputted")
    def GenSampledDTMFSEQ(seq,Fs,durTone):
        Amp= 0.5
        startt=0
        faiz =0
        #we need to break the string up
        seqarray = list(seq)
        for count in range(0,len(seqarray)):
            x3_2, y3_2=DTMF.GenSampledDTMF(seqarray[count],Fs,durTone)
            music(y3_2, 16000, '3_2.wav', x=x3_2)

def lab3_2():
    myDSPfn = DTMF()#create new DTMF object
    #testing 1 button instead of multiple
    """
    (x3_2,y3_2) = DTMF.GenSampledDTMF('1',16000,1.0)
    music(y3_2,16000,'3_2.wav',x=x3_2)
    """
    #now to run multiple in sequence
    DTMF.GenSampledDTMFSEQ('0123#',16000,1.0)

def lab3_3():
    #since there was no mention of A or B
    Fs =60
    A=2
    B=0.5
    cycles = 5
    pointspercyc = 20
    numpoints = pointspercyc * cycles
    startt = 0
    endt=startt+1/Fs*numpoints

    t= np.arange(0,endt,1/Fs)
    #in this case F = 10 and 15
    y1 = A* np.cos(2 * np.pi * 10 * t)
    y2 = B* np.cos(2 * np.pi * 15 * t)

    plt.figure(figsize=(20, 10))
    plt.subplot(311)
    plt.stem(t, y1,use_line_collection=True,basefmt="b")
    plt.subplot(312)
    plt.stem(t, y2,use_line_collection=True,basefmt="b")
    plt.subplot(313)
    plt.stem(t, y2+y1, use_line_collection=True, basefmt="b")
    plt.show()
    """
    3b) period for a cos graph is naturally 2pi, in this case for y1[n],y2[n],y3[n] it is:
    F/Fs = 10/60 =1/6
    1/6 *2pi = 1/3pi(in radian) = 120degress[y1[n]]
    
    F/Fs = 15/60 =1/4
    1/4 *2pi = 1/2pi(in radian) = 180degress[y2[n]]
    
    period if y3[n] is the lcm of the periods
    
    1/3 *1/2 =1/6pi (in radian)=60 degress[y3[n]]
    
    3c) y3[n] 's period is dependant on both y1[n] and y2[n]'s period as it is the lcm of the product
    3d)since it is a periodic signal, it is a power signal as a result(infinite energy)
    P1 = A**2/2 
    P2 = B**2/2
    P3 = 
    to be explained in real live
    
    
    """

def lab3_4():

    print('Below is figure 2 shows complex exponential')
    numSamples = 50
    A = 0.95
    w1 = 2 * np.pi / 36
    n = np.arange(0, numSamples, 1)
    y1 = np.multiply(np.power(A, n), np.exp(1j * w1 * n))

    for zeta in np.arange(0,min(10,numSamples)):
        print('%d, %.4f + j%.4f'%(n[zeta],y1[zeta].real,y1[zeta].imag))


    # plotting in 2-D, the real and imag in the same figure
    plt.figure(1)
    plt.plot(n, y1[0:numSamples].real, 'r--o')
    plt.plot(n, y1[0:numSamples].imag, 'g--o')
    plt.xlabel('sample index n');
    plt.ylabel('y[n]')
    plt.title('Complex exponential (red=real) (green=imag)')
    plt.grid()
    plt.show()
    #a way to show the above is through a table, refer to ss

    # 3.42 phasor portion
    # plotting in polar, understand what the spokes are

    plt.figure(3)
    for x in y1:
        plt.polar([0, np.angle(x)], [0, np.abs(x)], marker='o')

    plt.title('Polar plot showing phasors at n=0..N')
    plt.show()
#time is not shown in the 2d plot hence u create a 3d plot
#how many points to make 1 round? ans is 36,72=2 complete rounds
#if they say sth like A = 1, and ask you what is happening, the phasor does not decay,
#your real and imaginary does not decay, its like a non decreasing hoop

# To make the "circle" spin for part b, you just need to set the amp to like 0.98
#they may ask you if the figure is spining anti or cw, its anti always right to left
# you just need to see the curvature of the second plot
#if you want to make it go clock wise, just add a negative to the w, like w1 = -2 * np.pi / 36




#3.43 3d plot
    # plotting 3D complex plane
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    reVal = y1[0:numSamples].real
    imgVal = y1[0:numSamples].imag
    ax.plot(n, reVal, imgVal, label='complex exponential phasor')
    ax.scatter(n, reVal, imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imag')
    ax.legend()
    plt.show()

#3.44
    w2 = 2 * np.pi / 18
    n = np.arange(0, numSamples, 1)
    y1 = np.multiply(np.power(A, n), np.exp(1j * w2 * n))

    for zeta in np.arange(0, min(10, numSamples)):
        print('%d, %.4f + j%.4f' % (n[zeta], y1[zeta].real, y1[zeta].imag))

    # plotting in 2-D, the real and imag in the same figure
    plt.figure(4)
    plt.plot(n, y1[0:numSamples].real, 'r--o')
    plt.plot(n, y1[0:numSamples].imag, 'g--o')
    plt.xlabel('sample index n');
    plt.ylabel('y[n]')
    plt.title('Complex exponential (red=real) (green=imag)')
    plt.grid()
    plt.show()

    plt.figure(5)
    for x in y1:
        plt.polar([0, np.angle(x)], [0, np.abs(x)], marker='o')

    plt.title('Polar plot showing phasors at n=0..N')
    plt.show()

    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    reVal = y1[0:numSamples].real
    imgVal = y1[0:numSamples].imag
    ax.plot(n, reVal, imgVal, label='complex exponential phasor')
    ax.scatter(n, reVal, imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imag')
    ax.legend()
    plt.show()





#lab3_1()
#lab3_2()
#lab3_3()
lab3_4()