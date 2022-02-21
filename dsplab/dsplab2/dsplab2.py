import os
from scipy.signal import butter
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

    This question is about passing a sinuisodal wave through an LTI system given a characteristic impulse respones
    It is meant to demonstrate how sine waves are eigen function (modified amplitude and phaseshift but frequency is the same)
    It also shown how convolution with impulses signals are meant to look like.
    Lastly it investigate how assuming a sine wave as finite might not be good for our analysis.
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

def compare_two_spec(sig1,sig2,Fs=16000):

    [f, t, Sxx] = signal.spectrogram(sig1, Fs, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.figure(1,figsize=(30,10))
    plt.subplot(121)
    plt.pcolormesh(t, f, 10*np.log10(Sxx),cmap='plasma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spec1')

    [f2, t2, Sxx2] = signal.spectrogram(sig2, Fs, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.subplot(122)
    plt.pcolormesh(t2, f2, 10*np.log10(Sxx2),cmap='plasma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spec 2')

    plt.show()
def soundextractor(input_file):
    [Fs, sampleX_16bit] = wavfile.read(input_file)
    if len(sampleX_16bit.shape) == 2 :
        sampleX_16bit = sampleX_16bit[:,1]
    sampleX_float = np.array([float(s/32767.0) for s in sampleX_16bit],dtype='float')
    #sampleX_float = np.multiply(3.0,sampleX_float)
    return [Fs,sampleX_float]

def music(y,Fs = 16000,filename='t1_16bit.wav'):
    y_16bit = np.array([int(s * 32767) for s in y], dtype='int16')
    # Lets save the file, fname, sequence, and samplingrate needed
    wavfile.write(filename, Fs, y_16bit)
    # Lets play the wavefile using winsound given the wavefile saved above
    # unfortunately winsound ONLY likes u16 bit values
    # thats why we had to normalize y->y_norm (16 bits) integers to play using winsounds
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    # cleanup
    os.remove(filename)
def impulseresp_plot(impulseH):
    numSamples = len(impulseH)
    n = np.arange(0, numSamples, 1)
    num = [1, -0.7653668, 0.99999]
    den = [1, -0.722744, 0.888622]
    y = signal.lfilter(num, den, impulseH)
    plt.figure(2, figsize=(30, 10))
    plt.stem(n, y, use_line_collection=True)
    plt.ylabel('impulse response of the IIR filter')
    plt.xlabel('sample n')
    plt.show()

def plotsoundwave(input_file):
    [Fs, sampleX_float ]= soundextractor(input_file)
    plt.figure(figsize=(30,10))
    plt.subplot(121)
    plt.plot(sampleX_float,'r')
    plt.ylabel('signal (float)')
    plt.xlabel('sample n')
    [f, t, Sxx_clean] = signal.spectrogram(sampleX_float, Fs)
    plt.subplot(122)
    plt.pcolormesh(t, f, 10*np.log10(Sxx_clean),cmap='plasma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
    plt.show()

def cust_conv(x,y,mode='full'):
    """
    assume x bigger than y as array. Default to valid
    """
    result = []
    a, v = x,y
    if (len(v) > len(a)):    #swap the longer array to a
        a, v = v, a

    if mode == 'full':      #pad to a

        pad = np.array([0 for _ in range(len(v)-1)])
        b = np.copy(a)
        a = np.concatenate((pad,b,pad))
    for j in range(len(a)-len(v)+1):
        result.append((np.sum(np.matmul(v[::-1],a[j:j+len(v)]))))
    return np.array(result)

def dsplab2_3():
    input_file ="helloWorld_16bit.wav"
    #winsound.PlaySound(input_file, winsound.SND_FILENAME)
    plotsoundwave(input_file)

    #impulse response from manual
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3

    #sketch impulse response
    impulseresp_plot(impulseH)


    #testing convulation function
    test_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    test_h = np.array([0, 1, 2, 0, 0, 3, 4, 0])
    val_y = np.convolve(test_x, test_h)
    pred_y = cust_conv(test_x, test_h)
    print("NP CONVOLVE", val_y)
    print("CUSTOM FUNC", pred_y)

    #playing sound
    [Fs, sampleX_float] = soundextractor(input_file)
    y_n_rir = np.convolve(sampleX_float, impulseH)
    music(y_n_rir,Fs)
    '''
    Obs: there is a delay in the audio, creating a form of echolation.
    for part c i have no idea what they mean by cheaper or more expensive
    '''
def dsplab2_4():
    #part a impulse response
    h1 = [0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523]
    h2 = [-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523]
    impulseresp_plot(h1)
    impulseresp_plot(h2)

    #part b
    #n-15 is delaying time so u would plus in the arraywise
    n = [0 for _ in range(15)] + [1] + [0 for _ in range(14)] + [-2] + [0 for _ in range(14)]

    #using h1
    syste=np.convolve(n,h1,'full')
    custres=cust_conv(n,h1,'full')
    print(syste)
    print(custres)
    #using h2
    syste2=np.convolve(n,h2,'full')
    custres2=cust_conv(n,h2,'full')
    print(syste2)
    print(custres2)

    '''
    Observation: no idea H(n) is just a delayed response
    '''

    F1 = 700
    F2 = 3333
    A = 0.1
    Fs = 16000
    startt=0
    endt=1
    faiz = 0
    n = np.arange(startt, endt, 1.0 / Fs)
    y1 = A * np.cos(2 * np.pi * F1 * n + faiz)
    y2 = A * np.cos(2 * np.pi * F2 * n + faiz)
    y3 = y1+y2

    Y_h1 = np.convolve(h1, y3)
    Y_h2 = np.convolve(h2, y3)
    Y_h3 = np.convolve(np.convolve(h1, h2), y3)

    # compare_two_spec(sinful,Y_h3)
    compare_two_spec(Y_h1, Y_h2)
    compare_two_spec(np.convolve(h2, Y_h1), Y_h3)
    '''
    Spectrogram plotted
    '''

    np.convolve(h1,h2)

    plt.figure(figsize=(30, 10))
    n_points = 200
    zoom = np.arange(0, 0 + n_points / Fs, 1 / Fs)
    plt.subplot(121)
    plt.plot(zoom, Y_h1[0:n_points], 'r')
    plt.ylabel('signal (float)')
    plt.xlabel('H1')

    plt.subplot(122)
    plt.plot(zoom, Y_h2[0:n_points], 'r')
    plt.ylabel('signal (float)')
    plt.xlabel('H2')
    plt.show()
    '''
    Output generated
    4.3: its a filter
    '''
def notchfiler(y,f0,Fs=16000):
    A = np.array([1, -0.722744, 0.888622])
    B = np.array([1, - 0.7653668, 0.99999])
    y_clean = signal.lfilter(B, A, y)
    return y_clean

def bandpass(fs,lowcut,highcut,order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a

def notchfilter2(y,f0,Fs=16000):
    # Design notch filter using Dr CES coefficients
    b,a = bandpass(Fs,3000*0.99,3000*1.01,6)
    y_clean = signal.lfilter(b,a, y)
    return y_clean
def dsplab2_5():
    #5a
    filename ="helloworld_noisy_16bit.wav"

    '''
    if u fvtool in mat lab u will get a band stop, it looke like a small dip
    in the frequency basicallly not allowing it to pass through.
    stops at 0.375 F
    
    '''
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    plotsoundwave(filename)

    #5b
    #df1 representation is through paper, u just need diff equation

    #5c
    #need filter
    [Fs, y_noisy] = soundextractor(filename)
    y_clean = notchfiler(y_noisy, 3000, Fs=16000)
    compare_two_spec(y_noisy, y_clean)
    music(y_noisy)
    music(y_clean)

    #5d
    y_unclean = notchfilter2(y_noisy, 3000, Fs=16000)
    compare_two_spec(y_noisy, y_unclean)
    music(y_unclean)




#dsplab2_1()
dsplab2_3()
#dsplab2_4()
#dsplab2_5()
'''
Q2 
The impulse response of an LTI can be obtained from the linear constant-coefficient difference equation
in the case the particular solution yp=0, since for x(n) = 0 for n>0 for an impulse function. 
The same can be done via convolution of impulse with the transfer function of the LTI.

An example is a FIR filter -- raised cosine filter, 
'''
