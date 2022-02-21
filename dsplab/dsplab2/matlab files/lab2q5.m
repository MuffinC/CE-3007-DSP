close all;
clear all;

Fs = 8000;
t = 0.0:1/Fs:2 %gets 2 seconds of the signal
x = chirp(t,200,1,1000,'q'); % start at 100Hz cross 200 Hz at t=1 ec
%when u change the 1k, to whatever value the chirp signal is also affected
% because a shirp is an increasing sin wave
figure(1)
plot(x(1:1000));
sound(x,Fs)
figure(2);
spectrogram(x,hamming(256),128,Fs,'yaxis');
title('spectrogram of chirp');
B = [1,-0.7653668, 0.99999];
A = [1, -0.722744, 0.888622];