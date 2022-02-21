close all;
clear all;

n = 0:8000;Fs= 8000;
w1 = 0.05*pi, w2 = 0.375*pi % u tweak the values beside the pi to change the signal
x1 = 3*cos(w1.*n); %0.05 signal
x2=1*cos(w2.*n);
xSum = x1+x2; %red = blue + green

Nsample = 130
figure(1);legend;
plot(x1(1:Nsample),'g','DisplayName','input x1'); hold on;
plot(x2(1:Nsample),'b','DisplayName','input x2');
plot(xSum(1:Nsample),'r-','DisplayName','xSum');legend;
disp('Figure 1 == showing the input signal');
pause

B = [1,-0.7653668, 0.99999],A = [1, -0.722744, 0.888622]
y = filter(B,A,xSum)
figure(3);hold on; plot(y(1:Nsample),'r','DisplayName','output y');hold on;
plot(x1(1:Nsample),'g','DisplayName','input x1');
plot(x2(1:Nsample),'b','DisplayName','input x2');
plot(xSum(1:Nsample),'r--','DisplayName','xSum'); legend;
disp('Figyew 3 showing the filtered signal vs the input signal');
pause

figure(4)
spectrogram(y,hamming(256),128,Fs,'yaxis');
title('spectrogram of filtered signal, we now observe only 1 line');
disp('Figure 4 spectrogram of the filtered signal');


