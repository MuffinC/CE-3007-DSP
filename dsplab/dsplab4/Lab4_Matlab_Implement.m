% Lab 4:
% generate a signal with sampling rate of 1KHz and also 100Hz.
% we then wish to reconstruct the signal sampled at 1KHz from 100Hz.
% signal reconstruction, upsampling
% The idea is to send a interpolated signal through a low pass filter!
% answer the following questions

close all; clear all;

Duration = 1;
%Wo = omg/Fs= 2 pi fo = 2 pi 2
F1000 = 1000;%Fs
T1000 = 1/F1000;%T
n=1:F1000*Duration;

for n=1:F1000*Duration
  y1000(n) = 2*sin(2*pi*2*(n-1)*T1000+(pi/4));  %Wo= 2 pi 2 /Fs, y(t) values at f=1000hz
  t1000(n) = (n-1)*T1000;
end
%  the signal frequency for y(t) is 2Hz evident in the equation

F100 = 100; T100 = 1/F100;
for (n=1:F100*Duration)
  y100(n) = 2*sin(2*pi*2*(n-1)*T100+(pi/4));  
  t100(n) = (n-1)*T100;
end
%=======================

figure(1);
plot(t1000, y1000); title('Signal of 1000Hz'); hold on;

fprintf('Press any key to continue ...'); pause;

stem(t100,y100);
title('Signal of 1000Hz & 100Hz');

fprintf('Press any key to continue ...'); pause;

figure(2)
rsig = upsample(y100,10); %upscale by 10
plot(t1000(1:length(rsig)), rsig,'r+'); title('Upsampling of 100Hz Signal');hold;

pause;
%red colour is the output

% Upsample, apply a lowpass FIR filter to rsig.

%Wn = 0.1;   % what is the correct value of Wn? (choose between 0..1)
%EXPLANATION:  L = upscale value
% Cutoff = Pi/L = Pi/10 = 0.1Pi
%               Frequency after upsampling = 1000Hz
%               2Pi => 1000Hz
%               0.1Pi => 50Hz; thus cut-off Frequency is 50Hz
%               Wn is in unit of Pi, thus Wn = 0.1

% This is the cutoff frequency of the interpolation filter                    
Freq_Signal = 2; %Hz  
Freq_Fn = 50;
Wn = Freq_Fn/F1000*2; % choose 50Hz as the cutoff (as the origianl signal is 2Hz only
                 % Wn = 0-1. where 1 corresponding to PI
%=======================
%This part will be taught in the coming weeks
%=======================
N=61; %choose N arbitrarily, which should be determined by Kaiser estimation                       
h = fir1(N,Wn);    % how does filter length affect the output?
%longer filter is more accurate (higher order), but also introduce
%significant phase shift (as it needs to wait for more non-zero points to
%convolve

h = F1000/F100*h;           %%% Question: why do we multiply with this constant?
%EXPLANATION:   When low-pass filter is performed, the nett power of the frequency spectrum is
%               reduced by h-fold. To make up for this loss of power, we introduce a gain of h
%               to the remaining signal, so that the effective power is the same as the signal
%               prior to upsampling. 
%=======================

rsig2 = filter(h,1,rsig);  % what am I doing to get rsig2?
%EXPLANATION:   Passing rsig through the low-pass filter with a gain of h
plot(t1000,rsig2,'g');         % rsig2 is plotted in green color
title('Green waveform: 1000Hz Signal after upsampling from 100Hz')