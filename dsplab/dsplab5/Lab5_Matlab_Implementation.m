% design a low pass filter
close all;
clear all;

% Question: Q5

% Designing the FIR filter directly by inverse Fourier Transform!
N = 31;
w_c = 2*pi*2.5/10; %2*pi*f/Fs???;   
% How to calculate the normalized cut-off frequeny given the sampling rate at 10Khz and the stop-band at 2.5Khz?
% Fs = 10 000hz,Fstopband = 2500hz
% Wcut = 2*pi*Fs/Fstop =0.5 pi hz
% then normalize over pi will give you 0.5hz
n_range = -15:1:15;
midX = length(n_range)/2;
for (n=1:length(n_range))
  h(n) = sin(w_c*n_range(n))/(pi*n_range(n)); %??? For non-sero n, sinc function
end

h(1+((length(n_range)-1)/2)) = w_c/pi;
%^this is to get h(0)
plot(h,'+-');

% Question: Q5(1)
h2 = h./sum(h); %???   % h2 is a normalized version of h
fvtool(h2,1); % examine using magnitude response at y-axis (not dB or magnitude^2)?
%specified in linear scale, can always be converted to mag from db

B = fir1(30,0.5, ones(31,1)','noscale')
freqz(B,1,512)%frequency response
    
% Question: Q5 (2)
% Lets study the performance of the designed filter,
% does it satisfy criteria? passband ripple < 0.1, stopband attenuation <
% 0.01?
fvtool(h,1);

%Passband ripple 
%We don't trust our eyes
meow = freqz(h,1,N); %This is the frequency response
cutoff_index = 15  %trial and error to get this to be the end of the ripple
ripple = max(abs(meow(1:cutoff_index)))-min(abs(meow(1:cutoff_index)))
% passband ripple is 0.1011 in this case, <0.02 pass
stopband_attn = max(abs(meow(18:length(meow))))  %try and error to get the first stopband
% In this case is 0.06 so yea cannot, cause requirement is 0.01 therefore
% fail

% Question: Q5 (3) Hamming window setting
% Using Matab's function fir1 to design, fill in the ???
%B_ham = fir1(???,???, hamming(???),'noscale')
B_ham = fir1(N-1,w_c/pi, hamming(N),'noscale');
fvtool(B_ham,1);

 