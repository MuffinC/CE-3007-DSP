close all;
clear all;

X = zeros(100,1);
X(1) = 1;
X(40) = -0.5;
impuXlseH(60) = 0.3;
n = 0:length(X)-1;
figure;
subplot(3,1,1);
stem(n,X,'DisplayName','impulseH');
legend

h = zeros(1,length(n));
h(1:10) = [1:10];
subplot(3,1,2)
stem(n,h,'DisplayName','impulse h');legend

y=conv(X,h)
hold on;
subplot(3,1,3)
stem(n,y(1:length(n)),'r','DisplayName','y=x*h');legend