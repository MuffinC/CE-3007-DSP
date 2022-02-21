B = [1,-0.7653668, 0.99999];
A = [1, -0.722744, 0.888622];
fvtool(B, A, 'Analysis','polezero')
z = roots(B);
p = roots(A);
results = abs(p);
h = impz(B,A);

% B = [1,-0.7653668, 0.99999],A = [1, -0.722744, 0.888622]
%fvtool(B,A)
%this is a bandstop filter
%make sure to use  y = db2mag(ydb) to get your mag in magnitude and phasor