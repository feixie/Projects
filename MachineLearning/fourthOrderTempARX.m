%Solves thermal coefficients for 4th order ARX
%based on: Jang thesis p106 (solution slightly altered)

%A(q)x4 = B1(q)Tout + B2(q)Qadj + B3(q)Iglobal(t) + B4(q)Qint
%A(q) = 1 + a1q^-1 + a2q^-2 + a3q^-3 + a4q^-4
%B1(q) = b11q^-1 + b12q^-2 +b13q^-3 + b14q^-4
%B2(q) = b22q^-2 + b23q^-3 + b24q^-4
%B4(q) = b41q^-1

%H(k) = [-x4(k-1)|-x4(k-2)|-x4(k-3)|-x4(k-4)|To(k-1)|To(k-2)|To(k-3)|...
%To(k-4)|Qadj(k-2)|Qadj(k-3)|Qadj(k-4)|Ig(k-2)|Ig(k-3)|Ig(k-4)|Qint(k-1)|]

%X(N) := [H(5),H(6),...,H(N)]
%Y(N) := [x4(5), x4(6),...,x4(N)]
%where x4 is indoor air temperature

%By Fei Xie

clear all; 
summerOrWinter = 1; %0 for Winter, 1 for Summer
start = 1;
stop = 10081;
N = stop-start;



%TRAIN:
if (summerOrWinter == 1)
        Tout = load('outtempNYSum.csv');
        Tair = load('intempNYSum.csv');
        Iglobal = load('solarNYSum.csv');
        Qadj = load('statusNYSum.csv');
        setpoint = load('setpointNYSum.csv');
    elseif (summerOrWinter == 0)
        Tout = load('outtempNYWin.csv');
        Tair = load('intempNYWin.csv');
        Iglobal = load('solarNYWin.csv');
        Qadj = load('statusNYWin.csv');
        setpoint = load('setpointNYWin.csv');
    
end

Tout = Tout(start:stop);
Tair = Tair(start:stop);
Iglobal = Iglobal(start:stop);
Qadj = Qadj(start:stop);
setpoint = setpoint(start:stop);

Qint = ones(1,N);
Y = Tair(5:N);
X = zeros(N-4,15);

%populate X:
for k=5:N
    H = [-Tair(k-1),-Tair(k-2),-Tair(k-3),-Tair(k-4),Tout(k-1),Tout(k-2), ...
        Tout(k-3),Tout(k-4),Qadj(k-2),Qadj(k-3),Qadj(k-4),Iglobal(k-2), ...
        Iglobal(k-3),Iglobal(k-4),Qint(k-1)];
    X(k-4,1:15) = H;
end

Coeffs = regress(Y,X);
a1 = Coeffs(1)
a2 = Coeffs(2)
a3 = Coeffs(3)
a4 = Coeffs(4)
b11 = Coeffs(5)
b12 = Coeffs(6)
b13 = Coeffs(7)
b14 = Coeffs(8)
b22 = Coeffs(9)
b23 = Coeffs(10)
b24 = Coeffs(11)
b32 = Coeffs(12)
b33 = Coeffs(13)
b34 = Coeffs(14)
b41 = Coeffs(15)


%GENERATE:
if (summerOrWinter == 1)
        Tair = load('intempNYSum.csv');
        Qadj = load('statusNYSum.csv');
    elseif (summerOrWinter == 0)      
        Tair = load('intempNYWin.csv');
        Qadj = load('statusNYWin.csv');
    
end

Tair = Tair(start:stop);
Qadj = Qadj(start:stop);

%initial indoor temperatures 
kMinus1 = Tair(4);
kMinus2 = Tair(3);
kMinus3 = Tair(2);
kMinus4 = Tair(1);

counter = 0;
CalcInTemp = zeros(1,N);
CalcInTemp(1:4) = [kMinus4, kMinus3, kMinus2, kMinus1];
for k=5:N
    
    %event based synchronization:
    if setpoint(k) ~= setpoint(k-1)
        counter = 4;
    end
    
    if counter <= 0
        
        x4 = -a1*kMinus1-a2*kMinus2-a3*kMinus3-a4*kMinus4+b11*Tout(k-1)+ ...
            b12*Tout(k-2)+b13*Tout(k-3)+b14*Tout(k-4)+b22*Qadj(k-2)+ ...
            b23*Qadj(k-3)+b24*Qadj(k-4)+b32*Iglobal(k-2)+b33*Iglobal(k-3)+...
            b34*Iglobal(k-4)+b41*Qint(k-1);
    else
        x4 = Tair(k);
        counter = counter-1;
        
    end
    CalcInTemp(k) = x4;
        
    
    kMinus1 = x4;
    kMinus2 = kMinus1;
    kMinus3 = kMinus2;
    kMinus4 = kMinus3;
end

figure, plot(Tout,'k'); hold on
xlabel('timestep');
ylabel('temperature');
plot(Tair, 'r');
plot(Qadj + 13, 'm');
plot(CalcInTemp, 'g');
plot(Iglobal/100, 'y');
plot(setpoint,'b');
legend('outTemp','inTemp','HVAC Energy','calcTemp', 'solar/1000', 'setpoint', 'Location','SouthEast');
hold off;





        

