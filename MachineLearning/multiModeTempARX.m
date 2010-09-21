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
summerOrWinter = 0; %0 for Winter, 1 for Summer
hrIntervals = 3; %generate at 3hr intervals 
num_models = 7; %three days of data is 4320 minutes
hr_prior = 3;
min_prior = hr_prior*60;
daymin = 24*60;


%Load two consecutive weeks of data:
if (summerOrWinter == 1)
        outtemp = load('outtempNYSum.csv');
        intemp = load('intempNYSum.csv');
        solar = load('solarNYSum.csv');
        status = load('statusNYSum.csv');
        setpoint = load('setpointNYSum.csv');
    elseif (summerOrWinter == 0)
        outtemp = load('outtempNYWin.csv');
        intemp = load('intempNYWin.csv');
        solar = load('solarNYWin.csv');
        status = load('statusNYWin.csv');
        setpoint = load('setpointNYWin.csv');
    
end



%Use one week of data to TRAIN 7 seperate models:
Qint = ones(1,daymin);
X = zeros(daymin-4,15);
Coefficients = cell(num_models,1);

for i=1:num_models
    start = ((i-1)*daymin)+1;
    stop = ((i-1)*daymin)+daymin;
    Tout = outtemp(start:stop);
    Tair = intemp(start:stop);
    Iglobal = solar(start:stop);
    Qadj = status(start:stop);
    
    %populate Y:
    Y = Tair(5:daymin);
    
    %populate X:
    for k=5:daymin
        H = [-Tair(k-1),-Tair(k-2),-Tair(k-3),-Tair(k-4),Tout(k-1),Tout(k-2), ...
            Tout(k-3),Tout(k-4),Qadj(k-2),Qadj(k-3),Qadj(k-4),Iglobal(k-2), ...
            Iglobal(k-3),Iglobal(k-4),Qint(k-1)];
        X(k-4,1:15) = H;
    end

    Coeffs = regress(Y,X);
    Coefficients{i} = Coeffs;
end


   


%GENERATE the next day:

s1 = stop + 1
s2 = stop+daymin

Tout = outtemp(stop+1:stop+daymin);
Tair = intemp(stop+1:stop+daymin);
Iglobal = solar(stop+1:stop+daymin);
Qadj = status(stop+1:stop+daymin);
Qint = ones(1,daymin);
SP = setpoint(stop+1:stop+daymin);

model_gen = cell(1,num_models);


%generate for each model
for j=1:num_models
    Coeffs = Coefficients{j};
    a1 = Coeffs(1);
    a2 = Coeffs(2);
    a3 = Coeffs(3);
    a4 = Coeffs(4);
    b11 = Coeffs(5);
    b12 = Coeffs(6);
    b13 = Coeffs(7);
    b14 = Coeffs(8);
    b22 = Coeffs(9);
    b23 = Coeffs(10);
    b24 = Coeffs(11);
    b32 = Coeffs(12);
    b33 = Coeffs(13);
    b34 = Coeffs(14);
    b41 = Coeffs(15);
   

    %initial indoor temperatures 
    kMinus1 = Tair(4);
    kMinus2 = Tair(3);
    kMinus3 = Tair(2);
    kMinus4 = Tair(1);

    counter = 0;
    %calculate the current set of indoor temperatures
    %NOTE: remember to change the kMinus assignments to not
    %reinitialize every new prior!!
    CalcInTemp = zeros(1,min_prior);
    CalcInTemp(1:4) = [kMinus4, kMinus3, kMinus2, kMinus1];
    for k=5:daymin
        
        %event based synchronization:
        if SP(k) ~= SP(k-1) %if setpoint changes
            counter = 4; %next four indoor temperatures will be synched
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
    model_gen{j} = CalcInTemp;  

end

%generate one day from multi-mode soft switching:
len = length(Tair);
MMSS_result = zeros(1,len);

for i=1:(24/hr_prior)
    start = ((i-1)*min_prior)+1;
    stop = ((i-1)*min_prior)+min_prior;
    Tair_i = Tair(start:stop);
    
    rms = zeros(1,num_models);
    for j=1:num_models
        modeled_i = model_gen{j}(start:stop)';
        model_rms = sqrt(sum((Tair_i - modeled_i).^2)/len);
        rms(j) = model_rms;
    end
    
    %weight model temperatures:
    weight = (1./rms)/sum(1./rms);
    weighted_sum = zeros(1,min_prior);
    for j=1:num_models
         weighted_sum = weighted_sum + weight(j)*model_gen{j}(start:stop);
    end
    
    MMSS_result(start:stop) = weighted_sum;
    
end
    
    
    
        




figure, plot(Tout,'k'); hold on
xlabel('timestep');
ylabel('temperature');
plot(Tair, 'r');
plot(Qadj+ 13, 'm');
plot(MMSS_result, 'g');
%plot(model_gen{7}, 'g');
plot(Iglobal/100, 'y');
plot(SP,'b');
legend('outTemp','inTemp','HVAC Energy', 'solar/1000', 'setpoint', 'Location','SouthEast');
hold off;





        

