%{

Algorithm does linear regression on outdoor temperature, solar irradiance, 
and HVAC status data in order to learn thermal characteristics

By Fei Xie
%}

clear all;
summerOrWinter = 0; %0 for winter, 1 for summer
simIterations = 2015; %numbers of iterations to model a week

%TRAIN:

if (summerOrWinter == 1)
    outtemp = load('outtempNYSum.csv');
    intemp = load('intempNYOFFSum.csv');
    solar = load('solarNYWin.csv');
elseif (summerOrWinter == 0)
    outtemp = load('outtempNYWin.csv');
    intemp = load('intempNYOFFWin.csv');
    solar = load('solarNYWin.csv');
    
end

len = length(intemp);
status = zeros(len);

[insample outsample solarsample dutycycle] = sample(intemp, outtemp, solar, status);


%calculate Tout, Tin, T(t), and T(t+dt)
InOutDiff = outsample-insample;
num_sample = length(InOutDiff);

InOutDiff = InOutDiff(1:num_sample-1);
solarsample = solarsample(1:num_sample-1);

InT = insample(1:num_sample-1);
InTplusdt = insample(2:num_sample);
InTimeDiff = InTplusdt - InT;

%take only nighttime data to find alpha and beta(where solar radiation is 0)
noSolarInOutDiff = InOutDiff(solarsample == 0);
noSolarInTimeDiff = InTimeDiff(solarsample == 0);


%linear regression to find alpha and beta
X = noSolarInOutDiff;
Y = noSolarInTimeDiff;
scatter(X, Y); hold on
ABline = regress(Y, [ones(size(X),1) X]);
%ABline = robustfit(X, Y);
plot(X, ABline(1)+ABline(2)*X);
hold off;

alpha = ABline(2);
beta = ABline(1);
    



%linear regression to find gamma
X = solarsample;
Y = InTimeDiff - alpha*InOutDiff - beta;
figure, scatter(X,Y); hold on
Gline = regress(Y,X);
plot(X, Gline*X);
hold off;

gamma = Gline;
    


%calculate Tout, Tin, T(t) and T(t+dt) for Heating and Cooling being ON
if (summerOrWinter == 1)
    intempHVAC = load('intempNYSum.csv');
    status = load('statusNYSum.csv');
   
elseif (summerOrWinter == 0)
    intempHVAC = load('intempNYWin.csv');
    status = load('statusNYWin.csv');
   
end


[insample outsample solarsample dutycycle] = sample(intempHVAC, outtemp, solar, status);

InOutDiffHVAC = outsample-insample;


InOutDiffHVAC = InOutDiffHVAC(1:num_sample-1);
solarsample = solarsample(1:num_sample-1);
dutycycle = dutycycle(1:num_sample-1);
InTHVAC = insample(1:num_sample-1);
InTplusdtHVAC = insample(2:num_sample);
InTimeDiffHVAC = InTplusdtHVAC - InTHVAC;


%linear regression to find delta
%X = onStatus;
X = dutycycle;
Y = InTimeDiffHVAC - alpha*InOutDiffHVAC - beta - gamma*solarsample;
%Y = onInTimeDiff - alpha*onInOutDiff - beta - gamma*onSolar;
figure, scatter(X,Y); hold on 
Dline = regress(Y,X);
plot(X, Dline*X);
hold off;

delta = Dline;
    


%GENERATE:

if (summerOrWinter == 1)
    intempP = load('NYM3Var_SumIn.csv');
    statusP = load('NYM3Var_SumCool.csv');
   
elseif (summerOrWinter == 0)
    intempP = load('intempNYVarWin.csv');
    statusP = load('statusNYVarWin.csv');

    
end

[insample outsample solarsample dutycycle] = sample(intempP, outtemp, solar, statusP);



%Simulate indoor temperature based on calculated coeff:
%initialize indoor temperature to first measured indoor temperature

calcIndoorTemp = zeros(1,simIterations);
indoorTemp = insample(1); 
for m=1:simIterations
    InOutDiffP = outsample(m)-indoorTemp;
    calcTemp = abs(alpha*InOutDiffP + beta + gamma*solarsample(m) + delta*dutycycle(m) + indoorTemp);
    calcIndoorTemp(1,m) = calcTemp;
    indoorTemp = calcTemp;
end

alpha
beta
gamma
delta


figure, plot(outsample,'k'); hold on
xlabel('timestep');
ylabel('temperature');
plot(insample, 'r');
plot(dutycycle+13, 'm');
plot(calcIndoorTemp, 'g');
plot(solarsample/1000, 'b');
legend('outTemp','inTemp','HVAC Energy','calcTemp', 'solar/1000', 'Location','SouthEast');
hold off;

