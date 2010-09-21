%Performs HVAC system identification training on indoor temperature and HVAC
%ON/OFF status
%based on: Bai2007 starting on p.3

%y(k) = theta'*h(k)
%theta = [a, b1, b2,..., bm]'
%h(k) = [-y(k-1),u(k-1),u(k-2),...,u(k-m)]'

%using a defined error to update parameter estimates
%theta(k) = theta(k-1) + G(k)e(k)

%where the estimator gain matrix is defined as:
%G(k) = p(k-1)h(k)/(rho+h'(k)*P(k-1)*h(k))
%with forgetting factor, rho, chosen by designer as a value between 0,1

%and the covariance matrix P is updated using:
%P(k) = (1/phi)*[I-G(k)*h'(k)]*P(k-1)

%initial values are P(0)=alpha*I and theta(0)=0, where alpha is a large
%scalar

%By Fei Xie

clear all; 
summerOrWinter = 1; %0 for Winter, 1 for Summer

%load data
if (summerOrWinter == 1)
        meas_temp = load('test.txt');
        meas_hvac = load('hvac.txt');
        intemp = load('probeSimTemp.csv');
        hvac = load('probeSimStatus.csv');
        
    elseif (summerOrWinter == 0)
        intemp = load('test.txt');
        hvac = load('hvac.txt');
    
end

%{
%BAI SIMULATED:
a = 0.98347;
b = 1.19;
d = 5;
len = 1000;
x = 1:len;
u = 0.5*sin((2*pi*x)/10)+ 0.5;
y = zeros(1,len);
for k=d+2:len
    y(k) = b*u(k-1-d) + a*y(k-1);
end
%}


%E+ SIMULATED 
len = length(intemp)/7; %one day of sim
u = hvac(1:len);
y = intemp(1:len);
d = 5;


%{
%MEASURED
[len,z] = size(meas_temp);
temp_key = meas_temp(1:len,1);
hvac_key = meas_hvac(1:len,1);
meas_intemp = meas_temp(1:len,3); %indoor temperature
meas_status = meas_hvac(1:len,2); %hvac status
start = temp_key(1);
stop = temp_key(len);
xi = start:60:stop;
y = interp1(temp_key, meas_intemp, xi);
u = interp1(hvac_key, meas_status, xi);
u(1) = 0; %hack!
len = length(y);
d=5;
%}

%{
%FUNCTION
%len = length(intemp)/2;
x = 1:len;
u = sin((2*pi*x)/10);
%y = intemp;
%}




%clear intemp;
%clear hvac;
alpha = 1000; %needs to be a large number; used to initialize P
rho = 0.99; %forgetting factor (chosen between 0-1)
m = 1; %biggest possible delay time (adjust to find optimal)


%initialize values:
P_kminus1 = alpha*eye(m+1);
theta_kminus1 = zeros(m+1,1);
h = zeros(m+1,1);
h(1) = y(d);
h(2) = u(d);
    
%array allocation:
calcTemp = zeros(1,len-1);
a_param = zeros(1,len-1);
b_param = zeros(1,len-1);
d_param = zeros(1,len-1);
Ts = zeros(1,len-1);
ks = zeros(1,len-1);
Td = zeros(1,len-1);

%storing intermediate values for development:
Pk_store = cell(1,len-1);
Gk_store = cell(1,len-1);
theta_store = cell(1,len-1);
h_store = cell(1,len-1);
Ek_store = cell(1,len-1);
eig_val = cell(1, len-1);

%joint identification:
for k=d+1:len
      
   
    E_k = y(k)-theta_kminus1'*h; %realign with actual data
    G_k = (P_kminus1*h)/(rho+h'*P_kminus1*h);
    theta_k = theta_kminus1 + G_k*E_k;
    P_k = (1/rho)*(eye(m+1) - G_k*h')*P_kminus1;
    
    
    %--------------------
    %diagnostic store:
    Pk_store{k-1} = P_k;
    Gk_store{k-1} = G_k;
    theta_store{k-1} = theta_k;
    h_store{k-1} = h;
    Ek_store{k-1} = E_k;
    [v,ei] = eig(P_k);
    eig_val{k-1} = diag(ei);
    %---------------------
    
    %[v,d] = eig(P_k);
    %diagonal = diag(d)'
   
    calcTemp(k) = theta_k'*h; %calculate temperature
    
    %identify a, b, d:
    a_param(k) = theta_k(1);
    
   
    b_param(k) = theta_k(2);
    d_param(k) = d;

    
    %calculate Ts, ks, Td
    Td(k) = d_param(k);
    Ts(k) = -1/log(a_param(k));
    ks(k) = b_param(k)/(1-exp(-1/Ts(k)));
    
    %update values
    P_kminus1 = P_k;
    theta_kminus1 = theta_k;
    %repopulate h(k-1):
    %h(3:m+1) = h(2:m);
    h(1) = y(k);
    h(2) = u(k-d); 
    
end



figure, subplot(2,1,1); plot(y(1:len),'k'); hold on
plot(calcTemp, 'r');
ylabel('actual/calculated outputs');
hold off;

subplot(2,1,2); plot(u(1:len), 'b');
xlabel('timestep');
ylabel('input');



figure, plot(a_param);
ylabel('a');
figure, plot(b_param);
ylabel('b');
figure, plot(d_param);
ylabel('d');


%{
figure, plot(Ts);
ylabel('Ts');
figure, plot(ks);
ylabel('ks');
figure, plot(Td);
ylabel('Td');
%}



