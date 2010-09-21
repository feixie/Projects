function [insample, outsample, solarsample, dutycycle] = sample(intemp, outtemp, solar, status)

len = length(intemp);
dutycycle = zeros(len/5,1);
insample = zeros(len/5,1);
outsample = zeros(len/5,1);
solarsample = zeros(len/5,1);

counter = 1;

for i = 5:5:len
    dutycycle(counter) = sum(status(i-4:i))/5;
    insample(counter) = mean(intemp(i-4:i));
    outsample(counter) = mean(outtemp(i-4:i));
    solarsample(counter) = sum(solar(i-4:i));
    counter = counter + 1;    
end