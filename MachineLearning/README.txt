*ALL MATLAB PROGRAMS IN THIS FOLDER REQUIRE .CSV FILES IN THIS FOLDER TO RUN

firstOrderTempRegress: a first order ARX model that does linear regression on outdoor temperature, solar irradiance, 
and HVAC status data in order to learn thermal characteristics

fourthOrderTempARX: a 4th order ARX model that does linear regression on outdoor temperature, solar irradiance, 
and HVAC status data in order to learn thermal characteristics

multiModeTempARX: a Soft Switching Multi-Mode ARX model based on the Jang Thesis p118. This method is an extention of fourthOrderTempARX and blends 7 models of 4th order ARX ran on seven consecutive days. 

sample: a helper function to firstOrderTempRegress that transforms input data to duty cycle data. 

CtrlSysID: a program that implements a joint system identification model that identifies coefficient for HVAC control adaptive tuning. 

CtrlSysID_wDelay: program that implements a joint system identification model that identifies coefficient for HVAC control adaptive tuning with delay coefficient.