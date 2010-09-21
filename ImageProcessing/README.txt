The following packages are required to run main_v1NN.py:

-scipy
-numpy
-Image
-sys
-os
-random
-mathelp
-pickle
-itertools
-PyML

The path contained in 3faces.list must be modified to the paths of actual facial images. 

To run in command line:

>>>import main_v1NN
>>>score = main_v1NN.run_v1NN()

-------------------------------

NOTE: RECOGNITION PERFORMANCE IS BAD BECAUSE ESSENTIAL PARTS OF THE ALGORITHM ARE INTENTIONALLY LEFT OUT WHILE THIS RESEARCH IS AWAITING PUBLICATION. 

FILES:

main_v1NN is the main file that runs the recognition algorithm contained in v1NN

v1NN is a recognition algorithm that improves the facial recognition of blurred face images both in recognition performance and in computation performance. Recognition is based on wavelet math feature extraction and nearest neighbor comparison.

v1s contains functions that perform wavelet math feature extraction

v1s_funcs contains functions that performs dimensional reduction and other types of operations on v1 type data.

v1s_math contains functions that perform math operations on v1 type data.

params_feret is a dictionary of parameters specific to the feret dataset

3faces contains the paths to feret face image data. This file must be changed to reflect actual location of these images. 

FERET-faces-none is a folder that contains feret face image data. 
