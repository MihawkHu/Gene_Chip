## file name: load.py
## function: read data from txt
##           return numpy matrix/vector

import sys
import numpy as np
import string

def load(file_path):
    f = open(file_path, 'r')
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split('\t')
        data[i] = map(string.atof, data[i])
    data = np.array(data)
    f.close()
    
    return data
    
    
