## file name: load.py

import sys
import numpy as np
import string

def load(file_path):
    f = open(file_path, 'r')
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split('\t')
        data[i] = [float(d) for d in data[i]]# map(string.atof, data[i])
    data = np.array(data)
    f.close()
    
    return data
    
def normalize(x):
    mean = x.mean()
    std = x.std()
    for xx in x:
        xx = (xx - mean) / std
    
    return x

def writ(file_path, x):
    f = open(file_path, 'w')
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            f.write(str(x[i][j]))
            if j != x.shape[1]-1:
                f.write('\t')
        if i != x.shape[0]-1:
            f.write("\n")
    f.close()

