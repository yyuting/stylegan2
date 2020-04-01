import numpy
import numpy as np

def read_npz(file):
    arrs = []
    with open(file, 'rb') as f:
        arr = np.load(f)
        for i in range(len(arr)):
            arrs.append(getattr(arr.f, 'arr_%d' % i))
            
    return arrs