
#lets build the convo function for 1D

import numpy as np 

def convo1d(x,w,s=1,p=0):

    w_rot = np.array(w[::-1])
    x_padded = np.array(x)

    if p >0 :
        zero_pad = np.zeros(shape=0)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
        res =[]

    for i in range(0, int((len(x_padded)-len(w_rot)))+1,s):
        res.append(np.sum(x_padded[i:o+w_rot.shape[0]]*w_rot))
    return np.array(res)




##testing 


