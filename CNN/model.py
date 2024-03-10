
#lets build the convo function for 1D

import numpy as np 

def convo1d(x,w,s=1,p=0):

    w_rot = np.array(w[::-1])
    x_padded = np.array(x)

    if p >0 :
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
        res =[]

    for i in range(0, int((len(x_padded)-len(w_rot)))+1,s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)

##testing 

x = [1,2,3,4,5,6,1,3]
w = [1,0,3,1,2]

print("convo1D RES : ", convo1d(x,w,p=2,s=1))

print("convo1d numpy: ", np.convolve(x,w,mode='same'))


