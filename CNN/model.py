#lets build the convo function for 1D
#:""
import numpy as np 
# def convo1d(x,w,s=1,p=0):

#     w_rot = np.array(w) #according to the original formula :)
#     x_padded = np.array(x)

#     if p >0 :
#         zero_pad = np.zeros(shape=p)
#         x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
#         res =[]

#     for i in range(0, int((len(x_padded)-len(w_rot)))+1,s):
#         res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot)) #here we sliding 
#     return np.array(res)                                       #our kernel through inputs




#lets build convo for 1d but with cross-corelation  

def convo1d(x,w,p=0,s=1):

    x_array = np.array(x)
    w = np.array(w)

    if p > 0 :
        padding = np.zeros(shape=p)
        x_padded = np.concatenate([padding,x_array,padding])


    res = []

    for i in range(0,int((len(x_padded)-len(w)))+1,s):
        
        res.append(np.dot(x_padded[i:i+w.shape[0]],w)) #its the length of the kernel that is start with i and ends with i + w.shape

    return res 

##testing 

x = [1,2,3,4,5,6,1,3]
w = [1,0,3]

print("convo1D RES : ", convo1d(x,w,p=1,s=1))

print("convo1d: full: ", np.convolve(x,w,mode='same'))

print("convo1d: same: ",np.convolve(x,w,mode='same'))
print("convo1d: valid: ",np.convolve(x,w,mode='valid'))


#let's build the convo2d 

def convo2d(x,w,s=(1,1),p=(0,0)):


    w_rot = np.array(w)[::-1,::-1]
    x_orig = np.array(x)
    
    n1 = x_orig.shape[0] + 2*p[0]
    n2 = x_orig.shape[0] + 2*p[1]

    x_padded = np.zeros(shape = (n1,n2))

    x_padded[p[0]:p[0]+x_orig[0],p[1]:p[1]+x_orig[1]] = x_orig 

    res = []

    for i in range(0, int((x_padded.shape[0] - w_rot.shape[0]/s[0])+1,s[0]):

        res.append([])

        for j in range(0, int(x_padded[1] -  w_rot.shape[1]/s[1])+1,s[1]):

            x_sub = x_padded[i:i+w_rot.shape[0],j:j+w_rot.shape[1]]
            res[-1].append(np.dot(x_sub,w) 

    return(np.array(res))


#lets provide the inputs for convo2d
 

x = [[1,3,2,4],[5,6,1,3],[1,2,0,2],[3,4,3,2]]

w = [[1,0,3],[1,2,1],[0,1,1]]

print("convo2d : ",convo2d(x,w,p=(1,2),s=(1,1)))
