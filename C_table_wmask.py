#!/home/ubuntu/anaconda3/bin//python
#Model diagnostics frame averages. Albaraa Khayat, 2019.In fulfiframesment of MRes.
import numpy as np
import hickle as hkl
import multiprocessing
from skimage import measure as evaluu
from tqdm import tqdm
#from numba import vectorize

#LOAD
prediction=hkl.load('X_hat.hkl')
observation=hkl.load('X_test.hkl')
ss='rates_extrap_mds'

#INIT
frames=24
#start=1
sequences=len(prediction)
#sequences=20
threshold=0.5 #0.5mm/h Rain threshold in (normalized)pixel value=0.330588,13.00365 dBZ,
width=160 #1st dim
height=160 #2nd dim
area=width*height


TP=np.zeros(frames)
FP=np.zeros(frames)
TN=np.zeros(frames)
FN=np.zeros(frames)
TPm=np.zeros((sequences,frames))
FPm=np.zeros((sequences,frames))
TNm=np.zeros((sequences,frames))
FNm=np.zeros((sequences,frames))
pTP=np.zeros(frames)
pFP=np.zeros(frames)
pTN=np.zeros(frames)
pFN=np.zeros(frames)
pTPm=np.zeros((sequences,frames))
pFPm=np.zeros((sequences,frames))
pTNm=np.zeros((sequences,frames))
pFNm=np.zeros((sequences,frames))

#CALC

#X_all:[1176,5,160,160,5]
#buff:[1176,1,160,160,25]
#target[1176,25,160,160,1]
def fixframes(X_all):
    buff = np.zeros((sequences, 1, 160, 160, 30), np.float32)
    for i in range(sequences):
        buff[i,:,:,:,:5]=X_all[i,0]
        buff[i,:,:,:,5:10]=X_all[i,1]
        buff[i,:,:,:,10:15]=X_all[i,2]
        buff[i,:,:,:,15:20]=X_all[i,3]
        buff[i,:,:,:,20:25]=X_all[i,4]
        buff[i,:,:,:,25:]=X_all[i,5]

    buff=np.swapaxes(buff,1,4)
    return buff


def pix2rate(data):
    data*=255 #offset normalization
    #data=((data-0.5)/3.6429)-10 #pixel to dBZ
    data-=0.5
    data/=3.6429
    data-=10
    #data=10**((data-17.6738)/15.6) #dBZ to rainfall (mm/h)
    data-=17.6738
    data/=15.6
    data=np.power(10,data)
    print(np.shape(data))
    return data

prediction=fixframes(prediction)
observation=fixframes(observation)
#prediction=pix2rate(prediction)
#observation=pix2rate(observation)

for i in tqdm(range(1,30)):
    if i<5: previous_frame=0
    if i>=5 and i<10: previous_frame=4
    if i>=10 and i<15: previous_frame=9
    if i>=15 and i<20: previous_frame=14
    if i>=20: previous_frame=19
    for z in range(sequences):
        for x in range(width):
            for y in range(height):
                if prediction[z,i,x,y,0] >= threshold and observation[z,i,x,y,0] >= threshold:
                  TPm[z,i-1]+=1
                elif prediction[z,i,x,y,0] >= threshold and observation[z,i,x,y,0] < threshold:
                  FPm[z,i-1]+=1
                elif prediction[z,i,x,y,0] < threshold and observation[z,i,x,y,0] < threshold:
                  TNm[z,i-1]+=1
                elif prediction[z,i,x,y,0] < threshold and observation[z,i,x,y,0] >= threshold:
                  FNm[z,i-1]+=1
                else:
                  print('Error:FP')
        if (TPm[z,i-1]+FNm[z,i-1]+TNm[z,i-1]+FPm[z,i-1]) != area:
           print('T-F/P-N inconsistent')
    TP[i-1]=np.mean(TPm[:,i-1])
    TN[i-1]=np.mean(TNm[:,i-1])
    FP[i-1]=np.mean(FPm[:,i-1])
    FN[i-1]=np.mean(FNm[:,i-1])
 
   for z in range(sequences):
        for x in range(width):
            for y in range(height):
                if observation[z,previous_frame,x,y,0] >= threshold and observation[z,i,x,y,0] >= threshold:
                  pTPm[z,i-1]+=1
                elif observation[z,previous_frame,x,y,0] >= threshold and observation[z,i,x,y,0] < threshold:
                  pFPm[z,i-1]+=1
                elif observation[z,previous_frame,x,y,0] < threshold and observation[z,i,x,y,0] < threshold:
                  pTNm[z,i-1]+=1
                elif observation[z,previous_frame,x,y,0] < threshold and observation[z,i,x,y,0] >= threshold:
                  pFNm[z,i-1]+=1
                else:
                  print('Error:FP')
    pTP[i-1]=np.mean(pTPm[:,i-1])
    pTN[i-1]=np.mean(pTNm[:,i-1])
    pFP[i-1]=np.mean(pFPm[:,i-1])
    pFN[i-1]=np.mean(pFNm[:,i-1])


#WRITE    
f=open(ss+'_fixed_pn.txt','w')
f.write("Model TP:%s\n" % TP)
f.write("Model FP:%s\n" % FP)
f.write("Model TN:%s\n" % TN)
f.write("Model FN:%s\n" % FN)
f.close()
