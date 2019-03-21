#Model diagnostics frame averages. Albaraa Khayat, 2019.In fulfiframesment of MRes.
import numpy as np
import hickle as hkl
import multiprocessing
from skimage import measure as evaluu
from tqdm import tqdm

#LOAD
prediction=hkl.load('X_hat.hkl')
observation=hkl.load('X_test.hkl')
ss='rates_mds'

#INIT
frames=10
start=20
sequences=len(prediction)
#sequences=20
threshold=0.5 #0.5mm/h Rain threshold in (normalized)pixel value=0.330588,13.00365 dBZ,
width=160 #1st dim
height=160 #2nd dim
area=width*height

xmae=np.zeros((sequences,frames))
xssim=np.zeros((sequences,frames))
xstd_prediction=np.zeros((sequences,frames))
xstd_observation=np.zeros((sequences,frames))
xmae_p=np.zeros((sequences,frames))
xssim_p=np.zeros((sequences,frames))
xstd_p=np.zeros((sequences,frames))

mae=np.zeros(frames)
ssim=np.zeros(frames)
std_prediction=np.zeros(frames)
std_observation=np.zeros(frames)
mae_p=np.zeros(frames)
ssim_p=np.zeros(frames)
std_p=np.zeros(frames)

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
prediction=pix2rate(prediction)
observation=pix2rate(observation)

for i in tqdm(range(20,30)):
    if i<5: previous_frame=0
    if i>=5 and i<10: previous_frame=4
    if i>=10 and i<15: previous_frame=9
    if i>=15 and i<20: previous_frame=14
    if i>=20: previous_frame=19
    for z in range(sequences):
        xmae[z,i-start]=np.mean(np.abs(observation[z,i]-prediction[z,i]))
        xssim[z,i-start]=evaluu.compare_ssim(observation[z,i],prediction[z,i],win_size=3,multichannel=True)
        xstd_prediction[z,i-start]=np.std(prediction[z,i])
        xstd_observation[z,i-start]=np.std(observation[z,i])
        xmae_p[z,i-start]=np.mean(np.abs(observation[z,i]-observation[z,previous_frame]))
        xssim_p[z,i-start]=evaluu.compare_ssim(observation[z,i],observation[z,previous_frame],win_size=3,multichannel=True)
        for x in range(width):
            for y in range(height):
                if prediction[z,i,x,y,0] >= threshold and observation[z,i,x,y,0] >= threshold:
                  TPm[z,i-start]+=1
                elif prediction[z,i,x,y,0] >= threshold and observation[z,i,x,y,0] < threshold:
                  FPm[z,i-start]+=1
                elif prediction[z,i,x,y,0] < threshold and observation[z,i,x,y,0] < threshold:
                  TNm[z,i-start]+=1
                elif prediction[z,i,x,y,0] < threshold and observation[z,i,x,y,0] >= threshold:
                  FNm[z,i-start]+=1
                if observation[z,previous_frame,x,y,0] >= threshold and observation[z,i,x,y,0] >= threshold:
                  pTPm[z,i-start]+=1
                elif observation[z,previous_frame,x,y,0] >= threshold and observation[z,i,x,y,0] < threshold:
                  pFPm[z,i-start]+=1
                elif observation[z,previous_frame,x,y,0] < threshold and observation[z,i,x,y,0] < threshold:
                  pTNm[z,i-start]+=1
                elif observation[z,previous_frame,x,y,0] < threshold and observation[z,i,x,y,0] >= threshold:
                  pFNm[z,i-start]+=1
                else:
                  print('Error:FP')
        if (TPm[z,i-start]+FNm[z,i-start]+TNm[z,i-start]+FPm[z,i-start]) != area:
           print('T-F/P-N inconsistent')
        elif (pTPm[z,i-start]+pFNm[z,i-start]+pTNm[z,i-start]+pFPm[z,i-start]) != area:
           print('T-F/P-N prevframe inconsistent')
	   zz=pTPm[z,i-start]+pFNm[z,i-start]+pTNm[z,i-start]+pFPm[z,i-start]
	   print(zz)
    TP[i-start]=np.mean(TPm[:,i-start])
    TN[i-start]=np.mean(TNm[:,i-start])
    FP[i-start]=np.mean(FPm[:,i-start])
    FN[i-start]=np.mean(FNm[:,i-start])
    mae[i-start]=np.mean(xmae[:,i-start])
    ssim[i-start]=np.mean(xssim[:,i-start])
    std_prediction[i-start]=np.mean(xstd_prediction[:,i-start])
    std_observation[i-start]=np.mean(xstd_observation[:,i-start])
    mae_p[i-start]=np.mean(xmae_p[:,i-start])
    ssim_p[i-start]=np.mean(xssim_p[:,i-start])
    pTP[i-start]=np.mean(pTPm[:,i-start])
    pTN[i-start]=np.mean(pTNm[:,i-start])
    pFP[i-start]=np.mean(pFPm[:,i-start])
    pFN[i-start]=np.mean(pFNm[:,i-start])

#WRITE    
hkl.dump('test1.hkl',observation[:100])
hkl.dump('hat1.hkl',prediction[:100])

f=open(ss+'_scores.txt','w')
f.write("Model MAE:%s\n" % mae)
f.write("Model SSIM:%s\n" % ssim)
f.write("Observation Stddev:%s\n" % std_observation)
f.write("Model Stddev:%s\n" % std_prediction)
f.write("Previous Frame MAE:%s\n" % mae_p)
f.write("Previous Frame SSIM:%s\n" % ssim_p)
f.close()
f=open(ss+'_pn.txt','w')
f.write("Model TP:%s\n" % TP)
f.write("Model FP:%s\n" % FP)
f.write("Model TN:%s\n" % TN)
f.write("Model FN:%s\n" % FN)
f.write("Previous frame:%s\n")
f.write("Model TP:%s\n" % pTP)
f.write("Model FP:%s\n" % pFP)
f.write("Model TN:%s\n" % pTN)
f.write("Model FN:%s\n" % pFN)
f.close()
