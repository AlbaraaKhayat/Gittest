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
ss='pixel'

#INIT
frames=24
#start=1
#sequences=len(prediction)
sequences=20
threshold=0.330588 #0.5mm/h Rain threshold in (normalized)pixel value=0.330588,13.00365 dBZ,
width=160 #1st dim
height=160 #2nd dim
area=width*height

xmse=np.zeros((sequences,frames))
xmae=np.zeros((sequences,frames))
xssim=np.zeros((sequences,frames))
xnse=np.zeros((sequences,frames))
xstd_prediction=np.zeros((sequences,frames))
xstd_observation=np.zeros((sequences,frames))
xmse_p=np.zeros((sequences,frames))
xmae_p=np.zeros((sequences,frames))
xssim_p=np.zeros((sequences,frames))
xnse_p=np.zeros((sequences,frames))
xstd_p=np.zeros((sequences,frames))
xrmsd=np.zeros((sequences,frames))
xrmsd_p=np.zeros((sequences,frames))

mse=np.zeros(frames)
mae=np.zeros(frames)
ssim=np.zeros(frames)
nse=np.zeros(frames)
std_prediction=np.zeros(frames)
std_observation=np.zeros(frames)
mse_p=np.zeros(frames)
mae_p=np.zeros(frames)
ssim_p=np.zeros(frames)
nse_p=np.zeros(frames)
std_p=np.zeros(frames)
rmsd=np.zeros(frames)
rmsd_p=np.zeros(frames)

TP=np.zeros(frames)
FP=np.zeros(frames)
TN=np.zeros(frames)
FN=np.zeros(frames)
TPm=np.zeros((sequences,frames))
FPm=np.zeros((sequences,frames))
TNm=np.zeros((sequences,frames))
FNm=np.zeros((sequences,frames))

#CALC

#X_all:[1176,5,160,160,5]
#buff:[1176,1,160,160,25]
#target[1176,25,160,160,1]
def fixframes(X_all):
    buff = np.zeros((sequences, 1, 160, 160, 25), np.float32)
    for i in range(sequences):
        buff[i,:,:,:,:5]=X_all[i,0]
        buff[i,:,:,:,5:10]=X_all[i,1]
        buff[i,:,:,:,10:15]=X_all[i,2]
        buff[i,:,:,:,15:20]=X_all[i,3]
        buff[i,:,:,:,20:]=X_all[i,4]
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

def worker(start,fin,previous_frame):
    for i in tqdm(range(start,fin)):
        for z in range(sequences):
            xmse[z,i-1]=np.mean((prediction[z,i]-observation[z,i])**2)
            xmae[z,i-1]=np.mean(np.abs(observation[z,i]-prediction[z,i]))
            xssim[z,i-1]=evaluu.compare_ssim(observation[z,i],prediction[z,i],win_size=3,multichannel=True)
            xnse[z,i-1]=1-(np.sum((prediction[z,i]-observation[z,i])**2)/np.sum((observation[z,i]-np.mean(observation[z,i]))**2))
            xstd_prediction[z,i-1]=np.std(prediction[z,i])
            xstd_observation[z,i-1]=np.std(observation[z,i])
            xrmsd[z,i-1]=np.sqrt(np.sum(np.square(prediction[z,i]-observation[z,i]))/area)
            xmse_p[z,i-1]=np.mean((observation[z,previous_frame]-observation[z,i])**2)
            xmae_p[z,i-1]=np.mean(np.abs(observation[z,i]-observation[z,previous_frame]))
            xssim_p[z,i-1]=evaluu.compare_ssim(observation[z,i],observation[z,previous_frame],win_size=3,multichannel=True)
            xnse_p[z,i-1]=1-(np.sum((observation[z,previous_frame]-observation[z,i])**2)/np.sum((observation[z,i]-np.mean(observation[z,i]))**2))
            xstd_p[z,i-1]=np.std(observation[z,previous_frame])
            xrmsd_p[z,i-1]=np.sqrt(np.sum(np.square(observation[z,previous_frame]-observation[z,i]))/area)
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
        mse[i-1]=np.mean(xmse[:,i-1])
        print(i)
        mae[i-1]=np.mean(xmae[:,i-1])
        ssim[i-1]=np.mean(xssim[:,i-1])
        nse[i-1]=np.mean(xnse[:,i-1])
        std_prediction[i-1]=np.mean(xstd_prediction[:,i-1])
        std_observation[i-1]=np.mean(xstd_observation[:,i-1])
        mse_p[i-1]=np.mean(xmse_p[:,i-1])
        mae_p[i-1]=np.mean(xmae_p[:,i-1])
        ssim_p[i-1]=np.mean(xssim_p[:,i-1])
        nse_p[i-1]=np.mean(xnse_p[:,i-1])
        std_p[i-1]=np.mean(xstd_p[:,i-1])
        rmsd[i-1]=np.mean(xrmsd[:,i-1])
        rmsd_p[i-1]=np.mean(xrmsd_p[:,i-1])
#WRITE    
    f=open(ss+'_mds_scores.txt','w')
    f.write("Model MSE:%s\n" % mse)
    f.write("Model MAE:%s\n" % mae)
    f.write("Model SSIM:%s\n" % ssim)
    f.write("Model NSE:%s\n" % nse)
    f.write("Observation Stddev:%s\n" % std_observation)
    f.write("Model Stddev:%s\n" % std_prediction)
    f.write("Previous Frame Stddev:%s\n" % std_p)
    f.write("Model RMSD:%s\n" % rmsd)
    f.write("Previous Frame MSE:%s\n" % mse_p)
    f.write("Previous Frame MAE:%s\n" % mae_p)
    f.write("Previous Frame SSIM:%s\n" % ssim_p)
    f.write("Previous Frame NSE:%s\n" % nse_p)
    f.write("Previous Frame RMSD:%s\n" % rmsd_p)
    f.close()
    f=open(ss+'_mds_pn.txt','w')
    f.write("Model TP:%s\n" % TP)
    f.write("Model FP:%s\n" % FP)
    f.write("Model TN:%s\n" % TN)
    f.write("Model FN:%s\n" % FN)
    f.close()
    return 

prediction=fixframes(prediction)
observation=fixframes(observation)
#prediction=pix2rate(prediction)
#observation=pix2rate(observation)

#Parralization
if __name__ == '__main__':
    p1=multiprocessing.Process(target=worker, args=(1,6,0))
    p2=multiprocessing.Process(target=worker, args=(6,11,5))
    p3=multiprocessing.Process(target=worker, args=(11,16,10))
    p4=multiprocessing.Process(target=worker, args=(16,21,15))
    p5=multiprocessing.Process(target=worker, args=(21,25,20))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
