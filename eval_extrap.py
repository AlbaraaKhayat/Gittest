#!/home/ubuntu/anaconda3/bin//python


#Model diagnostics frame averages. Albaraa Khayat, 2019.In fulfiframesment of MRes.
import numpy as np
import hickle as hkl
import multiprocessing
from skimage import measure as evaluu
from tqdm import tqdm

#LOAD
prediction=hkl.load('X_hat.hkl')
observation=hkl.load('X_test.hkl')

#INIT
frames=8
start=2
sequences=len(prediction)
#sequences=20
threshold=0.330588 #0.5mm/h Rain threshold in (normalized)pixel value=0.330588,13.00365 dBZ,
width=160
height=160
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

#prediction=pix2rate(prediction)
#observation=pix2rate(observation)
def worker(): 
    for i in tqdm(range(start,len(observation[0]))):
        for z in range(sequences):
            xmse[z,i-start]=np.mean((prediction[z,i]-observation[z,i])**2)
            xmae[z,i-start]=np.mean(np.abs(observation[z,i]-prediction[z,i]))
            xssim[z,i-start]=evaluu.compare_ssim(observation[z,i],prediction[z,i],win_size=3,multichannel=True)
            xnse[z,i-start]=1-(np.sum((prediction[z,i]-observation[z,i])**2)/np.sum((observation[z,i]-np.mean(observation[z,i]))**2))
            xstd_prediction[z,i-start]=np.std(prediction[z,i])
            xstd_observation[z,i-start]=np.std(observation[z,i])
            xrmsd[z,i-start]=np.sqrt(np.sum(np.square(prediction[z,i]-observation[z,i]))/area)
            xmse_p[z,i-start]=np.mean((observation[z,start-1]-observation[z,i])**2)
            xmae_p[z,i-start]=np.mean(np.abs(observation[z,i]-observation[z,start-1]))
            xssim_p[z,i-start]=evaluu.compare_ssim(observation[z,i],observation[z,start-1],win_size=3,multichannel=True)
            xnse_p[z,i-start]=1-(np.sum((observation[z,start-1]-observation[z,i])**2)/np.sum((observation[z,i]-np.mean(observation[z,i]))**2))
            xstd_p[z,i-start]=np.std(observation[z,start-1])
            xrmsd_p[z,i-start]=np.sqrt(np.sum(np.square(observation[z,start-1]-observation[z,i]))/area)
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
                    else:
                      print('Error:FP')
            if (TPm[z,i-start]+FNm[z,i-start]+TNm[z,i-start]+FPm[z,i-start]) != area:
               print('T-F/P-N inconsistent')
        TP[i-start]=np.mean(TPm[:,i-start])
        TN[i-start]=np.mean(TNm[:,i-start])
        FP[i-start]=np.mean(FPm[:,i-start])
        FN[i-start]=np.mean(FNm[:,i-start])
        mse[i-start]=np.mean(xmse[:,i-start])
        mae[i-start]=np.mean(xmae[:,i-start])
        ssim[i-start]=np.mean(xssim[:,i-start])
        nse[i-start]=np.mean(xnse[:,i-start])
        std_prediction[i-start]=np.mean(xstd_prediction[:,i-start])
        std_observation[i-start]=np.mean(xstd_observation[:,i-start])
        mse_p[i-start]=np.mean(xmse_p[:,i-start])
        mae_p[i-start]=np.mean(xmae_p[:,i-start])
        ssim_p[i-start]=np.mean(xssim_p[:,i-start])
        nse_p[i-start]=np.mean(xnse_p[:,i-start])
        std_p[i-start]=np.mean(xstd_p[:,i-start])
        rmsd[i-start]=np.mean(xrmsd[:,i-start])
        rmsd_p[i-start]=np.mean(xrmsd_p[:,i-start])
    return 

if __name__ == '__main__':
    multiprocessing.Process(target=worker).start()

#WRITE
f=open('rates_clean_scores.txt','w')
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
f=open('rates_clean_pn.txt','w')
f.write("Model TP:%s\n" % TP)
f.write("Model FP:%s\n" % FP)
f.write("Model TN:%s\n" % TN)
f.write("Model FN:%s\n" % FN)
f.close()
