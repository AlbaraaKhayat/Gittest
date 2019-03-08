#AlbaraaKhayat,2019.In fulfilmentof MRes.
import numpy as np
import hickle as hkl
from skimage import measure as evaluu
from tqdm import tqdm

#Load
set1='X_hat.hkl'
set2='X_test.hkl'
hat=hkl.load(set1)
val=hkl.load(set2)
#hat=hat1[:6]
#val=hat1[:6]
#Init
ll=9
lenn=len(hat)
zeros=np.zeros(ll)
mse=zeros
mae=zeros
ssim=zeros
nse=zeros
std_hat=zeros
std_val=zeros
mse_p=zeros
mae_p=zeros
ssim_p=zeros
nse_p=zeros
std_p=zeros
rmsd=zeros
rmsd_p=zeros
TP=zeros
FP=zeros
TN=zeros
FN=zeros
TPm=np.zeros(((lenn,ll)))
FPm=np.zeros((lenn,ll))
TNm=np.zeros((lenn,ll))
FNm=np.zeros((lenn,ll))
#Calc
for i in tqdm(range(1,10)):
    mse[i-1]=np.mean((hat[:,i]-val[:,i])**2)
    mae[i-1]=np.mean(np.abs(val[:,i]-hat[:,i]))
    ssim[i-1]=evaluu.compare_ssim(val[:,i],hat[:,i],win_size=3,multichannel=True)
    nse[i-1]=1-(np.sum((hat[:,i]-val[:,i])**2)/np.sum((val[:,i]-np.mean(val[:,i]))**2))
    std_hat[i-1]=np.std(hat[:,i])
    std_val[i-1]=np.std(val[:,i])
    rmsd[i-1]=np.sqrt(np.sum(np.square(hat[:,i]-val[:,i]))/25600)
    mse_p[i-1]=np.mean((val[:,i-1]-val[:,i])**2)
    mae_p[i-1]=np.mean(np.abs(val[:,i]-val[:,i-1]))
    ssim_p[i-1]=evaluu.compare_ssim(val[:,i],val[:,i-1],win_size=3,multichannel=True)
    nse_p[i-1]=1-(np.sum((val[:,i-1]-val[:,i])**2)/np.sum((val[:,i]-np.mean(val[:,i]))**2))
    std_p[i-1]=np.std(val[:,i-1])
    rmsd_p[i-1]=np.sqrt(np.sum(np.square(hat[:,i]-val[:,i]))/25600)
    for z in range(len(hat)):
        for x in range(160):
            for y in range(160):
                if hat[z,i,x,y]>=12.978 and val[z,i,x,y]>=12.978:
                  TPm[z,i-1]+=1
                if hat[z,i,x,y]>=12.978 and val[z,i,x,y]<12.978:
                  FPm[z,i-1]+=1
                if hat[z,i,x,y]<12.978 and val[z,i,x,y]<12.978:
                  TNm[z,i-1]+=1
                if hat[z,i,x,y]<12.978 and val[z,i,x,y]>=12.978:
                  FNm[z,i-1]+=1
    TP[i-1]=np.mean(TPm[:,i-1])
    TN[i-1]=np.mean(TNm[:,i-1])
    FP[i-1]=np.mean(FPm[:,i-1])
    FN[i-1]=np.mean(FNm[:,i-1])
#Write for line in
f=open('eval_scores.text','w')
f.write("Model MSE:%s\n" % (mse))
f.write("Model MAE:%s\n" % (mae))
f.write("Model SSIM:%s\n" % (ssim))
f.write("Model NSE:%s\n" % (nse))
f.write("Model Stddev:%s\n" % (std_hat))
f.write("Model RMSD:%s\n" % (rmsd))
f.write("Observation Stddev:%s\n" % (std_val))
f.write("Previous Frame MSE:%s\n" % (mse_p))
f.write("Previous Frame MAE:%s\n" % (mae_p))
f.write("Previous Frame SSIM:%s\n" % (ssim_p))
f.write("Previous Frame NSE:%s\n" % (nse_p))
f.write("Previous Frame Stddev:%s\n" % (std_p))
f.write("Previous Frame RMSD:%s\n" % (rmsd_p))
f.close()
f=open('eval_PN.text','w')
f.write("Model TP:%s\n"%TP)
f.write("Model FP:%s\n"%FP)
f.write("Model TN:%s\n"%TN)
f.write("Model FN:%s\n"%FN)
f.close()
