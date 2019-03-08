#Albaraa Khayat, 2019. In fulfilment of MRes. 
import numpy as np
import hickle as hkl
from skimage import measure as evaluu
from tqdm import tqdm

#Load
set1='X_hat.hkl'
set2='X_test.hkl'
hat=hkl.load(set1)
val=hkl.load(set2)

#Init
ll=9
np.zeros(ll)
mse=np.zeros(ll)
mae=np.zeros(ll)
ssim=np.zeros(ll)
nse=np.zeros(ll)
std_hat=np.zeros(ll)
std_val=np.zeros(ll)
mse_p=np.zeros(ll)
mae_p=np.zeros(ll)
ssim_p=np.zeros(ll)
nse_p=np.zeros(ll)
std_p=np.zeros(ll)
TP = np.zeros(ll)
FP = np.zeros(ll)
TN = np.zeros(ll)
FN = np.zeros(ll)


#Calc
for i in tqdm(range(1,10)):
    mse[i-1] = np.mean( (hat[:, i] - val[:, i])**2 )  
    mae[i-1] = np.mean(np.abs(val[:, i] - hat[:, i]))
    ssim[i-1]= evaluu.compare_ssim(val[:, i],hat[:, i],win_size=3,multichannel=True)
    nse[i-1] = 1 - (np.sum((hat[:,i] - val[:,i])**2)/np.sum((val[:,i] - np.mean(val[:,i]))**2))
    std_hat[i-1]=np.std(hat[:,i])
    std_val[i-1]=np.std(val[:,i])
    rmsd = np.sqrt(np.sum(np.square(hat[:,i] - val[:,i]))/25600)
    mse_p[i-1] = np.mean( (val[:, i-1] - val[:, i])**2 )  
    mae_p[i-1] = np.mean(np.abs(val[:, i] - val[:, i-1]))
    ssim_p[i-1]= evaluu.compare_ssim(val[:, i],val[:, i-1],win_size=3,multichannel=True)
    nse_p[i-1] = 1 - (np.sum((val[:,i-1] - val[:,i])**2)/np.sum((val[:,i] - np.mean(val[:,i]))**2))
    std_p[i-1]=np.std(val[:,i-1])
    rmsd_p = np.sqrt(np.sum(np.square(hat[:,i] - val[:,i]))/25600)
  for z in range(len(hat)):
        for x in range(160): 
          for y in range(160):
            if hat[z,i,x,y]>=12.978 and val[z,i,x,y]>=12.978:
              TP[i] += 1
            if hat[z,i,x,y]>=12.978 and val[z,i,x,y]<12.978:
              FP[i] += 1
            if hat[z,i,x,y]<12.978 and val[z,i,x,y]<12.978:
              TN[i] += 1
            if hat[z,i,x,y]<12.978 and val[z,i,x,y]>=12.978:
              FN[i] += 1

#Write
f = open('eval_scores.text', 'w')
f.write("Model MSE: %f\n" % mse)
f.write("Model MAE: %f\n" % mae)
f.write("Model SSIM: %f\n" % ssim )
f.write("Model NSE: %f\n" % nse)
f.write("Model Stddev: %f\n" % std_hat)
f.write("Model RMSD: %f\n" % rmsd)
f.write("Observation Stddev: %f\n" % std_val)
f.write("Previous Frame MSE: %f\n" % mse_p)
f.write("Previous Frame MAE: %f\n" % mae_p)
f.write("Previous Frame SSIM: %f\n" % ssim_p )
f.write("Previous Frame NSE: %f\n" % nse_p)
f.write("Previous Frame Stddev: %f\n" % std_p)
f.write("Previous Frame RMSD: %f\n" % rmsd_p)
f.close()
f = open('eval_PN.text', 'w')
f.write("Model TP: %f\n" % TP)
f.write("Model FP: %f\n" % FP)
f.write("Model TN: %f\n" % TN )
f.write("Model FN: %f\n" % FN)
f.close()
