import numpy as np
import hickle as hkl
import sys
from skimage import measure as evaluu
set1='X_hat.hkl'
set2='X_test.hkl'
hat=hkl.load(set1)
val=hkl.load(set2)
for x in range(1,11):
  mse[x-1] = np.mean( (hat[:, x] - val[:, x])**2 )  
  mae[x-1] = np.mean(np.abs(val[:, x] - hat[:, x]))
  ssim[x-1]= evaluu.compare_ssim(val[:, x],hat[:, x],win_size=3,multichannel=True)
  nse[x-1] = 1 - (np.sum((hat[:,x] - val[:,x])**2)/np.sum((val[:,x] - np.mean(val[:,x]))**2))
  std_hat[x-1]=np.std(hat[:,x])
  std_val[x-1]=np.std(val[:,x])

for x in range(1,11):
  mse_p[x-1] = np.mean( (val[:, x-1] - val[:, x])**2 )  
  mae_p[x-1] = np.mean(np.abs(val[:, x] - val[:, x-1]))
  ssim_p[x-1]= evaluu.compare_ssim(val[:, x],val[:, x-1],win_size=3,multichannel=True)
  nse_p[x-1] = 1 - (np.sum((val[:,x-1] - val[:,x])**2)/np.sum((val[:,x] - np.mean(val[:,x]))**2))
  std_p[x-1]=np.std(val[:,x-1])
  
  
f = open('eval_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse)
f.write("Model MAE: %f\n" % mae)
f.write("Model SSIM: %f\n" %ssim )
f.write("Model NSE: %f\n" % nse)
f.write("Model Stddev: %f\n" % std_hat)
f.write("Observation Stddev: %f\n" % std_val)
f.write("Previous Frame MSE: %f\n" % mse_p)
f.write("Previous Frame MAE: %f\n" % mae_p)
f.write("Previous Frame SSIM: %f\n" %ssim_p )
f.write("Previous Frame NSE: %f\n" % nse_p)
f.write("Previous Frame Stddev: %f\n" % std_p)
f.close()
