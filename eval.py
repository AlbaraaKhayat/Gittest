import numpy as np
import hickle as hkl
import sys
from skimage import measure as evaluu
set1=sys.argv[1]
set2=sys.argv[2]
hat=hkl.load(set1)
val=hkl.load(set2)
for x in range(1,11):
  mse[x-1] = np.mean( (hat[:, x] - val[:, x])**2 )  
  mae[x-1] = np.mean(np.abs(val[:, x] - hat[:, x]))
  ssim[x-1]= evaluu.compare_ssim(val[:, x],hat[:, x],win_size=3,multichannel=True)
  nse = 1 - (np.sum((hat[:,x] - val[:,x])**2)/np.sum((val[:,x] - np.mean(val[:,x]))**2))
  
