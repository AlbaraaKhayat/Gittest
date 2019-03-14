import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hickle as hkl 

X_hat=hkl.load('X_hat.hkl')
X_test=hkl.load('X_test.hkl')
RESULTS_SAVE_DIR='./'
sequences=len(X_test)
nt=30
n_plot = 40

#X_all:[1176,5,160,160,5]
#buff:[1176,1,160,160,25]
#target[1176,25,160,160,1]
def fixframes(X_all):
    buff = np.zeros((sequences, 1, 160, 160, nt), np.float32)
    for i in range(sequences):
        buff[i,:,:,:,:5]=X_all[i,0]
        buff[i,:,:,:,5:10]=X_all[i,1]
        buff[i,:,:,:,10:15]=X_all[i,2]
        buff[i,:,:,:,15:20]=X_all[i,3]
        buff[i,:,:,:,20:25]=X_all[i,4]
        buff[i,:,:,:,25:]=X_all[i,5]

    buff=np.swapaxes(buff,1,4)
    return buff
    
X_hat=fixframes(X_hat)
X_test=fixframes(X_test)

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t].squeeze(), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t].squeeze(), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()
