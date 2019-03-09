'''
Train Pred_rgcLSTM on KITTI sequences base on Lotter 2016 original code, . (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
modified by: Nelly Elsayed to be able to use the Pred_rgcLSTM model
Special thanks for Lotter for original code.

To apply the code on moving MNIST, Please read the comments within the code to apply the changes
'''

import os
import sys
import numpy as np
np.random.seed(123)
from six.moves import cPickle
import contextlib 
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
import pylab as plt
#from prednet import PredNet
from mnist_pred_rgcLSTM import Pred_rgcLSTM
from data_utils import SequenceGenerator
import time #to evaluate the process of training process
start_time = time.time()

WEIGHTS_DIR="./"
DATA_DIR="./"

save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'extrap_weights.hdf5')  # where weights will be saved
oldweights_file = os.path.join(WEIGHTS_DIR, 'oldpred_rgcLSTM_hko7_weights.hdf5')

json_file = os.path.join(WEIGHTS_DIR, 'extrap_model.json')
split='train' #valid,test or train
split2='valid'
# Data files
#train_file = os.path.join(DATA_DIR, 'hko7_valid_data.hkl')
#train_sources = os.path.join(DATA_DIR, 'src_valid_list.hkl')
#val_file = os.path.join(DATA_DIR, 'hko7_valid_data.hkl')
#val_sources = os.path.join(DATA_DIR, 'src_valid_list.hkl')

# Training parameters moving MNIST
nb_epoch = 100#150#30+1+
batch_size = 10#4
samples_per_epoch = 500#500
N_seq_val = 100  #100 number of sequences to use for validation
extrap_start_time = 1

def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

# Model parameters
n_channels, im_height, im_width = (1, 160, 160) # change to (1,64,64) in case of moving MNIST
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
layer_config['extrap_start_time'] = extrap_start_time

pred_rgcLSTM = Pred_rgcLSTM(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True, **layer_config)

inputs = Input(shape=(nt,) + input_shape)
errors = pred_rgcLSTM(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
#model.load_weights(oldweights_file)
#adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=extrap_loss, optimizer='adam')

train_generator = SequenceGenerator(split, DATA_DIR, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(split2, DATA_DIR, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
f.close()

# Saving the results and ploting the training process diagrams
print("Final loss: ", history.history['loss'])
print("Validation loss: ", history.history['val_loss'])
print("--- Run Time = %s seconds ---" % ((time.time() - start_time)))
print("--- Run Time = %s minutes ---" % ((time.time() - start_time)/60.0)) 
print("--- Run Time = %s hours ---" % ((time.time() - start_time)/(60.0*60)))
text_file = open("model_report.txt", "w")
text_file.write("Final loss: "+str(history.history['loss'])+"\n \n"+"Validation loss: "+str(history.history['val_loss'])+"\n"+
                "--- Run Time ="+ str(((time.time() - start_time)))+" seconds ---"
                    +"\n" +"--- Run Time = "+str(((time.time() - start_time)/60.0))+" minutes ---"+"\n"
                    +"--- Run Time = "+str(((time.time() - start_time)/(60.0*60)))+" hours ---"+"\n")
text_file.close()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Pred_rgcLSTM Train and Validation Loss",fontsize=20)
plt.xlabel('epoch',fontsize=16)
plt.ylabel('loss',fontsize=16)
#plt.ylim(ymax=0.933)
#plt.yticks(np.arange(0.920, 0.934, step=0.002))
plt.legend(['loss', 'cal_loss'], loc='lower right', fontsize ='large')
plt.savefig("loss-graph.jpg")
#plt.show()
#save into a file
text_file = open("process_loss_value.txt", "w")
text_file.write("Loss=\n")
text_file.write(str(np.asarray(history.history['loss'])))
text_file.write("\n Pred_rgcLSTM validation loss: \n")
text_file.write(str(np.asarray(history.history['val_loss'])))
text_file.close()


def redirect_stdout(target):
    original = sys.stdout
    sys.stdout = target
    yield
    sys.stdout = original

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
f.close()
model.summary()