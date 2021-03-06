import numpy as np
import hickle as hkl
from scipy.misc import imread 
import zlib
import os
import sys
#from tqdm import tqdm
from keras import backend as K
from keras.preprocessing.image import Iterator
#from datetime import datetime
# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, split, DATA_DIR, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
	self.DATA_DIR = DATA_DIR
	self.splitt=split
	assert split in {'train', 'valid', 'test'} 
	print("Loading data")
	self.X = hkl.load('hko7_'+self.splitt+'_newdata.hkl')
	sourcez = os.path.join(self.DATA_DIR,'src_'+self.splitt+'_newlist.hkl')
#	self.X = out  # X will be like (n_images, nb_cols, nb_rows, nb_channels)/hkl.load(data_file)
	print("Data loaded, loading sources")           
	self.sources = hkl.load(sourcez) # source for each image so when creating sequences can assure that consecutive frames are from same video
	print("Sources loaded")           
	self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            print("transpose triggered")
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - 15) if self.sources[i] == self.sources[i + 15 - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - 15 + 1:
                if self.sources[curr_location] == self.sources[curr_location + 15 - 1]:
                    possible_starts.append(curr_location)
                    curr_location += 15
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, 3, 160, 160, 5), np.float32)
        buff2 = np.zeros((15,160,160,1), np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            buff2 = np.copy(self.preprocess(self.X[idx:idx+15]))
            buff2 = np.swapaxes(buff2,0,3)
            batch_x[i,0]=buff2[:,:,:,:5]
            batch_x[i,1]=buff2[:,:,:,5:10]
            batch_x[i,2]=buff2[:,:,:,10:]
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

        
    def create_all(self):
        X_all = np.zeros((self.N_sequences, 3,160,160,5), np.float32)
        buff = np.zeros((15,160,160,1), np.float32)
        for i, idx in enumerate(self.possible_starts):
            buff = np.copy(self.preprocess(self.X[idx:idx+15]))
            buff = np.swapaxes(buff,0,3)
            X_all[i,0]=buff[:,:,:,:5]
            X_all[i,1]=buff[:,:,:,5:10]
            X_all[i,2]=buff[:,:,:,10:]
        return X_all
