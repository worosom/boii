from __future__ import absolute_import
from __future__ import print_function
import math
import sys
import numpy as np
import os
import nn_utils.network_utils as network_utils
import utils.config.nn_config as nn_config
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

global cur_iter
config = nn_config.get_neural_net_configuration(int(sys.argv[1]))
freq = config['samplerate']
batch_size = config['batch_size']	#Number of training examples pushed to the GPU per batch.
inputFile = config['dataset_file']
cur_iter = sys.argv[2]
modelweight_dir = config['modelweight_dir']
model_filename = '%s/%s_%s_%s' % (
	modelweight_dir,
	config['hidden_dimension_size'],
	config['recurrent_units'],
	config['dataset_name'])

num_iters = 10000					#Number of iterations for training
epochs_per_iter = 20				#Number of iterations before we save our model

def loadData():
	#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
	#Y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
	X_train = np.load(config['dataset_file'] + '_x.npy')
	Y_train = np.load(config['dataset_file'] + '_y.npy')
	print('Training data shape:')
	print(X_train.shape)
	if(config['stateful']):
		print('The network model you are loading the data for is set to be stateful.')
		length = int(X_train.shape[0] / float(batch_size)) * batch_size
		X_train = X_train[len(X_train)-length:]
		Y_train = Y_train[len(Y_train)-length:]
	return X_train, Y_train

def loadmodel():
	if not os.path.isfile('%s_model.h5' % (model_filename)):
		model = network_utils.create_lstm_rnn(
		input_shape = input_shape,
		num_hidden_dimensions = config['hidden_dimension_size'],
		max_hidden_dimension_size = config['max_hidden_dimension_size'],
		num_recurrent_units = config['recurrent_units'],
		stateful = config['stateful'],
		lrate=.001)
	else:
		model = load_model('%s_model.h5' % (model_filename))
	return model

"""
http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/	"""
def step_decay(epoch):
	train_progress = epochs_per_iter / float(cur_iter) + epochs_per_iter
	epochs_drop = 50.0 / train_progress
	drop = 0.5
	initial_lrate = .01 * train_progress
	drop *= train_progress
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	print('Learning rate: %s' % lrate)
	return lrate

def time_decay(epoch):
	initial_lrate = .1
	lrate = initial_lrate * 1 / (1 + epoch)
	print('Learning rate: %s' % lrate)
	return lrate

def run_training():
	global cur_iter
	
	if not os.path.isdir(modelweight_dir):
		os.makedirs(modelweight_dir)
	plot_path = '%s_network_plot.png' % (model_filename)
	plot(model, show_shapes=True, to_file=plot_path)
	model.save('%s_model.h5' % (model_filename))

	lrate = LearningRateScheduler(time_decay)
	checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1)
	callbacks_list = [checkpointer]
	
	print ('Starting training!')
	while cur_iter < xrange(num_iters):
		print('Iteration: ' + str(cur_iter))
		history = model.fit(
		X_train,
		Y_train,
		batch_size=batch_size,
		nb_epoch=epochs_per_iter,
		validation_split=0.0,
		shuffle=False,
		callbacks=callbacks_list,
		verbose=1)
		cur_iter += epochs_per_iter
		model.save_weights(model_filename + str(cur_iter))

if __name__ == '__main__':
	#Load up the training data
	print ('Loading training data')
	X_train, Y_train = loadData()
	print ('Finished loading training data')
	print ('Learning data shape:')
	print (X_train.shape)
	#Figure out how many frequencies we have in the data
	freq_dims = X_train.shape[2]
	input_shape = (batch_size, X_train.shape[1], freq_dims)
	
	print ('Number of Frequency Dimensions: ', freq_dims, '\nNumber of Hidden Dimensions: ', config['hidden_dimension_size'])
	print ('Number of recurrent units: ', config['recurrent_units'])
	
	#Creates a lstm network
	model = loadmodel()
	
	print ('Current iteration: ', cur_iter)
	
	#Load existing weights if available
	if os.path.isfile(model_filename + str(cur_iter)):
		model.load_weights(model_filename + str(cur_iter))
		print ('Loaded model weights from %s' % (model_filename + str(cur_iter)))
	else:
		cur_iter = 0
	
	run_training()

	print ('Training complete!')
	exit()	