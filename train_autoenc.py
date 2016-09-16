import sys
import numpy as np
import utils.config.nn_config as nn_config
import nn_utils.network_utils as nn_utils
from keras.utils.visualize_util import plot


global cur_iter
config = nn_config.get_neural_net_configuration(int(sys.argv[1]))
freq = config['samplerate']
batch_size = config['batch_size']	#Number of training examples pushed to the GPU per batch.
inputFile = config['model_file']
cur_iter = int(sys.argv[2])
model_name = config['model_name']
model_weight_dir = './model_weights/%s' % (model_name)
model_filename = '%s/%s' % (model_weight_dir, model_name)

def load_data():
	#X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
	#Y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
	X_train = np.load(inputFile + '_x.npy')
	Y_train = np.load(inputFile + '_y.npy')
	print('Training data shape:')
	print(X_train.shape)
	if(config['stateful']):
		print('The network model you are loading the data for is set to be stateful.')
		length = int(X_train.shape[0] / float(batch_size)) * batch_size
		X_train = X_train[len(X_train)-length:]
		Y_train = Y_train[len(Y_train)-length:]
	return X_train, Y_train

def load_model():
	freq_dims = X_train.shape[2]
	batch_input_shape = (batch_size, X_train.shape[1], freq_dims)
	model = nn_utils.create_variational_autoenc(
		batch_input_shape = batch_input_shape,
		latent_dim = 2,
		num_hidden_dimensions=config['hidden_dimension_size'])
	return model

if __name__ == '__main__':
	X_train, Y_train = load_data()
	#X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
	#Y_train = Y_train.reshape((Y_train.shape[0]*Y_train.shape[1],Y_train.shape[2]))
	model = load_model()
	nn_utils.plot(model, './autoenc_plot.png')
	max_iters=500
	while cur_iter < max_iters:
		for i in xrange(X_train.shape[0]):
			model.fit(X_train[i], Y_train[i], batch_size=batch_size, nb_epoch=1, verbose=1)
		cur_iter += 1
		model.save_weights('./model_weights/autoenc_weights' + str(cur_iter))