from __future__ import absolute_import
from __future__ import print_function
import sys
import pprint
import numpy as np
import alsaaudio as alsa
import os
import nn_utils.network_utils as network_utils
import utils.seed_generator as seed_generator
import utils.sequence_generator as sequence_generator
import utils.parse_files as files
import utils.config.nn_config as nn_config
import time as now
import utils.audio as au
from keras.callbacks import LearningRateScheduler

class Generate:

	"""
	docstring for Generate
	"""
	def __init__(self):
		self.models = []
		self.audio = au.Audio()
		self.freq_dims = self.audio.config['buffersize'] * 2 -2
		self.input_shape = (1, 1, self.freq_dims)
		#self.model = self.load_model(config_number=int(sys.argv[1]),num_iters=int(sys.argv[2]))

	def load_training_data(self, config_number):
		config = nn_config.get_neural_net_configuration(config_number)
		inputFile = config['model_file']
		print ('Loading mean and variance data...')
		self.X_mean = np.load(inputFile + '_mean.npy')
		self.X_var = np.load(inputFile + '_var.npy')
		#self.X_train = np.load(inputFile + '_x.npy')
		print ('Done.')
	
	def load_model(self, config_number, num_iters):
		"""
			Configurations for different models are stored in a list of dictionaries
			'config_number' is used to retrieve one particular config in form of a dictionary	"""
		print(config_number)
		config = nn_config.get_neural_net_configuration(config_number)
		pprint.PrettyPrinter(indent=4).pprint(config)
		
		model_name = config['model_name']
		model_weight_dir = './model_weights/%s' % (model_name)
		model_filename = '%s/%s' % (model_weight_dir, model_name)

		"""
			Run the method to compile the model	"""
		model = network_utils.create_blstm_rnn(
		num_hidden_dimensions = config['hidden_dimension_size'],
		max_hidden_dimension_size = config['max_hidden_dimension_size'],
		num_recurrent_units = config['recurrent_units'],
		input_shape = self.input_shape,
		stateful = config['stateful'],
		lrate = .1)
	
		"""
			Load the weights generated during training	"""
		print('Loading model weights from \n %s' % model_filename)
		model.load_weights(model_filename)
		return model
	
	def get_lrate(epoch):
		return

	def generate(self):
		#lrate = LearningRateScheduler(get_lrate)
		print ('Starting generation!')
		fb = np.zeros(self.input_shape)
		#self.models[0].predict(fb)
		self.audio.run()
		while True:
			data = self.audio.read()
			#data -= self.X_mean
			#data /= self.X_var
			seed = np.reshape(data, self.input_shape)
			seed = (fb + seed)
			#out = np.zeros(self.input_shape)
			#for model in self.models:
			#	out += model.predict(seed)
			#out /= len(self.models)
			out = self.models[0].predict(seed)
			#self.models[0].fit(out, seed, nb_epoch=1, verbose=0)
			#fb = self.models[1].predict(out)
			fb = out
			#output = seed
			output = np.reshape(out, (self.freq_dims,))
			#output *= self.X_var
			#output += self.X_mean
			self.audio.write(output)

	def load_models(self):
		args = sys.argv[1:]
		i = 0
		for arg in args:
			self.models.append(self.load_model(i+0, arg))
			i += 1
	
	def run(self):
		#generate(self.models[0], self.models[1], self.models[2], 4000)
		print('Initializing audio generation process...')
		self.load_training_data(2)
		self.load_models()
		self.generate()

if __name__ == '__main__':
	generate = Generate()
	generate.run()
