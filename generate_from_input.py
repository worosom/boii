from __future__ import absolute_import
from __future__ import print_function
import sys
import pprint
import numpy as np
import alsaaudio as alsa
import os
import nn_utils.network_utils as network_utils
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
import utils.parse_files as files
import config.nn_config as nn_config
import time as now
import utils.audio as au

class Generate:

	"""
	docstring for Generate
	"""
	def __init__(self):
		self.models = []
		self.audio = au.Audio()
		self.freq_dims = self.audio.config['buffersize'] * 2 - 2
		self.input_shape = (1, 1, self.freq_dims)
		#self.model = self.load_model(config_number=int(sys.argv[1]),num_iters=int(sys.argv[2]))

	def load_training_data(self, config_number):
		config = nn_config.get_neural_net_configuration(config_number)
		inputFile = config['model_file']
		print ('Loading mean and variance data...')
		self.X_mean = np.load(inputFile + '_mean.npy')
		self.X_var = np.load(inputFile + '_var.npy')
		self.X_train = np.load(inputFile + '_x.npy')
		print ('Done.')
	
	def load_model(self, config_number, num_iters):
		"""
			Configurations for different models are stored in a list of dictionaries
			'config_number' is used to retrieve one particular config in form of a dictionary	"""
		print(config_number)
		config = nn_config.get_neural_net_configuration(config_number)
		pprint.PrettyPrinter(indent=4).pprint(config)
		
		"""
			Run a method to compile the model	"""
		model = network_utils.create_lstm_rnn(
		num_hidden_dimensions = config['hidden_dimension_size'],
		num_recurrent_units = config['recurrent_units'],
		input_shape = self.input_shape,
		stateful = config['stateful'])
	
		"""
			Load the weights generated during training	"""
		model_filename = config['model_basename'] + str(num_iters)
		print('Loading model weights from \n %s' % model_filename)
		model.load_weights(model_filename)
		return model
	
	def generate(self):
		print ('Starting generation!')
		fb = np.zeros(self.input_shape)
		while True:
			data = self.audio.read()
			seed = np.reshape(data, self.input_shape)
			seed = (fb + seed) * (2./3.)
			#out = np.zeros(self.input_shape)
			#for model in self.models:
			#	out += model.predict(seed)
			#out /= len(self.models)
			out = self.models[0].predict(seed)
			fb = self.models[1].predict(out)
			#fb = (seed - out) * .5
			#output = seed
			output = np.reshape(out, (self.freq_dims,))
			self.audio.write(output)
	
	def generate_batch(self):
		print ('Starting generation!')
		fb = np.zeros(self.input_shape)
		while True:
			data = []
			while len(data) < self.input_shape[1]:
				data.append(self.audio.read())
			seed = np.reshape(data, self.input_shape)
			seed = fb + seed
			out = np.zeros(self.input_shape)
			out = self.models[0].predict_on_batch(seed)
			fb = self.models[1].predict_on_batch(out)
			output = np.reshape(out, self.input_shape[1:])
			for output in out:
				self.audio.write(output)

	def load_models(self):
		args = sys.argv[1:]
		i = 0
		for arg in args:
			self.models.append(self.load_model(i+1, arg))
			i += 1
	
	def run(self):
		#generate(self.models[0], self.models[1], self.models[2], 4000)
		print('Initializing audio generation process...')
		self.load_models()
		self.audio.run()
		self.generate()

if __name__ == '__main__':
	generate = Generate()
	generate.run()