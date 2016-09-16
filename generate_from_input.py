from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K
import sys, os, signal
import pprint
import numpy as np
import alsaaudio as alsa
import nn_utils.network_utils as network_utils
import utils.seed_generator as seed_generator
import utils.sequence_generator as sequence_generator
import utils.parse_files as files
import utils.config.nn_config as nn_config
import time as now
import utils.audio as au
import utils.midi as mid
from keras.models import load_model

class Generate:

	"""
	docstring for Generate
	"""
	def __init__(self):
		self.models = []
		self.model_weights = []
		self.model_weights_noise = []
		self.audio = au.Audio(time_domain=False)
		au.config['quantize'] = True
		self.midi = mid.Midi()
		if au.config['time_domain']:
			self.freq_dims = au.config['buffersize']
		else:
			self.freq_dims = au.config['buffersize'] * 2 - 2
		self.input_shape = (1, 1, self.freq_dims)
		# Capture KeyboardInterrupt w/o try-except
		# http://stackoverflow.com/questions/4205317/ddg#4205386
		signal.signal(signal.SIGINT, self.stop)
	
	def load_model(self, config_number, num_iters):
		"""
			Configurations for different models are stored in a list of dictionaries
			'config_number' is used to retrieve one particular config in form of a dictionary	"""
		print('Neural Net Config #%d:' % config_number)
		config = nn_config.get_neural_net_configuration(config_number)
		pprint.PrettyPrinter(indent=4).pprint(config)
		model_weight_dir = config['modelweight_dir']
		model_filename = '%s/%s_%s_%s' % (model_weight_dir,
			config['hidden_dimension_size'],
			config['recurrent_units'],
			config['dataset_name'])

		if not os.path.isfile('%s_model.h5' % (model_filename)):
			"""
				Run the method to compile the model	"""
			if config_number < 3:
				model = network_utils.create_lstm_rnn(
				num_hidden_dimensions = config['hidden_dimension_size'],
				max_hidden_dimension_size = config['max_hidden_dimension_size'],
				num_recurrent_units = config['recurrent_units'],
				input_shape = self.input_shape,
				stateful = config['stateful'],
				lrate = .0)
			else:
				model = network_utils.create_wave_net(
					self.input_shape, 10)
		else:
			model = load_model('%s_model.h5' % (model_filename))
		"""
			Load the weights generated during training	"""
		print('Loading model weights from \n %s' % model_filename)
		model.load_weights(model_filename + '')
		return model
	
	def input_volume(self, epoch):
		return self.audio.get_input_level()

	def generate(self, run=True):
		self.run = run
		print ('Starting generation!')
		fb = np.zeros(self.input_shape)
		self.models[0].predict(fb)
		self.audio.run()
		self.midi.run()
		while self.run:
			cc1 = self.midi.read_cc(mid.a49_cc['cc1'])
			cc2 = self.midi.read_cc(mid.a49_cc['cc2'])
			data = self.audio.read()
			seed = np.reshape(data, self.input_shape)
			seed = (fb + seed)
			out1 = self.models[0].predict(seed)
			out2 = self.models[1].predict(seed) * (1.-cc2) + self.models[2].predict(seed) * cc2
			out = out2 * cc1 + out1 * (1.-cc1)
			output = np.reshape(out, (self.freq_dims,))
			self.audio.write(output)
		return

	def load_models(self):
		args = sys.argv[1:]
		i = 0
		for arg in args:
			self.models.append(self.load_model(i+0, arg))
			weights = np.asarray(self.models[-1].get_weights())
			self.model_weights.append(weights)
			self.model_weights_noise.append(np.random.random(np.asarray(self.model_weights[-1]).shape))
			i += 1
	
	def run(self):
		#generate(self.models[0], self.models[1], self.models[2], 4000)
		print('Initializing audio generation process...')
		#self.X_train, self.Y_train = files.load_training_data(2)
		self.load_models()
		self.generate()

	def stop(self, signal, frame):
		print('Stopping generation.')
		self.run = False
		sys.exit(0)

if __name__ == '__main__':
	generate = Generate()
	generate.run()
