import sys, os
import numpy as np
import utils.config.nn_config as nn_config
import utils.parse_files as files
import nn_utils.network_utils as nn_utils
from keras.utils.visualize_util import plot

class WaveNet:

	def __init__(self, args):
		if len(args) > 0:
			self.cur_iter = args[0]
		else:
			self.cur_iter = 0
		self.config_num = 3
		self.config = nn_config.get_neural_net_configuration(self.config_num)
		self.freq = self.config['samplerate']
		self.batch_size = self.config['batch_size']	#Number of training examples pushed to the GPU per batch.
		self.dataset_name = self.config['dataset_name']
		self.modelweight_dir = self.config['modelweight_dir']
		self.model_filename = '%s/%s' % (self.modelweight_dir, self.dataset_name)
		self.load_data()

	def load_data(self):
		self.X_train, self.Y_train, self.X_mean, self.X_var = files.load_training_data(self.config_num)
		init_shape = self.X_train.shape
		shape = (self.batch_size, init_shape[0] / self.batch_size) + init_shape[1:]
		print('Original Training data shape: %d | %d | %d' % init_shape)
		print('Final Training data shape: %d | %d | %d | %d' % shape)
		self.X_train = np.reshape(self.X_train, shape)
		self.Y_train = np.reshape(self.Y_train, shape)
		
	def run(self):
		if not os.path.isdir(self.modelweight_dir):
			os.makedirs(self.modelweight_dir)

		seq_len = self.X_train.shape[2]
		time_dim = self.X_train.shape[3]
		input_shape = (self.X_train.shape[1], seq_len, time_dim)
		model = nn_utils.create_wave_net(batch_input_shape=input_shape, num_hidden=self.config['hidden_dimension_size'])
		model.load_weights('%s/%s' % (self.modelweight_dir, self.config['dataset_name']))
		nn_utils.plot([model, self.modelweight_dir + '/wavenet_plot.png'])

		while True:
			step = 13
			#step = 200
			cur_step = 0
			i = 0
			while cur_step < input_shape[0]:
				i += 1
				prev_step = cur_step
				cur_step = step * i
				print('%d / %d' % (prev_step, cur_step))
				model.fit(
					self.X_train[prev_step:cur_step],
					self.Y_train[prev_step:cur_step],
					batch_size=step,
					nb_epoch=100)
			model.save_weights('%s/%s' % (self.modelweight_dir, self.config['dataset_name']), overwrite=True)

if __name__ == '__main__':
	wn = WaveNet(sys.argv[1:])
	wn.run()