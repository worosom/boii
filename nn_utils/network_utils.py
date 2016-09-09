import math
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

def create_lstm_rnn(num_hidden_dimensions, max_hidden_dimension_size, num_recurrent_units, input_shape, stateful=False, lrate=0.):
	model = Sequential()
	print ('Created Sequential Model.\nAdding layer to convert frequency space to hidden space...\n')
	model.add(
		TimeDistributed(
			Dense(
				num_hidden_dimensions
				),
			batch_input_shape=input_shape
			)
		)
	print ('Done.\nCreating recurrent units...')
	for cur_unit in xrange(num_recurrent_units):
		print('Layer: ' + str(cur_unit+1))
		input_dim, output_dim = get_hidden_count(cur_unit, num_recurrent_units, num_hidden_dimensions, max_hidden_dimension_size)
		print('Input Dimensions: %d Output Dimensions: %d' % (input_dim, output_dim))
		model.add(LSTM(
			input_dim=input_dim,
			output_dim=output_dim,
			return_sequences=True,
			init='glorot_uniform',
			inner_init='orthogonal',
			forget_bias_init='one',
			activation='tanh',
			inner_activation='sigmoid'))
	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
	model.add(
		TimeDistributed(
			Dense(
				input_shape[-1]
				)
			)
		)
	print ('Done.\nCompiling model...')
	sgd = SGD(lr=0.0, momentum=.9, decay=0.0, nesterov=False)
	rmsprop = 'rmsprop'
	model.compile(
		loss='mean_squared_error',
		optimizer=sgd,
		metrics=['accuracy'])
	return model

def create_blstm_rnn(num_hidden_dimensions, max_hidden_dimension_size, num_recurrent_units, input_shape, stateful=False, lrate=0.):
	model = Sequential()
	print ('Created Sequential Model.\nAdding layer to convert frequency space to hidden space...\n')
	model.add(
		TimeDistributed(
			Dense(
				num_hidden_dimensions
				),
			batch_input_shape=input_shape
			)
		)
	print ('Done.\nCreating recurrent units...')
	for cur_unit in xrange(num_recurrent_units):
		print('Layer: ' + str(cur_unit+1))
		input_dim, output_dim = get_hidden_count(cur_unit, num_recurrent_units, num_hidden_dimensions, max_hidden_dimension_size)
		print('Input Dimensions: %d Output Dimensions: %d' % (input_dim, output_dim))
		model.add(Bidirectional(LSTM(
			input_dim=input_dim,
			output_dim=output_dim,
			return_sequences=True,
			init='glorot_uniform',
			inner_init='orthogonal',
			forget_bias_init='one',
			activation='sigmoid',
			inner_activation='sigmoid')))
	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
	model.add(
		TimeDistributed(
			Dense(
				input_shape[-1]
				),
			)
		)
	print ('Done.\nCompiling model...')
	sgd = SGD(lr=lrate, momentum=.9, decay=0.0, nesterov=True)
	#rmsprop = 'rmsprop'
	model.compile(
		loss='mean_squared_error',
		optimizer=sgd,
		metrics=['accuracy'])
	return model

def get_hidden_count(layer_num, layer_count, num_hidden, num_max_hidden):
	x = __linear_rise_and_fall(layer_num, layer_count, num_hidden, num_max_hidden)
	input_dim = x * num_max_hidden
	layer_num += 1
	x = __linear_rise_and_fall(layer_num, layer_count, num_hidden, num_max_hidden)
	output_dim = x * num_max_hidden
	return int(input_dim), int(output_dim)

def __linear_rise_and_fall(layer_num, layer_count, num_hidden, num_max_hidden):
	x = float(layer_num) / float(layer_count)
	x = abs(x*2.-1.)
	r = float(num_hidden) / float(num_max_hidden)
	return -x+1.+r*x