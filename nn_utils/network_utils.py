import math
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.wrappers import Bidirectional
from keras.layers import TimeDistributed, Dense
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD

def create_lstm_rnn(num_hidden_dimensions, num_recurrent_units, input_shape, stateful=False):
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
	model.add(LSTM(
	input_dim=num_hidden_dimensions,
	output_dim=num_hidden_dimensions,
	return_sequences=True,
	stateful=stateful))
	for cur_unit in xrange(num_recurrent_units-1):
		print(str(cur_unit+1))
		model.add(LSTM(
		input_dim=num_hidden_dimensions,
		output_dim=num_hidden_dimensions,
		return_sequences=True))
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

def create_blstm_rnn(num_hidden_dimensions, num_recurrent_units, input_shape, stateful=False):
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
	model.add(LSTM(
	input_dim=num_hidden_dimensions,
	output_dim=num_hidden_dimensions,
	return_sequences=True,
	stateful=stateful))
	for cur_unit in xrange(num_recurrent_units-1):
		print(str(cur_unit+2))
		model.add(Bidirectional(
			LSTM(
			input_dim=num_hidden_dimensions,
			output_dim=num_hidden_dimensions,
			return_sequences=True)))
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
