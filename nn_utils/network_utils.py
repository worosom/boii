import math
from keras import backend as K
from keras import objectives
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import SGD, RMSprop
from keras.utils.visualize_util import plot as plt
from residual_blocks import build_residual_block
def plot(args):
	model, filename = args
	plt(model, show_shapes=True, to_file=filename)

def create_wave_net(batch_input_shape, num_hidden):
	rows = batch_input_shape[0]
	cols = batch_input_shape[1]
	model = Sequential()
	input_layer = TimeDistributed(
		Convolution1D(nb_filter=batch_input_shape[2], filter_length=1),
		input_shape=batch_input_shape)
	model.add(input_layer)
	print(input_layer.output_shape)
	i = 0
	while i < 10:
		residual = build_residual_block(
			input_shape=batch_input_shape,
			n_feature_maps=batch_input_shape[0],
			kernel_sizes=(78,512), verbose=0)
		model.add(residual)
		i += 1
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=batch_input_shape[0], nb_row=1, nb_col=1))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=batch_input_shape[0], nb_row=1, nb_col=1))
	model.add(TimeDistributed(Activation('softmax')))
	rmsprop = RMSprop()
	model.compile(optimizer=rmsprop, loss='mean_squared_error', metrics=['accuracy'])
	return model

# https://blog.keras.io/building-autoencoders-in-keras.html
def create_variational_autoenc(batch_input_shape, latent_dim, num_hidden_dimensions, lrate=.000001):
	original_dim = batch_input_shape[-1]
	# First, here's our encoder network, mapping inputs to our latent distribution parameters:
	x = Input(batch_shape=batch_input_shape)
	h = TimeDistributed(Dense(num_hidden_dimensions, activation='tanh'))(x)
	z_mean = LSTM(latent_dim, return_sequences=True)(h)
	z_log_var = LSTM(latent_dim, return_sequences=True)(h)

	# We can use these parameters to sample new similar points from the latent space:
	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(batch_input_shape[1], latent_dim), mean=0.)
		return z_mean + K.exp(z_log_var / 2) * epsilon
	
	# note that "output_shape" isn't necessary with the TensorFlow backend
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
	
	# Finally, we can map these sampled latent points back to reconstructed inputs:
	# we instantiate these layers separately so as to reuse them later
	decoder_h = LSTM(num_hidden_dimensions, activation='tanh', input_shape=(143, 2))
	decoder_mean = TimeDistributed(Dense(original_dim, activation='sigmoid', input_dim=num_hidden_dimensions))
	h_decoded = decoder_h(z)
	x_decoded_mean = decoder_mean(h_decoded)
	#
	## end-to-end autoencoder
	vae = Model(x, x_decoded_mean)
	#
	## encoder, from inputs to latent space
	#encoder = Model(x, z_mean)
	#
	## generator, from latent space to reconstructed inputs
	#decoder_input = Input(shape=(latent_dim,))
	#_h_decoded = decoder_h(decoder_input)
	#_x_decoded_mean = decoder_mean(_h_decoded)
	#generator = Model(decoder_input, _x_decoded_mean)

	# We train the model using the end-to-end model, with a custom loss function: the sum of a reconstruction term, and the KL divergence regularization term.
	def vae_loss(x, x_decoded_mean):
		xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return xent_loss + kl_loss

	rmsprop = RMSprop(lr=lrate)
	vae.compile(optimizer=rmsprop, loss=vae_loss, metrics=['accuracy'])
	return vae

def create_lstm_autoenc(input_shape, num_hidden_dimensions):
	input_dim = input_shape[-1]
	inputs = Input(shape=input_shape)
	#inputs = TimeDistributed(Dense(num_hidden_dimensions))(inputs)
	encoded = LSTM(num_hidden_dimensions)(inputs)

	decoded = RepeatVector(input_shape[:1])(encoded)
	decoded = LSTM(num_hidden_dimensions, return_sequences=True)(decoded)

	decoder = Model(inputs, decoded)
	encoder = Model(inputs, encoded)
	return encoder, decoder

def create_lstm_rnn(num_hidden_dimensions, max_hidden_dimension_size, num_recurrent_units, input_shape, stateful=False, lrate=0.):
	model = Sequential()
	print ('Created Sequential Model.\nAdding layer to convert frequency space to hidden space...\n')
	model.add(
		TimeDistributed(
			Dense(
				num_hidden_dimensions,
				),
			batch_input_shape=input_shape
			)
		)
	model.add(Activation('linear'))
	print ('Done.\nCreating recurrent units...')
	for cur_unit in xrange(num_recurrent_units):
		print('Layer: ' + str(cur_unit+1))
		input_dim, output_dim = get_hidden_count(cur_unit, num_recurrent_units, num_hidden_dimensions, max_hidden_dimension_size)
		print('Input Dimensions: %d Output Dimensions: %d' % (input_dim, output_dim))
		model.add(LSTM(
			input_dim=input_dim,
			output_dim=output_dim,
			return_sequences=True,
			stateful=stateful,
			init='glorot_uniform',
			inner_init='orthogonal',
			forget_bias_init='one',
			activation='sigmoid',
			inner_activation='tanh'))
	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
	model.add(
		TimeDistributed(
			Dense(
				input_shape[-1],
				)
			)
		)
	print ('Done.\nCompiling model...')
	sgd = SGD(lr=lrate, momentum=.9, decay=0.0, nesterov=False)
	rmsprop = RMSprop(lr=lrate)
	model.compile(
		loss='mean_squared_error',
		optimizer=rmsprop,
		metrics=['accuracy'])
	return model

#def create_lstm_rnn(num_hidden_dimensions, max_hidden_dimension_size, num_recurrent_units, input_shape, stateful=False, lrate=0.):
#	x = TimeDistributed(
#			Dense(
#				num_hidden_dimensions,
#				activation='linear'
#				),
#			batch_input_shape=input_shape
#		)
#	x = Activation('linear', input_shape=input_shape)(x)
#	l = []
#	for cur_unit in xrange(num_recurrent_units):
#		print('Layer: ' + str(cur_unit+1))
#		input_dim, output_dim = get_hidden_count(cur_unit, num_recurrent_units, num_hidden_dimensions, max_hidden_dimension_size)
#		print('Input Dimensions: %d Output Dimensions: %d' % (input_dim, output_dim))
#		if cur_unit == 0:
#			inp = x
#		else:
#			inp = l[cur_unit-1]
#		if not cur_unit == 0:
#			inp = Activation('linear')(l[cur_unit-1])
#		l.append(LSTM(
#			input_dim=input_dim,
#			output_dim=output_dim,
#			return_sequences=True,
#			init='glorot_normal',
#			inner_init='glorot_normal',
#			forget_bias_init='one',
#			activation='sigmoid',
#			inner_activation='tanh')(inp))
#	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
#	y = l[-1]
#	y = TimeDistributed(Dense(input_shape[-1]))(y)
#	model = Model(x, y)
#	rmsprop = RMSprop(lr=lrate)
#	model.compile(
#		loss='categorical_crossentropy',
#		optimizer=rmsprop,
#		metrics=['accuracy'])
#	return model


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
			batch_input_shape=(input_shape[0], input_shape[1], input_dim),
			input_dim=input_dim,
			output_dim=output_dim,
			return_sequences=True,
			init='glorot_normal',
			inner_init='glorot_normal',
			forget_bias_init='one',
			activation='sigmoid',
			inner_activation='tanh')))
	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
	model.add(
		TimeDistributed(
			Dense(
				input_shape[-1]
				)
			)
		)
	print ('Done.\nCompiling model...')
	sgd = SGD(lr=lrate, momentum=.9, decay=0.0, nesterov=True)
	rmsprop = RMSprop(lr=lrate)
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