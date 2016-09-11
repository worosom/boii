import math
from keras import backend as K
from keras import objectives
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Dense, Lambda
from keras.layers import Bidirectional, RepeatVector
from keras.layers import LSTM
from keras.optimizers import SGD, RMSprop

def create_variational_autoenc(batch_input_shape, latent_dim, num_hidden_dimensions):
	original_dim = batch_input_shape[-1]
	# First, here's our encoder network, mapping inputs to our latent distribution parameters:
	x = Input(batch_shape=batch_input_shape)
	h = Dense(num_hidden_dimensions, activation='relu')(x)
	z_mean = Dense(latent_dim)(h)
	z_log_sigma = Dense(latent_dim)(h)

	# We can use these parameters to sample new similar points from the latent space:
	def sampling(args):
		z_mean, z_log_sigma = args
		epsilon = K.random_normal(shape=(batch_input_shape[0], latent_dim), mean=0.)
		return z_mean + K.exp(z_log_sigma) * epsilon
	
	# note that "output_shape" isn't necessary with the TensorFlow backend
	# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

	# Finally, we can map these sampled latent points back to reconstructed inputs:
	# we instantiate these layers separately so as to reuse them later
	decoder_h = Dense(num_hidden_dimensions, activation='relu')
	decoder_mean = Dense(original_dim, activation='sigmoid')
	h_decoded = decoder_h(z)
	x_decoded_mean = decoder_mean(h_decoded)
	
	# end-to-end autoencoder
	vae = Model(x, x_decoded_mean)
	
	# encoder, from inputs to latent space
	encoder = Model(x, z_mean)
	
	# generator, from latent space to reconstructed inputs
	decoder_input = Input(shape=(latent_dim,))
	_h_decoded = decoder_h(decoder_input)
	_x_decoded_mean = decoder_mean(_h_decoded)
	generator = Model(decoder_input, _x_decoded_mean)

	# We train the model using the end-to-end model, with a custom loss function: the sum of a reconstruction term, and the KL divergence regularization term.
	def vae_loss(x, x_decoded_mean):
		xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
		kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
		return xent_loss + kl_loss

	vae.compile(optimizer='rmsprop', loss=vae_loss)
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
			activation='sigmoid',
			inner_activation='tanh'))
	print ('Done.\nCreating layer to convert hidden space back to frequency space...')
	model.add(
		TimeDistributed(
			Dense(
				input_shape[-1]
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
			init='uniform',
			inner_init='uniform',
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