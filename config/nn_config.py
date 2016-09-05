def load_paramenters():
	nn_params = [{},{},{}]
	
	"""
			PRESET 0
	"""
	nn_params[0]['dataset_name'] = 'Chopin'
	nn_params[0]['sampling_frequency'] = 8000
	
	"""
	Number of hidden dimensions.
	For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes	"""
	nn_params[0]['hidden_dimension_size'] = 511
	#nn_params[0]['hidden_dimension_size'] = 2555
	"""
	Number of recurrent units.	"""
	nn_params[0]['recurrent_units'] = 1
	
	"""
	Training batch size.
	Bigger batches are quicker while training but require more memory.
	Setting this variable too high can result in an error
	occuring sooner or later during the training process. 	"""
	#nn_params[0]['batch_size'] = 118
	#nn_params[0]['batch_size'] = 128
	#[1, 11, 13, 53, 143, 583, 689]
	nn_params[0]['batch_size'] = 583
	
	"""
	If the model is stateful, the current states
	of the model depend on the previous states.	"""
	nn_params[0]['stateful'] = True
	
	"""
	The weights filename for saving/loading trained models	"""
	nn_params[0]['model_basename'] = get_model_basename(
		nn_params[0]['hidden_dimension_size'],
		nn_params[0]['recurrent_units'],
		nn_params[0]['dataset_name'])
	
	"""
	The model filename for the training data	"""
	nn_params[0]['model_file'] = './datasets/%sNP' % (nn_params[0]['dataset_name'])
	
	"""
	The dataset directory	"""
	nn_params[0]['dataset_directory'] = './datasets/%s/' % (nn_params[0]['dataset_name'])
	
	"""
			PRESET 1
		same as 'PRESET 0' but with
		recurrent_units = 2
	"""
	nn_params[1] = nn_params[0].copy()
	nn_params[1]['batch_size'] = 143
	nn_params[1]['recurrent_units'] = 6
	nn_params[1]['hidden_dimension_size'] = 511
	#nn_params[1]['hidden_dimension_size'] = 1022
	nn_params[1]['model_basename'] = get_model_basename(
		nn_params[1]['hidden_dimension_size'],
		nn_params[1]['recurrent_units'],
		nn_params[1]['dataset_name'])
	
	"""
	The model filename for the training data	"""
	nn_params[1]['model_file'] = './datasets/%sNP' % (nn_params[1]['dataset_name'])
	
	"""
			PRESET 2
		same as 'PRESET 0' but with
		recurrent_units = 3
	"""
	nn_params[2] = nn_params[0].copy()
	nn_params[2]['batch_size'] = 143
	nn_params[2]['recurrent_units'] = 8
	nn_params[2]['hidden_dimension_size'] = 768
	nn_params[2]['model_basename'] = get_model_basename(
		nn_params[2]['hidden_dimension_size'],
		nn_params[2]['recurrent_units'],
		nn_params[2]['dataset_name'])
	
	"""
	The model filename for the training data	"""
	nn_params[2]['model_file'] = './datasets/%sNP' % (nn_params[2]['dataset_name'])

	return nn_params

def get_neural_net_configuration(preset_number):
	nn_params = load_paramenters()
	return nn_params[preset_number]

def get_model_basename(hidden_dimension_size, recurrent_units, dataset_name):
	return './%s_%s_%s/%s' % (hidden_dimension_size, recurrent_units, dataset_name, dataset_name)