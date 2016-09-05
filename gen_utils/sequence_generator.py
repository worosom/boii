import numpy as np
import utils.parse_files as files

#Extrapolates from a given seed sequence
def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
	seedSeq = seed.copy()
	output = []

	#The generation algorithm is simple:
	#Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
	#Step 2 - Concatenate X_n + 1 onto A
	#Step 3 - Repeat MAX_SEQ_LEN times
	for it in xrange(sequence_length):
		seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
		#Step 2. Append it to the sequence
		if it == 0:
			for i in xrange(seedSeqNew.shape[1]):
				output.append(seedSeqNew[0][i].copy())
		else:
			output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
		newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
		newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
		seedSeq = np.concatenate((seedSeq, newSeq), axis=1)

	#Finally, post-process the generated sequence so that we have valid frequencies
	#We're essentially just undo-ing the data centering process
	for i in xrange(len(output)):
		output[i] *= data_variance
		output[i] += data_mean
	return output

def generate_from_file(model, block_size, filename, data_variance, data_mean, input_shape):
	song = np.asarray(files.load_fft_blocks_from_wav(filename, block_size))
	print("Song length: %s, Frequency Dimensions: %s" % (song.shape[0], song.shape[1]))
	for i in xrange(len(song)):
		song[i] -= data_mean
		song[i] /= data_variance

	seedSeq = np.resize(song, input_shape)
	output = []
	for x in xrange(seedSeq.shape[0]):
		seedSeq[x] = model.predict(seedSeq[x].reshape((1,input_shape[1],input_shape[2])))
		prog = x*seedSeq.shape[1]
		print('%s / %s' % (prog, song.shape[0]))
		output.append(seedSeq[x])
	#output=model.predict_on_batch(seedSeq)
	output = np.reshape(output, (input_shape[0]*input_shape[1], input_shape[2]))
	for i in xrange(len(output)):
		output[i] *= data_variance
		output[i] += data_mean

	return output

