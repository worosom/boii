from __future__ import absolute_import
from __future__ import print_function
import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from utils.config import nn_config
from utils.audio import *

def convert_mp3_to_wav(filename, sample_frequency):
	ext = filename[-4:]
	if(ext != '.mp3'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-4]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in xrange(len(files)-1):
		new_path += files[i]+'/'
	tmp_path = new_path + 'tmp'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
	new_name = new_path + '/' + orig_filename + '.wav'
	sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
	print('Set resampling to ' + sample_freq_str + ' kHz.')
	cmd = 'lame -a -m m {0} {1} --resample {2}'.format(quote(filename), quote(filename_tmp), sample_freq_str)
	os.system(cmd)
	return new_name

def convert_mp3s_to_wavs(filenames, sample_frequency):
	for filename in filenames:
		new_name = convert_mp3_to_wav(filename, sample_frequency)
	return new_name

def convert_flac_to_wav(filename, sample_frequency):
	ext = filename[-5:]
	if(ext != '.flac'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-5]
	orig_path = filename[0:-len(files[-1])]
	new_path = ''
	if(filename[0] == '/'):
		new_path = '/'
	for i in xrange(len(files)-1):
		new_path += files[i]+'/'
	new_path += 'wave'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	new_name = new_path + '/' + orig_filename + '.wav'
	cmd = 'sox {0} {1} channels 1 rate {2}'.format(quote(filename), quote(new_name), sample_frequency)
	os.system(cmd)
	return new_name

def convert_folder_to_wav(directory, sample_rate):
	files = os.listdir(directory)
	i = 0
	while i < len(files):
		fullfilenames = (directory+files[i+0], directory+files[i+1], directory+files[i+2])
		convert_mp3s_to_wavs(filenames=fullfilenames, sample_frequency=sample_rate)
		i += 3
	return directory + 'wave/'

def read_wav_as_np(filename):
	data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
	#np_arr = np.array(np_arr)
	return np_arr, data[0]

def write_np_as_wav(X, sample_rate, filename):
	Xnew = X * 32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=1000, time_domain=False):
	files = []
	for file in os.listdir(directory):
		if file.endswith('.wav'):
			files.append(directory+file)
	chunks_X = []
	chunks_Y = []
	num_files = len(files)
	if(num_files > max_files):
		num_files = max_files
	print('Number of wav files: ', num_files)
	for file_idx in xrange(num_files):
		file = files[file_idx]
		print('Processing: ', file_idx, '/', num_files)
		print('Filename: ', file)
		X, Y = load_training_example(file, block_size, time_domain=time_domain)
		cur_seq = 0
		total_seq = len(X)
		print('Total Length of example: %d' % total_seq)
		print('Total Length of example Y: %d' % len(Y))
		print('Maximum sequence length: %d' % max_seq_len)
		while cur_seq + max_seq_len < total_seq:
			chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
			chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
			cur_seq += max_seq_len
		print('Saved example ', (file_idx+1), ' / ',num_files)
	num_examples = len(chunks_X)
	num_dims_out = 518
	if(time_domain):
		num_dims_out = block_size
	out_shape = (num_examples, max_seq_len, num_dims_out)
	
	print('Flushing to disk... Out')
	mean_x = np.mean(np.mean(chunks_X, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(chunks_X-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	chunks_X[:][:] -= mean_x #Mean 0
	chunks_X[:][:] /= std_x #Variance 1
	chunks_Y[:][:] -= mean_x #Mean 0
	chunks_Y[:][:] /= std_x #Variance 1

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	print('Writing NP_x... ')
	np.save(out_file+'_x', np.float32(chunks_X))
	print('Writing NP_y... ')
	np.save(out_file+'_y', np.float32(chunks_Y))
	print('Done!')

def convert_nptensor_to_wav_files(tensor, indices, filename, time_domain=False):
	num_seqs = tensor.shape[1]
	for i in indices:
		chunks = []
		for x in xrange(num_seqs):
			chunks.append(tensor[i][x])
		save_generated_example(filename+str(i)+'.wav', chunks,time_domain=time_domain)

def save_generated_example(filename, generated_sequence, time_domain=False, sample_frequency=44100):
	if time_domain:
		time_blocks = generated_sequence
	else:
		time_blocks = fft_blocks_to_time_blocks(generated_sequence)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, sample_frequency, filename)
	return

def load_training_example(filename, block_size, time_domain=False):
	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size)) #Add special end block composed of all zeros
	if time_domain:
		return x_t, y_t
	X = time_blocks_to_fft_blocks(x_t)
	Y = time_blocks_to_fft_blocks(y_t)
	return X, Y

def load_fft_blocks_from_wav(filename, block_size):
	data, bitrate = read_wav_as_np(filename)
	time_blocks = convert_np_audio_to_sample_blocks(data, block_size)
	return time_blocks_to_fft_blocks(time_blocks)

def audio_unit_test(filename, filename2):
	data, bitrate = read_wav_as_np(filename)
	time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
	ft_blocks = time_blocks_to_fft_blocks(time_blocks)
	time_blocks = fft_blocks_to_time_blocks(ft_blocks)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, bitrate, filename2)
	return
