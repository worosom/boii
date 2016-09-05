from utils.parse_files import *
import config.nn_config as nn_config
import sys


config = nn_config.get_neural_net_configuration(0)
input_directory = config['dataset_directory']
output_filename = config['model_file']
freq = config['sampling_frequency'] #sample frequency in Hz

clip_len = 5 		#length of clips for training. Defined in seconds
block_size = 512 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
#Step 1 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
convert_wav_files_to_nptensor(input_directory+'wave/', block_size, max_seq_len, output_filename, 200)


#./datasets/Chopin/wave/
