"""
	audio.py

	A class that implements realtime buffered audio I/O.

	Alexander Morosow, 2016
	
	reference:
	http://stackoverflow.com/q/34619779
"""
debug = False
import pprint
import numpy as np
import scipy
import alsaaudio as alsa
from multiprocessing import Process, Queue
from config.audio_config import get_audio_config
import pywt

class Audio:
	
	"""
	Initialize the audio buffer.
	"""
	def __init__(self):
		self.config = get_audio_config()
		print('Audio configuration:')
		pprint.PrettyPrinter(indent=4).pprint(self.config)
		print('__')
		self.__read_queue = Queue()
		self.__write_queue = Queue()
		self.__input_level = 0.
	
	"""
	Set up audio capture.
	"""
	def __configure_input(self):
		inp = alsa.PCM(
			alsa.PCM_CAPTURE,
			self.config['in']['mode'],
			device=self.config['in']['device'])
		inp.setchannels(self.config['channel_count'])
		inp.setrate(self.config['samplerate'])
		inp.setformat(self.config['sampleformat'])
		inp.setperiodsize(self.config['in']['buffersize'])
		return inp

	"""
	Set up audio playback.
	"""
	def __configure_output(self):
		outp = alsa.PCM(
			alsa.PCM_PLAYBACK,
			self.config['out']['mode'],
			device=self.config['out']['device'])
		outp.setchannels(self.config['channel_count'])
		outp.setrate(self.config['samplerate'])
		outp.setformat(self.config['sampleformat'])
		outp.setperiodsize(self.config['out']['buffersize'])
		return outp

	"""
	Reads audio from an ALSA audio device into the read queue.
	Supposed to run in its own process.
	"""
	def __read(self):
		inp = self.__configure_input()
		print('\nAudio input device initialized.\nStarting read Thread.')
		while True:
			_, data = inp.read()
			data = np.fromstring(data, dtype=self.config['datatype'])
			buf = data.astype(float)
			buf /= self.config['maxvalue']
			self.__read_queue.put(buf)

	"""
	Writes audio to an ALSA audio device from the write queue.
	Supposed to run in its own process.
	"""
	def __write(self):
		outp = self.__configure_output()
		print('\nAudio output device initialized.\nStarting write Thread.')
		while True:
			np_audio = np.asarray(self.__write_queue.get(),dtype=float)
			np_audio *= self.config['maxvalue']
			data = np_audio.astype(self.config['datatype'])
			outp.write(data)

	"""
	Pre-post data into the output buffer to avoid buffer underrun.
	"""
	def __pre_post_data(self):
		zeros = np.zeros(self.config['out']['buffersize'], dtype = self.config['datatype'])
		for i in range(0, self.config['pre_post']):
		    self.__write_queue.put(zeros)

	"""
	Runs the read and write processes.
	"""
	def run(self):
		self.__pre_post_data()
		read_process = Process(target = self.__read)
		write_process = Process(target = self.__write)
		read_process.start()
		write_process.start()

	"""
	Reads audio samples from the queue captured from the reading thread.
	"""
	def read(self):
		buf = []
		while len(buf) < self.config['buffersize']:
			buf.extend(self.__read_queue.get())
		self.__input_level = float(np.mean(np.abs(buf)))
		if not self.config['time_domain']:
			buf = time_block_to_fft_block(block=buf)
			#buf = time_block_to_wavelet_block(buf)
		return buf

	"""
	Writes audio samples to the queue to be played by the writing thread.
	"""
	def write(self, np_audio):
		if not self.config['time_domain']:
			data = fft_block_to_time_block(block=np_audio.copy())
			#data = wavelet_block_to_time_block(np_audio)
		else:
			data = np_audio
		#print(data)
		self.__write_queue.put(data)
		#print(self.__write_queue.qsize())
	
	def get_input_level(self):
		return self.__input_level

	def loopback(self):
		while True:
			data = self.read()
			print(data)
			self.write(data)

"""
Returns an numpy array of signed random samples centered around 0
"""
def noise(bufsize=512, amplitude=.5):
	sample = np.random.random_sample(size=bufsize)
	sample -= .5
	sample *= amplitude
	return sample
				
def convert_np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	num_samples_so_far = 0
	while(num_samples_so_far < total_samples):
		block = song_np[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0],))
			block = np.concatenate((block, padding))
		block_lists.append(block)
		num_samples_so_far += block_size
	return block_lists

def convert_sample_blocks_to_np_audio(blocks):
	song_np = np.concatenate(blocks)
	return song_np

"""
Converts a list of real numbers into frequency space. 
"""
def time_block_to_fft_block(block):
	fft_block = np.fft.hfft(block)
	new_block = np.real(fft_block)
	return new_block

def time_blocks_to_fft_blocks(blocks):
	fft_blocks = []
	for block in blocks:
		fft_blocks.append(time_block_to_fft_block(block))
	return fft_blocks

"""
Converts a real signal in frequency space into time space. 
"""
def fft_block_to_time_block(block):
	time_block = np.fft.ihfft(block)
	return np.real(time_block)

def fft_blocks_to_time_blocks(blocks):
	time_blocks = []
	for block in blocks:
		time_blocks.append(fft_block_to_time_block(block))
	return time_blocks

"""

"""
def time_block_to_wavelet_block(block):
	cA, cD = pywt.dwt(block, 'db4')
	return np.concatenate((cA,cD,))

def wavelet_block_to_time_block(block):
	cA = block[:len(block)/2]
	cD = block[len(block)/2:]
	return pywt.idwt(cA, cD, 'db4')

if __name__ == '__main__':
	a = Audio()
	a.run()
	a.loopback()