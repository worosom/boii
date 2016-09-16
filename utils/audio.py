"""
	audio.py

	A class that implements realtime buffered audio I/O
	using pyalsaaudio.

	Alexander Morosow, 2016
	
	reference:
	http://stackoverflow.com/q/34619779
"""
debug = False
import pprint
import numpy as np
import scipy
import struct
import alsaaudio as alsa
from multiprocessing import Process, Queue
from config.audio_config import audio_conf as config
import pywt

class Audio:
	
	"""
	Initialize the audio buffer.
	"""
	def __init__(self, time_domain=False):
		config['time_domain'] = time_domain
		config['quantize'] = False
		print('Audio configuration:')
		pprint.PrettyPrinter(indent=4).pprint(config)
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
			config['in']['mode'],
			device=config['in']['device'])
		inp.setchannels(config['channel_count'])
		inp.setrate(config['samplerate'])
		inp.setformat(config['sampleformat'])
		inp.setperiodsize(config['in']['buffersize'])
		return inp

	"""
	Set up audio playback.
	"""
	def __configure_output(self):
		outp = alsa.PCM(
			alsa.PCM_PLAYBACK,
			config['out']['mode'],
			device=config['out']['device'])
		outp.setchannels(config['channel_count'])
		outp.setrate(config['samplerate'])
		outp.setformat(config['sampleformat'])
		outp.setperiodsize(config['out']['buffersize'])
		return outp

	"""
	Reads audio from an ALSA audio device into the read queue.
	Supposed to run in its own process.
	"""
	def __read(self):
		inp = self.__configure_input()
		print('\nAudio input device initialized.\nStarting read Thread.')
		while True:
			len, data = inp.read()
			data = np.fromstring(data, dtype=config['datatype'])
			buf = data.astype(float)
			buf /= config['maxvalue']
			if config['quantize']:
				buf = quantize(m_law(buf, m=config['m_law_coefficient']))
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
			np_audio *= config['maxvalue']
			data = np_audio.astype(config['datatype'])
			outp.write(data)

	"""
	Pre-post data into the output buffer to avoid buffer underrun.
	"""
	def __pre_post_data(self):
		zeros = np.zeros(config['out']['buffersize'], dtype = config['datatype'])
		for i in range(0, config['pre_post']):
			self.__write_queue.put(zeros)

	"""
	Runs the read and write processes.
	"""
	def run(self):
		self.__pre_post_data()
		self.__read_process = Process(target = self.__read)
		self.__write_process = Process(target = self.__write)
		self.__read_process.start()
		self.__write_process.start()
		return

	def stop(self):
		print('Stopping read process...')
		self.__read_process.terminate()
		print('Stopping write process...')
		self.__write_process.terminate()
		return

	"""
	Reads audio samples from the queue captured from the reading thread.
	"""
	def read(self):
		buf = []
		while len(buf) < config['buffersize']:
			buf.extend(self.__read_queue.get())
		self.__input_level = float(np.mean(np.abs(buf)))
		#print(np.mean(np.abs(buf) - config['maxvalue']))
		#print(config['maxvalue'])
		if not config['time_domain']:
			buf = time_block_to_fft_block(block=buf)
			#buf = time_block_to_wavelet_block(buf)
		return buf

	"""
	Writes audio samples to the queue to be played by the writing thread.
	"""
	def write(self, np_audio):
		if not config['time_domain']:
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
			#print(np.mean(np.abs(data)))
			print(np.max(data))
			self.write(data)

# ITU-T. Recommendation G. 711. Pulse Code Modulation (PCM) of voice frequencies, 1988.
def m_law(x, m=64):
	return np.sign(x)*(np.log(1+m*np.abs(x))/np.log(1+m))

def quantize(x,steps=256):
	return np.floor(steps*x)/steps

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