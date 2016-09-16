import numpy as np
import alsaaudio as alsa

"""
_______Audio__I/O__Configuration________
	
	

	These are settings that are common
	for the input and output channels:
		
		channel_count
	this class has only been tested with channel_count = 1

		samplerate
	integer in Hz. Depends on your hardware,
	e.g. 8000, 22050, 44100, ...

		sampleformat
	The sound format of the I/O devices.
	Sound format controls how the PCM device
	interpret data for playback, and how data
	is encoded in captures.
	see this link for available formats:
	https://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#alsaaudio.PCM.setformat
		
		datatype
	This depends on 'sampleformat' and must be
	set to the corresponding numpy datatype,
	consult the numpy docs for available datatypes.
	Setting this wrongly will turn the audio into
	complete garbage due to faulty data conversion.

		maxvalue
	The maximum value determined by the datatype.

		time_domain
	if set to False, 'read()' will return the signal in freq domain,
	'write(np_audio)' will expect the signal to be in freq domain as well.


		pre_post
	determines how many buffers of silence are written
	to the output at init. This serves to prevent buffer underruns.

		buffersize
	this determines the size of the sample buffer within this class.
if timedomain is False:
	each time 'read()' is called, it will return an array of this size.
else:
	'read()' will return an array of size 'buffersize/2+1'
________________________________________
		
	The following settings can be set
	individually
		
_______I/O______________________________
		
		device
	ALSA device to be used,
	string as listed in 'aplay -L' such as 'sysdefault'

		mode
	'alsaaudio.PCM_NORMAL' or 'alsaaudio.PCM_NONBLOCK'

		buffersize (i.e. period_size)
________________________________________
"""

audio_conf = {
	'channel_count': 1,
	'sampleformat': alsa.PCM_FORMAT_S16_LE,
	'samplerate': 8000,
	'datatype': np.int16,
	'time_domain': True,
	'pre_post' : 8,
	'buffersize' : 512,
	'quantize' : True,
	'm_law_coefficient' : 4,
	'in': {
		'device': 'pulse',
		'mode' : alsa.PCM_NORMAL,
		'buffersize': 64},
	'out': {
		'device': 'pulse',
		'mode' : alsa.PCM_NONBLOCK,
		'buffersize': 256}}
audio_conf['maxvalue'] =  float(np.iinfo(audio_conf['datatype']).max)


def get_samplerate():
	return audio_conf['samplerate']

def get_datatype():
	return audio_conf['datatype']

def get_maxvalue():
	return audio_conf['maxvalue']

def get_m_law_coefficient():
	return audio_conf['get_m_law_coefficient']





