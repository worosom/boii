"""
	midi.py

	A class that implements MIDI input
	using the pygame portmidi bindings.

	Because I am using ALSA for an Interface to raw audio
	and pygame.midi is using portmidi, this has to run
	in a seperate thread.

	Alexander Morosow, 2016
"""
from multiprocessing import Process, Value
from time import sleep
import pygame.midi as midi
import numpy as np
import time

midi_events = {
	'note_on' : 144,
	'note_off' : 128,
	'pitch' : 224,
	'mod' : 176,
	'aft' : 208
}

a49_cc = {
	'cc1' : 74,
	'cc2' : 71,
	'mod' : 1,
	'vol' : 7
}

class Midi:

	"""
	Initialize MIDI
	"""
	def __init__(self, device=3):
		self.device = device
		self.ccs = a49_cc.copy()
		self.pitch = Value('f', 0.)
		self.aftrtch = Value('f', 0.)

	def read_cc(self, cc_id):
		return self.ccs[cc_id].value

	def get_pw(self):
		return self.pitch.value

	def get_aftrtch(self):
		return self.aftrtch.value

	def __read(self, args):
		midi.init()
		midi_in = midi.Input(3)
		while True:
			data = midi_in.read(1)
			if len(data) > 0:
				message = data[0][0]
				#print(message)
				if message[0] == midi_events['mod']:
					for cc in args:
						if message[1] == cc:
							self.ccs[cc].value = message[2] / 127.
				if message[0] == midi_events['pitch']:
					self.pitch.value = int((message[2] - 63.5) / 63.5 * 100) / 100.
				if message[0] == midi_events['aft']:
					self.aftrtch = max(0., message[1] - 22. / (127.-22.))
			sleep(.001)

	def run(self):
		print('Starting MIDI process.')
		for cc in a49_cc:
			self.ccs[a49_cc[cc]] = Value('f', .5) 
		self.midi_process = Process(target=self.__read, args=(self.ccs,))
		self.midi_process.start()
		return

	def stop(self):
		self.midi_process.terminate()
		return

if __name__ == '__main__':
	mid = Midi()
	mid.run()
	while True:
		#print(mid.read_cc(a49_cc['cc1']))
		#print(mid.read_cc(a49_cc['cc2']))
		#print(mid.get_aftrtch())
		sleep(.01)


def midi_to_freq(midi_note):
	if midi_note >= 0 and midi_note <= 119:
		return 440.*np.pow((midi_note-57)/12., 2)
