# boii
You have just found a project for generating real time audio using artificial neural networks.
* [Introduction](#introduction)
* [ALSA disclaimer](#alsa-disclaimer)
* [Dependencies](#dependencies)
* [Getting the sources](#getting-the-sources)
* [Contents](#contents)
* [Usage](#usage)
* [Addendum](#addendum)

## Introduction

	bidirectionally  
	oriented  
	intelligent  
	interface-agent

in short boii.  
To read about what that means, read this website: [worosom.github.io/boii](https://worosom.github.io/boii)

The code in this repository contains everything you need for training and running 

## ALSA disclaimer
If you are using a system such as Mac OS X or Windows, where ALSA is currently not supported, you will have to adapt 
	utils/audio.py
To support other audio drivers.

The Instructions work with all Debian based Linux Distributions, but due to the fact that the audio generation process depends on low latency processing capabilities of the kernel, I recommend using  
[Ubuntu Studio](http://ubuntustudio.org)  
for all experiments in this direction.  
Go here to download Ubuntu Studio:  
[ubuntustudio.org/download/](http://ubuntustudio.org/download/)

All the following instructions imply that you have installed a working version of Ubuntu Studio or a Debian based Linux distribution with a _lowlatency_ kernel and [ALSA](http://www.alsa.org).

## Dependencies
These dependencies are essential for generating sound using the code in this repository:
* Python 2.7
* Theano
* keras
* pyalsaaudio

Optional, if you want to train models using your GPU(s):
* CUDA

For easy installation of the dependencies:
* pip
		
		sudo apt-get install python python-pip python-dev build-essential 
		sudo pip install --upgrade pip 

If you want to train models using your own datasets, GPU acceleration is highly recommended.

### Theano
To install the most recent version of Theano, simply issue this command in the Terminal:
	
	pip install git+git://github.com/Theano/Theano.git

To install Theano with GPU support using CUDA, please refer to this guide:  
[Install Theano - Theano 0.8.2 # Using the GPU](http://deeplearning.net/software/theano/install.html#using-the-gpu)

### Keras
Install the most recent version of Keras by issuing the following command in the Terminal:

	pip install git+git://github.com/fchollet/keras.git
If you run into issues, consult the documentation:  
[keras.io # Installation](https://keras.io/#installation)  
It is very well written.

### pyalsaaudio
Run this to install pyalsaaudio:

	pip install pyalsaaudio
If you run into issues concerning this package, try installing the debian package:

	sudo apt-get install python-pyalsaaudio

## Getting the sources
Go to the directory you want boii to be in, for example:

	cd ~/
	mkdir ai
	cd ai
Make sure you have installed git:

	sudo apt-get install git

Run the `git clone` command to `clone the repository into the diretory `./boii`:

	git clone https://github.com/worosom/boii.git

## Usage

Examples:
Assuming you have 
	python generate.py 0 1 2

### Contents

This repository has the following structure:

* `docs/`
	
	contains [worosom.github.io/boii](https://worosom.github.io/boii)
* `utils/`
	* `config/`
		* `audio_config.py`  
			Change the audio configuration in this file.
		* `nn_config.py`  
			This file contains an array of dictionaries for holding configurations of different neural-network implementations, so they can be easily retrieved just by passing an integer to respective funtions.
	* `audio.py`  
		Run `python utils/audio.py` to check if the audio driver is working as expected.
		If you send a signal to your default input on your default soundcard, the same signal (though downsampled) should be sent to the default output of that soundcard.
	* `midi.py`  
		Provides MIDI access to some parameters of the audio generation process.
	* `parse_files.py`  
		Provides utility functions for converting wav files to tensors and back.
		The original code stems from [GitHub MattVitelli/GRUV](https://github.com/MattVitelli/GRUV)
	* `seed_generator.py`  
		Utility functions for generating seeds to send to the model.
		Most of the code in here comes from [GitHub MattVitelli/GRUV](https://github.com/MattVitelli/GRUV)
	* `sequence_generator.py`
* `nn_utils/`
	* `network_utils.py`  
		This file contains methods to compile different kinds of artificial neural networks.
		Currently the only thoroughly tested variant is a Deep LSTM RNN, but hopefully soon we can have a version with the Google WaveNet running.
		I already started implementing the Google WaveNet algorithm, but with no (easy, for me) way to create _recurrent projection layers_ in Keras, this endeavor had to be interrupted prematurely.
	* `residual_blocks.py`  
		Implements the Method needed to create a _residual network_ in Keras. (for Google's WaveNet)
* `model_weights/`  
	Models are saved here during training.
	The directory structure inside this folder is generated using the configuration stored in `utils/config/nn_config.py`.
	The naming convention I have chosen goes like this:
	
		<hidden dimension size>_<depth>_<dataset name>
* `datasets/`  
	For training, put your .wav files into `datasets/<dataset_name>/wave`
* `convert_wave.py`  
	After changing `nn_params[<CONFIG_#>]['dataset_name']` in `utils/config/nn_config.py` to the name of your dataset, run this to convert the wav files into a dataset file suitable for training.
* `generate.py <CONFIG_#> [<CONFIG_#>, <CONFIG_#>]`  
	You can pass up to three different integers according to the different configurations in `utils/config/nn_config.py`.
	Audio from your input device as cofigured in `utils/config/audio_config.py` will be fed to up to three networks and mixed according to values coming from `utils/midi.py`.
* `train.py <CONFIG_#> [<CURRENT_ITERATION>]`  
	Pass the integer corresponding to the model you want to train as specified in `utils/config/nn_config.py`.
	If you already started training, pass the number of training iterations already run as an argument.

### Running pretrained models
If the required dependencies have been installed correctly and 
To run `boii` in the default configuration, run the following command while at the root of the source tree:

	python generate.py 0 1 2

This program reads audio from the default input of the default soundcard as defined in the Pulse configuration, and writes processed audio to the default output of said soundcard.  
That's it.  
If you want to use MIDI to control this process, have a look at `utils/midi.py` and adapt the CC values to your MIDI hardware.  
The default configuration is set up for a Roland A49, with the following cc channels:

	a49_cc = {
		'cc1' : 74,
		'cc2' : 71,
		'mod' : 1,
		'vol' : 7
	}

### Training models yourself
To train a model yourself, you will need to tinker with the configuration.  
Open `utils/config/nn_config.py`. This file contains an array of dictionaries for holding configurations of different neural-network implementations, so they can be easily retrieved just by passing an integer to respective funtions.

To add a new configuration, choose one you like, e.g. **#3**, and add the following line above `return nn_params`:

	nn_params.append(nn_params[3].copy())

This copies the dict containing the configuration of Network 3 into a new configuration.

Next, you need to decide on a name for your dataset. The models that are included in this repository have been trained on about 10h of solo piano music composed by Frederic Chopin, so the default `dataset_name` is `Chopin`. Change the dataset_name in your configuration to the one of your dataset by adding the following line:

	nn_params[4]['dataset_name'] = 'YOUR NAME HERE'

You may also need to adjust the `batch_size` according to your hardware. If you encounter crashes during training, try lowering this value.  
Now you need to copy your sound files to the folder determined by your `dataset_name`:
	
	datasets/YOUR NAME HERE/wave/

Currently the only format boii supports is 16bit 8khz mono .wav files.

To convert the .wav files into tensors for training, we run the `convert.py` script:

	python convert.py

This will take a while, but when it is done, you should be ready for training!
Issue the following command to generate your model and start training:
	
	python train.py 4 0

The first command line argument `4` tells the script which configuration to load, while the `0` indicates how many iterations you have trained for before.

That's all! The `train.py` script will save a png of the model graph, the model file and training weight checkpoints to a single unique folder under `model_weights/`.

## Addendum

The code in this repository was inspired and resulted from    
[GitHub MattVitelli/GRUV](https://github.com/MattVitelli/GRUV)

Here is a curated List of resources dedicated to recurrent neural networks:  
https://github.com/kjw0612/awesome-rnn

Here is some really good information on RNN training and gradient descent:  
http://sebastianruder.com/optimizing-gradient-descent/

Example sound sample by user NLM from http://freesound.org  
https://www.freesound.org/people/NLM/sounds/178316/
