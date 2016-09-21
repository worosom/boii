# boii
You have just a project for generating real time audio using artificial neural networks.
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

The Instructions work with all Debian based Linux Distributions, but due to the fact that the audio generation process depends on low latency processing capabilities I recommend using  
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

Run the `git clone` command to clone the repository into the diretory `./boii`:

	git clone https://github.com/worosom/boii.git

## Contents

This repository has the following structure:

	docs/
	utils/

	nn_utils/
	model_weights/
	datasets/

## Usage

### Running pretrained models

### Training models yourself

## Addendum
Here is a curated List of resources dedicated to recurrent neural networks:
https://github.com/kjw0612/awesome-rnn
http://sebastianruder.com/optimizing-gradient-descent/

Example sound sample by user NLM from http://freesound.org
https://www.freesound.org/people/NLM/sounds/178316/
