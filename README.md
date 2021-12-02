# DL project: Voice Cloning

Most of the codes are adopted from https://github.com/CorentinJ/Real-Time-Voice-Cloning.

The goal of this project is to replicate the paper "Neural Voice Cloning with a Few Samples".


There are two main models: 

1. Speaker verification model to determine whether a test audio comes from the same speaker as the enrollment audios.
2. Speaker encoder to quickly learn the speaker embedding from the few audios.

## Training Data

There are two data sources for the speech related tasks: 1. LibriSpeech, 2. VCTK.


## Speaker verification model

### Model architecture

### Data

LibriSpeech. 

Need to convert the audios to its Mel-specturam. 



## Training infrastructure

I compared AWS and GCP. Maybe because I'm more familar with Google product, I find that GCP is way more intuitive than AWS. GCP provides a $300.00 free credits. I will use a CPU machine (cannot afford a GPU machine). Use Ubuntu (not Debian), and Google's original machine (not DL images from some marketplace). It's much easier to install python related packages and system updates this way. 
