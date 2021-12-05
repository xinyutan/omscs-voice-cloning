# DL project: Voice Cloning

Most of the codes are adopted from https://github.com/CorentinJ/Real-Time-Voice-Cloning.

The goal of this project is to replicate the paper "Neural Voice Cloning with a Few Samples".


There are two main models: 

1. Speaker verification model to determine whether a test audio comes from the same speaker as the enrollment audios.
2. Speaker encoder to quickly learn the speaker embedding from the few audios.

## Train the Model

```
python main.py \
--train_dataset_path '/Users/xinyutan/Documents/SV2TTS/encoder/'\
--dev_dataset_path '/Users/xinyutan/Documents/SV2TTS/encoder/' \
--saved_models_dir './saved_models' \
--num_epochs 1000 --save_every 100 --print_every 100
```


## Training Data

There are two raw data sources for the speech related tasks: 1. LibriSpeech, 2. VCTK.

1. `wget https://us.openslr.org/resources/12/train-clean-360.tar.gz` to get the raw data.
2. Run `python3 encoder_preprocess.py '{dataset_root}'` from [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). `{dataset_root}` is the root directory of the downloaded dataset. This step is to clean and transform the raw audio data into mel-spectogram (2D data).
3. Step 2 will put the preprocessd data into `dataset_root/SV2TTS/encoder/`, which should be the `train_dataset_path` in above command.


## Speaker verification model

### Model architecture

### Data

LibriSpeech. 

Need to convert the audios to its Mel-spectogram. 



## Training infrastructure

I compared AWS and GCP. Maybe because I'm more familar with Google product, I find that GCP is way more intuitive than AWS. GCP provides a $300.00 free credits. I will use a CPU machine (cannot afford a GPU machine). Use Ubuntu (not Debian), and Google's original machine (not DL images from some marketplace). It's much easier to install python related packages and system updates this way. 
