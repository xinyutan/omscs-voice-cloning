from pathlib import Path
from models.train import *
from data.dataset import *

if __name__ == '__main__':
    
    train_dataset = SpeakerVerifierDataset(Path('/home/tanxy_nju/omscs-voice-cloning/SV2TTS/encoder/'))
    num_epochs = 100_000_000
    model_dir = Path('saved_models')
    train('120421', train_dataset, train_dataset, \
        num_epochs, model_dir, 2000, 2000)