from pathlib import Path
import argparse
from models.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default="test", help="Identifier for this run.")
    parser.add_argument('--train_dataset_path', type=Path,\
        default=Path('/home/tanxy_nju/omscs-voice-cloning/SV2TTS/encoder/'), \
        help="The directory to the preprocessed data for training.")
    parser.add_argument('--dev_dataset_path', type=Path,\
        help="The directory to the preprocessed data for training.")
    parser.add_argument('--saved_models_dir', type=Path,\
        help="Checkpoints will be saved here.")
    parser.add_argument('--num_epochs', type=int, default=100_000_000)
    parser.add_argument('--save_every', type=int, default=2000, \
        help="Save the checkpoint every this many steps.")
    parser.add_argument('--print_every', type=int, default=2000, \
        help="Print the evaluation on the dev dataset every this many steps.")

    opts = parser.parse_args()
    opts=vars(opts)
    train(**opts)