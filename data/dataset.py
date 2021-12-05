import torch
from torch.utils.data import Dataset
import numpy as np
from data.random_cycler import RandomCycler
import data.data_params as dp


class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath

    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)


class Speaker:
    def __init__(self, root):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.speaker_cycler = None

    def _load_utterances(self):
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.split(",") for l in sources_file]
        sources = {frames_fname: wave_fpath for frames_fname,
                   wave_fpath in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w)
                           for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count, n_frames):
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)
        a = [(u,) + u.random_partial(n_frames) for u in utterances]
        return a


class SpeakerVerifierDataset(Dataset):
    def __init__(self, dataset_root):
        self.root = dataset_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception(f"No speaker found. {self.root.path}" +
                            f"does not contain the speaker directories.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.num_speakers = len(self.speakers)

    def __len__(self):
        return int(1e10)

    def __getitem__(self, idx):
        label = np.random.binomial(1, dp.positive_probability)
        tid = np.random.randint(0, self.num_speakers, 1)
        eid = tid
        if label == 0:
            # from the different speaker
            while eid[0] == tid[0]:
                eid = np.random.randint(0, self.num_speakers, 1)
        eid = eid[0]
        tid = tid[0]

        if eid == tid:
            utterances = self.speakers[eid].random_partial(
                dp.num_enrollment_audios + 1, dp.partials_n_frames)
            return [torch.from_numpy(u) for _, u, _ in utterances], label

        e_utterances = self.speakers[eid].random_partial(
            dp.num_enrollment_audios, dp.partials_n_frames)

        t_utterances = self.speakers[eid].random_partial(
            1, dp.partials_n_frames)

        return [torch.from_numpy(u) for _, u, _ in e_utterances] + \
            [torch.from_numpy(u) for _, u, _ in t_utterances], label
