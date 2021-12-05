import torch
import numpy as np
import data.data_params as dp


class DataLoader:
    def __init__(self, dataset, batch_size, trial_num=5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.trial_num = trial_num

    def _create_batch(self):
        labels = []
        audios = [[] for _ in range(dp.num_enrollment_audios)]
        for i in range(self.batch_size):
            x, y = self.dataset[i]
            labels.append(y)
            for j in range(dp.num_enrollment_audios):
                audios[j].append(x[j])

        return [torch.stack(x, axis=0).unsqueeze(axis=1) for x in audios], \
            torch.tensor(labels, dtype=torch.float)

    def __next__(self):
        success = False
        fail_times = 0
        X, y = None, None
        while not success and fail_times < self.trial_num:
            try:
                X, y = self._create_batch()
                success = True
            except:
                fail_times += 1
        if not success:
            raise RuntimeError("dataset seems to be empty.")
        return X, y

