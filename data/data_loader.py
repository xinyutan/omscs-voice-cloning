import torch
import numpy as np
import data.data_params as dp


class DataLoader:
    def __init__(self, dataset, batch_size, normalization=False, trial_num=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.trial_num = trial_num
        self.normalization = normalization

    def _create_batch(self):
        labels = []
        audios = [[] for _ in range(dp.num_enrollment_audios + 1)]
        for i in range(self.batch_size):
            x, y = self.dataset[i]
            labels.append(y)
            for j in range(dp.num_enrollment_audios + 1):
                audios[j].append(x[j])

        X, y = [torch.stack(x, axis=0).unsqueeze(axis=1) for x in audios], \
            torch.tensor(labels, dtype=torch.float)
        
        if not self.normalization:
            return X, y

        all_data = torch.cat(X, dim=1)
        mean, std = all_data.mean(), all_data.std()
        return [(x - mean) / std for x in X], y
        

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

