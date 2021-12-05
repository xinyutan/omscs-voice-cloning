from torch import nn
import torch

from models.hyperparameters import SpeakerVerifierHyperparameters as hp
import data.data_params as dp


class SpeakerVerifier(nn.Module):
    def __init__(self):
        super(SpeakerVerifier, self).__init__()
        self.conv = nn.Conv2d(
            hp.conv2d_in_channels,
            hp.conv2d_out_channels,
            hp.conv2d_kernel_size,
            hp.conv2d_strides)
        self.conv_bn = nn.BatchNorm2d(hp.conv2d_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hp.dropout_probability)

        conv_output_dim = (dp.mel_n_channels - hp.conv2d_kernel_size[1]) // \
            hp.conv2d_strides[1] + 1
        gru_input_dim = conv_output_dim * hp.conv2d_out_channels
        self.gru = nn.GRU(
            gru_input_dim,
            hp.recurrent_hidden_size,
            batch_first=True,
        )
        self.linear = nn.Linear(
            hp.recurrent_hidden_size,
            hp.fully_connected_size,
        )
        self.weight = nn.Parameter(torch.tensor([10.]))
        self.bias = nn.Parameter(torch.tensor([-5.]))
        self.half_similarity_matrix = nn.Parameter(
            torch.randn(hp.fully_connected_size, hp.fully_connected_size)
        )

    def _compute_audio_embedding(self, x):
        output = self.conv(x)
        output = self.conv_bn(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = output.permute(0, 2, 1, 3)
        N, T, _, _ = output.shape
        output = output.reshape((N, T, -1))
        output, _ = self.gru(output)
        output = torch.mean(output, axis=1)
        output = self.linear(output)
        return output

    def _batch_dot_mul(self, x, y):
        return torch.sum(x * y, dim=1, keepdims=True)

    def forward(self, enrollment_mels, test_mel):
        enrollment_embeddings = []
        for mel in enrollment_mels:
            enrollment_embeddings.append(self._compute_audio_embedding(mel))
        e_emb = torch.mean(
            torch.stack(enrollment_embeddings, axis=2),
            axis=2,
        )
        t_emb = self._compute_audio_embedding(test_mel)

        s = (self.half_similarity_matrix + self.half_similarity_matrix.T) / 2.0

        score = self.weight * self._batch_dot_mul(e_emb, e_emb) - \
            self._batch_dot_mul(e_emb, torch.matmul(t_emb, s)) - \
            self._batch_dot_mul(t_emb, t_emb) + self.bias

        return score
