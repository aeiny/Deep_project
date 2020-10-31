import numpy as np
import torch
from torch import nn, Tensor
import math

class STA(nn.Module):
    def __init__(self, L: int = 1, K: int = 1000, B: int = 1):
        super(STA, self).__init__()
        self.L = L
        self.K = K
        self.B = B

        self.W_1x = torch.nn.Parameter(torch.normal(0, 1, (L, L)), requires_grad=True)

        self.W_1s = torch.nn.Parameter(torch.normal(0, 1, (L, L)), requires_grad=True)

        self.W_2x = torch.nn.Parameter(torch.normal(0, 1, (K, K)), requires_grad=True)

        self.W_2s = torch.nn.Parameter(torch.normal(0, 1, (K, K)), requires_grad=True)

        self.conv_layer_x = nn.Conv2d(in_channels=2 * self.L, out_channels=2 * self.L, kernel_size=3, stride=1,
                                      padding=1)
        self.conv_layer_s = nn.Conv2d(in_channels=2 * self.L, out_channels=2 * self.L, kernel_size=3, stride=1,
                                      padding=1)

        self.positional_encoding = PositionalEncoding(self.K, self.B)

    def forward(self, X: Tensor, S: Tensor):
        batch_size = X.shape[0]

        assert (X.shape == (batch_size, self.L, self.K * self.B))
        assert (S.shape == (batch_size, self.L, self.K * self.B))

        Xe = (S.transpose(1,2) @ self.W_1x).transpose(1,2)
        Se = (X.transpose(1,2) @ self.W_1s).transpose(1,2)

        assert (Xe.shape == (batch_size, self.L, self.K * self.B))
        assert (Se.shape == (batch_size, self.L, self.K * self.B))

        Ax = torch.softmax(torch.bmm(Xe.transpose(1, 2), Xe), dim=2)
        As = torch.softmax(torch.bmm(Se.transpose(1, 2), Se), dim=2)

        assert (Ax.shape == (batch_size, self.K * self.B, self.K * self.B))
        assert (As.shape == (batch_size, self.K * self.B, self.K * self.B))

        Xa = torch.bmm(Xe, Ax)
        Sa = torch.bmm(Se, As)

        assert (Xa.shape == (batch_size, self.L, self.K * self.B))
        assert (Sa.shape == (batch_size, self.L, self.K * self.B))

        Xa = Xa.view(batch_size, self.L, self.K, self.B)
        Sa = Sa.view(batch_size, self.L, self.K, self.B)

        Xa = Xa.permute(0, 1, 3, 2)
        Xa = Xa.reshape(batch_size, self.L * self.B, self.K)
        Mx = Xa @ self.W_2x
        assert (Mx.shape == (batch_size, self.L * self.B, self.K))
        Mx = Mx.reshape(batch_size, self.L, self.B, self.K)
        Mx = Mx.permute(0, 1, 3, 2)

        Sa = Sa.permute(0, 1, 3, 2)
        Sa = Sa.reshape(batch_size, self.L * self.B, self.K)
        Ms = Sa @ self.W_2s
        assert (Ms.shape == (batch_size, self.L * self.B, self.K))
        Ms = Ms.reshape(batch_size, self.L, self.B, self.K)
        Ms = Ms.permute(0, 1, 3, 2)

        Mx = Mx.view(batch_size, self.L, self.K * self.B)
        Ms = Ms.view(batch_size, self.L, self.K * self.B)

        # Apply the attention masks
        assert (Mx.shape == X.shape)
        assert (Ms.shape == S.shape)
        Xm = Mx * X
        Sm = Ms * S

        X_hat = torch.cat((Xm, X), dim=1)
        S_hat = torch.cat((Sm, S), dim=1)

        assert (X_hat.shape == (batch_size, 2 * self.L, self.K * self.B))
        assert (S_hat.shape == (batch_size, 2 * self.L, self.K * self.B))

        X_hat = X_hat.view(batch_size, 2 * self.L, self.K, self.B)
        S_hat = S_hat.view(batch_size, 2 * self.L, self.K, self.B)

        # add positional encoding
        S_hat = self.positional_encoding(S_hat)

        # add convolution
        X_e_hat = self.conv_layer_x(X_hat)
        S_e_hat = self.conv_layer_s(S_hat)

        return torch.cat((X_e_hat, S_e_hat), dim=1)


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()
#
#         # Not a parameter
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
#
#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         # n_position - B
#         # d_hid - K
#         # We need to add this 2L times
#
#         ''' Sinusoid position encoding table '''
#
#         # TODO: make it with torch instead of numpy
#
#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#
#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#
#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)
#
#     def forward(self, x):
#         return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding(nn.Module):
    """

    """

    def __init__(self, features_dimensionality: int, num_of_beats: int = 60, dropout_rate: float = 0.1):
        """

        :param features_dimensionality:
        :param num_of_beats:
        :param dropout_rate:
        """

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(features_dimensionality, num_of_beats)
        position = torch.arange(0, features_dimensionality, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_of_beats, 2).float() * (-math.log(10000.0) /
                                                                         num_of_beats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x:
        :return:
        """

        tmp = self.pe.squeeze(1).repeat(x.shape[0], x.shape[1], 1, 1)
        x = x + self.pe.squeeze(1).repeat(x.shape[0], x.shape[1], 1, 1)

        return self.dropout(x)


SIGNAL_LEN = 450 # K in research proposal
N_INTERVALS = 1 # B in research proposal
N_CHANNELS = 1 # L in research proposal

class Encoder(nn.Module):

    def __init__(self, nonlinearities: bool = False, gets_spectral_input: bool = False, sta_enabled: bool = False):
        super().__init__()

        if sta_enabled:
            self.input_size = SIGNAL_LEN * 4
            self.sta = STA(N_CHANNELS, SIGNAL_LEN, N_INTERVALS)
        else:
            if gets_spectral_input:
                self.input_size = SIGNAL_LEN * 2
            else:
                self.input_size = SIGNAL_LEN

        layers = [nn.Linear(self.input_size, self.input_size)]

        if nonlinearities:
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.input_size, SIGNAL_LEN))

        self.network = nn.Sequential(*layers)
        self.gets_spectral_input = gets_spectral_input
        self.sta_enabled = sta_enabled

    def forward(self, input: Tensor):
        if self.gets_spectral_input or self.sta_enabled:
            # in this case input = (X,S)
            X, S = input
            assert(X.shape[0] == S.shape[0]) # same number of batches
            assert(X.shape[1:] == (N_CHANNELS, SIGNAL_LEN * N_INTERVALS))
            assert(S.shape[1:] == (N_CHANNELS, SIGNAL_LEN * N_INTERVALS))
            if self.sta_enabled:
                x = self.sta(X, S)
                assert(x.shape == (X.shape[0], 4 * N_CHANNELS, SIGNAL_LEN, N_INTERVALS))
                x = x.view(X.shape[0], 4 * N_CHANNELS * SIGNAL_LEN * N_INTERVALS)
            else:
                x = torch.cat((X, S), dim=1)
                assert(x.shape == (X.shape[0], 2 * N_CHANNELS, SIGNAL_LEN * N_INTERVALS))
                x = x.view(X.shape[0], 2 * N_CHANNELS * SIGNAL_LEN * N_INTERVALS)
        # not spectral, no sta
        else:
            assert (input.shape[1:] == (N_CHANNELS, SIGNAL_LEN * N_INTERVALS))
            X = input
            x = X
            x = x.view(X.shape[0], N_CHANNELS * SIGNAL_LEN * N_INTERVALS)

        return self.network(x).view(X.shape[0], N_CHANNELS, SIGNAL_LEN * N_INTERVALS)
