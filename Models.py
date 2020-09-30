import numpy as np
import torch
from torch import nn, Tensor


class STA(nn.Module):
    def __init__(self, L : int=1, K : int=1000, B : int=1):
        super(STA, self).__init__()
        self.L = L
        self.K = K
        self.B = B

        self.W_1x = torch.nn.Parameter(torch.normal(0, 1, (L, L)))
        self.W_1x.requires_grad = True

        self.W_1s = torch.nn.Parameter(torch.normal(0, 1, (L, L)))
        self.W_1s.requires_grad = True

        self.W_2x = torch.nn.Parameter(torch.normal(0, 1, (K, K)))
        self.W_2x.requires_grad = True

        self.W_2s = torch.nn.Parameter(torch.normal(0, 1, (K, K)))
        self.W_2s.requires_grad = True

        self.conv_layer_x = nn.Conv2d(in_channels=2*self.L, out_channels=2*self.L, kernel_size=3, stride=1, padding=1)
        self.conv_layer_s = nn.Conv2d(in_channels=2*self.L, out_channels=2*self.L, kernel_size=3, stride=1, padding=1)

        self.positional_encoding = PositionalEncoding(self.B, self.K)

    def forward(self, X : Tensor, S : Tensor):
        batch_size = X.shape[0]

        assert(X.shape == (batch_size, self.L, self.K * self.B))
        assert(S.shape == (batch_size, self.L, self.K * self.B))


        Xe = S @ self.W_1x
        Se = X @ self.W_1s

        assert(Xe.shape == (batch_size, self.L, self.K * self.B))
        assert(Se.shape == (batch_size, self.L, self.K * self.B))

        Ax = torch.softmax(torch.bmm(Xe.transpose(1,2), Xe), dim=2)
        As = torch.softmax(torch.bmm(Se.transpose(1,2), Se), dim=2)

        assert(Ax.shape == (batch_size, self.K * self.B, self.K * self.B))
        assert(As.shape == (batch_size, self.K * self.B, self.K * self.B))

        Xa = torch.bmm(Xe, Ax)
        Sa = torch.bmm(Se, As)

        assert(Xa.shape == (batch_size, self.L, self.K * self.B))
        assert(Sa.shape == (batch_size, self.L, self.K * self.B))

        Xa = Xa.view(batch_size, self.L, self.K, self.B)
        Sa = Sa.view(batch_size, self.L, self.K, self.B)

        Xa = Xa.permute(0, 1, 3, 2)
        Xa = Xa.reshape(batch_size, self.L * self.B, self.K)
        Mx = Xa @ self.W_2x
        assert(Mx.shape == (batch_size, self.K, self.L * self.B))
        Mx = Mx.reshape(batch_size, self.L, self.B, self.K)
        Mx = Mx.permute(0, 1, 3, 2)

        Sa = Sa.permute(0, 1, 3, 2)
        Sa = Sa.reshape(batch_size, self.L * self.B, self.K)
        Ms = Sa @ self.W_2s
        assert(Ms.shape == (batch_size, self.K, self.L * self.B))
        Ms = Ms.reshape(batch_size, self.L, self.B, self.K)
        Ms = Ms.permute(0, 1, 3, 2)


        Mx = Mx.view(batch_size, self.L, self.K * self.B)
        Ms = Ms.view(batch_size, self.L, self.K * self.B)

        # Apply the attention masks
        assert(Mx.shape == X.shape)
        assert(Ms.shape == S.shape)
        Xm = Mx * X
        Sm = Ms * S

        X_hat = torch.cat((Xm, X), dim=1)
        S_hat = torch.cat((Sm, S), dim=1)

        assert(X_hat.shape == (batch_size, 2*self.L, self.K * self.B))
        assert(S_hat.shape == (batch_size, 2*self.L, self.K * self.B))

        X_hat = X_hat.view(2*self.L, self.K, self.B)
        S_hat = S_hat.view(2*self.L, self.K, self.B)

        # add positional encoding
        S_hat = self.positional_encoding(S_hat)

        # add convolution
        X_e_hat = self.conv_layer_x(X_hat)
        S_e_hat = self.conv_layer_s(S_hat)

        return torch.cat((X_e_hat, S_e_hat), dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        # n_position - B
        # d_hid - K
        # We need to add this 2L times

        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

SIGNAL_LEN = 1000
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(SIGNAL_LEN, SIGNAL_LEN), nn.Linear(SIGNAL_LEN, SIGNAL_LEN))

    def forward(self, x:Tensor):
        assert(x.shape == (1, SIGNAL_LEN))
        return self.linear(x)

class EncoderSTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(SIGNAL_LEN * 4, SIGNAL_LEN * 4), nn.Linear(SIGNAL_LEN * 4, SIGNAL_LEN))

    def forward(self, x:Tensor):
        assert(x.shape == (2, 2, SIGNAL_LEN, 1))
        x = x.view(1, SIGNAL_LEN * 4)
        return self.linear(x)