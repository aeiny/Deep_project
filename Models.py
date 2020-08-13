import torch
from torch import nn, Tensor


class STA(nn.Module):
    def __init__(self, L : int, K : int, B : int):
        super(STA, self).__init__()
        self.L = L
        self.K = K
        self.B = B

        self.W_1x = torch.normal(0, 1, (L, L))
        self.W_1x.requires_grad = True

        self.W_1s = torch.normal(0, 1, (L, L))
        self.W_1s.requires_grad = True

        self.W_2x = torch.normal(0, 1, (K, K))
        self.W_2x.requires_grad = True

        self.W_2s = torch.normal(0, 1, (K, K))
        self.W_2s.requires_grad = True

        self.conv_layer = nn.Conv2d(2*self.L, 2*self.L, 3, stride=1, padding=1)

    def forward(self, S : Tensor, X : Tensor):
        Xe : Tensor = self.W_1x @ S
        Se : Tensor = self.W_1s @ X

        assert(Xe.shape == (self.L, self.K * self.B))
        assert(Se.shape == (self.L, self.K * self.B))

        Ax = torch.softmax(Xe.T @ Xe, dim=0)
        As = torch.softmax(Se.T @ Se, dim=0)

        Xa = Xe @ Ax
        Sa = Se @ As

        assert(Xe.shape == (self.L, self.K * self.B))
        assert(Se.shape == (self.L, self.K * self.B))

        Mx = self.W_2x @ Xa.view(self.K, self.L * self.B)
        Ms = self.W_2x @ Sa.view(self.K, self.L * self.B)

        assert(Mx.shape == (self.K, self.L * self.B))
        assert(Ms.shape == (self.K, self.L * self.B))

        Mx = Mx.view(self.L, self.K * self.B)
        Ms = Ms.view(self.L, self.K * self.B)

        assert(Mx.shape == X.shape)
        assert(Ms.shape == S.shape)
        Xm = Mx * X
        Sm = Ms * X

        X_hat = torch.cat((Xm, X), dim=0)
        S_hat = torch.cat((Sm, S), dim=0)

        assert(X_hat.shape == (2*self.L, self.K * self.B))
        assert(S_hat.shape == (2*self.L, self.K * self.B))

        X_hat = X_hat.view(2*self.L, self.K, self.B)
        S_hat = S_hat.view(2*self.L, self.K, self.B)

        # add positional encoding
        # add convolution

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        # n_position - B
        # d_hid - K]
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