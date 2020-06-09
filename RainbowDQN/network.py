from math import sqrt
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import RainbowDQN as src

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, std_init, bias=True):
        super().__init__(in_features, out_features, bias)

        self.noise_weight = nn.Parameter(Tensor(out_features, in_features))
        self.noise_weight.data.fill_(std_init / sqrt(in_features))
        
        if bias:
            self.noise_bias = nn.Parameter(Tensor(out_features))
            self.noise_bias.data.fill_(std_init / sqrt(out_features))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('weps', torch.zeros(in_features))
        self.register_buffer('beps', torch.zeros(out_features))

    def forward(self, x):
        if self.training:
            epsin = self.get_noise(self.in_features)
            epsout = self.get_noise(self.out_features)
            self.weps.copy_(epsout.ger(epsin))
            self.beps.copy_(self.get_noise(self.out_features))

            return super().forward(x) + F.linear(x, self.noise_weight * self.weps, self.noise_bias * self.beps)
        else:
            return super().forward(x)

    @torch.jit.script
    def get_noise(self, size: int) -> Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt_()


class RainbowNet(nn.Module):
    def __init__(self, config: 'src.RainbowConfig'):
        super().__init__()

        self.config = config

        # Build network applied before separated into value and advantage streams
        if len(config.pre_stream_hidden_layer_sizes) > 0:
            seq = [NoisyLinear(config.state_dim, config.pre_stream_hidden_layer_sizes[0], config.std_init), nn.ReLU(inplace=True)]
            for i in range(len(config.pre_stream_hidden_layer_sizes) - 1):
                seq.extend([NoisyLinear(config.pre_stream_hidden_layer_sizes[i], config.pre_stream_hidden_layer_sizes[i+1], config.std_init), nn.ReLU(inplace=True)])
            self.pre_stream = nn.Sequential(*seq)
            pre_outdim = config.pre_stream_hidden_layer_sizes[-1]
        else:
            pre_outdim = config.state_dim
            self.pre_stream = None

        # Build value stream
        seq = [NoisyLinear(pre_outdim, config.value_stream_hidden_layer_sizes[0], config.std_init), nn.ReLU(inplace=True)]
        for i in range(len(config.value_stream_hidden_layer_sizes) - 1):
            seq.extend([NoisyLinear(config.value_stream_hidden_layer_sizes[i], config.value_stream_hidden_layer_sizes[i+1], config.std_init), nn.ReLU(inplace=True)])
        seq.append(NoisyLinear(config.value_stream_hidden_layer_sizes[-1], config.no_atoms, config.std_init))
        self.val_stream = nn.Sequential(*seq)

        # Build advantage stream
        seq = [NoisyLinear(pre_outdim, config.advantage_stream_hidden_layer_sizes[0], config.std_init), nn.ReLU(inplace=True)]
        for i in range(len(config.advantage_stream_hidden_layer_sizes) - 1):
            seq.extend([NoisyLinear(config.advantage_stream_hidden_layer_sizes[i], config.advantage_stream_hidden_layer_sizes[i+1], config.std_init), nn.ReLU(inplace=True)])
        seq.append(NoisyLinear(config.advantage_stream_hidden_layer_sizes[-1], config.no_atoms * config.action_dim, config.std_init))
        self.adv_stream = nn.Sequential(*seq)

    def forward(self, x):
        
        if self.pre_stream is not None:
            x = self.pre_stream(x)

        v = self.val_stream(x).unsqueeze_(1)
        a = self.adv_stream(x).view(-1, self.config.action_dim, self.config.no_atoms)
        abar = a.mean(dim=1, keepdim=True)

        return torch.softmax(
            v + a - abar, dim=2
        )