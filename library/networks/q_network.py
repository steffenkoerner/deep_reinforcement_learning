import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QNetwork(nn.Module):
    def __init__(self, config, seed):
        super(QNetwork, self).__init__()
        torch.manual_seed(config.seed)
        self.model = config.model
        self.init_weights((self.model))

    def init_weights(self,layer):
        if type(layer) == nn.Linear:
            if layer.out_features == self.actor_output_size:
                layer.weight.data.uniform_(-3e-3, 3e-3)
            else:
                 layer.weight.data.uniform_(*hidden_init(layer))

    

    def forward(self, state):
        return self.model(state)