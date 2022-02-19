import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, config, seed):
        super(QNetwork, self).__init__()
        torch.manual_seed(config.seed)
        self.model = config.model

    def forward(self, state):
        return self.model(state)