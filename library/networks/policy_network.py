import torch.nn as nn
import torch
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        # torch.manual_seed(config.seed)
        self.model = config.model


        # self.fc1 = nn.Linear(s_size, h_size)
        # self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, state):
        return self.model(state)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return F.softmax(x, dim=1)
    
