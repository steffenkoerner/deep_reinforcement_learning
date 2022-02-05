import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

    def __init__(self, actor_input_size, actor_output_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_output_size = actor_output_size
        self.layers = nn.Sequential(
            nn.Linear(actor_input_size, fc1_units),
            # nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            # nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, actor_output_size),
            nn.Tanh()
        )
        self.layers.apply(self.init_weights)
        self.to(device)
        
    def init_weights(self,layer):
        if type(layer) == nn.Linear:
            if layer.out_features == self.actor_output_size:
                layer.weight.data.uniform_(-3e-3, 3e-3)
            else:
                 layer.weight.data.uniform_(*hidden_init(layer))

    def forward(self, state):
        return self.layers(state) 


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+ action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
