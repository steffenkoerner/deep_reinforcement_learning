import torch.nn as nn
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
class Config:
  def __init__(self):

    self.seed = 0
    self.logger = SummaryWriter()

    self.replay_buffer_size = int(1e5)
    self.batch_size = 64
    self.gamma = 0.99
    self.tau = 1e-3
    self.learning_rate = 1e-3
    
    self.number_episodes = 2000
    self.eps_start = 0.7
    self.eps_end = 0.01
    self.eps_decay = 0.995
    
    self.stop_return = 13
    self.save_each_return_step = 0.5
    self.episode_length = 2000

    #TODO Move config for Neural Network to here
    self.input_layer_size = 105
    self.output_layer_size = 7
    self.model = nn.Sequential(
          nn.Linear(self.input_layer_size,300),
          nn.ReLU(),
          nn.Linear(300,200),
          nn.ReLU(),
          nn.Linear(200, self.output_layer_size),
        )
