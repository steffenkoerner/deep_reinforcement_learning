from algorithms.dqn_unity import dqn_unity
import torch.nn as nn
from agent.dqn_agent import DQNAgent
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
class Config:
  def __init__(self):

    self.env_path = "/home/steffen/workspace/Environments/Basic/Basic.x86_64"
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
    self.eps_decay = 0.9
    
    self.stop_return = 13
    self.save_each_return_step = 0.1
    self.episode_length = 5000

    #TODO Move config for Neural Network to here
    self.input_layer_size = 20
    self.output_layer_size = 3
    self.model = nn.Sequential(
          nn.Linear(self.input_layer_size,300),
          nn.ReLU(),
          nn.Linear(300,200),
          nn.ReLU(),
          nn.Linear(200, self.output_layer_size),
        )

    self.agent = DQNAgent(self)
    self.path_to_stored_weights = ""
    self.algorithm = dqn_unity
