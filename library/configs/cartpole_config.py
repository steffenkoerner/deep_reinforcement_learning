from library.algorithms.reinforce import reinforce
import torch.nn as nn
import gym
from library.agent.reinforce_agent import ReinforceAgent
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


class Config:
  def __init__(self):

    self.gym = gym.make('CartPole-v1')
    self.seed = 0
    self.logger = SummaryWriter()

    self.gamma = 1.0
    self.learning_rate = 1e-2
    
    self.number_episodes = 20000
    
    self.save_each_return_step = 20
    self.stop_return = 190
    self.episode_length = 5000

    self.input_layer_size = 4 
    self.output_layer_size = 2
    self.model = nn.Sequential(
          nn.Linear(self.input_layer_size,16),
          nn.ReLU(),
          nn.Linear(16,self.output_layer_size),
          nn.Softmax()
        )

    self.agent = ReinforceAgent(self)
    self.path_to_stored_weights = '/home/steffen/workspace/deep_reinforcement_learning/library/examples/cartpole_checkpoint.pth'
    self.algorithm = reinforce