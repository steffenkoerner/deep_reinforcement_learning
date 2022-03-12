from library.networks.policy_network import PolicyNetwork

import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReinforceAgent():
    def __init__(self, config):
        self.seed = random.seed(config.seed)
        self.iter = 0
        self.config = config

        # Q-Network
        self.policy_network = PolicyNetwork(config)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def learn(self,return_trajectory,log_probs):
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * return_trajectory)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def load_weights(self, config_path):
         self.policy_network.load_state_dict(torch.load(config_path))
        