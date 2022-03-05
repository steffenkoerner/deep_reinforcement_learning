
from networks.q_network import QNetwork
from memory.replay_buffer import ReplayBuffer

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DQNAgent():

    def __init__(self, config):
        self.seed = random.seed(config.seed)
        self.iter = 0
        self.config = config

        # Q-Network
        self.qnetwork_local = QNetwork(config, self.seed).to(device)
        self.qnetwork_target = QNetwork(config, self.seed).to(device)
        self.set_target_weights_to_local()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.learning_rate)

        self.memory = ReplayBuffer(config.replay_buffer_size, self.config.batch_size, self.seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn()

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.config.output_layer_size))

    def learn(self):
        if len(self.memory) < self.config.batch_size:
            return

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.iter += 1
        self.config.logger.add_scalar("loss", loss, self.iter)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Updates the parameters of the target network  """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def set_target_weights_to_local(self):
        self.soft_update(self.qnetwork_local, self.qnetwork_target,1.0)

    def load_weights(self, path_to_stored_weights):
        self.qnetwork_local.load_state_dict(torch.load(path_to_stored_weights))
        self.set_target_weights_to_local()