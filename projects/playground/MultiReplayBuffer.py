import torch
import numpy as np
import random
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        tmp_states =np.array([e.state for e in experiences if e is not None])
        states = torch.from_numpy(np.array([tmp_states[:,0,:],tmp_states[:,1,:]])).float().to(device)


        tmp_rewards =np.array([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(np.array([tmp_rewards[:,0],tmp_rewards[:,1]])).float().to(device)

        tmp_next_states =np.array([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(np.array([tmp_next_states[:,0,:],tmp_next_states[:,1,:]])).float().to(device)

        tmp_dones =np.array([e.done for e in experiences if e is not None])
        dones = torch.from_numpy(np.array([tmp_dones[:,0],tmp_dones[:,1]]).astype(np.uint8)).float().to(device)

        # blub = [[],[]]
        # for e in experiences:
        #     blub[0].append(e.action[0])
        #     blub[1].append(e.action[1])
        # blus = np.array(blub)
        # g = torch.from_numpy(blus).float().to(device)
        tmp_actions =np.array([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(np.array([tmp_actions[:,0,:],tmp_actions[:,1,:]])).float().to(device)

        # actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        # rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class GaussianNoise:
    def __init__(self, size, std_start, std_end=None, steps=None):
        if std_end is None:
            std_end = std_start
            steps = 1
        self.inc = (std_end - std_start) / float(steps)
        self.current_std = std_start
        self.std_end = std_end
        self.std_start = std_start
        self.size = size

    def __call__(self):
        self.current_std = np.clip(self.current_std - self.inc, self.std_end, self.std_start)
        return np.random.normal(0,self.current_std, self.size)



class GaussianProcess():
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std()



class LinearIncrements:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()