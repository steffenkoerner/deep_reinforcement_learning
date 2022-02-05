import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from buffer import ReplayBuffer
from noise import OUNoise, GaussianNoise
from torch.utils.tensorboard import SummaryWriter



logger = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6) 
BATCH_SIZE = 512       
GAMMA = 0.99          
TAU = 1e-2              
ACTOR_LEARNING_RATE = 1e-3         
CRITIC_LEARNING_RATE = 1e-3     
WEIGHT_DECAY = 0        


class Agent():
    
    def __init__(self, state_size, action_size, random_seed, id):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LEARNING_RATE)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.gaussian_noise = GaussianNoise(size=action_size, std_start=0.4, std_end=0.4,steps=10)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.set_weights_of_target_to_local()
        self.id = id
        self.iter = 0

    def set_weights_of_target_to_local(self):
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn()

    def act(self, states, noise=1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(states).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise * self.gaussian_noise()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
            self.iter += 1

            with torch.no_grad():
                y = rewards + (1 - dones) * GAMMA * self.critic_target(next_states, self.actor_target(next_states))
            critic_loss = F.mse_loss(y , self.critic_local(states,actions))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            actor_loss = self.critic_local(states.detach(), self.actor_local(states))
            actor_loss = -actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            logger.add_scalars('agent%i/losses' % self.id,{'critic_loss': critic_loss, 'actor_loss': actor_loss, },self.iter)
            
            self.update_weights()

    def update_weights(self):
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


