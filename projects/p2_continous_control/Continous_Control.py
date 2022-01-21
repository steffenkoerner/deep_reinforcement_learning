from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReplayBuffer import ReplayBuffer, GaussianNoise

LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
TAU = 1e-3              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Needs to be 1 as this is the max(Q(s,a)) that is learned
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(x)

class DDPGNetwork():
    def __init__(self, state_size, action_size, seed):
        self.actor_network = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic_network = CriticNetwork(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)
        

    def actor(self, state):
        return self.actor_network(state)
    
    def critic(self, states,actions):
        return self.critic_network(torch.cat((states, actions), 1))


class DDPGAgent():
    def __init__(self, state_size, action_size, seed, warmup = 100):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_lowest_value = -1
        self.action_highest_value = 1
        self.warump = warmup
        self.gaussian_noise = GaussianNoise(size=action_size, std_start=1, std_end=0.1,steps=100) 
        #self.state = None # Does it make sense to have this variable and if its none than initalise the network correspondingly or directly here

        # DDPG-Network
        self.local_network = DDPGNetwork(state_size, action_size, seed)
        self.target_network = DDPGNetwork(state_size, action_size, seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size=action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)

    def save_experience_in_replay_buffer(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def get_action_of_target_policy_for(self,state):
        pass # same as the below one but using the target network

    def get_acion_per_current_policy_for(self, state , number_episode):
        # TODO: Does it make sense to normalise the input layer as in the paper

        if number_episode < self.warump:
            actions = np.random.randn(action_size) 
            actions = np.clip(actions, self.action_lowest_value, self.action_highest_value)   
        else:       
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            actions = self.local_network.actor(state)
            actions = actions.detach().numpy()
            noise = self.gaussian_noise()
            actions = np.clip(actions + noise, self.action_lowest_value, self.action_highest_value)
        return actions


    def learn(self, gamma):
        """Update parameters.
        """
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

            y = rewards + (1 - dones) * gamma * self.target_network.critic(next_states, self.target_network.actor(next_states))
            critic_loss = F.mse_loss(y,self.local_network.critic(states,actions).detach())
            self.local_network.critic_network.zero_grad()
            critic_loss.backward()
            self.local_network.critic_optimizer.step()

            # TODO_Calculate_Actor_Critic_Loss
            actor_loss = self.local_network.critic(states.detach(),self.local_network.actor(states))
            actor_loss = actor_loss.mean()
            self.local_network.actor_network.zero_grad()
            actor_loss.backward()
            self.local_network.actor_optimizer.step()

            self.soft_update(self.local_network, self.target_network, TAU) 
      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters for actor and critic of target network.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.target_network.actor_network.parameters(), self.local_network.actor_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        for target_param, local_param in zip(self.target_network.critic_network.parameters(), self.local_network.critic_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def plot_scores(scores, number):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    figure_name = "scores_" + str(number) +".png"
    plt.savefig(figure_name)
    #plt.show()
    

def ddpg(env, agent, n_episodes=2000, max_t=1000, gamma=0.9):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    max_score_value = 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0] 
        score = 0
        for t in range(max_t):
            action = agent.get_acion_per_current_policy_for(state, i_episode)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.save_experience_in_replay_buffer(state, action, reward, next_state, done)
            state = next_state
            score += reward
            agent.learn(gamma)
            if done:
                break 
        
        scores_window.append(score)
        scores.append(score)
        if i_episode % 200 == 0:
            plot_scores(scores,i_episode)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > max_score_value + 3:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.local_network.actor_network.state_dict(), 'intermediate_weight_actor.pth')
            torch.save(agent.local_network.critic_network.state_dict(), 'intermediate_weight_critic.pth')
            max_score_value = np.mean(scores_window)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.local_network.actor_network.state_dict(), 'final_weight_actor.pth')
            torch.save(agent.local_network.critic_network.state_dict(), 'final_weight_critic.pth')
            break
    return scores


def take_random_action():
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

if __name__ == '__main__':
    env = UnityEnvironment(file_name='/home/steffen/workspace/deep_reinforcement_learning/projects/p2_continous_control/Reacher_Linux/Reacher.x86_64')
    #take_random_action()
    brain = env.brains[env.brain_names[0]]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    agent = DDPGAgent(state_size= state_size, action_size = action_size, seed = 0 , warmup = 50)
    scores = ddpg(env,agent, n_episodes=5000)
    plot_scores(scores,0)
