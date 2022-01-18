from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReplayBuffer import ReplayBuffer

LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
TAU = 1e-3              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 64, fc2_units = 64):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 64, fc2_units = 64):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc4 = nn.Softmax(dim=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DDPGNetwork():
    def __init__(self, state_size, action_size, seed):
        self.actor_network = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic_network = CriticNetwork(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)


class DDPGAgent():
    def __init__(self, state_size, action_size, seed):
        """
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_prob_low = 0.05
        self.action_prob_high = 0.95
        #self.state = None # Does it make sense to have this variable and if its none than initalise the network correspondingly or directly here

        # DDPG-Network
        self.network_local = DDPGNetwork(state_size, action_size, seed)
        self.network_target = DDPGNetwork(state_size, action_size, seed)
        

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def save_experience_in_replay_buffer(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def something_else():
        pass
        # # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # if self.t_step == 0:
        #     # If enough samples are available in memory, get random subset and learn
        #     if len(self.memory) > BATCH_SIZE:
        #         experiences = self.memory.sample()
        #         self.learn(experiences, GAMMA)

    def get_acion_per_current_policy_for(self, state):
        """
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = self.network_local.actor_network(state)
        action = action.cpu().detach().numpy()
        #action += self.random_process.sample() #add some random noise
        action = np.clip(action, self.action_prob_low, self.action_prob_high)
        return action
        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #     action_values = self.qnetwork_local(state)
        # self.qnetwork_local.train()

        # # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return np.argmax(action_values.cpu().data.numpy())
        # else:
        #     return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update parameters.
        """
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states 
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)

        # # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU) 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        pass
        # for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        #     target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("scores.png")
    #plt.show()
    

def ddpg(env, agent, n_episodes=2000, max_t=1000):
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
            action = agent.get_acion_per_current_policy_for(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.save_experience_in_replay_buffer(state, action, reward, next_state, done)
            state = next_state
            score += reward
            agent.learn()
            # sample minibatch # Do we want to sample each timestep ???
            # set y_i
            # Update critic by minimizing loss
            # upddate the actor policy by using the sampled policy gradient
            if done:
                break 


        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > max_score_value + 3:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.ddpgnetwork_local.state_dict(), 'checkpoint_intermediate.pth')
            max_score_value = np.mean(scores_window)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.ddpgnetwork_local.state_dict(), 'checkpoint.pth')
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

    agent = DDPGAgent(state_size= state_size, action_size = action_size, seed = 0)
    scores = ddpg(env,agent, n_episodes=1)
    plot_scores(scores)
