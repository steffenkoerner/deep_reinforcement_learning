from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import ReplayBuffer, GaussianNoise, OUNoise

LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 128
TAU = 1e-3              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = SummaryWriter()

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Needs to be 1 as this is the max(Q(s,a)) that is learned
        self.fc1.weight.data.uniform_(0.0,1.0)
        self.fc2.weight.data.uniform_(0.0,1.0)
        self.fc3.weight.data.uniform_(0.0,1.0)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc1.weight.data.uniform_(0.0,1.0)
        self.fc2.weight.data.uniform_(0.0,1.0)
        self.fc3.weight.data.uniform_(0.0,1.0)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

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
    def __init__(self, state_size, action_size, seed, warmup, id):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_lowest_value = -1
        self.action_highest_value = 1
        self.warump = warmup
        self.id = id
        self.iter = 0
        #self.gaussian_noise = GaussianNoise(size=action_size, std_start=0.3, std_end=0.01,steps=1000000) 
        self.ou_noise = OUNoise(action_size, scale=1.0 )

        # DDPG-Network
        self.local_network = DDPGNetwork(state_size, action_size, seed)
        self.target_network = DDPGNetwork(state_size, action_size, seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size=action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)   
        
    def get_acion_per_current_policy_for(self, state , number_episode, noise):
        if number_episode < self.warump:
            actions = np.random.randn(self.action_size) 
            actions = np.clip(actions, self.action_lowest_value, self.action_highest_value)   
        else:       
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.local_network.actor_network.eval()
            with torch.no_grad():
                actions = self.local_network.actor(state).cpu().data.numpy()
            self.local_network.actor_network.train()
            noise_applied = noise * self.ou_noise.noise().numpy() # TODO: Investigate noise function
            #noise_applied = self.gaussian_noise()
            actions = np.clip(actions + noise_applied, self.action_lowest_value, self.action_highest_value)
        return actions


    def learn(self, gamma):
        """Update parameters.
        """
        if len(self.memory) > BATCH_SIZE:
            self.iter += 1
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
            with torch.no_grad():
                y = rewards + (1 - dones) * gamma * self.target_network.critic(next_states, self.target_network.actor(next_states))
            critic_loss = F.mse_loss(y , self.local_network.critic(states,actions))
            self.local_network.critic_network.zero_grad()
            critic_loss.backward()
            self.local_network.critic_optimizer.step()

            actor_loss = self.local_network.critic(states.detach(), self.local_network.actor(states))
            actor_loss = -actor_loss.mean()
            self.local_network.actor_network.zero_grad()
            actor_loss.backward()
            self.local_network.actor_optimizer.step()
            logger.add_scalars('agent%i/losses' % self.id,{'critic_loss': critic_loss.cpu().detach().item(), 'actor_loss': actor_loss.cpu().detach().item()},self.iter)

            self.soft_update(self.local_network, self.target_network, TAU) 

    def step(self,state, action, reward, next_state, done, gamma):
        self.memory.add(state, action, reward, next_state, done)
        self.learn(gamma)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters for actor and critic of target network.
        ??_target = ??*??_local + (1 - ??)*??_target

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
    

def ddpg(env, agent1, agent2, n_episodes=2000, max_t=1000, gamma=0.9):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    max_score_value = 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    noise = 2
    noise_decay = 0.9999
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations
        score = [0,0]
        while True:
            noise *= noise_decay
            action1 = agent1.get_acion_per_current_policy_for(state[0], i_episode, noise)
            action2 = agent2.get_acion_per_current_policy_for(state[1], i_episode, noise)
            action = np.vstack((action1,action2))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent1.step(state[0], action1, reward[0], next_state[0], done[0], gamma)
            agent2.step(state[1], action2, reward[1], next_state[1], done[1], gamma)
            state = next_state
            score += reward
            if reward[0] !=0 or reward[1] != 0:
                blub = 1

            if done[0] or done[1]:
                break 
        max_score = np.max(score)
        logger.add_scalar("score/train", max_score, i_episode)
        scores_window.append(max_score)
        scores.append(max_score)
        if i_episode % 200 == 0:
            #plot_scores(scores,i_episode)
            pass

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > max_score_value + 0.1:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent1.local_network.actor_network.state_dict(), 'intermediate_weight_actor1.pth')
            torch.save(agent1.local_network.critic_network.state_dict(), 'intermediate_weight_critic1.pth')
            torch.save(agent2.local_network.actor_network.state_dict(), 'intermediate_weight_actor2.pth')
            torch.save(agent2.local_network.critic_network.state_dict(), 'intermediate_weight_critic2.pth')
            max_score_value = np.mean(scores_window)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent1.local_network.actor_network.state_dict(), 'final_weight_actor1.pth')
            torch.save(agent1.local_network.critic_network.state_dict(), 'final_weight_critic1.pth')
            torch.save(agent2.local_network.actor_network.state_dict(), 'final_weight_actor2.pth')
            torch.save(agent2.local_network.critic_network.state_dict(), 'final_weight_critic2.pth')
            break
    return scores


if __name__ == '__main__':
    
    env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/projects/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent1 = DDPGAgent(state_size= state_size, action_size = action_size, seed = 0 , warmup = 200, id=1)
    agent2 = DDPGAgent(state_size= state_size, action_size = action_size, seed = 0 , warmup = 200, id=2)
    scores = ddpg(env,agent1, agent2, n_episodes=5000000000)

    logger.flush()
    logger.close()
    env.close()