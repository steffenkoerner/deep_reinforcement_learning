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
from MultiReplayBuffer import ReplayBuffer, GaussianNoise, OUNoise

LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
TAU = 1e-3              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Needs to be 1 as this is the max(Q(s,a)) that is learned
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
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DDPGNetwork():
    def __init__(self, state_size, action_size, critic_input_size, seed):
        self.actor_network = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic_network = CriticNetwork(critic_input_size, 1, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)
        

    def actor(self, state):
        return self.actor_network(state)
    
    def critic(self, states,actions):
        result = torch.cat((states[0], states[1] ,actions[0], actions[1]), 1)
        return self.critic_network(result)




class DDPGAgent():
    def __init__(self, state_size, action_size, seed,critic_input_size, warmup):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_lowest_value = -1
        self.action_highest_value = 1
        self.warump = warmup
        self.gaussian_noise = GaussianNoise(size=action_size, std_start=0.2, std_end=0.01,steps=1000000) 
        self.ou_noise = OUNoise(action_size, scale=1.0 )

        # DDPG-Network
        self.local_network = DDPGNetwork(state_size=state_size, action_size=action_size, critic_input_size =critic_input_size ,seed=seed) 
        self.target_network = DDPGNetwork(state_size=state_size, action_size= action_size, critic_input_size=critic_input_size ,seed=seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size=action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)   
        
    def get_acion_per_current_policy_for(self, state , number_episode, noise=0):
        if number_episode < self.warump:
            actions = np.random.randn(self.action_size) 
            actions = np.clip(actions, self.action_lowest_value, self.action_highest_value)   
        else:         
            state = torch.from_numpy(state).float().to(device)
            self.local_network.actor_network.eval()
            with torch.no_grad():
                actions = self.local_network.actor(state).cpu().data.numpy()
            self.local_network.actor_network.train()
            noise_applied = noise * self.ou_noise.noise().numpy()
            actions = np.clip(actions + noise_applied, self.action_lowest_value, self.action_highest_value)
        return actions


    def soft_update(self, tau):
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


class MADDPGAgent():
    def __init__(self, state_size, action_size, seed, warmup):
        self.number_agents = 2
        self.critic_input_size=state_size *self.number_agents+ self.number_agents*action_size
        self.agents = [DDPGAgent(state_size=state_size,action_size=action_size,critic_input_size=self.critic_input_size, seed=seed, warmup=warmup) for i in range(0,self.number_agents)]
        self.memory = ReplayBuffer(action_size=action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)   

    def get_acion_per_current_policy_for(self, observation_all , number_episode, noise=0):
        assert len(observation_all) == len(self.agents)
        actions = [agent.get_acion_per_current_policy_for(obs,number_episode) for agent,obs in zip(self.agents, observation_all)]
        result = np.array(actions)
        assert result.ndim == 2
        return result

    def step(self,state, action, reward, next_state, done, gamma,i_episode,episode_step):
        self.memory.add(state, action, reward, next_state, done)
        self.learn(gamma,i_episode,episode_step)
    
    def learn(self, gamma,i_episode,episode_step):
        if len(self.memory) > BATCH_SIZE:
            for i in range(0,2):
                experiences = self.memory.sample()
                states, actions, rewards, next_states, dones = experiences

                ## TODO: Somwehow the sizes don't seem to match
                q_next = self.agents[i].target_network.critic(next_states, [self.agents[0].target_network.actor(next_states[0]),self.agents[1].target_network.actor(next_states[1])])
                y = rewards[i] + (1 - dones[i]) * gamma * q_next
                critic_loss = F.mse_loss(y , self.agents[i].local_network.critic(states,actions)) 
                self.agents[i].local_network.critic_network.zero_grad()
                critic_loss.backward()
                self.agents[i].local_network.critic_optimizer.step()
                #writer.add_scalar("multi/train/critic_loss" +str(i), critic_loss, i_episode)

                actor_loss = self.agents[i].local_network.critic(states.detach(), [self.agents[0].local_network.actor(states[0]), self.agents[1].local_network.actor(states[1])])
                actor_loss = -actor_loss.mean()
                self.agents[i].local_network.actor_network.zero_grad()
                actor_loss.backward()
                self.agents[i].local_network.actor_optimizer.step()
                #writer.add_scalar("multi/train/actor_loss" + str(i), -actor_loss, i_episode)

                self.agents[i].soft_update(TAU) 

    def soft_update(self, local_model, target_model, tau):
        for agent in self.agents:
            agent.soft_update(tau)

          

def plot_scores(scores, number):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    figure_name = "scores_" + str(number) +".png"
    plt.savefig(figure_name)
    

def mddpg(env, agent, n_episodes=2000, max_t=1000, gamma=0.9):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    max_score_value = 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    noise = 1
    noise_decay = 0.9999
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations
        score = [0,0]
        noise *= noise_decay
        episode_step = 0
        while True:
            episode_step += 1
            action = agent.get_acion_per_current_policy_for(state, i_episode, noise)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            agent.step(state, action,reward,next_state,done, gamma,i_episode,episode_step)
            state = next_state
            score += reward
            if done[0] or  done[1]:
                break 
        max_score = np.max(score)
        writer.add_scalar("multi/train/score", max_score, i_episode)
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
            torch.save(agent.agents[0].local_network.actor_network.state_dict(), 'multi_intermediate_weight_actor1.pth')
            torch.save(agent.agents[0].local_network.critic_network.state_dict(), 'multi_intermediate_weight_critic1.pth')
            torch.save(agent.agents[1].local_network.actor_network.state_dict(), 'multi_intermediate_weight_actor2.pth')
            torch.save(agent.agents[1].local_network.critic_network.state_dict(), 'multi_intermediate_weight_critic2.pth')
            max_score_value = np.mean(scores_window)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.agents[0].local_network.actor_network.state_dict(), 'multi_final_weight_actor1.pth')
            torch.save(agent.agents[0].local_network.critic_network.state_dict(), 'multi_final_weight_critic1.pth')
            torch.save(agent.agents[1].local_network.actor_network.state_dict(), 'multi_final_weight_actor2.pth')
            torch.save(agent.agents[1].local_network.critic_network.state_dict(), 'multi_final_weight_critic2.pth')
        #     break
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


    agent = MADDPGAgent(state_size= state_size, action_size = action_size, seed = 0 , warmup = 0)
    scores = mddpg(env,agent, n_episodes=5000000000)

    writer.flush()
    writer.close()
    env.close()