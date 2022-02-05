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
from NewReplayBuffer import ReplayBuffer, GaussianNoise, OUNoise

LEARNING_RATE = 1e-2
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
TAU = 1e-2              # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = SummaryWriter()



agent1_actor_weights  = ""
agent1_critic_weights = ""
agent2_actor_weights  = ""
agent2_critic_weights = ""

class CriticNetwork(nn.Module):
    def __init__(self,critic_input_size, critic_output_size, seed, fc1_units = 400, fc2_units = 300):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(critic_input_size, fc1_units),
            # nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            # nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, critic_output_size) 
        )
        self.to(device)

    def forward(self, state):
        return self.layers(state) # TODO: Introduce batch normalization

class ActorNetwork(nn.Module):
    def __init__(self, actor_input_size, actor_output_size, seed, fc1_units = 400, fc2_units = 300):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(actor_input_size, fc1_units),
            # nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            # nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, actor_output_size),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, state):
        return self.layers(state) # TODO: Introduce batch normalization

class DDPGNetwork():
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, critic_output_size, seed):
        self.actor_network = ActorNetwork(actor_input_size, actor_output_size, seed).to(device)
        self.critic_network = CriticNetwork(critic_input_size, critic_output_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)
        

    def actor(self, state):
        return self.actor_network(state)
    
    def critic(self, states,actions):
        return self.critic_network(torch.cat((states, actions), 1))


class DDPGAgent():
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, critic_output_size, seed, warmup):
        self.actor_input_size = actor_input_size
        self.actor_output_size = actor_output_size
        self.seed = random.seed(seed)
        self.action_lowest_value = -0.9
        self.action_highest_value = 0.9
        self.warump = warmup
        #self.gaussian_noise = GaussianNoise(size=action_size, std_start=0.8, std_end=0.01,steps=1000000) 
        self.ou_noise = OUNoise(action_size,seed )

        # DDPG-Network
        # TODO: Make parameter of both networks identical at beginnings
        self.local_network = DDPGNetwork(actor_input_size, actor_output_size, critic_input_size, critic_output_size, seed)
        self.target_network = DDPGNetwork(actor_input_size, actor_output_size, critic_input_size, critic_output_size, seed)
        self.set_parameters_of_target_and_local_equal()
    
    def reset_noise(self):
        self.ou_noise.reset()

    def set_parameters_of_target_and_local_equal(self):
        self.soft_update(1.0)

    def get_acion_per_current_policy_for(self, state , number_episode, train_mode):
        if number_episode < self.warump and train_mode:
            actions = np.random.randn(self.actor_output_size) 
            actions = np.clip(actions, self.action_lowest_value, self.action_highest_value)   
        else:       
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.local_network.actor_network.eval()
            with torch.no_grad():
                actions = self.local_network.actor(state).cpu().data.numpy()
            self.local_network.actor_network.train()
            noise_applied = self.ou_noise.noise().numpy() # TODO: Investigate noise function
            #noise_applied =  self.gaussian_noise()
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


def plot_scores(scores, number):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    figure_name = "scores_" + str(number) +".png"
    plt.savefig(figure_name)
    

def maddpg(env, agent, n_episodes=2000, max_t=1000, gamma=0.9):
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
        agent.reset_noise()
        score = [0,0]
        while True:
            noise *= noise_decay
            action = agent.get_acion_per_current_policy_for(state, i_episode, True)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done, gamma)
            state = next_state
            score += reward
            if np.any(done): 
                break 
        logger.add_scalars('agent/scores',{'agent1': score[0], 'agent2': score[1], },i_episode)
        max_score = np.max(score)
        scores_window.append(max_score)
        scores.append(max_score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > max_score_value + 0.1:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.agents[0].local_network.actor_network.state_dict(), 'multi_intermediate_weight_actor1.pth')
            torch.save(agent.agents[0].local_network.critic_network.state_dict(),'multi_intermediate_weight_critic1.pth')
            torch.save(agent.agents[1].local_network.actor_network.state_dict(), 'multi_intermediate_weight_actor2.pth')
            torch.save(agent.agents[1].local_network.critic_network.state_dict(),'multi_intermediate_weight_critic2.pth')
            max_score_value = np.mean(scores_window)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.agents[0].local_network.actor_network.state_dict(),  'multi_final_weight_actor1.pth')
            torch.save(agent.agents[0].local_network.critic_network.state_dict(), 'multi_final_weight_critic1.pth')
            torch.save(agent.agents[1].local_network.actor_network.state_dict(),  'multi_final_weight_actor2.pth')
            torch.save(agent.agents[1].local_network.critic_network.state_dict(), 'multi_final_weight_critic2.pth')
            break
    return scores

class MADDPGAgent():
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, critic_output_size, seed, warmup):
        self.agents = [
            DDPGAgent(actor_input_size=actor_input_size, actor_output_size=actor_output_size, critic_input_size=critic_input_size, critic_output_size=critic_output_size, seed=seed, warmup = warmup),
            DDPGAgent(actor_input_size=actor_input_size, actor_output_size=actor_output_size, critic_input_size=critic_input_size, critic_output_size=critic_output_size, seed=seed, warmup = warmup)
            ]
        self.memory = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=seed)
        self.iter = 0
        self.load_weights()

    def load_weights(self):
        if agent1_actor_weights != "":
            self.agents[0].local_network.actor_network.load_state_dict(torch.load(agent1_actor_weights))
            self.agents[0].local_network.critic_network.load_state_dict(torch.load(agent1_critic_weights))
            self.agents[0].set_parameters_of_target_and_local_equal()

            self.agents[1].local_network.actor_network.load_state_dict(torch.load(agent2_actor_weights))
            self.agents[1].local_network.critic_network.load_state_dict(torch.load(agent2_critic_weights))
            self.agents[1].set_parameters_of_target_and_local_equal()

    def reset_noise(self):
        self.agents[0].reset_noise()
        self.agents[1].reset_noise()

    def get_acion_per_current_policy_for(self, all_states , number_episode, train_mode):
        action0 = self.agents[0].get_acion_per_current_policy_for(all_states[0],number_episode, True)
        action1 = self.agents[1].get_acion_per_current_policy_for(all_states[1],number_episode, True)
        actions = np.vstack((action0,action1))
        return actions

    def step(self,state, action, reward, next_state, done, gamma):
        self.memory.add(state[0], state[1] ,action[0], action[1] ,reward, next_state[0], next_state[1], done)
        self.learn(gamma)
    
    def learn(self, gamma):
        """Update parameters.
        """
        if len(self.memory) > BATCH_SIZE:
            self.iter += 1

            for i in range(0,2):
                experiences = self.memory.sample()
                states0, states1, actions0, actions1, rewards, next_states0, next_states1, dones = experiences
                next_states = torch.cat((next_states0,next_states1), 1)
                states = torch.cat((states0,states1),1)
                actions = torch.cat((actions0,actions1),1)
                next_action0 = self.agents[0].target_network.actor(next_states0)
                next_action1 = self.agents[1].target_network.actor((next_states1))
                next_actions = torch.cat((next_action0,next_action1),1)

                self.agents[i].local_network.critic_optimizer.zero_grad()
                with torch.no_grad():
                    critic_values = torch.squeeze(self.agents[i].target_network.critic(next_states, next_actions))
                y = rewards[:,i] + (1 - dones[:,i]) * gamma * critic_values
                q_value =torch.squeeze(self.agents[i].local_network.critic(states,actions))
                huber_loss = torch.nn.SmoothL1Loss()
                critic_loss = huber_loss(y.detach() , q_value)
                critic_loss.backward()
                self.agents[i].local_network.critic_optimizer.step()

                estimated_action0 = self.agents[0].local_network.actor(states0)
                estimated_action1 = self.agents[1].local_network.actor(states1)
                estimated_actions = torch.cat((estimated_action0,estimated_action1),1)

                self.agents[i].local_network.actor_optimizer.zero_grad()
                actor_loss = -self.agents[i].local_network.critic(states, estimated_actions).mean()
                actor_loss.backward()
                self.agents[i].local_network.actor_optimizer.step()
                self.agents[i].soft_update(TAU)
                al = -actor_loss.cpu().detach().item()
                cl = critic_loss.cpu().detach().item()
                q_val = q_value.cpu().detach().mean().item()
                reward = rewards[:,i].cpu().detach().mean()
                logger.add_scalars('agent%i/losses' % i,{'critic_loss': cl, 'actor_loss': al, 'q_value' : q_val, 'reward' : reward},self.iter)


def test_performance(agent, env, brain_name):
    for i in range(1, 1000):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        #states = env_info.vector_observations                  # get the current state (for each agent)
        scores = [0,0]                       # initialize the score (for each agent)
        while True:
            #actions = agent.get_acion_per_current_policy_for(states , 0, False) # select an action (for each agent)
            actions_random = np.random.randn(2, 2) 
            actions = actions_random
            #actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
    


if __name__ == '__main__':  
    env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/projects/p3_collab-compet/Tennis_Linux/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]


    

    critic_input_size =2*state_size+2*action_size
    agent = MADDPGAgent(actor_input_size=state_size, actor_output_size=action_size, critic_input_size=critic_input_size, critic_output_size=1, seed=0, warmup=0)
    scores = maddpg(env=env,agent=agent, n_episodes=5000000000, gamma=0.99)
    #test_performance(agent,env,brain_name)
    logger.flush()
    logger.close()
    env.close()