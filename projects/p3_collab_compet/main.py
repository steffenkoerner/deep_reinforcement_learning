from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from torch.utils.tensorboard import SummaryWriter


logger = SummaryWriter()

env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/projects/p3_collab-compet/Tennis_Linux/Tennis.x86_64")   
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
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

agent_0 = Agent(state_size, action_size, random_seed=0, id=0)
agent_1 = Agent(state_size, action_size, random_seed=0,id=1)    

def maddpg(n_episodes=2000, solve_score = 0.5):
    scores_window = deque(maxlen=100)
    scores = []
    moving_average = []
    max_score_value = 0
    max_average = 0
    noise = 1
    noise_discount = 0.999
    for i_episode in range(1, n_episodes+1):
        noise *= noise_discount
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        
        time_episode_start = time.time()
        
        while True:
            action_0 = agent_0.act(states[0], noise)
            action_1 = agent_1.act(states[1], noise)
            actions = np.concatenate((action_0, action_1), axis=0).flatten()
            env_info = env.step(actions)[brain_name] 
            next_states = env_info.vector_observations
            rewards = env_info.rewards                        
            dones = env_info.local_done 
            agent_0.step(states[0], action_0, rewards[0], next_states[0], dones[0])
            agent_1.step(states[1], action_1, rewards[1], next_states[1], dones[1])         
            states = next_states
            score += rewards  
            if np.any(dones):                                  
                break

        logger.add_scalars('agent/scores',{'agent0': score[0], 'agent1': score[1], },i_episode)
        if np.max(score) > max_score_value:
            max_score_value = np.max(score)
        scores_window.append(np.max(score))
        scores.append(np.max(score))
        mean = np.mean(scores_window)
        moving_average.append(mean)
        

        print('\rEpisode {}\tBest Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode,max_score_value ,mean), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tBest Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode,max_score_value ,mean))
        if mean > max_average + 0.1:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean))
            torch.save(agent_0.actor_local.state_dict(), 'multi_intermediate_weight_actor1.pth')
            torch.save(agent_0.critic_local.state_dict(),'multi_intermediate_weight_critic1.pth')
            torch.save(agent_1.actor_local.state_dict(), 'multi_intermediate_weight_actor2.pth')
            torch.save(agent_1.critic_local.state_dict(),'multi_intermediate_weight_critic2.pth')
            max_average = mean
        if mean >= 0.7:
            print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean))
            torch.save(agent_0.actor_local.state_dict(),  'multi_final_weight_actor1.pth')
            torch.save(agent_0.critic_local.state_dict(), 'multi_final_weight_critic1.pth')
            torch.save(agent_1.actor_local.state_dict(),  'multi_final_weight_actor2.pth')
            torch.save(agent_1.critic_local.state_dict(), 'multi_final_weight_critic2.pth')
            break
    return scores, moving_average

def plot_scores(scores, number):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    figure_name = "scores_" + str(number) +".png"
    plt.savefig(figure_name)
    

train = True
if train:

    scores, moving_average = maddpg()
    plot_scores(scores,0)