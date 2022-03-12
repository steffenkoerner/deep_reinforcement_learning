from collections import deque
import torch
import numpy as np
from library.utils.logging import log_and_save

def reinforce(env, config):
    scores_deque = deque(maxlen=100)
    scores = []
    agent = config.agent
    for i_episode in range(1, config.number_episodes+1):
        log_ptrobs = []
        rewards = []
        state = env.reset()
        while True:
            action, log_prob = agent.act(state)
            log_ptrobs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        sum_rewards = sum(rewards)
        scores_deque.append(sum_rewards)
        scores.append(sum_rewards)
        
        discounts = [config.gamma**i for i in range(len(rewards)+1)]
        return_trajectory = sum([a*b for a,b in zip(discounts, rewards)])
        
        agent.learn(return_trajectory,log_ptrobs)
        log_and_save(agent.policy_network,i_episode,np.mean(scores_deque),config)

        
    return scores