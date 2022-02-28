from collections import deque
from utils.logging import log_and_save
import numpy as np
import torch


def dqn(env,agent, config):
    scores = []
    scores_window = deque(maxlen=100)
    eps = config.eps_start
    brain_name = env.brain_names[0]
    for i_episode in range(1, config.number_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0] 
        score = 0
        while True:
            action = agent.act(state, eps)            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]         
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)
        scores.append(score)
        config.logger.add_scalar("score/train", score, i_episode)
        eps = max(config.eps_end, config.eps_decay*eps)
        mean_value = np.mean(scores_window)
        log_and_save(agent,i_episode,mean_value, config)
        if mean_value >= config.stop_return:
            break
    return scores

