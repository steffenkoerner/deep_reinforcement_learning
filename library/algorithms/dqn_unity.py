from collections import deque
from utils.logging import log_and_save
from mlagents_envs.environment import ActionTuple
import numpy as np
import torch


def dqn_unity(env,agent, config):
    scores = []
    scores_window = deque(maxlen=100)
    eps = config.eps_start
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    agent_id = 0
    for i_episode in range(1, config.number_episodes+1):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_id = decision_steps.agent_id[0]
        agent_index = decision_steps.agent_id_to_index[agent_id]
        state = decision_steps.obs[0][agent_index]

        score = 0
        done = False
        for episode_length in range(1, config.episode_length):
            action = agent.act(state, eps)  
            action_reshaped = action.reshape((1,1))
            action_tuple = ActionTuple(discrete=action_reshaped)
            env.set_actions(behavior_name,action_tuple)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if agent_id in decision_steps.agent_id:
                agent_index = decision_steps.agent_id_to_index[agent_id]
                reward = decision_steps.reward[agent_index]
                next_state = decision_steps.obs[0][agent_index]
            elif agent_id in terminal_steps.agent_id:
                agent_index = terminal_steps.agent_id_to_index[agent_id]
                reward = terminal_steps.reward[agent_index]
                next_state = terminal_steps.obs[0][agent_index]
                done = True
            else:
                print("Something is wrong !!!")
                break

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(config.eps_end, config.eps_decay*eps)
        mean_value = np.mean(scores_window)
        config.logger.add_scalar("score", score, i_episode)    
        config.logger.add_scalar("mean scores", mean_value, i_episode)   
        log_and_save(agent,i_episode,mean_value, config)
        if mean_value >= config.stop_return:
            break
    return scores

