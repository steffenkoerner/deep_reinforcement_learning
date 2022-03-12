# from mlagents_envs.environment import ActionTuple
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
# from pyvirtualdisplay import Display



def evaluate_unity(env, config):
    agent = config.agent 
    agent.load_weights(config.path_to_stored_weights)

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    scores = []   
    
    for i in range(0,50):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_id = decision_steps.agent_id[0]
        agent_index = decision_steps.agent_id_to_index[agent_id]
        state = decision_steps.obs[0][agent_index]
        score = 0
        done = False
        while True:
            action = agent.act(state, 0)  

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
            
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

    print("Score: {}".format(np.mean(scores)))


def evaluate_gym(env,config):
    agent = config.agent
    agent.load_weights(config.path_to_stored_weights)
    scores_window = deque(maxlen=100)
    rewards = []
    for t in range(0,100):
        state = env.reset()
        while True:
            action, _ = agent.act(state)
            state, reward, done, _ = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                break 
        scores_window.append(np.sum(rewards))
        rewards.clear()
        print('Episode {}\tAverage Score: {:.2f}'.format(t, np.mean(scores_window)))


       