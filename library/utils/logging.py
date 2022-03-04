import torch
from mlagents_envs.environment import ActionTuple
import numpy as np
import matplotlib.pyplot as plt

def log_and_save(agent,i_episode, mean_value, config):
     
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value))
    if mean_value > log_and_save.max_score_value + config.save_each_return_step:
        print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_intermediate.pth')
        log_and_save.max_score_value = mean_value
    if mean_value >= config.stop_return:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
log_and_save.max_score_value = 0

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


# def evaluate(env, agent):
#     brain_name = env.brain_names[0]
#     agent.load_weights()
#     env_info = env.reset(train_mode=False)[brain_name]
#     state = env_info.vector_observations[0]
#     score = 0
#     while True:
#         action = agent.act(state, 0.)
#         env_info = env.step(action)[brain_name]
#         next_state = env_info.vector_observations[0]
#         reward = env_info.rewards[0]
#         done = env_info.local_done[0]
#         score += reward
#         state = next_state
#         if done:
#             break

#     print("Score: {}".format(score))


def evaluate(env, agent):
    agent.load_weights()

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