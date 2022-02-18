import torch
import numpy as np
import matplotlib.pyplot as plt

def log_and_save(agent,i_episode, mean_value, max_score_value):
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value))
    if mean_value>=0 and mean_value > max_score_value + 3:
        print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_intermediate.pth')
        max_score_value = mean_value
    if mean_value >= 13:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def evaluate(env, agent):
    brain_name = env.brain_names[0]
    # agent.load_weights()
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state, 0.)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    print("Score: {}".format(score))