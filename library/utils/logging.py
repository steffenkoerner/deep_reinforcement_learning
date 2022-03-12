import torch
import numpy as np
import matplotlib.pyplot as plt

def log_and_save(network,i_episode, mean_value, config):
     
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_value))
    if mean_value > log_and_save.max_score_value + config.save_each_return_step:
        print('\nEnvironment saved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(network.state_dict(), 'checkpoint_intermediate.pth')
        log_and_save.max_score_value = mean_value
    if mean_value >= config.stop_return:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_value))
        torch.save(network.state_dict(), 'checkpoint.pth')
log_and_save.max_score_value = 0

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


