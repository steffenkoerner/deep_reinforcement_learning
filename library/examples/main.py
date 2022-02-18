from algorithms.dqn import dqn
from agent.dqn_agent import DQNAgent
from unityagents import UnityEnvironment

from argparse import ArgumentParser







def train(env, agent):
    dqn(env,agent)

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



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help = "The mode should be either train or eval")
    args = parser.parse_args()
    env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/library/examples/banana_env/Banana_Linux/Banana.x86_64")
    agent = DQNAgent(state_size=37, action_size=4, seed=0)
    if(args.mode == "train"):
        train(env,agent)
    else:
        evaluate(env,agent)