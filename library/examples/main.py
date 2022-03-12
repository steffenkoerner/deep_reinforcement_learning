from algorithms.dqn_unity import dqn_unity
from utils.evaluate import evaluate
from agent.dqn_agent import DQNAgent
# from unityagents import UnityEnvironment
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from configs.dqn_config import Config
import os

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help = "The mode should be either train or eval")   
    args = parser.parse_args()

    config = Config()
    env = UnityEnvironment(file_name=config.env_path)
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    if(args.mode == "train"):
        scores = config.algorithm(env,config)
    else:
        evaluate(env,config)