from algorithms.dqn import dqn
from utils.logging import evaluate
from agent.dqn_agent import DQNAgent
from unityagents import UnityEnvironment
from configs.dqn_config import Config
import os

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help = "The mode should be either train or eval")
    args = parser.parse_args()
    env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/library/examples/banana_env/Banana_Linux/Banana.x86_64")

    config = Config()
    agent = DQNAgent(config=config,seed=0)
    if(args.mode == "train"):
        scores = dqn(env,agent,config)
    else:
        evaluate(env,agent) #TODO Define that path to the weights to load