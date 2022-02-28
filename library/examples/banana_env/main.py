from algorithms.dqn_unity import dqn_unity
from utils.logging import evaluate
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
    env = UnityEnvironment(file_name="/home/steffen/workspace/PushBlockNonDevelopment/PushBlockCustom.x86_64")

    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]

    # Examine the number of observations per Agent
    print("Number of observations : ", len(spec.observation_specs))
    print("Observation: " , spec.observation_specs)

    # Is there a visual observation ?
    # Visual observation have 3 dimensions: Height, Width and number of channels
    vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
    print("Is there a visual observation ?", vis_obs)
    
    # Is the Action continuous or multi-discrete ?
    if spec.action_spec.continuous_size > 0:
      print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
      print(f"There are {spec.action_spec.discrete_size} discrete actions")


    config = Config()
    agent = DQNAgent(config=config)
    if(args.mode == "train"):
        scores = dqn_unity(env,agent,config)
    else:
        evaluate(env,agent) #TODO Define that path to the weights to load