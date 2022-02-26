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
    env = UnityEnvironment(file_name="/home/steffen/workspace/PushBlockCustom/PushBlockCustom.x86_64")

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


    # How many actions are possible ?

    # For discrete actions only : How many different options does each action has ?
    # if spec.action_spec.discrete_size > 0:
    #   for action, branch_size in enumerate(spec.action_spec.discrete_branches):
    #     print(f"Action number {action} has {branch_size} different options")

    # decision_steps, terminal_steps = env.get_steps(behavior_name)

    # action = spec.action_spec.random_action(len(decision_steps))
    # env.set_actions(behavior_name, action)
    # Perform a step in the simulation

    
    # ###### old stuff
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    # # number of agents in the environment
    # print('Number of agents:', len(env.agents))

    # # number of actions
    # action_size = brain.vector_action_space_size
    # print('Number of actions:', action_size)

    # # examine the state space 
    # state = env_info.vector_observations[0]
    # print('States look like:', state)
    # state_size = len(state)
    # print('States have length:', state_size)



    config = Config()
    agent = DQNAgent(config=config)
    if(args.mode == "train"):
        scores = dqn_unity(env,agent,config)
    else:
        evaluate(env,agent) #TODO Define that path to the weights to load