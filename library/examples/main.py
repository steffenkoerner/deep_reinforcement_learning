from algorithms.dqn import dqn
from agent.dqn_agent import DQNAgent
from unityagents import UnityEnvironment

if __name__ == "__main__":
    env = UnityEnvironment(file_name="/home/steffen/workspace/deep_reinforcement_learning/library/examples/banana_env/Banana_Linux/Banana.x86_64")
    agent = DQNAgent(state_size=37, action_size=4, seed=0)
    dqn(env,agent)