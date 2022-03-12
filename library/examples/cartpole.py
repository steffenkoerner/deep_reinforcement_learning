
from library.configs.cartpole_config import Config
from library.utils.evaluate import evaluate_gym

from argparse import ArgumentParser
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help = "The mode should be either train or eval")   
    args = parser.parse_args()
    config = Config()
    env = config.gym


    if(args.mode == "train"):
        scores = config.algorithm(env,config)
    else:
        evaluate_gym(env,config)