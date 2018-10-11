from gibson.envs.minitaur_env import MinitaurVectorServoingEnv
import argparse
import math
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'minitaur_vector_servoing.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = MinitaurForwardWalkEnv(config = args.config)

    env.reset()
    action = [0]*8
    while True:
      env.step(action)
