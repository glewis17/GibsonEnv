from gibson.envs.minitaur_env import MinitaurVectorServoingEnv
import argparse
import math
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'minitaur_vector_servoing.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = MinitaurVectorServoingEnv(config = args.config)

    env.reset()
    phi = 0
    speed_frac = .5
    action = [phi, speed_frac]
    while True:
      env.step(action)
