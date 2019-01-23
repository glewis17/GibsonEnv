from gibson.envs.turtlebot_env import TurtlebotRealEnv
import argparse
import math
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'turtlebot_real.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotRealEnv(config = args.config)

    print(env.reset().shape)
    """
    print(env.action_space)
    action = [.02, -0.02]
    while True:
      print(env.step(action))
    """
