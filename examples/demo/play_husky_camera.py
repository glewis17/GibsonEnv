from gibson.envs.turtlebot_env import TurtlebotRealEnv
from gibson.utils.play import play
import os

config_file = '/home/bradleyemi/sim2real/TEAS/teas/env/gibson/turtlebot_navigate_real.yaml'
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()
    env = TurtlebotRealEnv(config=args.config, gpu_count = 0)
    play(env, zoom=4)