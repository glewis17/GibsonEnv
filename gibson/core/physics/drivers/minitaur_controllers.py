import numpy as np
import math
import gym

class SinePolicyController:
    """Have minitaur walk with a sine gait."""
    # Leg model enabled
    # Accurate model enabled
    # pd control enabled
    def __init__(self, time_step=.001, gait_period=60):
        self.gait_period = gait_period
        self.step_count = 0
        self.time_step = time_step
        self.action_space = None
        self.t = 0

    def get_motor_commands(self, amplitude_1, amplitude_2, go_left=False):
        self.t = self.step_count * self.time_step
        left_add = 0
        if go_left:
            left_add = math.pi
        right_add = math.pi - left_add

        a1 = math.sin(self.t * self.gait_period + left_add) * amplitude_1
        a2 = math.sin(self.t * self.gait_period + right_add) * amplitude_2
        a3 = math.sin(self.t * self.gait_period + left_add) * amplitude_2
        a4 = math.sin(self.t * self.gait_period + right_add) * amplitude_1

        action = [a1, a2, a2, a1, a3, a4, a4, a3]

        self.step_count = self.step_count + 1

        return action

class ForwardSinePolicyController(SinePolicyController):
    def __init__(self, time_step=.001):
        SinePolicyController.__init__(self, time_step=time_step)

    def translate_action_to_motor_commands(self, a):
        return self.get_motor_commands(.7, .7)

class VectorSinePolicyController(SinePolicyController):
    def __init__(self, time_step=.001):
        SinePolicyController.__init__(self, time_step=time_step)
        self.action_lower_bounds = np.array([-np.pi/3., .5])
        self.action_upper_bounds = np.array([np.pi/3., 1.])
        self.action_space = gym.spaces.Box(self.action_lower_bounds, self.action_upper_bounds)

    def translate_action_to_motor_commands(self, a):
        #print("VectorSinePolicyController: input action: " + str(a))
        phi, r = a
        # this constant translates +- pi/3 constraint to +- 2 steering amplitude
        c = -6. / np.pi
        steering_amplitude = c*phi
        amplitude_1 = max(r + steering_amplitude, self.action_lower_bounds[1])
        amplitude_2 = max(r - steering_amplitude, self.action_lower_bounds[1])
        left_add = phi >= 0
        return self.get_motor_commands(amplitude_1, amplitude_2, left_add)
