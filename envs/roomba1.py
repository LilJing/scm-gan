import time
import numpy as np
import random

from tqdm import tqdm
from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete

CHANNELS = 3
GAME_SIZE = 64
BALL_RADIUS = 3
ROBOT_RADIUS = 4

MARGIN_Y = 4
MARGIN_X = 5


class RealpongEnv():
    def __init__(self):
        self.reset()
        self.action_space = Discrete(4)

    def reset(self):
        self.robot_x = np.random.randint(MARGIN_X, 64 - MARGIN_X)
        self.robot_y = np.random.randint(MARGIN_Y, 64 - MARGIN_Y)

        self.ball_x = np.random.randint(32 - 10, 32 + 10)
        self.ball_y = np.random.randint(32 - 10, 32 + 10)

        self.state = render_state(self.robot_x, self.robot_y, self.ball_x, self.ball_y)

    # The agent can press one of four buttons
    def step(self, a):
        # Move left/right/up/down, pushing the ball if in contact with it
        robot_speed = 3
        radius = BALL_RADIUS + ROBOT_RADIUS
        reward = 0
        if a == 0:
            if (self.ball_y - radius < self.robot_y < self.ball_y + radius
                and self.ball_x <= self.robot_x <= self.ball_x + radius + robot_speed):
                reward = 1
                self.ball_x = self.robot_x - robot_speed - radius
            self.robot_x -= robot_speed
        elif a == 1:
            if (self.ball_y - radius < self.robot_y < self.ball_y + radius
                and self.ball_x - radius - robot_speed <= self.robot_x <= self.ball_x):
                reward = 1
                self.ball_x = self.robot_x + robot_speed + radius
            self.robot_x += robot_speed
        elif a == 2:
            if (self.ball_x - radius < self.robot_x < self.ball_x + radius
                and self.ball_y <= self.robot_y <= self.ball_y + radius + robot_speed):
                reward = 1
                self.ball_y = self.robot_y - robot_speed - radius
            self.robot_y -= robot_speed
        elif a == 3:
            if (self.ball_x - radius < self.robot_x < self.ball_x + radius
                and self.ball_y - radius - robot_speed <= self.robot_y <= self.ball_y):
                reward = 1
                self.ball_y = self.robot_y + robot_speed + radius
            self.robot_y += robot_speed
        self.robot_x = max(MARGIN_X, min(self.robot_x, GAME_SIZE - MARGIN_X))
        self.robot_y = max(MARGIN_Y, min(self.robot_y, GAME_SIZE - MARGIN_Y))

        self.state = render_state(self.robot_x, self.robot_y, self.ball_x, self.ball_y)
        done = False
        info = {}
        return self.state, reward, done, info


def render_state(robot_x, robot_y, ball_x, ball_y):
    state = np.ones((CHANNELS, GAME_SIZE, GAME_SIZE)) * .0

    # Robot is a square, ball is a square
    draw_rect(state, robot_x, robot_y, ROBOT_RADIUS, ROBOT_RADIUS, color=2)
    draw_rect(state, ball_x, ball_y, BALL_RADIUS, BALL_RADIUS, color=0)
    return state


def draw_rect(pixels, center_x, center_y, width, height, color):
    img_channels, img_height, img_width = pixels.shape
    left = max(center_x - width, 0)
    right = min(center_x + width, img_width - 1)
    top = max(center_y - height, 0)
    bottom = min(center_y + height, img_height - 1)
    pixels[color, top:bottom, left:right] = 1


def get_trajectories(batch_size=32, timesteps=10, policy='random'):
    envs = MultiEnvironment([RealpongEnv() for _ in range(batch_size)])
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    for t in range(timesteps):
        if policy == 'random':
            actions = np.random.randint(envs.action_space.n, size=(batch_size,))
        if policy == 'repeat':
            actions = [i % envs.action_space.n for i in range(batch_size)]
        states, rewards, dones, _ = envs.step(actions)
        t_states.append(states)
        t_rewards.append(rewards)
        t_dones.append(dones)
        t_actions.append(actions)
    # Reshape to (batch_size, timesteps, ...)
    states = np.swapaxes(t_states, 0, 1)
    rewards = np.swapaxes(t_rewards, 0, 1)
    dones = np.swapaxes(t_dones, 0, 1)
    actions = np.swapaxes(t_actions, 0, 1)
    return states, rewards, dones, actions


if __name__ == '__main__':
    states, rewards, dones, actions = get_trajectories(batch_size=1, timesteps=200)
    import imutil
    vid = imutil.Video('roomba1.mp4', framerate=10)
    for state, action, reward in zip(states[0], actions[0], rewards[0]):
        pixels = np.transpose(state, (1, 2, 0))
        caption = "Action {} Reward {}".format(action, reward)
        vid.write_frame(pixels, img_padding=8, resize_to=(512,512), caption=caption)
    vid.finish()
