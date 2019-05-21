import time
import numpy as np
import random

from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete

CHANNELS = 3
GAME_SIZE = 16
BALL_RADIUS = 2
NUM_ACTIONS = 4
NUM_REWARDS = 1
MARGIN_X = 2
MARGIN_Y = 2


class Env():
    def __init__(self):
        self.reset()
        self.action_space = Discrete(4)

    def reset(self):
        self.ball_x = np.random.randint(0 + MARGIN_X, GAME_SIZE - MARGIN_X)
        self.ball_y = np.random.randint(0 + MARGIN_Y, GAME_SIZE - MARGIN_Y)
        self.state = render_state(self.ball_x, self.ball_y)

    # The agent can press one of four buttons
    def step(self, a):
        # Move ball up/down
        if a == 0:
            self.ball_y -= 3
        elif a == 1:
            self.ball_y += 3
        self.ball_y = max(MARGIN_Y, min(self.ball_y, GAME_SIZE - MARGIN_Y))

        # Move ball left/right
        if a == 2:
            self.ball_x -= 3
        elif a == 3:
            self.ball_x += 3
        self.ball_x = max(0, min(self.ball_x, GAME_SIZE))

        # If the ball goes out of the court, the episode ends
        done = False
        reward = 0
        if self.ball_x >= GAME_SIZE:
            # Ball went right
            reward = 1

        if self.ball_x <= 0:
            # Ball went left
            reward = -1

        self.state = render_state(self.ball_x, self.ball_y)
        info = {}
        return self.state, reward, done, info


def render_state(ball_x, ball_y):
    state = np.ones((CHANNELS, GAME_SIZE, GAME_SIZE)) * .0
    # Ball moves around
    draw_rect(state, ball_x, ball_y, BALL_RADIUS, BALL_RADIUS, color=1)
    return state


def draw_rect(pixels, center_x, center_y, width, height, color):
    img_channels, img_height, img_width = pixels.shape
    left = max(center_x - width, 0)
    right = min(center_x + width, img_width - 1)
    top = max(center_y - height, 0)
    bottom = min(center_y + height, img_height - 1)
    pixels[color, top:bottom, left:right] = 1


def get_trajectories(batch_size=32, timesteps=10, policy='random', random_start=False, training=True):
    envs = MultiEnvironment([Env() for _ in range(batch_size)])
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    # Initial actions/stats
    actions = np.random.randint(envs.action_space.n, size=(batch_size,))
    for t in range(timesteps):
        states, rewards, dones, _ = envs.step(actions)
        rewards = [rewards]
        if policy == 'random':
            actions = np.random.randint(envs.action_space.n, size=(batch_size,))
        if policy == 'repeat':
            actions = [i % envs.action_space.n for i in range(batch_size)]
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
    states, rewards, dones, actions = get_trajectories(batch_size=1, timesteps=100)
    import imutil
    vid = imutil.Video('gridworld.mp4', framerate=5)
    for state, action, reward in zip(states[0], actions[0], rewards[0]):
        pixels = np.transpose(state, (1, 2, 0))
        caption = "Prev. Action {} Prev Reward {}".format(action, reward)
        vid.write_frame(pixels, img_padding=8, resize_to=(512,512), caption=caption)
    vid.finish()
