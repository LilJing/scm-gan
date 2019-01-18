import time
import numpy as np
import random

from tqdm import tqdm
from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete

CHANNELS = 3
GAME_SIZE = 64
PADDLE_WIDTH = 1
PADDLE_HEIGHT = 4
BALL_RADIUS = 2

MARGIN_Y = 4
MARGIN_X = 5


class RealpongEnv():
    def __init__(self):
        self.reset()
        self.action_space = Discrete(4)

    def reset(self):
        self.left_y = np.random.randint(MARGIN_Y, 64 - MARGIN_Y)
        self.right_y = np.random.randint(MARGIN_Y, 64 - MARGIN_Y)
        self.ball_x = np.random.randint(0 + MARGIN_X, 64 - MARGIN_X)
        self.ball_y = np.random.randint(0 + MARGIN_Y, 64 - MARGIN_Y)
        self.ball_velocity_x = random.choice([-3, -2, 2, +3])
        self.ball_velocity_y = random.choice([-3, -2, 2, +3])
        self.state = render_state(self.left_y, self.right_y, self.ball_x, self.ball_y, self.ball_velocity_x, self.ball_velocity_y)

    # The agent can press one of four buttons
    def step(self, a):
        # Move the red paddle up/down
        if a == 0:
            self.right_y -= 3
        elif a == 1:
            self.right_y += 3
        self.right_y = max(0, min(self.right_y, GAME_SIZE))

        # Move the blue paddle up/down
        if a == 2:
            self.left_y -= 3
        elif a == 3:
            self.left_y += 3
        self.left_y = max(0, min(self.left_y, GAME_SIZE))

        # The ball moves and interacts with the paddles
        self.ball_x += self.ball_velocity_x
        self.ball_y += self.ball_velocity_y

        # Ball bouncing from the paddles
        bounce_right = GAME_SIZE - MARGIN_X - BALL_RADIUS - PADDLE_WIDTH
        bounce_left = MARGIN_X + BALL_RADIUS + PADDLE_WIDTH
        if (self.ball_x >= bounce_right and self.ball_velocity_x > 0 and
            self.right_y - PADDLE_HEIGHT <= self.ball_y <= self.right_y + PADDLE_HEIGHT):
            self.ball_velocity_x *= -1
        if (self.ball_x <= bounce_left and self.ball_velocity_x < 0 and
            self.left_y - PADDLE_HEIGHT <= self.ball_y <= self.left_y + PADDLE_HEIGHT):
            self.ball_velocity_x *= -1

        # Ball bounces off the top and bottom
        if self.ball_y >= GAME_SIZE - 2 and self.ball_velocity_y > 0:
            self.ball_velocity_y *= -1
        if self.ball_y <= 2 and self.ball_velocity_y < 0:
            self.ball_velocity_y *= -1

        # If the ball goes out of the court, the episode ends
        done = False
        if self.ball_x >= GAME_SIZE and self.ball_velocity_x > 0:
            # Blue player scores, episode is over
            reward = 1
            done = True

        if self.ball_x <= 0 and self.ball_velocity_x < 0:
            # Red player scores, episode is over
            reward = -1
            done = True

        self.state = render_state(self.left_y, self.right_y, self.ball_x, self.ball_y, self.ball_velocity_x, self.ball_velocity_y)
        reward = 0
        info = {}
        return self.state, reward, done, info


def render_state(left_y, right_y, ball_x, ball_y, ball_velocity_x, ball_velocity_y):
    state = np.ones((CHANNELS, GAME_SIZE, GAME_SIZE)) * .0

    # Blue paddle on the left, red on the right
    left_x = MARGIN_X
    right_x = GAME_SIZE - MARGIN_X
    state[2, left_y - PADDLE_HEIGHT:left_y + PADDLE_HEIGHT,
               left_x - PADDLE_WIDTH: left_x + PADDLE_WIDTH] = 1
    state[0, right_y - PADDLE_HEIGHT:right_y + PADDLE_HEIGHT,
               right_x - PADDLE_WIDTH: right_x + PADDLE_WIDTH] = 1

    # Green ball with a little tail to indicate direction
    ball_color = (0, 1, 0)
    for idx, c in enumerate(ball_color):
        state[idx, ball_y-BALL_RADIUS:ball_y+BALL_RADIUS,
                   ball_x-BALL_RADIUS:ball_x+BALL_RADIUS] = c
        tail_x = ball_x - ball_velocity_x
        tail_y = ball_y - ball_velocity_y
        tail_size = BALL_RADIUS - 1
        state[idx, tail_y-1:tail_y+1, tail_x-tail_size:tail_x+tail_size] = c
    return state


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
