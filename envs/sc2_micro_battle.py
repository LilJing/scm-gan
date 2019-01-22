import time
import json
import numpy as np
import os

import imutil

from multi_env import MultiEnvironment
from sc2env.environments.micro_battle import MicroBattleEnvironment
from gym.spaces.discrete import Discrete


class RandomAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, state):
        return np.random.randint(0, self.num_actions)


def generate_trajectories(policy='random'):
    # This environment teaches win/loss outcomes vs different enemies
    env = MicroBattleEnvironment(render=True)
    if policy == 'random':
        agent = RandomAgent(num_actions=env.actions())

    # TODO: after too many trajectories, restart environment
    while True:
        states, actions, rewards = play_episode(env, agent)
        yield states, actions, rewards
    print('Finished playing {} episodes'.format(train_episodes))


def play_episode(env, agent):
    start_time = time.time()
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action = agent.step(state)
        states.append(state)
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        imutil.show(np.array(state[3]), filename='state.jpg')
    states.append(state)
    print('Finished episode ({} actions) in {:.3f} sec'.format(
        len(actions), time.time() - start_time))
    return states, actions, rewards


class MicroBattlePixels():
    """ A wrapper that extracts only the rendered game frames """
    def __init__(self):
        self.env = MicroBattleEnvironment(render=True)
        self.action_space = self.env.action_space
        self.output_size = (64, 64)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        feature_map, feature_screen, pixels_minimap, pixels = state
        pixels = imutil.show(pixels, display=False, save=False, resize_to=self.output_size, return_pixels=True)
        pixels = pixels.transpose((2,0,1)) / 255.
        return pixels, reward, done, info

    def reset(self):
        return self.env.reset()


# Hack
batch_size=8
envs = MultiEnvironment([MicroBattlePixels() for _ in range(batch_size)])
def get_trajectories(batch_size=8, timesteps=10, policy='random'):
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    for t in range(timesteps):
        if policy == 'random':
            actions = np.random.randint(envs.action_space.n, size=(batch_size,))
        if policy == 'repeat':
            actions = [i % envs.action_space.n for i in range(batch_size)]
        print('Simulating timestep {}'.format(t))
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
    states, rewards, dones, actions = get_trajectories(batch_size=1, timesteps=10)
    import imutil
    vid = imutil.Video('sc2_micro_battle.mp4', framerate=10)
    for state, action, reward in zip(states[0], actions[0], rewards[0]):
        feature_map, feature_screen, pixels_minimap, pixels = state
        caption = "Action {} Reward {}".format(action, reward)
        vid.write_frame(pixels, img_padding=8, resize_to=(512,512), caption=caption)
    vid.finish()
