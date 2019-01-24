import sys
import random
import gym
import time
import numpy as np
import imutil
from threading import Thread

from sc2env.environments.micro_battle import MicroBattleEnvironment

MAX_BUFFER_LEN = 100
replay_buffer = []


def play_game_thread():
    env = MicroBattleEnvironment(render=True)
    policy = lambda: env.action_space.sample()
    while True:
        play_episode(env, policy)


def play_episode(env, policy):
    trajectory = []
    state = env.reset()
    reward = 0
    done = False
    while True:
        action = policy()
        state = state[3]  # Rendered game pixels
        state = state.transpose((2,0,1))  # HWC -> CHW
        state = state * (1/255)  # [0,1]
        trajectory.append((state, reward, action))
        if done:
            break
        state, reward, done, info = env.step(action)
    add_to_replay_buffer(trajectory)


def add_to_replay_buffer(episode):
    if len(replay_buffer) < MAX_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(1, MAX_BUFFER_LEN)
        replay_buffer[idx] = episode


simulator_running = False
def get_trajectories(batch_size=8, timesteps=10, policy='random'):
    global simulator_running
    if not simulator_running:
        print('Starting simulator thread...')
        init()
        simulator_running = True

    while len(replay_buffer) < batch_size:
        print('Waiting for replay buffer to fill, {}/{}...'.format(
            len(replay_buffer), batch_size))
        time.sleep(1)

    # Create a batch
    states_batch, rewards_batch, dones_batch, actions_batch = [], [], [], []
    for batch_idx in range(batch_size):

        # Accumulate trajectory clips
        trajectory = []
        timesteps_remaining = timesteps
        dones = []
        while timesteps_remaining > 0:
            selected_trajectory = random.choice(replay_buffer)
            start_idx = np.random.randint(0, len(selected_trajectory) - 3)
            end_idx = min(start_idx + timesteps_remaining, len(selected_trajectory) - 1)
            duration = end_idx - start_idx
            trajectory.extend(selected_trajectory[start_idx:end_idx])
            dones.extend([False for _ in range(duration - 1)] + [True])
            timesteps_remaining -= duration
        states, rewards, actions = zip(*trajectory[:timesteps])

        states_batch.append(np.array(states))  # BHWC
        rewards_batch.append(np.array(rewards))
        dones_batch.append(np.array(dones))
        actions_batch.append(np.array(actions))
    return np.array(states_batch), np.array(rewards_batch), np.array(dones_batch), np.array(actions_batch)


def init():
    thread = Thread(target=play_game_thread)
    thread.daemon = True  # Hack to kill thread on ctrl+C
    thread.start()


if __name__ == '__main__':
    init()
    start_time = time.time()
    while True:
        duration = time.time() - start_time
        print('At t={:.03f} buffer contains {} items ({:.2f} games/minute)'.format(
            duration, len(replay_buffer), 60*len(replay_buffer)/duration))
        time.sleep(1)
        if not thread.is_alive():
            raise Exception('Simulator thread crashed')
        if len(replay_buffer) == MAX_BUFFER_LEN:
            print('Filled replay buffer in {:03f} sec'.format(duration))
            break
    thread.join()
    print('Finished')
