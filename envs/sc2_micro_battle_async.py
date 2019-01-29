import sys
import math
import random
import time
import numpy as np
from threading import Thread

from sc2env.environments.micro_battle import MicroBattleEnvironment

REPLAY_BUFFER_LEN = 500
MAX_TRAJECTORY_LEN = 100
MAX_EPISODES_PER_ENVIRONMENT = 500
replay_buffer = []
initialized = False
env = None
simulation_iters = 0


def init():
    global initialized
    thread = Thread(target=play_game_thread)
    thread.daemon = True  # hack to kill on ctrl+C
    thread.start()
    initialized = True


def play_game_thread():
    while True:
        simulate_to_replay_buffer(1)


# Simulate batch_size episodes and add them to the replay buffer
def simulate_to_replay_buffer(batch_size):
    global simulation_iters
    global env
    policy = lambda: env.action_space.sample()
    for _ in range(batch_size):
        if simulation_iters % MAX_EPISODES_PER_ENVIRONMENT == 0:
            print('At iteration {}, rebuilding SC2 simulator...'.format(simulation_iters))
            if env:
                env.sc2env.close()
            env = MicroBattleEnvironment(render=True)
            print('Built SC2 simulator successfully')
        play_episode(env, policy)
        simulation_iters += 1


def play_episode(env, policy):
    states, rewards, actions = [], [], []
    state = env.reset()
    # hack: skip first few steps
    for _ in range(3):
        state, _, _, _ = env.step(0)
    reward = 0
    done = False
    while True:
        action = policy()
        state = state[3]  # Rendered game pixels
        state = state.transpose((2,0,1))  # HWC -> CHW
        state = state * (1/255)  # [0,1]
        state[:, :80] = 0
        state[:, -80:] = 0
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if len(states) >= MAX_TRAJECTORY_LEN:
            print('Warning: ending trajectory at {} timesteps'.format(len(states)))
            done = True
        if done:
            break
        state, reward, done, info = env.step(action)
    trajectory = (np.array(states), np.array(rewards), np.array(actions))
    add_to_replay_buffer(trajectory)


def add_to_replay_buffer(episode):
    if len(replay_buffer) < REPLAY_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(1, REPLAY_BUFFER_LEN)
        replay_buffer[idx] = episode


def get_trajectories(batch_size=8, timesteps=10, random_start=True):
    if not initialized:
        init()

    # Run the game and add new episodes into the replay buffer
    while len(replay_buffer) < batch_size:
        print('Waiting for simulator, buffer size {}...'.format(len(replay_buffer)))
        time.sleep(0.5)

    # Sample episodes from the replay buffer
    states_batch, rewards_batch, dones_batch, actions_batch = [], [], [], []
    for batch_idx in range(batch_size):
        # Accumulate trajectory clips
        states, rewards, actions = [], [], []
        timesteps_remaining = timesteps
        dones = []
        while timesteps_remaining > 0:
            selected_states, selected_rewards, selected_actions = random.choice(replay_buffer)
            if random_start:
                start_idx = np.random.randint(0, len(selected_states) - 3)
            else:
                start_idx = 0
            end_idx = min(start_idx + timesteps_remaining, len(selected_states) - 1)
            duration = end_idx - start_idx
            states.extend(selected_states[start_idx:end_idx])
            rewards.extend(selected_rewards[start_idx:end_idx])
            actions.extend(selected_actions[start_idx:end_idx])
            dones.extend([False for _ in range(duration - 1)] + [True])
            timesteps_remaining -= duration

        states_batch.append(np.array(states))  # BHWC
        rewards_batch.append(np.array(rewards))
        dones_batch.append(np.array(dones))
        actions_batch.append(np.array(actions))
    return np.array(states_batch), np.array(rewards_batch), np.array(dones_batch), np.array(actions_batch)


if __name__ == '__main__':
    start_time = time.time()
    batch_size = 8
    MAX_ITERS = 1000 * 10
    for i in range(0, MAX_ITERS, batch_size):
        get_trajectories(batch_size)
        duration = time.time() - start_time
        print('Generated {} trajectories in {:.03f} sec ({:.02f} traj/sec) replay buffer len {}'.format(
            i, duration, i / duration, len(replay_buffer)))
