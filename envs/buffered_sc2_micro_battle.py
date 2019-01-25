import sys
import random
import gym
import time
import numpy as np
import imutil
from threading import Thread

from sc2env.environments.micro_battle import MicroBattleEnvironment

MAX_BUFFER_LEN = 128
replay_buffer = []


def play_game_thread():
    policy = lambda: env.action_space.sample()
    iters = 0
    while True:
        if iters % 1000 == 0:
            print('Rebuilding SC2 simulator...')
            env = MicroBattleEnvironment(render=True)
            print('Built SC2 simulator successfully')
        play_episode(env, policy)
        time.sleep(.01)
        iters += 1

def play_episode(env, policy):
    states, rewards, actions = [], [], []
    state = env.reset()
    reward = 0
    done = False
    while True:
        action = policy()
        state = state[3]  # Rendered game pixels
        state = state.transpose((2,0,1))  # HWC -> CHW
        state = state * (1/255)  # [0,1]
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        state, reward, done, info = env.step(action)
    trajectory = (np.array(states), np.array(rewards), np.array(actions))
    add_to_replay_buffer(trajectory)


def add_to_replay_buffer(episode):
    if len(replay_buffer) < MAX_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(1, MAX_BUFFER_LEN)
        replay_buffer[idx] = episode


simulator_running = False
def get_trajectories(batch_size=8, timesteps=10, random_start=True):
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
