import sys
import gym
import time
import numpy as np

from sc2env.environments.micro_battle import MicroBattleEnvironment

MAX_BUFFER_LEN = 1000
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


from threading import Thread
if __name__ == '__main__':
    thread = Thread(target=play_game_thread)
    thread.daemon = True  # Hack to kill thread on ctrl+C
    thread.start()
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
