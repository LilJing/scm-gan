import gym
import numpy as np
import imutil
import time
from concurrent import futures


def map_fn(fn, *iterables):
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        result_iterator = executor.map(fn, *iterables)
    return [i for i in result_iterator]


class MultiEnvironment():
    def __init__(self, name, batch_size):
        start_time = time.time()
        self.batch_size = batch_size
        #self.envs = map_fn(lambda idx: gym.make(name), range(batch_size))
        # ALE is non-threadsafe
        self.envs = [gym.make(name) for i in range(batch_size)]
        # Disable frameskip
        for env in self.envs:
            env.unwrapped.frameskip = 1
        self.reset()
        print('Initialized {} environments in {:.03f}s'.format(self.batch_size, time.time() - start_time))

    def reset(self):
        for env in self.envs:
            reset_env(env)

    def step(self, actions):
        start_time = time.time()
        assert len(actions) == len(self.envs)

        def run_one_step(env, action):
            new_state = []
            cumulative_reward = 0
            for _ in range(3):
                state, reward, done, info = env.step(action)
                if done:
                    reset_env(env)
                cumulative_reward += reward
                new_state.append(state)
            frames = convert_pong(np.array(new_state))
            return frames, cumulative_reward, done, info

        results = map_fn(run_one_step, self.envs, actions)
        #print('Ran {} environments one step in {:.03f}s'.format(self.batch_size, time.time() - start_time))
        states, rewards, dones, infos = zip(*results)
        return states, rewards, dones, infos


def reset_env(env):
    # Pong: wait until the enemy paddle appears
    env.reset()
    for _ in range(100):
        env.step(0)


def convert_pong(img_sequence):
    from skimage.measure import block_reduce
    pixels = img_sequence.mean(-1)  # Convert to monochrome
    pixels = np.array(pixels)[:,34:-16,:]  # Crop along height dimension
    assert pixels.shape == (3, 160, 160)
    # Downsample to 32x32x3
    pixels = np.array([block_reduce(frame, (5,5), np.max) for frame in pixels])
    pixels -= pixels.min()
    pixels[np.where(pixels > 0)] = 1.0  # Binarize
    return pixels


if __name__ == '__main__':
    batch_size = 64
    env = MultiEnvironment('Pong-v0', batch_size)
    for i in range(10):
        actions = np.random.randint(0, 4, size=batch_size)
        states, rewards, dones, infos = env.step(actions)
        imutil.show(states[:4])

