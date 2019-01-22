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
    def __init__(self, envs):
        start_time = time.time()
        self.batch_size = len(envs)
        self.envs = envs
        self.reset()
        self.action_space = self.envs[0].action_space
        #print('Initialized {} environments in {:.03f}s'.format(self.batch_size, time.time() - start_time))

    def reset(self):
        map_fn(lambda env: env.reset(), self.envs)

    def step(self, actions):
        start_time = time.time()
        assert len(actions) == len(self.envs)

        def run_one_step(env, action):
            state, reward, done, info = env.step(action)
            if done:
                env.reset()
            return state, reward, done, info

        results = map_fn(run_one_step, self.envs, actions)
        states, rewards, dones, infos = zip(*results)
        #print('Ran {} environments one step in {:.03f}s'.format(self.batch_size, time.time() - start_time))
        return np.array(states), np.array(rewards), np.array(dones), infos


if __name__ == '__main__':
    batch_size = 8
    envs = [gym.make('Pong-v0') for i in range(batch_size)]
    env = MultiEnvironment(envs)
    num_actions = env.action_space.n
    for i in range(100):
        actions = np.random.randint(0, num_actions, batch_size)
        states, rewards, dones, infos = env.step(actions)
        print('Step {}, Average reward {:.02f}'.format(i, np.mean(rewards)))
        imutil.show(states)
        time.sleep(1)
