from envs import sc2_zone_intruders
from envs.sc2_zone_intruders import ZoneIntrudersEnvironment
from envs import betterpong


def allocate_datasource(datasource_name):
    if datasource_name == 'sc2_zone_intruders':
        return SC2ZoneIntruders()
    elif datasource_name == 'pong':
        return Pong()
    msg = 'Failed to find datasource with name {}'.format(datasource_name)
    raise ValueError(msg)


class Datasource():
    def __init__(self):
        pass


class SC2ZoneIntruders(Datasource):
    def __init__(self):
        self.NUM_ACTIONS = sc2_zone_intruders.NUM_ACTIONS
        self.NUM_REWARDS = sc2_zone_intruders.NUM_REWARDS
        self.RGB_SIZE = sc2_zone_intruders.RGB_SIZE

    def make_env(self):
        return ZoneIntrudersEnvironment()

    def convert_frame(self, state):
        return sc2_zone_intruders.convert_frame(state)

    def get_trajectories(self, *args, **kwargs):
        return sc2_zone_intruders.get_trajectories(*args, **kwargs)


class Pong(Datasource):
    def __init__(self):
        self.NUM_ACTIONS = betterpong.NUM_ACTIONS
        self.NUM_REWARDS = betterpong.NUM_REWARDS
        self.RGB_SIZE = betterpong.RGB_SIZE

    def make_env(self):
        return betterpong.BetterPongEnv

    def convert_frame(self, state):
        return state

    def get_trajectories(self, *args, **kwargs):
        states, rewards, dones, actions = betterpong.get_trajectories(*args, **kwargs)
        return states, states, rewards, dones, actions
