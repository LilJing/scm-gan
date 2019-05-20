from envs import sc2_star_intruders
from envs import betterpong
from envs import gridworld
from envs import gameoflife


def allocate_datasource(datasource_name):
    if datasource_name == 'sc2_star_intruders':
        return SC2StarIntruders()
    elif datasource_name == 'sc2_star_intruders_variant_a':
        return SC2StarIntruders('StarIntrudersVariantA')
    elif datasource_name == 'sc2_star_intruders_variant_b':
        return SC2StarIntruders('StarIntrudersVariantB')
    elif datasource_name == 'sc2_star_intruders_variant_c':
        return SC2StarIntruders('StarIntrudersVariantC')
    elif datasource_name == 'pong':
        return Pong()
    elif datasource_name == 'gridworld':
        return GridWorld()
    elif datasource_name == 'gameoflife':
        return GameOfLife()
    msg = 'Failed to find datasource with name {}'.format(datasource_name)
    raise ValueError(msg)


class Datasource():
    def __init__(self):
        pass
    def convert_frame(self, state):
        return state


class SC2StarIntruders(Datasource):
    def __init__(self, map_name=None):
        # global map filename hack
        if map_name:
            sc2_star_intruders.MAP_NAME = map_name
        self.map_name = map_name
        self.binary_input_channels = sc2_star_intruders.NUM_ACTIONS
        self.scalar_output_channels = sc2_star_intruders.NUM_REWARDS
        self.conv_input_channels = 4
        self.conv_output_channels = 4

    def make_env(self):
        return sc2_star_intruders.StarIntrudersEnvironment(map_name=self.map_name)

    def convert_frame(self, state):
        return sc2_star_intruders.convert_frame(state)

    def get_trajectories(self, *args, **kwargs):
        return sc2_star_intruders.get_trajectories(*args, **kwargs)


class Pong(Datasource):
    def __init__(self):
        self.binary_input_channels = betterpong.NUM_ACTIONS
        self.scalar_output_channels = betterpong.NUM_REWARDS
        self.conv_input_channels = 3
        self.conv_output_channels = 3

    def make_env(self):
        return betterpong.BetterPongEnv()

    def get_trajectories(self, *args, **kwargs):
        states, rewards, dones, actions = betterpong.get_trajectories(*args, **kwargs)
        return states, rewards, dones, actions


class GridWorld(Datasource):
    def __init__(self):
        self.binary_input_channels = gridworld.NUM_ACTIONS
        self.scalar_output_channels = gridworld.NUM_REWARDS
        self.conv_input_channels = 3
        self.conv_output_channels = 3

    def make_env(self):
        return gridworld.Env()

    def get_trajectories(self, *args, **kwargs):
        states, rewards, dones, actions = gridworld.get_trajectories(*args, **kwargs)
        return states, rewards, dones, actions


class GameOfLife(Datasource):
    def __init__(self):
        self.binary_input_channels = 1
        self.scalar_output_channels = 1
        self.conv_input_channels = 1
        self.conv_output_channels = 1

    def make_env(self):
        return gameoflife.Env()

    def get_trajectories(self, *args, **kwargs):
        states, rewards, dones, actions = gameoflife.get_trajectories(*args, **kwargs)
        return states, rewards, dones, actions
