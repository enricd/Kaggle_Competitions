
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

def agent(obs_dict, config_dict):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    food = observation.food[0]
    food_row, food_column = row_col(food, configuration.columns)

    if player_column % 2 == 0:
        return Action.WEST.name
    if player_column % 2 == 1:
        return Action.EAST.name
    return Action.WEST.name
