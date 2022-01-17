
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate, adjacent_positions
import numpy as np

def min_dist(position: int, food: List[int], columns: int, rows: int):
    row, column = row_col(position, columns)
    return min(
        min(abs(row - food_row), (abs(row - food_row)-rows) + min(abs(column - food_column), (abs(column - food_column)-columns))
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )
        
def score(next_position, food, columns, rows, bodies, self_len):
    
    body_collision = next_position in bodies
    danger_move = next_position in head_adjacent_positions
    food_dist = min_dist(next_position, food, columns, rows) if (self_len < 4) else 15
    
    score = (15-food_dist) + (body_collision*(-1e6)) + (danger_move*(-1e4))
    return score    
        

class MyAgent:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation: Observation):
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]
    
        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, rows, columns)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        
        # Move to the closest food
        position = geese[observation.index][0]
        
        actions = {
            action : score(new_position, food, columns, rows)
            #action: min_dist(new_position, food, columns, rows)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if self.last_action is None or action != self.last_action.opposite()
        }        
        
        action = max(actions, key=actions.get) if any(actions) else choice([action for action in Action])
        self.last_action = action
        return action.name


cached_my_agents = {}
        
def my_agent(obs, config):
    index = obs["index"]
    if index not in cached_my_agents:
        cached_my_agents[index] = MyAgent(Configuration(config))
    return cached_my_agents[index](Observation(obs))
        
        

        
