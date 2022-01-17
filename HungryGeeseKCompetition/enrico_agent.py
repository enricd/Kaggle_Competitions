
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate, adjacent_positions
import numpy as np

bucle = False

def min_food_dist(position, food, columns, rows):
    row, column = row_col(position, columns)
    return min(
        min(abs(row - food_row), (abs(abs(row - food_row)-rows))) + min(abs(column - food_column), (abs(abs(column - food_column)-columns)))
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )

def min_tail_dist(position, my_tail, columns, rows):
    row, column = row_col(position, columns)
    tail_row, tail_column = row_col(my_tail, columns)
    return min(abs(row - tail_row), (abs(abs(row - tail_row)-rows))) + min(abs(column - tail_column), (abs(abs(column - tail_column)-columns)))

def opposite(action):
    if action == Action.NORTH:
        return Action.SOUTH
    if action == Action.SOUTH:
        return Action.NORTH
    if action == Action.EAST:
        return Action.WEST
    if action == Action.WEST:
        return Action.EAST
    raise TypeError(str(action) + " is not a valid Action.")
        
        
def score(next_position, food, columns, rows, bodies, self_len, head_adjacent_positions, my_tail, tails, action, opponents, step):
    
    global bucle
    
    if self_len > 4 and step < 100:
        bucle = True
    else:
        bucle = False
    
    if self_len > 8:
        bucle = True
    
    if step > 150:
        if self_len <= max([len(op) for op in opponents]):
            bucle = False
    
    body_collision = next_position in bodies
    danger_move = (next_position in head_adjacent_positions) if (min_tail_dist(next_position, my_tail, columns, rows) > 0) else 0
    food_dist = min_food_dist(next_position, food, columns, rows) if (self_len < 9 or not bucle) else min_tail_dist(next_position, my_tail, columns, rows)
    to_enemy_tail = next_position in tails
    
    if bucle:
        if self_len < 7:
            bucle = False
            
    food_safe = 1
    near_ops = 0
    if (self_len > 4) and not bucle:
        for op in opponents:
            if min_tail_dist(next_position, op[0], columns, rows) < 5:
                near_ops += 1    
        if near_ops > 1:
            food_safe = -1  
            
    way_out = 0
    max_way_out = 0
    for next_action in Action:
        if way_out == 4:
            max_way_out = 4
            break 
        
        if way_out > max_way_out:
            max_way_out = way_out
            
        way_out = 0
        if next_action != opposite(action):   
            next2_position = translate(next_position, next_action, columns, rows)
            if next2_position not in bodies:
                way_out = 1
                if next2_position in head_adjacent_positions:
                    way_out -= 0.5               
                
                    for next2_action in Action: 
                        if way_out == 4:
                            break
                        if next2_action != opposite(next_action):
                            next3_position = translate(next2_position, next2_action, columns, rows)
                            if next3_position not in bodies:
                                way_out += 1
                                if next3_position in head_adjacent_positions:
                                    way_out -= 0.5 

                                    for next3_action in Action:
                                        if way_out == 4:
                                            break
                                        if next3_action != opposite(next2_action):  
                                            next4_position = translate(next3_position, next3_action, columns, rows)
                                            if next4_position not in bodies:
                                                way_out += 1
                                                if next4_position in head_adjacent_positions:
                                                    way_out -= 0.5 

                                                    for next4_action in Action: 
                                                        if next4_action != opposite(next3_action):
                                                            next5_position = translate(next2_position, next2_action, columns, rows)
                                                            if next5_position not in bodies:
                                                                way_out += 1
                                                                if next5_position in head_adjacent_positions:
                                                                    way_out -= 0.5
                                                                break
        
        
    score = (15-(food_dist*food_safe)) + (body_collision*(-1e6+food_dist)) + (max_way_out*1e4) + (2*danger_move*(-1e4+food_dist)) + (to_enemy_tail*(-1e2)) 
    
    return score    
        

class MyAgent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation):
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
            for opponent_head_adjacent in adjacent_positions(opponent[0], columns, rows)
        }

        
        # Don't move into any bodies
        
        bodies = []
        for goose in geese:
            if len(goose)==1:
                bodies += [position for position in goose]
            elif len(goose)>1:
                bodies += [position for position in goose[:-1]]
        
        #bodies = {position for goose in geese for position in goose[:-1]} 
        tails = {goose[-1] for goose in opponents if len(goose)>1}

        
        # Move to the closest food
        position = geese[observation.index][0]
        
        actions = {
            action : score(new_position, food, columns, rows, bodies, len(geese[observation.index]), head_adjacent_positions, geese[observation.index][-1], tails, action, opponents, observation.step)
            #action: min_dist(new_position, food, columns, rows)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (self.last_action is None or action != opposite(self.last_action))
        }  
        
        action = max(actions, key=actions.get) if any(actions) else choice([action for action in Action if (self.last_action is None or action != opposite(self.last_action))])
        self.last_action = action
        
        if(observation.index == 0):    
            print(len(geese[observation.index]), "\n", action.name, "\n", actions, "\n", opposite(action), "\n player: ", geese[observation.index], " head adj: ", head_adjacent_positions)
        
        return action.name


cached_my_agents = {}
        
def my_agent(obs, config):
    index = obs["index"]
    if index not in cached_my_agents:
        cached_my_agents[index] = MyAgent(Configuration(config))
    return cached_my_agents[index](Observation(obs))
        
        

        
