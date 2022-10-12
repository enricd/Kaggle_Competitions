# This line creates the "agent.py" file that you need to download and submit to the competition page
# Otherwise, you can copy and paste this entire cell (without the first line), save it into a plain text file and name it "baseline_agent.py"

# Creating a besline strategy from the tutorial notebooks slightly improved with some extra conditions
# Original baseline: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py

from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def agent(obs, config):
    board = Board(obs, config)

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    for shipyard in me.shipyards:
        
        if shipyard.ship_count > (10 + int(turn/8)):
            n_ships = 2 + int(turn/10)
            direction = Direction.from_index(randint(1,2))
            if n_ships > 21:
                lateral_dev = randint(1,9)
                if lateral_dev == 2:
                    lateral_dev = 0
                flight_plan = [f'E{lateral_dev}N9W{lateral_dev}N', 
                               f'W{lateral_dev}N9E{lateral_dev}N'][randint(0,1)] 
            else:
                flight_plan = Direction.from_index(randint(0,1)).to_char()
            action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)
            shipyard.next_action = action
        
        elif kore_left > spawn_cost * shipyard.max_spawn:
            if turn > 200:
                n_ships = int(shipyard.max_spawn / 2)
            else:
                n_ships = shipyard.max_spawn
            action = ShipyardAction.spawn_ships(n_ships)
            shipyard.next_action = action
            kore_left -= spawn_cost * n_ships
        
        elif kore_left > spawn_cost:
            action = ShipyardAction.spawn_ships(1)
            shipyard.next_action = action
            kore_left -= spawn_cost

    return me.next_actions
