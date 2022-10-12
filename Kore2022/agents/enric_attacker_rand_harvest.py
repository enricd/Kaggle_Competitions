# from https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py
   
from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def get_closest_enemy_shipyard(board, position, me):
    min_dist = 1000000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard


def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    for shipyard in me.shipyards:
        action = None
        
        if turn > 100 and shipyard.ship_count >= 50:
            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
            if not closest_enemy_shipyard:
                print("No enemy shipyard found")
                continue
            if (closest_enemy_shipyard.ship_count + 20) > shipyard.ship_count:
                print("closest_enemy_shipyard.ship_count:", closest_enemy_shipyard.ship_count)
                print("shipyard.ship_count:", shipyard.ship_count)
                n_ships = shipyard.max_spawn
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships
                continue
            enemy_pos = closest_enemy_shipyard.position
            my_pos = shipyard.position
            flight_plan = "N" if enemy_pos.y > my_pos.y else "S"
            flight_plan += str(abs(enemy_pos.y - my_pos.y) - 1)
            flight_plan += "W" if enemy_pos.x < my_pos.x else "E"
            action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)  

        elif turn > 100 and shipyard.max_spawn < 5:
                n_ships = int(kore_left / spawn_cost)
                n_ships = shipyard.max_spawn if n_ships > shipyard.max_spawn else n_ships
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships         

        elif (turn % 5 == 0) and shipyard.ship_count > (10 + int(turn/5)) or (not me.fleets and shipyard.ship_count > 10):
            n_ships = int(shipyard.ship_count * 0.7)
            n_ships = (21 + randint(0,3)) if n_ships > 24 else n_ships
            if n_ships >= 21:
                lateral_dev = randint(0,8)
                flight_plan = [f'E{lateral_dev}N9W{lateral_dev}N', 
                               f'W{lateral_dev}N9E{lateral_dev}N',
                               f'E{lateral_dev}S9W{lateral_dev}S',
                               f'W{lateral_dev}S9E{lateral_dev}S'][randint(0,3)] 
            else:
                flight_plan = Direction.from_index(randint(0,1)).to_char()
            action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)
            shipyard.next_action = action

        elif kore_left > spawn_cost * shipyard.max_spawn:
            if turn > 300:
                n_ships = int(shipyard.max_spawn / 3)
            elif turn > 200:
                n_ships = int(shipyard.max_spawn / 1.5)
            else:
                n_ships = shipyard.max_spawn
            action = ShipyardAction.spawn_ships(n_ships)
            shipyard.next_action = action
            kore_left -= spawn_cost * n_ships

        elif kore_left > spawn_cost:
            action = ShipyardAction.spawn_ships(1)
            kore_left -= spawn_cost

        shipyard.next_action = action

    return me.next_actions
