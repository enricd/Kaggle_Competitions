# from https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py
   
from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def get_closest_enemy_shipyard(board: Board, position: Point, me: Player):
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


def shipyard_is_targeted(shipyard: Shipyard, board: Board, me: Player):
    is_targeted = False
    attacking_ships = 0
    if board.fleets.values():
        for fleet in board.fleets.values():
            if fleet.player_id == me.id:
                continue
            flight_plan = fleet.flight_plan
            current_pos = fleet.position
            flight_plan_cells = [current_pos]
            pos = current_pos
            dir = fleet.direction
            for step in flight_plan:
                if step.isdigit():
                    for _ in range(int(step)):
                        pos += dir.to_point()
                        flight_plan_cells.append(pos)
                elif step == "C":
                    break
                else:
                    dir = Direction.from_char(step)
                    pos += dir.to_point()
                    flight_plan_cells.append(pos)   
            
            if shipyard.position in flight_plan_cells:
                is_targeted = True
                attacking_ships += fleet.ship_count 
        
    return is_targeted, attacking_ships


def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    for shipyard in me.shipyards:
        action = None
        targeted, attacking_ships = shipyard_is_targeted(shipyard, board, me)
        available_ships = shipyard.ship_count

        # If shipyard is targeted and there are 
        if targeted:
            if (available_ships + 20) < attacking_ships:
                n_ships = shipyard.max_spawn
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships
                continue
            else: 
                available_ships -= (attacking_ships + 20)
        
        # If shipyard has been conquered recently, just spawn ships, don't harvest nor attack
        if turn > 100 and shipyard.max_spawn < 4:
                n_ships = int(kore_left / spawn_cost)
                n_ships = shipyard.max_spawn if n_ships > shipyard.max_spawn else n_ships
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships     

        # Conquer enemy shipyard
        elif turn % 50 < 20 and available_ships >= 50:
            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
            if not closest_enemy_shipyard:
                continue
            if (closest_enemy_shipyard.ship_count + 25) > available_ships:
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
            action = ShipyardAction.launch_fleet_with_flight_plan(available_ships, flight_plan)    

        elif (turn % 3 == 0) and (available_ships > (8 + int(turn/5)) or (not me.fleets)):
            n_ships = int(available_ships * 0.7)
            n_ships = (21 + randint(0,3)) if n_ships > 24 else n_ships
            n_ships = 3 if n_ships < 21 else n_ships
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
