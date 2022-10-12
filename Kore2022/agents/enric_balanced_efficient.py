# from https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py
   
from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def get_closest_enemy_shipyard(board: Board, position: Point, me: Player):
    min_dist = 10000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard, min_dist


def flight_plan_to_cells(board: Board, initial_pos: Point, flight_plan: str, fleet: Fleet = None):
    size = board.configuration.size
    shipyards_pos = [shipyard.position for shipyard in board.shipyards.values()]
    flight_plan_cells = [initial_pos]
    pos = initial_pos
    if fleet is not None:
        dir = fleet.direction
    else:
        dir = Direction.from_char(flight_plan[0])

    for step in flight_plan:
        if step.isdigit():
            for _ in range(int(step)):
                pos = pos.translate(dir.to_point(), size)
                flight_plan_cells.append(pos)
        elif step == "C":
            return flight_plan_cells
        else:
            dir = Direction.from_char(step)
            pos = pos.translate(dir.to_point(), size)
            flight_plan_cells.append(pos)
        if pos in shipyards_pos:
            return flight_plan_cells

    for _ in range(21):
        pos = pos.translate(dir.to_point(), size)
        flight_plan_cells.append(pos)
        if pos in shipyards_pos:
            return flight_plan_cells

    return flight_plan_cells


def route_kore_density(shipyard: Shipyard, flight_plan: str, board: Board):
    flight_plan_cells = flight_plan_to_cells(board, shipyard.position, flight_plan)
    kore = 0
    for cell in board.cells.values():
        if cell.position in flight_plan_cells:
            kore += cell.kore

    if len(flight_plan_cells) <= 1:
        return 0

    return kore / (len(flight_plan_cells)-1)


def shipyard_already_created(shipyard: Shipyard, flight_plan: str, board: Board, me: Player):
    # flight plan should end with a "C"
    flight_plan_cells = flight_plan_to_cells(board, shipyard.position, flight_plan)
    final_pos = flight_plan_cells[-1]
    for other_shipyard in me.shipyards:
        if other_shipyard.position == final_pos:
            return True
    
    return False


def enemy_fleets_plan_pos(board: Board, me: Player):
    enemy_fleets_pos = []
    my_shipyards_pos = [shipyard.position for shipyard in board.shipyards.values() if shipyard.player_id == me.id]
    my_shipyards_ids = [shipyard.id for shipyard in board.shipyards.values() if shipyard.player_id == me.id]
    if board.fleets.values():
        for fleet in board.fleets.values():
            my_shipyard_conquered = None
            if fleet.player_id == me.id:
                continue

            flight_plan_cells = flight_plan_to_cells(board, fleet.position, fleet.flight_plan, fleet)
            if flight_plan_cells[-1] in my_shipyards_pos:
                my_shipyard_conquered = my_shipyards_ids[my_shipyards_pos.index(flight_plan_cells[-1])]

            enemy_fleets_pos.append([my_shipyard_conquered, fleet.ship_count, flight_plan_cells])

        return enemy_fleets_pos    
    return [[None, None, None]]


def detect_fleet_collision(start_pos: Point, flight_plan: str, board: Board, enemy_fleets_pos):
    collision = False
    n_ships = 0
    flight_plan_cells = flight_plan_to_cells(board, start_pos, flight_plan)
    for enemy_fleet in enemy_fleets_pos:
        for my_cell, enemy_cell in zip(flight_plan_cells, enemy_fleet[2]):
            if my_cell == enemy_cell:
                collision = True
                n_ships += enemy_fleet[1]

    return collision, n_ships


def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    enemy_fleets_pos = enemy_fleets_plan_pos(board, me)
    enemy_ships = sum([f[1] for f in enemy_fleets_pos if f[1] is not None]) + sum([shipyard.ship_count for shipyard in board.shipyards.values() if shipyard.player_id != me.id])
    my_targeted_shipyards = [(f[0], f[1]) for f in enemy_fleets_pos if f[0] is not None]

    for shipyard in me.shipyards:
        action = None
        targeted = shipyard.id in [s[0] for s in my_targeted_shipyards]
        available_ships = shipyard.ship_count
        _, dist_to_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)

        # If shipyard is targeted
        if targeted:
            attacking_ships = sum([f[1] for f in enemy_fleets_pos if f[0] == shipyard.id])
            if (available_ships + 5) < attacking_ships:
                if not me.fleets and kore_left < spawn_cost:
                    closest_enemy_shipyard, _ = get_closest_enemy_shipyard(board, shipyard.position, me)
                    if not closest_enemy_shipyard:
                        continue
                    if closest_enemy_shipyard.ship_count < shipyard.ship_count:
                        enemy_pos = closest_enemy_shipyard.position
                        my_pos = shipyard.position
                        if enemy_pos.y - my_pos.y == 0:
                            flight_plan = "W" if enemy_pos.x < my_pos.x else "E"
                        else:
                            flight_plan = "N" if enemy_pos.y > my_pos.y else "S"
                            flight_plan += str(abs(enemy_pos.y - my_pos.y) - 1)
                            flight_plan += "W" if enemy_pos.x < my_pos.x else "E"
                        n_ships = available_ships if available_ships < (closest_enemy_shipyard.ship_count + 50) else (closest_enemy_shipyard.ship_count + 50)
                        action = ShipyardAction.launch_fleet_with_flight_plan(available_ships, flight_plan)
                else:    
                    n_ships = int(kore_left / spawn_cost)
                    n_ships = n_ships if (n_ships <= shipyard.max_spawn and n_ships >= 1) else shipyard.max_spawn
                    action = ShipyardAction.spawn_ships(n_ships)
                    shipyard.next_action = action
                    kore_left -= spawn_cost * n_ships
                    shipyard.next_action = action
                    continue
            else: 
                if turn < 100:
                    available_ships -= int(attacking_ships + 5)
                else:
                    available_ships -= int(attacking_ships + 10)
        
        # If shipyard has been conquered recently, just spawn ships, dont harvest nor attack
        if (turn > 105 and shipyard.max_spawn < 4) and (shipyard.ship_count < 60 or shipyard.max_spawn < 2):
                n_ships = int(kore_left / spawn_cost)
                n_ships = shipyard.max_spawn if n_ships > shipyard.max_spawn else n_ships
                n_ships = n_ships if n_ships >= 1 else 1
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships     

        # Conquer enemy shipyard
        elif turn > 80 and (turn % 60 < 20) and (turn % 3 == 1) and available_ships >= 25:
            closest_enemy_shipyard, _ = get_closest_enemy_shipyard(board, shipyard.position, me)
            if not closest_enemy_shipyard:
                continue
            if (
                (closest_enemy_shipyard.max_spawn <= 3 and (closest_enemy_shipyard.ship_count + 25) < available_ships) or 
                (closest_enemy_shipyard.ship_count + 40) < available_ships
                ):
                enemy_pos = closest_enemy_shipyard.position
                my_pos = shipyard.position
                if enemy_pos.y - my_pos.y == 0:
                    flight_plan = "W" if enemy_pos.x < my_pos.x else "E"
                else:
                    flight_plan = "N" if enemy_pos.y > my_pos.y else "S"
                    flight_plan += str(abs(enemy_pos.y - my_pos.y) - 1)
                    flight_plan += "W" if enemy_pos.x < my_pos.x else "E"
                n_ships = (available_ships - 10) if available_ships < (closest_enemy_shipyard.ship_count + 70) else (closest_enemy_shipyard.ship_count + 70)
                action = ShipyardAction.launch_fleet_with_flight_plan(available_ships, flight_plan)
                shipyard.next_action = action
                continue
            
            # Spawning ships instead of conquering as there are not enough ships
            if turn < 300:
                n_ships = shipyard.max_spawn
            else:
                n_ships = int(shipyard.max_spawn / 3)
            n_ships = n_ships if n_ships >= 1 else 1
            action = ShipyardAction.spawn_ships(n_ships)
            kore_left -= spawn_cost * n_ships

        # Creating a new shipyard
        elif (80 < turn < 300) and available_ships > 80 and len(me.shipyards) <= 4:
            n_ships = available_ships - 23
            flight_plan = "S" if shipyard.position.y > board.configuration.size // 2 else "N"
            flight_plan += "2"
            flight_plan += "E" if flight_plan[0] == "S" else "W"
            flight_plan += "3"
            flight_plan += "C"
            if shipyard_already_created(shipyard, flight_plan, board, me):
                flight_plan = "S" if shipyard.position.y < board.configuration.size // 2 else "N"
                flight_plan += "2"
                flight_plan += "W" if flight_plan[0] == "S" else "E"
                flight_plan += "3"
                flight_plan += "C"
                if shipyard_already_created(shipyard, flight_plan, board, me):
                    flight_plan = "S" if shipyard.position.y < board.configuration.size // 2 else "N"
                    flight_plan += "2"
                    flight_plan += "E" if flight_plan[0] == "S" else "W"
                    flight_plan += "3"
                    flight_plan += "C"
            action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)

        # Harvesting
        elif (turn % 3 == 0) and (available_ships > 23 or (len(me.fleets) < 3) and available_ships > 2):
            n_ships = int(available_ships * ((23 + dist_to_enemy_shipyard) / 46))
            n_ships = (22 + randint(int(0 + turn*0.02), int(4 + turn*0.04))) if n_ships > 27 else n_ships
            n_ships = 3 if n_ships < 21 else n_ships
            if n_ships >= 21:
                best_kore_density = 0
                best_flight_plan = "N"
                for _ in range(15):
                    lateral_dev = randint(1,5)
                    vertical_dev = randint(0,5)
                    flight_plan = [f"E{lateral_dev}N9W{lateral_dev}N", 
                                f"W{lateral_dev}N9E{lateral_dev}N",
                                f"N{lateral_dev}E{vertical_dev}S{lateral_dev}W",
                                f"E{lateral_dev}S{vertical_dev}W{lateral_dev}N",
                                f"S{lateral_dev}W{vertical_dev}N{lateral_dev}E",
                                f"W{lateral_dev}N{vertical_dev}E{lateral_dev}S",
                                f"N{lateral_dev}E{vertical_dev}W{vertical_dev}S",
                                f"N{lateral_dev}W{vertical_dev}E{vertical_dev}S",
                                f"S{lateral_dev}E{vertical_dev}W{vertical_dev}N",
                                f"S{lateral_dev}W{vertical_dev}E{vertical_dev}N",
                                ][randint(0,9)] 
                    kore_density = route_kore_density(shipyard, flight_plan, board)
                    if kore_density > best_kore_density:
                        collision, enemy_n_ships = detect_fleet_collision(shipyard.position, flight_plan, board, enemy_fleets_pos)
                        if collision and enemy_n_ships >= n_ships:
                            if (enemy_n_ships + 5) < available_ships:
                                post_n_ships = enemy_n_ships + 5
                            else:
                                continue
                        else:
                            post_n_ships = n_ships
                        best_kore_density = kore_density
                        best_flight_plan = flight_plan
                flight_plan = best_flight_plan
                action = ShipyardAction.launch_fleet_with_flight_plan(post_n_ships, flight_plan)    
            else:
                flight_plan = Direction.from_index(randint(0,1)).to_char()
                action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)

        # Attacking enemy fleets
        #elif (turn % 3 == 2) and (available_ships > 23):
    

        # Spawining ships
        elif kore_left > spawn_cost * shipyard.max_spawn:
            if turn > 370:
                n_ships = 1
            if turn > 300:
                n_ships = int(shipyard.max_spawn / 3)
            elif turn > 200:
                n_ships = int(shipyard.max_spawn / 1.5)
            else:
                n_ships = shipyard.max_spawn
            n_ships = n_ships if n_ships >= 1 else 1
            action = ShipyardAction.spawn_ships(n_ships)
            shipyard.next_action = action
            kore_left -= spawn_cost * n_ships

        elif kore_left > spawn_cost:
            action = ShipyardAction.spawn_ships(1)
            kore_left -= spawn_cost

        shipyard.next_action = action

    return me.next_actions
