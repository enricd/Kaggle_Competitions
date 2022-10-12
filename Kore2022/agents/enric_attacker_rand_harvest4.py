# ---------- OLD VERSION ----------
   
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
                    
            for _ in range(21):
                pos += dir.to_point()
                flight_plan_cells.append(pos)
            
            if shipyard.position in flight_plan_cells:
                is_targeted = True
                attacking_ships += fleet.ship_count 
        
    return is_targeted, attacking_ships


def route_kore_density(shipyard: Shipyard, flight_plan: str, board: Board):
    initial_pos = shipyard.position
    flight_plan_cells = [initial_pos]
    pos = initial_pos
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
            
    for _ in range(21):
        pos += dir.to_point()
        flight_plan_cells.append(pos)
        if pos == initial_pos:
            break
    
    kore = 0
    for cell in board.cells.values():
        if cell.position in flight_plan_cells:
            kore += cell.kore

    return kore / (len(flight_plan_cells)-2)


def shipyard_already_created(shipyard: Shipyard, flight_plan: str, me: Player):
    initial_pos = shipyard.position
    flight_plan_cells = [initial_pos]
    pos = initial_pos
    if len(me.shipyards) <= 1:
        return False

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
    
    final_pos = flight_plan_cells[-1]
    for other_shipyard in me.shipyards:
        if other_shipyard.position == final_pos:
            return True
    
    return False


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

        # If shipyard is targeted
        if targeted:
            if (available_ships + 5) < attacking_ships:
                if not me.fleets and kore_left < spawn_cost:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
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
                    n_ships = shipyard.max_spawn
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
        if (turn > 105 and shipyard.max_spawn < 4) and (shipyard.ship_count < 70 or shipyard.max_spawn < 2):
                n_ships = int(kore_left / spawn_cost)
                n_ships = shipyard.max_spawn if n_ships > shipyard.max_spawn else n_ships
                n_ships = n_ships if n_ships >= 1 else 1
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships     

        # Conquer enemy shipyard
        elif turn > 80 and (turn % 40 < 20) and (turn % 4 == 1) and available_ships >= 25:
            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
            if not closest_enemy_shipyard:
                continue
            if (closest_enemy_shipyard.max_spawn <= 3 and (closest_enemy_shipyard.ship_count + 20) > available_ships) or (closest_enemy_shipyard.ship_count + 40) > available_ships:
                if turn < 300:
                    n_ships = shipyard.max_spawn
                else:
                    n_ships = int(shipyard.max_spawn / 3)
                n_ships = n_ships if n_ships >= 1 else 1
                action = ShipyardAction.spawn_ships(n_ships)
                shipyard.next_action = action
                kore_left -= spawn_cost * n_ships
                shipyard.next_action = action
                continue
            enemy_pos = closest_enemy_shipyard.position
            my_pos = shipyard.position
            if enemy_pos.y - my_pos.y == 0:
                flight_plan = "W" if enemy_pos.x < my_pos.x else "E"
            else:
                flight_plan = "N" if enemy_pos.y > my_pos.y else "S"
                flight_plan += str(abs(enemy_pos.y - my_pos.y) - 1)
                flight_plan += "W" if enemy_pos.x < my_pos.x else "E"
            n_ships = available_ships if available_ships < (closest_enemy_shipyard.ship_count + 70) else (closest_enemy_shipyard.ship_count + 70)
            action = ShipyardAction.launch_fleet_with_flight_plan(available_ships, flight_plan)

        # Creating a new shipyard
        elif (80 < turn < 300) and available_ships > 80 and len(me.shipyards) <= 4:
            n_ships = available_ships - 21
            flight_plan = "S" if shipyard.position.y > board.configuration.size // 2 else "N"
            flight_plan += "3"
            flight_plan += "E" if flight_plan[0] == "S" else "W"
            flight_plan += "3"
            flight_plan += "C"
            if shipyard_already_created(shipyard, flight_plan, me):
                flight_plan = "S" if shipyard.position.y < board.configuration.size // 2 else "N"
                flight_plan += "3"
                flight_plan += "W" if flight_plan[0] == "S" else "E"
                flight_plan += "3"
                flight_plan += "C"
            action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)

        # Harvesting
        elif (turn % 4 == 0) and (available_ships > 23 or (len(me.fleets) < 3) and available_ships > 2):
            n_ships = int(available_ships * 0.9)
            n_ships = (22 + randint(int(0 + turn*0.02), int(5 + turn*0.04))) if n_ships > 27 else n_ships
            n_ships = 3 if n_ships < 21 else n_ships
            if n_ships >= 21:
                best_kore_density = 0
                best_flight_plan = "N"
                for i in range(15):
                    lateral_dev = randint(1,6)
                    vertical_dev = randint(0,6)
                    flight_plan = [f"E{lateral_dev}N9W{lateral_dev}N", 
                                f"W{lateral_dev}N9E{lateral_dev}N",
                                f"E{lateral_dev}S9W{lateral_dev}S",
                                f"W{lateral_dev}S9E{lateral_dev}S",
                                f"N{lateral_dev}E{vertical_dev}S{lateral_dev}W",
                                f"E{lateral_dev}S{vertical_dev}W{lateral_dev}N",
                                f"S{lateral_dev}W{vertical_dev}N{lateral_dev}E",
                                f"W{lateral_dev}N{vertical_dev}E{lateral_dev}S",
                                ][randint(0,7)] 
                    kore_density = route_kore_density(shipyard, flight_plan, board)
                    if kore_density > best_kore_density:
                        best_kore_density = kore_density
                        best_flight_plan = flight_plan
                flight_plan = best_flight_plan    
            else:
                flight_plan = Direction.from_index(randint(0,1)).to_char()
            action = ShipyardAction.launch_fleet_with_flight_plan(n_ships, flight_plan)

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
