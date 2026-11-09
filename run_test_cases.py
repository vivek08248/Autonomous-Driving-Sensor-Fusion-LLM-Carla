# run_test_cases.py

import sys
import carla
import time
import config

from evaluation_logger import MetricsLogger
from main_closed_loop import (
    run_simulation_cycle,
    update_spectator,
    setup_simulation,
    restore_simulation,
    spawn_ego_vehicle,
)


def _safe_blueprint(blueprint_lib, pattern, fallback="vehicle.*"):
    matches = list(blueprint_lib.filter(pattern))
    if matches:
        return matches[0]
    clean = pattern.split(".")[-1]
    matches = list(blueprint_lib.filter(f"*{clean}*"))
    if matches:
        return matches[0]
    if fallback:
        matches = list(blueprint_lib.filter(fallback))
        if matches:
            return matches[0]
    all_v = list(blueprint_lib.filter("vehicle.*"))
    if all_v:
        return all_v[0]
    raise RuntimeError(f"No blueprints available in CARLA library matching {pattern}")


_blueprint = _safe_blueprint


def _road_transform_ahead(world, ego_transform, distance_m, lateral_m=0.0, yaw_offset_deg=0.0, z_lift=0.45):
    """Computes a transform following the actual OpenDRIVE lane geometry ahead of ego."""
    wmap = world.get_map()
    ego_wp = wmap.get_waypoint(ego_transform.location, project_to_road=True)
    if ego_wp is not None and distance_m > 0.5:
        nxt = ego_wp.next(float(distance_m))
        if nxt:
            base_tf = nxt[0].transform
            right = base_tf.get_right_vector()
            loc = base_tf.location + right * float(lateral_m) + carla.Location(z=float(z_lift))
            rot = carla.Rotation(
                pitch=base_tf.rotation.pitch,
                yaw=base_tf.rotation.yaw + float(yaw_offset_deg),
                roll=base_tf.rotation.roll,
            )
            return carla.Transform(loc, rot)

    fwd = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    loc = ego_transform.location + fwd * float(distance_m) + right * float(lateral_m) + carla.Location(z=float(z_lift))
    rot = carla.Rotation(
        pitch=ego_transform.rotation.pitch,
        yaw=ego_transform.rotation.yaw + float(yaw_offset_deg),
        roll=ego_transform.rotation.roll,
    )
    return carla.Transform(loc, rot)


def _spawn_vehicle(world, blueprint_lib, transform, pattern="vehicle.*",
                   autopilot=False, traffic_manager_port=None):
    bp = _blueprint(blueprint_lib, pattern)
    actor = _safe_spawn_obstacle(world, bp, transform)
    if actor is None:
        return None
    if autopilot:
        if traffic_manager_port is None:
            traffic_manager_port = config.TRAFFIC_MANAGER_PORT
        try:
            actor.set_autopilot(True, traffic_manager_port)
        except Exception:
            pass
    return actor


def _safe_spawn_obstacle(world, bp, transform, max_retries=6):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "npc")

    loc = carla.Location(transform.location.x, transform.location.y, transform.location.z)
    rot = carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
    fwd = rot.get_forward_vector()
    right = rot.get_right_vector()

    candidate_transforms = [
        carla.Transform(loc, rot),
        carla.Transform(loc + carla.Location(z=0.35), rot),
        carla.Transform(loc - right * 0.35 + carla.Location(z=0.25), rot),
        carla.Transform(loc + right * 0.35 + carla.Location(z=0.25), rot),
        carla.Transform(loc + fwd * 1.2 + carla.Location(z=0.25), rot),
        carla.Transform(loc - fwd * 1.2 + carla.Location(z=0.25), rot),
    ]

    for tf in candidate_transforms:
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            return actor

    return None


def _spawn_walker(world, blueprint_lib, transform, direction, speed=1.5):
    bp = _blueprint(blueprint_lib, "walker.pedestrian.*")
    tf = carla.Transform(transform.location + carla.Location(z=0.5), transform.rotation)
    walker = world.try_spawn_actor(bp, tf)
    if walker is None:
        tf2 = carla.Transform(transform.location + carla.Location(z=1.0), transform.rotation)
        walker = world.try_spawn_actor(bp, tf2)
    if walker is None:
        return None
    control = carla.WalkerControl()
    control.direction = direction
    control.speed = speed
    walker.apply_control(control)
    return walker


def _opposite_direction(vector):
    return carla.Vector3D(-vector.x, -vector.y, -vector.z)


def _spawn_animal(world, blueprint_lib, transform, direction, speed=1.0):
    animal_patterns = ("static.prop.deer", "static.prop.dog", "static.prop.cat")
    for pattern in animal_patterns:
        matches = blueprint_lib.filter(pattern)
        if matches:
            animal = world.try_spawn_actor(matches[0], transform)
            if animal is not None:
                return animal
    return _spawn_walker(world, blueprint_lib, transform, direction, speed)


def _spawn_traffic(world, blueprint_lib, ego_transform, count=8):
    spawn_points = list(world.get_map().get_spawn_points())
    ego_location = ego_transform.location
    spawn_points.sort(key=lambda sp: sp.location.distance(ego_location), reverse=True)
    traffic = []
    for transform in spawn_points:
        if len(traffic) >= count:
            break
        if transform.location.distance(ego_location) < 40.0:
            continue
        vehicle = _spawn_vehicle(world, blueprint_lib, transform, autopilot=True)
        if vehicle is not None:
            traffic.append(vehicle)
    return traffic


def _set_nearest_traffic_light(world, ego_transform, state):
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    if not lights:
        return
    loc = ego_transform.location
    fwd = ego_transform.get_forward_vector()
    target, best_along = None, 1e9
    for light in lights:
        try:
            stops = light.get_stop_waypoints()
        except Exception:
            continue
        for sw in stops:
            p = sw.transform.location
            dx, dy = p.x - loc.x, p.y - loc.y
            along = dx * fwd.x + dy * fwd.y
            lateral = abs(dx * fwd.y - dy * fwd.x)
            if 3.0 < along < 120.0 and lateral < 3.5 and along < best_along:
                target, best_along = light, along
    if target is None:
        target = min(lights, key=lambda l: l.get_location().distance(loc))
    target.set_state(state)
    target.set_red_time(12.0)
    target.set_green_time(12.0)
    target.set_yellow_time(3.0)


def setup_scenario(world, scenario_name):
    """Spawns ego + scenario actors aligned with OpenDRIVE road geometry."""
    try:
        import extra_scenarios as extra
        if hasattr(extra, "CASES") and scenario_name in extra.CASES:
            ego, obstacles, runtime = extra.build(world, scenario_name)
            ego.runtime = runtime
            update_spectator(world, ego)
            return ego, obstacles
    except Exception as e:
        print(f"[Setup Notice] extra_scenarios builder: {e}")

    blueprint_lib = world.get_blueprint_library()
    ego_vehicle = spawn_ego_vehicle(world)

    # CRITICAL FIX: Tick the world once so ego_vehicle.get_transform() is initialized
    # and does NOT return (0, 0, 0)!
    if config.SYNCHRONOUS_MODE:
        world.tick()
    else:
        world.wait_for_tick()

    ego_transform = ego_vehicle.get_transform()
    if abs(ego_transform.location.x) < 1e-3 and abs(ego_transform.location.y) < 1e-3:
        sps = world.get_map().get_spawn_points()
        if sps:
            ego_transform = sps[0]

    update_spectator(world, ego_vehicle)
    obstacles = []

    if scenario_name == "TC_01_CLEAR_LANE":
        print("\n[Setup] Clear Lane")
        world.set_weather(carla.WeatherParameters.ClearNoon)

    elif scenario_name == "TC_02_STATIC_OBSTACLE":
        print("\n[Setup] Static Obstacle (18m ahead)")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = _safe_blueprint(blueprint_lib, "vehicle.audi.tt")
        tf = _road_transform_ahead(world, ego_transform, 18.0)
        obs = _safe_spawn_obstacle(world, bp, tf)
        if obs is not None:
            obstacles.append(obs)

    elif scenario_name == "TC_03_RAIN_AND_OBSTACLE":
        print("\n[Setup] Heavy Rain + Obstacle (14m ahead)")
        world.set_weather(carla.WeatherParameters.HardRainNoon)
        bp = _safe_blueprint(blueprint_lib, "vehicle.mini.cooper_s")
        tf = _road_transform_ahead(world, ego_transform, 14.0)
        obs = _safe_spawn_obstacle(world, bp, tf)
        if obs is not None:
            obstacles.append(obs)

    elif scenario_name == "TC_04_URBAN_JAYWALKER":
        print("\n[Setup] Urban Jaywalker")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        parked_bp = _safe_blueprint(blueprint_lib, "vehicle.nissan.micra")
        # Keep the occluding vehicle at the kerb.  At -1.6 m it overlaps the
        # driving corridor on narrow Town10 lanes and turns this pedestrian
        # scenario into a permanent blocked-lane scenario.
        parked_tf = _road_transform_ahead(world, ego_transform, 15.0, lateral_m=-3.8)
        obs = _safe_spawn_obstacle(world, parked_bp, parked_tf)
        if obs is not None:
            obstacles.append(obs)
        # Start at the road edge so this is a genuine crossing event rather
        # than a pedestrian spawned inside the ego lane.
        ped_tf = _road_transform_ahead(world, ego_transform, 17.5, lateral_m=-3.2)
        right = ped_tf.get_right_vector()
        ped = _spawn_walker(world, blueprint_lib, ped_tf, right, speed=1.6)
        if ped is not None:
            obstacles.append(ped)

    elif scenario_name == "TC_05_SEMI_RURAL_ENCROACH":
        print("\n[Setup] Semi-Rural Encroachment")
        world.set_weather(carla.WeatherParameters.CloudySunset)
        bp = _safe_blueprint(blueprint_lib, "vehicle.tata.nano")
        tf = _road_transform_ahead(world, ego_transform, 20.0, lateral_m=-1.1, yaw_offset_deg=15.0)
        obs = _safe_spawn_obstacle(world, bp, tf)
        if obs is not None:
            obstacles.append(obs)

    elif scenario_name == "TC_06_RURAL_SLOW_TRUCK":
        print("\n[Setup] Rural Slow Truck")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = _safe_blueprint(blueprint_lib, "vehicle.carlamotors.carlacola")
        tf = _road_transform_ahead(world, ego_transform, 22.0)
        truck = _safe_spawn_obstacle(world, bp, tf)
        if truck is not None:
            truck.set_target_velocity(tf.get_forward_vector() * 2.7)
            setattr(truck, "_dynamic_speed", 2.7)
            truck.apply_control(carla.VehicleControl(throttle=0.45, steer=0.0, brake=0.0, hand_brake=False))
            obstacles.append(truck)

    elif scenario_name == "TC_07_2WHEELER_CUT_IN":
        print("\n[Setup] 2-Wheeler Cut-In")
        world.set_weather(carla.WeatherParameters.ClearSunset)
        bp = _safe_blueprint(blueprint_lib, "vehicle.yamaha.yzf")
        tf = _road_transform_ahead(world, ego_transform, 14.0, lateral_m=2.8, yaw_offset_deg=-25.0)
        bike = _safe_spawn_obstacle(world, bp, tf)
        if bike is not None:
            bike.set_target_velocity(tf.get_forward_vector() * 3.8)
            obstacles.append(bike)

    elif scenario_name == "TC_08_NIGHT_VISION_LIMIT":
        print("\n[Setup] Night Vision Limit")
        world.set_weather(carla.WeatherParameters.ClearNight)
        bp = _safe_blueprint(blueprint_lib, "vehicle.charger.2020")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "0,0,0")
        tf = _road_transform_ahead(world, ego_transform, 16.0)
        obs = _safe_spawn_obstacle(world, bp, tf)
        if obs is not None:
            obstacles.append(obs)

    elif scenario_name == "TC_09_WRONG_WAY_ONCOMING":
        print("\n[Setup] Wrong Way Oncoming")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = _safe_blueprint(blueprint_lib, "vehicle.lincoln.mkz_2020")
        tf = _road_transform_ahead(world, ego_transform, 32.0, yaw_offset_deg=180.0)
        car = _safe_spawn_obstacle(world, bp, tf)
        if car is not None:
            car.set_target_velocity(tf.get_forward_vector() * 5.0)
            obstacles.append(car)

    elif scenario_name == "TC_10_HEAVY_FOG_GRIDLOCK":
        print("\n[Setup] Heavy Fog Gridlock")
        weather = carla.WeatherParameters.CloudyNoon
        weather.fog_density = 70.0
        weather.fog_distance = 15.0
        world.set_weather(weather)
        bp1 = _safe_blueprint(blueprint_lib, "vehicle.tesla.model3")
        obs1 = _safe_spawn_obstacle(world, bp1, _road_transform_ahead(world, ego_transform, 15.0))
        if obs1 is not None:
            obstacles.append(obs1)
        bp2 = _safe_blueprint(blueprint_lib, "vehicle.audi.tt")
        obs2 = _safe_spawn_obstacle(world, bp2, _road_transform_ahead(world, ego_transform, 23.0))
        if obs2 is not None:
            obstacles.append(obs2)
        _set_nearest_traffic_light(world, ego_transform, carla.TrafficLightState.Red)

    elif scenario_name == "TC_11_ERRATIC_ANIMAL":
        print("\n[Setup] Erratic Animal")
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        ped_tf = _road_transform_ahead(world, ego_transform, 18.0, lateral_m=-3.0)
        fwd = ped_tf.get_forward_vector()
        right = ped_tf.get_right_vector()
        animal = _spawn_animal(world, blueprint_lib, ped_tf, fwd + right, speed=0.9)
        if animal is not None:
            obstacles.append(animal)

    elif scenario_name == "TC_12_SUN_GLARE":
        print("\n[Setup] Sun Glare")
        world.set_weather(carla.WeatherParameters.WetSunset)
        bp = _safe_blueprint(blueprint_lib, "vehicle.carlamotors.carlacola")
        tf = _road_transform_ahead(world, ego_transform, 20.0)
        truck = _safe_spawn_obstacle(world, bp, tf)
        if truck is not None:
            truck.set_target_velocity(tf.get_forward_vector() * 2.2)
            obstacles.append(truck)

    elif scenario_name == "TC_13_TIGHT_CORRIDOR":
        print("\n[Setup] Tight Corridor")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp_l = _safe_blueprint(blueprint_lib, "vehicle.mini.cooper_s")
        obs_l = _safe_spawn_obstacle(world, bp_l, _road_transform_ahead(world, ego_transform, 15.0, lateral_m=-1.9))
        if obs_l is not None:
            obstacles.append(obs_l)
        bp_r = _safe_blueprint(blueprint_lib, "vehicle.nissan.micra")
        obs_r = _safe_spawn_obstacle(world, bp_r, _road_transform_ahead(world, ego_transform, 18.0, lateral_m=1.9))
        if obs_r is not None:
            obstacles.append(obs_r)

    elif scenario_name == "TC_14_CROSS_TRAFFIC":
        print("\n[Setup] Cross Traffic")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = _safe_blueprint(blueprint_lib, "vehicle.mustang.mustang")
        tf = _road_transform_ahead(world, ego_transform, 20.0, lateral_m=10.0, yaw_offset_deg=-90.0)
        car = _safe_spawn_obstacle(world, bp, tf)
        if car is not None:
            car.set_target_velocity(tf.get_forward_vector() * 8.0)
            obstacles.append(car)
        _set_nearest_traffic_light(world, ego_transform, carla.TrafficLightState.Green)

    elif scenario_name == "TC_15_CHAOTIC_MIXED":
        print("\n[Setup] Chaotic Mixed Traffic")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp_t = _safe_blueprint(blueprint_lib, "vehicle.carlamotors.carlacola")
        tf_t = _road_transform_ahead(world, ego_transform, 25.0)
        truck = _safe_spawn_obstacle(world, bp_t, tf_t)
        if truck is not None:
            truck.set_target_velocity(tf_t.get_forward_vector() * 3.0)
            obstacles.append(truck)
        bp_p = _safe_blueprint(blueprint_lib, "vehicle.audi.tt")
        parked = _safe_spawn_obstacle(world, bp_p, _road_transform_ahead(world, ego_transform, 12.0, lateral_m=-1.8))
        if parked is not None:
            obstacles.append(parked)
        ped_tf = _road_transform_ahead(world, ego_transform, 18.0, lateral_m=2.8)
        ped = _spawn_walker(world, blueprint_lib, ped_tf, _opposite_direction(ped_tf.get_right_vector()), speed=1.4)
        if ped is not None:
            obstacles.append(ped)

    elif scenario_name == "TC_16_TRAFFIC_SIGNAL_STOP":
        print("\n[Setup] Traffic Signal Stop")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        _set_nearest_traffic_light(world, ego_transform, carla.TrafficLightState.Red)
        car = _spawn_vehicle(world, blueprint_lib, _road_transform_ahead(world, ego_transform, 22.0), autopilot=True)
        if car is not None:
            obstacles.append(car)

    elif scenario_name == "TC_17_PEDESTRIAN_CROWD":
        print("\n[Setup] Pedestrian Crowd")
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        for index in range(5):
            ped_tf = _road_transform_ahead(
                world, ego_transform, 14.0 + index * 2.0, lateral_m=-3.0 + index * 1.5
            )
            right = ped_tf.get_right_vector()
            ped = _spawn_walker(
                world, blueprint_lib, ped_tf,
                right if index % 2 == 0 else _opposite_direction(right),
                speed=1.0 + index * 0.15,
            )
            if ped is not None:
                obstacles.append(ped)

    elif scenario_name == "TC_18_ANIMAL_CROSSING":
        print("\n[Setup] Animal Crossing")
        world.set_weather(carla.WeatherParameters.WetCloudyNoon)
        ped_tf = _road_transform_ahead(world, ego_transform, 20.0, lateral_m=-3.5)
        animal = _spawn_animal(world, blueprint_lib, ped_tf, ped_tf.get_right_vector(), speed=1.2)
        if animal is not None:
            obstacles.append(animal)

    elif scenario_name == "TC_19_DENSE_TRAFFIC_CONGESTION":
        print("\n[Setup] Dense Traffic Congestion")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        obstacles.extend(_spawn_traffic(world, blueprint_lib, ego_transform, 16))
        _set_nearest_traffic_light(world, ego_transform, carla.TrafficLightState.Red)

    elif scenario_name == "TC_20_ADVERSE_WEATHER":
        print("\n[Setup] Adverse Weather")
        world.set_weather(carla.WeatherParameters.HardRainSunset)
        car = _spawn_vehicle(world, blueprint_lib, _road_transform_ahead(world, ego_transform, 18.0))
        if car is not None:
            obstacles.append(car)

    elif scenario_name == "TC_21_AMBIGUOUS_INTERSECTION":
        print("\n[Setup] Ambiguous Intersection")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for offset, yaw in ((14.0, 90.0), (18.0, -90.0), (24.0, 0.0)):
            car = _spawn_vehicle(
                world, blueprint_lib,
                _road_transform_ahead(world, ego_transform, offset, lateral_m=3.2, yaw_offset_deg=yaw),
                autopilot=True,
            )
            if car is not None:
                obstacles.append(car)

    elif scenario_name == "TC_22_OCCLUDED_PEDESTRIAN":
        print("\n[Setup] Occluded Pedestrian")
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        parked = _spawn_vehicle(
            world, blueprint_lib, _road_transform_ahead(world, ego_transform, 14.0, lateral_m=1.6)
        )
        ped_tf = _road_transform_ahead(world, ego_transform, 17.0, lateral_m=1.8)
        ped = _spawn_walker(
            world, blueprint_lib, ped_tf, _opposite_direction(ped_tf.get_right_vector()), speed=1.4
        )
        obstacles.extend(actor for actor in (parked, ped) if actor is not None)

    elif scenario_name == "TC_23_SIMULTANEOUS_CROSSING":
        print("\n[Setup] Simultaneous Vehicle and Pedestrian Crossing")
        world.set_weather(carla.WeatherParameters.ClearSunset)
        car = _spawn_vehicle(
            world, blueprint_lib,
            _road_transform_ahead(world, ego_transform, 18.0, lateral_m=8.0, yaw_offset_deg=-90.0),
            autopilot=True,
        )
        ped_tf = _road_transform_ahead(world, ego_transform, 19.0, lateral_m=-3.5)
        ped = _spawn_walker(world, blueprint_lib, ped_tf, ped_tf.get_right_vector(), speed=1.5)
        obstacles.extend(actor for actor in (car, ped) if actor is not None)

    elif scenario_name == "TC_24_CONFLICTING_SIGNALS":
        print("\n[Setup] Conflicting Traffic Signals")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        lights = list(world.get_actors().filter("traffic.traffic_light"))
        for index, light in enumerate(lights[:2]):
            light.set_state(
                carla.TrafficLightState.Red if index == 0 else carla.TrafficLightState.Green
            )
        obstacles.extend(_spawn_traffic(world, blueprint_lib, ego_transform, 4))

    elif scenario_name == "TC_25_SUDDEN_LEAD_BRAKE":
        print("\n[Setup] Sudden Lead Vehicle Brake")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        tf = _road_transform_ahead(world, ego_transform, 18.0)
        lead = _spawn_vehicle(world, blueprint_lib, tf)
        if lead is not None:
            lead.set_target_velocity(tf.get_forward_vector() * 6.0)
            obstacles.append(lead)

    elif scenario_name == "TC_26_LOW_VISIBILITY_MIXED":
        print("\n[Setup] Low Visibility Mixed Hazards")
        world.set_weather(carla.WeatherParameters.HardRainNoon)
        vehicle = _spawn_vehicle(
            world, blueprint_lib,
            _road_transform_ahead(world, ego_transform, 20.0, lateral_m=1.8),
            autopilot=True,
        )
        ped_tf = _road_transform_ahead(world, ego_transform, 22.0, lateral_m=-2.8)
        ped = _spawn_walker(world, blueprint_lib, ped_tf, ped_tf.get_right_vector(), speed=1.2)
        obstacles.extend(actor for actor in (vehicle, ped) if actor is not None)

    return ego_vehicle, obstacles
