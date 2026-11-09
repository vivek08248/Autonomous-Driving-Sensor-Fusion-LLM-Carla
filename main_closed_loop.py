# main_closed_loop.py

import carla
import time
import math
import random
import threading

import config

from sensor_rig import MultiSensorRig
from perception_processor import SensorProcessor
from prompt_builder import generate_multisensor_prompt
from llm_planner import LLMAgent
from safety_filter import SafetyFilter
from pid_controller import VehiclePIDController
from ambiguity_detector import AmbiguityDetector
from drivegpt4_v2 import PrivilegedExpertSupervisor


# ============================================================
# SIMULATION SETUP (from carla PythonAPI/util/config.py)
# ============================================================

def setup_simulation(client, map_name=None, reload=False):
    """
    Configure CARLA for our pipeline.

    Mirrors the official carla PythonAPI/util/config.py pattern:
      1. Optionally load a specific map
      2. Set spectator (best-effort; CARLA window may ignore it)
      3. Set synchronous mode
      4. Set weather

    Returns (world, original_settings).
    """
    original_settings = client.get_world().get_settings()

    if reload and map_name:
        print(f"[CARLA] Loading map {map_name} ...")
        world = client.load_world(map_name)
    else:
        world = client.get_world()

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = config.SYNCHRONOUS_MODE
    settings.fixed_delta_seconds = config.SIM_DELTA_SECONDS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(config.TRAFFIC_MANAGER_PORT)
    traffic_manager.set_synchronous_mode(config.SYNCHRONOUS_MODE)
    traffic_manager.set_random_device_seed(config.TRAFFIC_MANAGER_SEED)
    traffic_manager.set_hybrid_physics_mode(
        config.TRAFFIC_MANAGER_HYBRID_PHYSICS
    )
    traffic_manager.set_hybrid_physics_radius(
        config.TRAFFIC_MANAGER_HYBRID_RADIUS
    )

    # Clear weather (from carla examples)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    print(f"[CARLA] Map: {world.get_map().name}")
    print(f"[CARLA] Sync mode: {settings.synchronous_mode}")
    return world, original_settings


def restore_simulation(client, original_settings):
    """Restore the original settings (always run in finally)."""
    try:
        world = client.get_world()
        world.apply_settings(original_settings)
    except Exception as e:
        print(f"[CARLA] Could not restore settings: {e}")


# ============================================================
# SPAWN EGO VEHICLE (retry pattern from generate_traffic.py)
# ============================================================

def spawn_ego_vehicle(world):
    """
    Spawn the ego Tesla using the official retry pattern.
    Tries every spawn point until one succeeds.
    """
    bp_lib = world.get_blueprint_library()
    matches = list(bp_lib.filter(config.EGO_VEHICLE_BLUEPRINT))
    if not matches:
        matches = list(bp_lib.filter("vehicle.tesla.*")) or list(bp_lib.filter("vehicle.*"))
    vehicle_bp = matches[0]
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available")

    order = list(enumerate(spawn_points))
    if getattr(config, "PREFER_MULTILANE_START", 1):
        # Lane changes need a SAME-direction neighbour lane and no junction right
        # ahead; try spawn points that satisfy this first.
        from lane_change_planner import no_junction_ahead
        wmap = world.get_map()

        def _multilane_ok(sp):
            try:
                wp = wmap.get_waypoint(sp.location)
                if wp is None or wp.is_junction:
                    return False
                has_nb = any(
                    n is not None and n.lane_type == carla.LaneType.Driving
                    and n.lane_id * wp.lane_id > 0
                    for n in (wp.get_left_lane(), wp.get_right_lane())
                )
                return has_nb and no_junction_ahead(wp, 90.0)
            except Exception:
                return False

        good = [(i, sp) for i, sp in order if _multilane_ok(sp)]
        rest = [(i, sp) for i, sp in order if (i, sp) not in good]
        if good:
            print(f"[CARLA] Using multi-lane spawn point #{good[0][0]} (of {len(good)} found).")
        else:
            print("[CARLA] No multi-lane spawn point found: lane changes will be impossible on this map.")
        order = good + rest

    for i, sp in order:
        sp.location.z += 0.5
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            print(f"[CARLA] Ego spawned at spawn point #{i}")
            return vehicle

    raise RuntimeError("All spawn points blocked. Restart CARLA.")


# ============================================================
# SPECTATOR CAMERA (best-effort)
# ============================================================

def get_spectator_transform(vehicle,
                            distance=8.0,
                            height=4.0,
                            extra_pitch_down=15.0):
    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()
    cam_loc = vt.location - forward * distance + carla.Location(z=height)
    direction = vt.location - cam_loc
    yaw = math.degrees(math.atan2(direction.y, direction.x))
    horizontal = math.sqrt(direction.x ** 2 + direction.y ** 2)
    pitch = math.degrees(math.atan2(direction.z, horizontal)) - extra_pitch_down
    return carla.Transform(
        cam_loc,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
    )


def update_spectator(world, ego_vehicle):
    if not config.SPECTATOR_FOLLOW:
        return
    if ego_vehicle is None or not ego_vehicle.is_alive:
        return
    try:
        spectator = world.get_spectator()
        spectator.set_transform(get_spectator_transform(ego_vehicle))
    except Exception:
        pass


# ============================================================
# ASYNC LLM
# ============================================================

class AsyncLLM:
    """Non-blocking LLM wrapper.

    Key fixes:
    - Single LLM call per submit (no duplicate).
    - Caches the last good result so the car always has a decision.
    - Malformed / None results are replaced with the safe FALLBACK.
    """

    def __init__(self, agent):
        self.agent = agent
        self._lock = threading.Lock()
        self._state = {"result": None, "latency_ms": 0.0, "ready": False}
        self._running = threading.Event()
        self._last_good = None          # cache of most recent valid decision

    def submit(self, prompt):
        if self._running.is_set():
            return                       # previous call still in-flight
        self._running.set()
        threading.Thread(
            target=self._worker, args=(prompt,), daemon=True
        ).start()

    def _worker(self, prompt):
        t0 = time.time()
        try:
            result = self.agent.get_driving_decision(prompt)
        except Exception as exc:
            print(f"[AsyncLLM] inference error: {exc}")
            result = None
        latency = (time.time() - t0) * 1000.0
        # Guard against None / non-dict responses
        if not isinstance(result, dict) or not result:
            from llm_planner import FALLBACK
            result = dict(FALLBACK)
        with self._lock:
            self._state["result"] = result
            self._state["latency_ms"] = latency
            self._state["ready"] = True
            self._last_good = result
        self._running.clear()

    def poll(self):
        """Return (decision, latency_ms).

        Fresh result returned when available.
        Otherwise the most recent cached result is returned with latency 0.
        If no result has ever been received, returns (None, None).
        """
        with self._lock:
            if self._state["ready"]:
                r = self._state["result"]
                lat = self._state["latency_ms"]
                self._state["ready"] = False
                return r, lat
            # No fresh result - reuse the last known-good decision
            if self._last_good is not None:
                return self._last_good, 0.0
        return None, None


# ============================================================
# MAIN LOOP
# ============================================================

def run_simulation_cycle(world, ego_vehicle, logger, num_frames=100, runtime=None):

    sensor_rig = MultiSensorRig(world, ego_vehicle)
    async_llm = AsyncLLM(LLMAgent())
    safety_layer = SafetyFilter()
    ambiguity_detector = AmbiguityDetector()
    controller = VehiclePIDController()
    drivegpt4 = PrivilegedExpertSupervisor()

    is_sync = config.SYNCHRONOUS_MODE

    print()
    print("=" * 60)
    print("STARTING CLOSED-LOOP SIMULATION")
    print("=" * 60)
    print(f"Ego Vehicle ID : {ego_vehicle.id}")
    print(f"Frames         : {num_frames}")
    print(f"Sync mode      : {is_sync}")
    print(f"CV window      : {config.SHOW_CV_WINDOW}")

    validated_decision = {
        "intent": "KEEP_LANE",
        "target_speed_kmh": 20.0,
        "override_active": False,
    }
    latency_ms = 0.0
    is_override = False

    try:
        for frame in range(num_frames):

            if is_sync:
                world.tick()

            if runtime is not None:
                try:
                    runtime.tick(getattr(config, "SIM_DELTA_SECONDS", 0.05))
                except Exception:
                    pass

            update_spectator(world, ego_vehicle)
            sensor_rig.render()          # <-- ADDED: draw frame on main thread

            speed_kmh = SensorProcessor.get_ego_velocity(ego_vehicle)
            lidar_dist = SensorProcessor.process_lidar_obstacle_distance(
                sensor_rig.latest_lidar_points
            )
            ambiguity = ambiguity_detector.assess(
                world, ego_vehicle, lidar_dist, speed_kmh
            )
            has_crashed = sensor_rig.collision_detected

            raw_decision, new_latency = async_llm.poll()
            if raw_decision is not None:
                latency_ms = new_latency
                if new_latency <= getattr(config, "LLM_MAX_DECISION_AGE_S", 3.0) * 1000.0:
                    validated_decision = safety_layer.validate_intent(
                        raw_decision, lidar_dist, speed_kmh, ambiguity
                    )
                    is_override = validated_decision.get("override_active", False)

            if ambiguity.get("crossing_actor_count", 0) > 0:
                validated_decision = {
                    "intent": "EMERGENCY_STOP",
                    "target_speed_kmh": 0.0,
                    "override_active": True,
                    "override_reason": "Crossing road user",
                }
                is_override = True

            prompt = generate_multisensor_prompt(
                speed_kmh, lidar_dist, navigation_goal="Maintain Lane"
            )
            async_llm.submit(prompt)

            # DriveGPT4-v2 Privileged Expert Supervision & DeciHead Control
            sup_ctrl, sup_override, sup_reason = drivegpt4.supervise_and_control(
                ego=ego_vehicle,
                point_cloud=sensor_rig.latest_lidar_points,
                base_agent_ctrl=carla.VehicleControl(steer=0.0),
                llm_decision=validated_decision,
                current_speed_kmh=speed_kmh,
                dt=config.SIM_DELTA_SECONDS,
            )

            # Fail-safe AEB guard
            if lidar_dist <= 3.8 and speed_kmh > 1.0:
                sup_ctrl.throttle = 0.0
                sup_ctrl.brake = 1.0
            elif lidar_dist <= 2.2:
                sup_ctrl.throttle = 0.0
                sup_ctrl.brake = max(sup_ctrl.brake, 0.85)

            ego_vehicle.apply_control(sup_ctrl)

            logger.log_cycle(
                speed_kmh,
                lidar_dist,
                latency_ms,
                validated_decision["intent"],
                is_override,
                has_crashed,
                ambiguity["active"],
            )

            if has_crashed:
                print("[WARNING] Collision detected!")
                sensor_rig.collision_detected = False

            print(
                f"Frame {frame:03d} | "
                f"Speed: {speed_kmh:5.1f} km/h | "
                f"LiDAR: {lidar_dist:5.1f} m | "
                f"Intent: {validated_decision['intent']:14s} | "
                f"Override: {is_override} | "
                f"LLM: {latency_ms:6.1f} ms"
            )

            if not is_sync:
                time.sleep(0.05)

    finally:
        sensor_rig.destroy()
        print("[Loop] Sensors destroyed.")


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CARLA CLOSED-LOOP AUTONOMOUS DRIVING")
    print("=" * 60)

    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT)

    world, original_settings = setup_simulation(
        client,
        map_name=config.CARLA_MAP,
        reload=config.RELOAD_WORLD_PER_TEST,
    )

    ego_vehicle = spawn_ego_vehicle(world)

    if config.SYNCHRONOUS_MODE:
        world.tick()
    else:
        time.sleep(0.5)

    update_spectator(world, ego_vehicle)

    from evaluation_logger import MetricsLogger
    logger = MetricsLogger(filename="default_run.csv")

    try:
        run_simulation_cycle(world, ego_vehicle, logger, num_frames=150)
    finally:
        restore_simulation(client, original_settings)

        if ego_vehicle is not None and ego_vehicle.is_alive:
            ego_vehicle.destroy()

        logger.print_summary()
        print("[CARLA] Simulation finished.")