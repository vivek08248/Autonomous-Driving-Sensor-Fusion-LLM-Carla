"""
drive_in_carla.py  (v20 - DriveGPT4-V2 closed-loop stack, rebuilt)

LLM + sensor fusion autonomous driving in CARLA 0.9.16, tuned for an RTX 3050 6 GB laptop.

Pipeline per 50 ms tick (paper: Xu et al., "DriveGPT4-V2", CVPR 2025):

    sensors (LiDAR, radar, IMU, GNSS, collision) + object list
        -> Perception: objects in the ROUTE frame, weather, fused front gap      (ad_world.py)
        -> Expert planner with privileged info: target speed + path offset      (ad_planner.py)
        -> LLM agent (Ollama, async, compact prompt): action + target speed     (ad_llm.py)
        -> DAgger arbiter: LLM drives inside the expert envelope; large
           discrepancy -> expert takes over + sample aggregated (dataset)       (ad_llm.py)
        -> PID controllers: target speed / target angle -> throttle, brake, steer (ad_control.py)
        -> leaderboard metrics DS / RC / IS, trace of the 4 decision variables  (ad_metrics.py)

The car never waits for the LLM: the expert decision is always available, so slow or
missing LLM replies can no longer freeze the vehicle.

Controls (CARLA window shows the spectator view, no extra render window):
    ESC quit | M / TAB  AI <-> MANUAL | N / P next / previous test case | R restart
    L next LLM model | C chase / driver view | T route preference | B AEB assist (manual)
    Manual: W/S or arrows = throttle / brake, A/D = steer, Q = reverse gear, SPACE = hand brake

Examples:
    python drive_in_carla.py                                   # interactive, AI mode
    python drive_in_carla.py --scenario TC_47_RED_LIGHT_STOP
    python drive_in_carla.py --batch                           # run all test cases, write report
    python drive_in_carla.py --batch --from 27 --to 54 --model qwen2.5:0.5b
    python drive_in_carla.py --batch --llm off                 # expert-only baseline
"""
import argparse
import glob
import math
import os
import random
import sys
import time
from typing import Any

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "PythonAPI", "carla"), os.path.join(_here, "PythonAPI"),
           os.path.join(_here, ".venv", "Lib", "site-packages")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
for _egg in glob.glob(os.path.join(_here, "PythonAPI", "carla", "dist", "carla-*.egg")):
    if _egg not in sys.path:
        sys.path.insert(0, _egg)
os.chdir(_here)

import carla  # noqa: E402

import config  # noqa: E402
from ad_types import Scene  # noqa: E402
from ad_planner import BehaviorPlanner, KMH  # noqa: E402
from ad_control import SpeedPID, SteeringPID  # noqa: E402
from ad_world import RouteManager, LightIndex, Perception, SensorRig  # noqa: E402
from ad_llm import (LLMWorker, DaggerArbiter, build_prompt, list_ollama_models,  # noqa: E402
                    INTENT_TO_ACTION)
from ad_metrics import EpisodeMetrics  # noqa: E402

try:
    import keyboard  # type: ignore
except Exception:
    keyboard = None

try:
    import extra_scenarios as extra  # type: ignore
    EXTRA_NAMES = list(extra.EXTRA_NAMES)
except Exception as _e:
    print(f"[warn] extra_scenarios not loaded: {_e}")
    extra, EXTRA_NAMES = None, []

DT = float(getattr(config, "SIM_DELTA_SECONDS", 0.05))
MANUAL_MAX_KMH = float(getattr(config, "MANUAL_MAX_KMH", 40.0))
REVERSE_MAX_KMH = float(getattr(config, "REVERSE_MAX_KMH", 12.0))
HUD_N = max(2, int(getattr(config, "HUD_UPDATE_EVERY_N_FRAMES", 5)))
STATUS_N = max(5, int(getattr(config, "STATUS_UPDATE_EVERY_N_FRAMES", 10)))
LLM_PERIOD_S = float(getattr(config, "AD_LLM_PERIOD_S", 0.5))
DRAW_PLAN = bool(getattr(config, "AD_DRAW_PLAN", True))

BASE_SCENARIOS = [
    "TC_01_CLEAR_LANE", "TC_02_STATIC_OBSTACLE", "TC_03_RAIN_AND_OBSTACLE",
    "TC_04_URBAN_JAYWALKER", "TC_05_SEMI_RURAL_ENCROACH", "TC_06_RURAL_SLOW_TRUCK",
    "TC_07_2WHEELER_CUT_IN", "TC_08_NIGHT_VISION_LIMIT", "TC_09_WRONG_WAY_ONCOMING",
    "TC_10_HEAVY_FOG_GRIDLOCK", "TC_11_ERRATIC_ANIMAL", "TC_12_SUN_GLARE",
    "TC_13_TIGHT_CORRIDOR", "TC_14_CROSS_TRAFFIC", "TC_15_CHAOTIC_MIXED",
    "TC_16_TRAFFIC_SIGNAL_STOP", "TC_17_PEDESTRIAN_CROWD", "TC_18_ANIMAL_CROSSING",
    "TC_19_DENSE_TRAFFIC_CONGESTION", "TC_20_ADVERSE_WEATHER",
    "TC_21_AMBIGUOUS_INTERSECTION", "TC_22_OCCLUDED_PEDESTRIAN",
    "TC_23_SIMULTANEOUS_CROSSING", "TC_24_CONFLICTING_SIGNALS",
    "TC_25_SUDDEN_LEAD_BRAKE", "TC_26_LOW_VISIBILITY_MIXED",
]
SCENARIOS = BASE_SCENARIOS + [n for n in EXTRA_NAMES if n not in BASE_SCENARIOS]

# Test cases whose correct outcome is to stop and WAIT (permanent red / yellow, closed lane,
# stationary queue).  For these, route completion is credited when the car holds correctly.
EXPECT_HOLD = set(getattr(config, "AD_EXPECT_HOLD", [
    "TC_10_HEAVY_FOG_GRIDLOCK", "TC_31_STATIONARY_QUEUE", "TC_47_RED_LIGHT_STOP",
    "TC_49_YELLOW_STOPPABLE", "TC_50_GREEN_TO_RED", "TC_52_RED_WITH_QUEUE", "TC_54_RED_NIGHT_RAIN",
]))
ROUTE_TARGET_M = float(getattr(config, "AD_ROUTE_TARGET_M", 150.0))
EPISODE_TIMEOUT_S = float(getattr(config, "AD_EPISODE_TIMEOUT_S", 70.0))
BLOCK_LIMIT_S = float(getattr(config, "AD_BLOCK_LIMIT_S", 30.0))
HOLD_DONE_S = float(getattr(config, "AD_HOLD_DONE_S", 15.0))

ROUTE_PREFS = ["straight", "right", "left", "random"]
HOTKEYS = ["esc", "tab", "m", "b", "c", "q", "n", "p", "r", "t", "l", "g"]

CONTROLS = """
 ------------------------------------------------------------------------------
  ESC      quit (metrics + report are saved)
  M / TAB  switch AI  <->  MANUAL driving
  N / P    next / previous test case          R   restart current test case
  L        next LLM model (Ollama)            C   chase / driver camera
  T        route at next junction: straight -> right -> left -> random
  G        pause / resume automatic advance to the next test case (tour mode)
  B        AEB assist ON/OFF (manual mode)    Q   reverse gear (manual mode)
  Manual:  W/S or arrows = throttle / brake, A/D = steer, SPACE = hand brake
 ------------------------------------------------------------------------------"""


# =============================================================================
# Keyboard input (global hook - no extra window needed)
# =============================================================================
class KeyboardInput:
    def __init__(self):
        self.edges = {k: False for k in HOTKEYS}
        for k in HOTKEYS:
            try:
                keyboard.add_hotkey(k, (lambda kk=k: self.edges.__setitem__(kk, True)), suppress=False)
            except Exception:
                pass

    def down(self, key):
        try:
            return bool(keyboard.is_pressed(key))
        except Exception:
            return False

    def update(self):
        return True

    def once(self, key):
        v = self.edges.get(key, False)
        self.edges[key] = False
        return v


class PygameInput:
    """Fallback when the `keyboard` package is missing (small focus window)."""

    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((420, 60))
        pygame.display.set_caption("Drive pad - keep focused")
        self.edges = {k: False for k in HOTKEYS}
        self.pressed = pygame.key.get_pressed()
        self._quit = False
        self.map = {"w": pygame.K_w, "a": pygame.K_a, "s": pygame.K_s, "d": pygame.K_d,
                    "up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT,
                    "right": pygame.K_RIGHT, "space": pygame.K_SPACE, "tab": pygame.K_TAB,
                    "m": pygame.K_m, "b": pygame.K_b, "c": pygame.K_c, "q": pygame.K_q,
                    "n": pygame.K_n, "p": pygame.K_p, "r": pygame.K_r, "t": pygame.K_t,
                    "l": pygame.K_l, "g": pygame.K_g, "esc": pygame.K_ESCAPE}

    def down(self, key):
        return bool(self.pressed[self.map[key]]) if key in self.map else False

    def update(self):
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                self._quit = True
            elif e.type == self.pg.KEYDOWN:
                for k, code in self.map.items():
                    if e.key == code and k in HOTKEYS:
                        self.edges[k] = True
        self.pressed = self.pg.key.get_pressed()
        return not self._quit

    def once(self, key):
        v = self.edges.get(key, False)
        self.edges[key] = False
        return v


class NoInput:
    def down(self, key):
        return False

    def update(self):
        return True

    def once(self, key):
        return False


def make_input(kind, allow_window=True):
    if kind == "none":
        return NoInput()
    if kind in ("auto", "keyboard") and keyboard is not None:
        return KeyboardInput()
    if not allow_window:
        return NoInput()
    try:
        return PygameInput()
    except Exception:
        print("[input] neither `keyboard` nor `pygame` available -> no keyboard control")
        return NoInput()


# =============================================================================
# Simulation setup
# =============================================================================
def setup_world(client):
    original = client.get_world().get_settings()
    world = client.get_world()
    want = getattr(config, "CARLA_MAP", None)
    try:
        if getattr(config, "AD_LOAD_MAP", False) and want and want not in world.get_map().name:
            print(f"[CARLA] loading {want} ...")
            world = client.load_world(want)
    except Exception as e:
        print(f"[CARLA] could not load map {want}: {e}")
    # Layered (_Opt) maps: drop GPU-heavy / non-actor layers.  Map-baked parked cars are NOT
    # actors, so no object list can see them -> removing them prevents "invisible" static crashes.
    layers = getattr(config, "AD_UNLOAD_MAP_LAYERS", ["ParkedVehicles", "Particles", "Foliage"])
    if "_Opt" in world.get_map().name and layers:
        done = []
        for name in layers:
            layer = getattr(carla.MapLayer, name, None)
            if layer is None:
                continue
            try:
                world.unload_map_layer(layer)
                done.append(name)
            except Exception as e:
                print(f"[CARLA] could not unload layer {name}: {e}")
        if done:
            print(f"[CARLA] unloaded map layers: {', '.join(done)}")
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = DT
    s.no_rendering_mode = bool(getattr(config, "AD_NO_RENDERING", False))
    world.apply_settings(s)
    tm = client.get_trafficmanager(int(getattr(config, "TRAFFIC_MANAGER_PORT", 8000)))
    tm.set_synchronous_mode(True)
    try:
        tm.set_random_device_seed(int(getattr(config, "TRAFFIC_MANAGER_SEED", 42)))
        tm.set_hybrid_physics_mode(bool(getattr(config, "TRAFFIC_MANAGER_HYBRID_PHYSICS", True)))
        tm.set_hybrid_physics_radius(float(getattr(config, "TRAFFIC_MANAGER_HYBRID_RADIUS", 70.0)))
    except Exception:
        pass
    world.tick()
    print(f"[CARLA] map {world.get_map().name}, synchronous {DT * 1000:.0f} ms")
    return world, original, tm


def spectator_tf(ego, view):
    vt = ego.get_transform()
    fwd = vt.get_forward_vector()
    if view == "driver":
        right = vt.get_right_vector()
        loc = vt.location + fwd * 0.3 - right * 0.35 + carla.Location(z=1.25)
        return carla.Transform(loc, carla.Rotation(pitch=vt.rotation.pitch, yaw=vt.rotation.yaw))
    loc = vt.location - fwd * 7.5 + carla.Location(z=3.6)
    d = vt.location - loc
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))) - 6.0
    return carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw))


# =============================================================================
# Session
# =============================================================================
class Session:
    def __init__(self, world, client, tm, args):
        self.world, self.client, self.tm, self.args = world, client, tm, args
        self.map = world.get_map()
        self.spectator = world.get_spectator()
        self.mode = args.mode
        self.view = "chase"
        self.assist = True
        self.idx = 0
        self.ego: Any = None
        self.actors = []
        self.runtime = None
        self.rig = None
        self.route = None
        self.perc = None
        self.lights = None
        self.metrics = None
        self.planner = BehaviorPlanner()
        self.speed_pid = SpeedPID()
        self.steer_pid = SteeringPID()
        self.reverse_gear = False
        self.thr_cache = self.brk_cache = self.steer_cache = 0.0
        self.route_pref = "straight"
        self.sim_t = 0.0
        self.frame = 0
        self.plan = None
        self.arb = None
        self.final_v = 0.0
        self.alpha = 0.0
        self.applied = (0.0, 0.0, 0.0)
        self.fused_gap = None
        self.light = None
        self.collisions_total = 0
        self.crash_flash = 0
        self._last_submit = -99.0
        self._spec_loc = None
        self._prev_loc = None
        self._loop_t = time.perf_counter()
        self._hold_t = 0.0
        self.end_reason = ""
        self.base_npcs = []
        self._lights_set = None
        self.ego_hl, self.ego_hw = 2.4, 1.0
        self.episode_done = ""
        self.auto_next = bool(args.demo and not args.batch)

        # ---- LLM
        self.llm_on = args.llm != "off"
        installed = list_ollama_models() if self.llm_on else []
        wanted = args.model or getattr(config, "OLLAMA_MODEL", "qwen2.5:1.5b")
        pool = [m for m in getattr(config, "AVAILABLE_LLM_MODELS", []) if m in installed] or installed
        if self.llm_on and installed and wanted not in installed:
            print(f"[LLM] model '{wanted}' is not installed in Ollama ({', '.join(installed)}). "
                  f"Run:  ollama pull {wanted}")
        self.models = pool or [wanted]
        self.model = wanted if (wanted in installed or not installed) else self.models[0]
        if self.llm_on and not installed:
            print("[LLM] Ollama not reachable on", getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
                  "-> expert keeps driving; start `ollama serve` to enable the LLM agent.")
        authority = "expert" if not self.llm_on else args.authority
        self.arbiter = DaggerArbiter(authority=authority)
        self.worker = LLMWorker(self.model) if self.llm_on else None
        if self.worker is not None:
            self.worker.warmup()

    # ------------------------------------------------------------- helpers
    @property
    def scenario(self):
        return SCENARIOS[self.idx]

    def _tick(self):
        self.world.tick()

    def model_label(self):
        return self.model if self.llm_on else "EXPERT_ONLY"

    # ------------------------------------------------------------- loading
    def teardown(self, reason="user_switch"):
        if self.metrics is not None:
            row = self.metrics.finish(reason, self._ego_speed())
            if row:
                print(f"\n[Metrics] {row['Test_Case']}: DS {row['Driving_Score_DS']} RC {row['Route_Completion_RC_%']}% "
                      f"IS {row['Infraction_Score_IS']} (ped {row['Pedestrian_Collisions']} veh "
                      f"{row['Vehicle_Collisions']} stat {row['Static_Collisions']} red "
                      f"{row['Red_Light_Violations']}) end={reason}")
            self.metrics = None
        if self.rig is not None:
            self.rig.destroy()
            self.rig = None
        ids = []
        for a in list(self.actors) + [self.ego]:
            if a is not None:
                try:
                    if a.is_alive:
                        ids.append(a.id)
                except Exception:
                    pass
        if self.runtime is not None:
            for it in getattr(self.runtime, "items", []):
                try:
                    if it["actor"].is_alive and it["actor"].id not in ids:
                        ids.append(it["actor"].id)
                except Exception:
                    pass
        if ids:
            try:
                self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
            except Exception:
                for a in list(self.actors) + [self.ego]:
                    try:
                        a.destroy()
                    except Exception:
                        pass
        self.ego, self.actors, self.runtime, self.base_npcs = None, [], None, []
        try:
            self.world.freeze_all_traffic_lights(False)
            self.world.reset_all_traffic_lights()
        except Exception:
            pass
        try:
            self._tick()
        except Exception:
            pass

    def load(self, idx):
        self.teardown()
        self.idx = idx % len(SCENARIOS)
        name = self.scenario
        print(f"\n[Session] ===== {name} ({self.idx + 1}/{len(SCENARIOS)}) =====")
        try:
            if extra is not None and name in EXTRA_NAMES:
                self.ego, self.actors, self.runtime = extra.build(self.world, name)
                self.route_pref = getattr(self.runtime, "route_pref", None) or "straight"
                desc = getattr(self.runtime, "desc", "")
                if desc:
                    print(f"[Scenario] {desc}  ->  expected: {getattr(self.runtime, 'expect', '')}")
            else:
                from run_test_cases import setup_scenario
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
                self.ego, self.actors = setup_scenario(self.world, name)
                self.runtime = getattr(self.ego, "runtime", None)
                self.route_pref = "straight"
                self._prepare_base_npcs(name)
        except Exception as e:
            print(f"[ERROR] could not set up {name}: {e}")
            self.ego = None
            return False
        for _ in range(3):
            self._tick()
        self._spawn_background()
        self.rig = SensorRig(self.world, self.ego)
        self.perc = Perception(self.world, self.ego)
        self.lights = LightIndex(self.world)
        self.route = RouteManager(self.map, self.ego.get_transform(), self.route_pref)
        self._read_ego_physics()
        try:
            li, _ = self.lights.find(self.route, 150.0)
            print(f"[Light] {'signal ' + li.state + ' on the route, stop line %.0f m ahead' % (li.s_line - self.route.ego_s) if li else 'no traffic light on the route within 150 m'}")
        except Exception as e:
            print(f"[Light] check failed: {e}")
        s, d, _ = self.route.update(self.ego.get_location())
        self.planner.reset(s, d)
        self.speed_pid.reset()
        self.steer_pid.reset()
        self.arbiter.reset()
        self.sim_t = 0.0
        self.frame = 0
        self._hold_t = 0.0
        self.collisions_total = 0
        self._prev_loc = self.ego.get_location()
        self._spec_loc = None
        self.reverse_gear = False
        self.episode_done = ""
        self.metrics = EpisodeMetrics(_here, name, self.model_label(), self.mode, self.arbiter.authority,
                                      route_target_m=ROUTE_TARGET_M, expect_hold=name in EXPECT_HOLD)
        self._lights_set = None
        self.move_spectator(force=True)
        return True

    def _read_ego_physics(self):
        ext = self.ego.bounding_box.extent
        self.ego_hl, self.ego_hw = float(ext.x), float(ext.y)
        try:
            pc = self.ego.get_physics_control()
            w = pc.wheels
            self.steer_pid.max_steer = math.radians(max(30.0, w[0].max_steer_angle))
            wb = math.hypot(w[0].position.x - w[2].position.x, w[0].position.y - w[2].position.y) / 100.0
            if 2.0 < wb < 4.5:
                self.steer_pid.wheelbase = wb
        except Exception:
            pass

    def _prepare_base_npcs(self, name):
        """Base test cases relied on a per-tick velocity hack for these two NPCs."""
        speed = 2.7 if "SLOW_TRUCK" in name else (5.0 if "WRONG_WAY" in name else None)
        if speed is None:
            return
        for a in self.actors:
            if a is not None and a.type_id.startswith("vehicle."):
                self.base_npcs.append((a, speed, "WRONG_WAY" in name))

    def _drive_base_npcs(self):
        for a, spd, backwards in self.base_npcs:
            try:
                if not a.is_alive:
                    continue
                wp = self.map.get_waypoint(a.get_location(), project_to_road=True)
                if wp is None:
                    continue
                f = wp.transform.get_forward_vector()
                if backwards:
                    f = carla.Vector3D(-f.x, -f.y, 0.0)
                a.set_target_velocity(carla.Vector3D(f.x * spd, f.y * spd, 0.0))
            except Exception:
                pass

    def _spawn_background(self):
        n = int(getattr(config, "BACKGROUND_VEHICLES", 0))
        if n <= 0 or self.ego is None:
            return
        lib = self.world.get_blueprint_library()
        bps = [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        sps = list(self.map.get_spawn_points())
        random.shuffle(sps)
        eloc = self.ego.get_location()
        port = self.tm.get_port()
        for sp in sps:
            if n <= 0:
                break
            if sp.location.distance(eloc) < 45.0:
                continue
            v = self.world.try_spawn_actor(random.choice(bps), sp)
            if v is not None:
                v.set_autopilot(True, port)
                self.actors.append(v)
                n -= 1

    # ------------------------------------------------------------- control
    def _ego_speed(self):
        if self.ego is None:
            return 0.0
        try:
            v = self.ego.get_velocity()
            return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        except Exception:
            return 0.0

    def set_mode(self, mode):
        self.mode = mode
        self.thr_cache = self.brk_cache = 0.0
        if self.route is not None and self.ego is not None:
            s, d, _ = self.route.update(self.ego.get_location())
            self.planner.reset(s, d)
        self.speed_pid.reset()
        self.steer_pid.reset()
        print(f"\n[Mode] -> {mode}")

    def cycle_model(self):
        if not self.llm_on or not self.models:
            print("\n[LLM] disabled")
            return
        i = (self.models.index(self.model) + 1) % len(self.models) if self.model in self.models else 0
        self.model = self.models[i]
        self.worker.set_model(self.model)
        self.worker.warmup()
        self.arbiter.reset()
        print(f"\n[LLM] -> {self.model} (restart the test case with R to log it as a new episode)")

    def cycle_route(self):
        i = (ROUTE_PREFS.index(self.route_pref) + 1) % len(ROUTE_PREFS)
        self.route_pref = ROUTE_PREFS[i]
        self.route = RouteManager(self.map, self.ego.get_transform(), self.route_pref)
        s, d, _ = self.route.update(self.ego.get_location())
        self.planner.reset(s, d)
        print(f"\n[Route] -> {self.route_pref.upper()}")

    def toggle_reverse(self):
        if self._ego_speed() > 1.0:
            print("\n[Gear] stop first")
            return
        self.reverse_gear = not self.reverse_gear
        print(f"\n[Gear] {'REVERSE' if self.reverse_gear else 'DRIVE'}")

    # ------------------------------------------------------------------ step
    def step(self, keys):
        """Advance one tick.  Returns an end reason (batch mode) or ''."""
        if self.ego is None or not self.ego.is_alive:
            return "no_ego"
        now = time.perf_counter()
        loop_dt = now - self._loop_t
        self._loop_t = now
        self._tick()
        self.sim_t += DT
        self.frame += 1
        if self.runtime is not None:
            try:
                self.runtime.tick()
            except TypeError:
                self.runtime.tick(DT)
            except Exception:
                pass
        self._drive_base_npcs()

        ego_tf = self.ego.get_transform()
        vel = self.ego.get_velocity()
        fwd = ego_tf.get_forward_vector()
        v_signed = vel.x * fwd.x + vel.y * fwd.y
        v = max(0.0, v_signed)
        loc = ego_tf.location
        dist_step = loc.distance(self._prev_loc) if self._prev_loc is not None else 0.0
        self._prev_loc = loc

        # ---------------- perception
        ego_s, ego_d, _ = self.route.update(loc)
        self.perc.update_weather(self.sim_t)
        objs = self.perc.objects(self.route, ego_tf)
        geo = self.route.lane_geo()
        self.light, _tl = self.lights.find(self.route)
        in_junction = bool(self.route.wp_at(ego_s).is_junction)
        self.fused_gap = self.perc.fused_sensor_gap(self.rig, ego_tf, self.route, self.planner.profile,
                                                    self.ego_hw, self.ego_hl, objs)
        scene = Scene(t=self.sim_t, dt=DT, ego_s=ego_s, ego_d=ego_d, ego_v=v,
                      ego_half_len=self.ego_hl, ego_half_wid=self.ego_hw, objs=objs, geo=geo,
                      light=self.light, in_junction=in_junction, weather=self.perc.weather,
                      friction=self.perc.friction, sensor_gap=self.fused_gap,
                      route_pref=self.route_pref, ego_reversing=v_signed < -0.3,
                      rear_gap=getattr(self.rig, "rear_gap", None))
        plan = self.planner.step(scene)
        self.plan = plan

        # ---------------- LLM agent (async)
        if self.worker is not None:
            gi = self.worker.gpu_info
            if gi is not None and gi is not getattr(self, "_gpu_printed", None):
                self._gpu_printed = gi
                where = ("fully on the GPU" if gi["gpu_pct"] > 99 else
                         f"only {gi['gpu_pct']:.0f}% on the GPU, the rest on the CPU -> slow replies")
                print(f"\n[LLM] {self.model} ({gi['size_gb']:.1f} GB) is {where}")
            dec = self.worker.poll()
            if dec is not None:
                self.arbiter.feed(dec)
                if self.metrics:
                    self.metrics.llm_result(dec)
            if not self.worker.busy and self.sim_t - self._last_submit >= LLM_PERIOD_S:
                self._last_submit = self.sim_t
                shape, jd = self.route.describe_ahead()
                tp_s = 40.0
                prompt = build_prompt(self.sim_t, self.perc.weather, v * 3.6,
                                      self.planner._cruise_cap(scene) * 3.6, shape, jd,
                                      (tp_s, self.planner.profile.value(ego_s + tp_s)),
                                      self.light, objs, ego_s + self.ego_hl, ego_d,
                                      lane_pos=self.route.lane_position())
                act = (("BYPASS_LEFT" if plan.intent.endswith("LEFT") else "BYPASS_RIGHT")
                       if plan.intent.startswith(("BYPASS", "EVADE", "LANE_CHANGE"))
                       else INTENT_TO_ACTION.get(plan.intent, "KEEP"))
                self.worker.submit(prompt, tag=self.scenario,
                                   meta={"action": act, "speed_kmh": plan.target_speed * 3.6,
                                         "reason": plan.reason, "sim_t": self.sim_t})

        collisions = self.rig.pop_collisions() if self.rig else []
        new_hits = self.metrics.collision(collisions, self.sim_t) if self.metrics else []
        if new_hits:
            self.collisions_total += len(new_hits)
            self.crash_flash = 30
            for cat, tid in new_hits:
                print(f"\n[COLLISION] {cat} with {tid} at {v * 3.6:.1f} km/h ({plan.intent}: {plan.reason[:60]})")

        # ---------------- control
        if self.mode == "AI":
            ctrl = self._ai_control(plan, ego_tf, v, v_signed, ego_s)
        else:
            ctrl = self._manual_control(keys, v_signed)
            self.final_v, self.arb = v, None
            if self.assist and (plan.accel_req is not None and plan.accel_req < -4.0
                                or plan.lead_gap < 1.5 and v > 1.0):
                ctrl.throttle = 0.0
                ctrl.brake = max(ctrl.brake, 1.0 if plan.lead_gap < 3.0 else 0.6)
        self.ego.apply_control(ctrl)
        self.applied = (ctrl.throttle, ctrl.brake, ctrl.steer)
        self._auto_lights()

        # ---------------- metrics / logs
        end = ""
        if self.metrics is not None:
            hw = geo.at(ego_s)[0]
            self.metrics.update(self.sim_t, DT, ego_s, v, ego_s + self.ego_hl, self.light, ego_d, hw,
                                self.planner.mode, plan, self.arb if self.mode == "AI" else None,
                                loop_dt, dist_step, new_hits)
            if self.frame % 10 == 0:
                wps, rps = self._paper_points(ego_tf, ego_s, v)
                self.metrics.log_2hz(self.sim_t, v, plan, self.final_v, self.arb, math.degrees(self.alpha),
                                     wps, rps, self.fused_gap, bool(new_hits), self.light, ego_d)
            end = self._episode_end(v, plan)
            if end and not self.args.batch:
                self.metrics.finish(end, v)
                self.episode_done = end
                if self.auto_next:
                    print(f"\n[Tour] {self.scenario} finished ({end}) -> next test case")
                else:
                    print(f"\n[Metrics] episode logged ({end}); keep driving or press N for the next test case")
                self.metrics = None
                end = ""

        self.move_spectator()
        if self.frame % HUD_N == 0:
            self.draw_hud(v)
        if DRAW_PLAN and self.frame % 4 == 0 and self.mode == "AI":
            self.draw_plan(ego_s)
        if self.frame % STATUS_N == 0:
            self.print_status(v)
        # real-time pacing: 1 simulated second = 1 wall second (x --speed).  Without it CARLA at
        # Low quality runs ~5x faster than real time: test cases look "short" and the LLM, whose
        # replies take real milliseconds, sees almost every scene too late.
        if not self.args.fast:
            period = DT / max(0.1, self.args.speed)
            wait = period - (time.perf_counter() - now)
            if wait > 0:
                time.sleep(wait)
        return end

    def _episode_end(self, v, plan):
        m = self.metrics
        if self.args.demo:                     # showcase: drive the whole duration, keep going
            return "duration" if self.sim_t >= self.args.timeout else ""
        if m.progress >= ROUTE_TARGET_M:
            return "route_completed"
        if self.scenario in EXPECT_HOLD and v < 0.3 and plan.hold:
            self._hold_t += DT
            if self._hold_t >= HOLD_DONE_S:
                return "hold_completed"
        else:
            self._hold_t = 0.0
        if m.block_t >= BLOCK_LIMIT_S:
            return "blocked"
        if self.sim_t >= self.args.timeout:
            return "timeout"
        return ""

    def _ai_control(self, plan, ego_tf, v, v_signed, ego_s):
        v_cmd, self.arb = self.arbiter.arbitrate(plan, v * 3.6, self.scenario,
                                                 collect=not getattr(self.args, "no_dagger_log", False),
                                                 sim_t=self.sim_t)
        self.final_v = v_cmd
        loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        if plan.reverse:
            thr, brk, _ = self.speed_pid.step(v_signed, plan.target_speed, None, DT, reverse=True)
            self.alpha = 0.0
            self.steer_pid.prev_steer *= 0.8
            return carla.VehicleControl(throttle=thr, brake=brk, steer=0.0, reverse=True)
        ld = SteeringPID.lookahead(v)
        s_t = ego_s + ld
        tx, ty, _, _ = self.route.point(s_t, self.planner.profile.value(s_t))
        ld_real = max(1.0, math.hypot(tx - loc.x, ty - loc.y))
        self.alpha = self.steer_pid.target_angle(loc.x, loc.y, yaw, tx, ty)
        steer = self.steer_pid.step(self.alpha, ld_real, DT, v)
        if v_signed < -0.3:                     # rolling back after a reverse -> stop first
            return carla.VehicleControl(throttle=0.0, brake=0.7, steer=steer)
        thr, brk, _ = self.speed_pid.step(v, v_cmd, plan.accel_req, DT)
        # last-line safety shield (independent of LLM / arbiter)
        if plan.lead_gap < 1.0 and v > 0.3:
            thr, brk = 0.0, 1.0
        return carla.VehicleControl(throttle=float(thr), brake=float(brk), steer=float(steer),
                                    hand_brake=False, reverse=False)

    def _manual_control(self, keys, v_signed):
        fwd_in = keys.down("w") or keys.down("up")
        back_in = keys.down("s") or keys.down("down")
        accel, brake = (back_in, fwd_in) if self.reverse_gear else (fwd_in, back_in)
        cap = (REVERSE_MAX_KMH if self.reverse_gear else MANUAL_MAX_KMH) / 3.6
        spd = abs(v_signed)
        self.thr_cache = min(1.0, self.thr_cache + 1.8 * DT) if (accel and not brake) else \
            max(0.0, self.thr_cache - 4.0 * DT)
        thr = self.thr_cache * max(0.0, min(1.0, (cap - spd) / 2.2))
        self.brk_cache = min(1.0, max(0.35, self.brk_cache) + 3.5 * DT) if brake else \
            max(0.0, self.brk_cache - 6.0 * DT)
        want = (1.0 if (keys.down("d") or keys.down("right")) else 0.0) - \
               (1.0 if (keys.down("a") or keys.down("left")) else 0.0)
        limit = max(0.22, 0.75 - 0.04 * spd)
        target = want * limit
        rate = (1.5 if want else 2.5) * DT
        self.steer_cache += max(-rate, min(rate, target - self.steer_cache))
        return carla.VehicleControl(throttle=thr, brake=self.brk_cache, steer=round(self.steer_cache, 3),
                                    hand_brake=keys.down("space"), reverse=self.reverse_gear)

    def _auto_lights(self):
        w = self.perc.weather if self.perc else "clear"
        want = w in ("night", "night_rain", "storm_night", "fog", "heavy_rain", "rain", "dusk")
        if want == self._lights_set:
            return
        self._lights_set = want
        try:
            st = carla.VehicleLightState.NONE
            if want:
                st = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                if "fog" in w:
                    st = carla.VehicleLightState(int(st) | int(carla.VehicleLightState.Fog))
            self.ego.set_light_state(st)
        except Exception:
            pass

    # ------------------------------------------------------- paper outputs
    def _paper_points(self, ego_tf, ego_s, v):
        """Waypoints: 8 future positions over 4 s; route points: 10 points ahead (ego frame)."""
        yaw = math.radians(ego_tf.rotation.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex, ey = ego_tf.location.x, ego_tf.location.y

        def to_ego(px, py):
            dx, dy = px - ex, py - ey
            return [round(dx * cy + dy * sy, 2), round(-dx * sy + dy * cy, 2)]
        wps, s, vv = [], ego_s, v
        for _ in range(8):
            for _ in range(5):
                a = max(-3.0, min(1.5, 1.1 * (self.final_v - vv)))
                vv = max(0.0, vv + a * 0.1)
                s += vv * 0.1
            px, py, _, _ = self.route.point(s, self.planner.profile.value(s))
            wps.append(to_ego(px, py))
        rps = []
        for k in range(1, 11):
            px, py, _, _ = self.route.point(ego_s + 3.0 * k)
            rps.append(to_ego(px, py))
        return wps, rps

    # ------------------------------------------------------------- display
    def move_spectator(self, force=False):
        if self.ego is None:
            return
        tf = spectator_tf(self.ego, self.view)
        if self.view == "chase" and self._spec_loc is not None and not force:
            a = 0.3
            l0 = self._spec_loc
            tf.location = carla.Location(l0.x + a * (tf.location.x - l0.x), l0.y + a * (tf.location.y - l0.y),
                                         l0.z + a * (tf.location.z - l0.z))
        self._spec_loc = tf.location
        try:
            self.spectator.set_transform(tf)
        except Exception:
            pass

    def draw_hud(self, v):
        p, a = self.plan, self.arb or {}
        if p is None:
            return
        lines = [(f"{self.scenario} ({self.idx + 1}/{len(SCENARIOS)})  {self.perc.weather}", (255, 255, 255)),
                 (f"{self.mode} | {self.model_label()} | {self.arbiter.authority} | lane "
                  f"{'%d/%d' % self.route.lane_position()} | t {self.sim_t:4.0f}/{self.args.timeout:.0f}s",
                  (120, 220, 255)),
                 (f"{v * 3.6:4.1f} km/h -> {self.final_v * 3.6:4.1f} | gap {min(p.lead_gap, 99):4.1f} m "
                  f"{p.lead_kind}", (255, 255, 255)),
                 (f"EXPERT {p.intent}: {p.reason[:58]}", (180, 255, 180))]
        if self.worker is not None:
            lat = a.get("latency_ms")
            tm = self.worker.last_timing or {}
            split = (f" (prompt {tm.get('prompt_ms', 0):.0f} + gen {tm.get('gen_ms', 0):.0f})"
                     if tm.get("total_ms") else "")
            lines.append((f"LLM {a.get('llm_action', '-')} {a.get('llm_speed_kmh') or 0:.0f} km/h "
                          f"[{a.get('source', '-')}] {'' if lat is None else f'{lat:.0f} ms'}{split}",
                          (255, 220, 120)))
        if not self.args.batch:
            if self.auto_next:
                left = max(0.0, self.args.timeout - self.sim_t)
                lines.append((f"TOUR: next test case in {left:3.0f}s  (G pause, N skip)", (200, 200, 255)))
            elif self.args.demo:
                lines.append(("TOUR PAUSED (G resume)", (200, 200, 255)))
        if self.light is not None:
            col = {"RED": (255, 60, 60), "YELLOW": (255, 220, 60)}.get(self.light.state, (80, 255, 80))
            lines.append((f"LIGHT {self.light.state} line {self.light.s_line - self.route.ego_s - self.ego_hl:.1f} m",
                          col))
        if self.crash_flash > 0:
            self.crash_flash -= HUD_N
            lines.append((f"COLLISION! (total {self.collisions_total})", (255, 40, 40)))
        base = self.ego.get_location() + carla.Location(z=2.6)
        n = len(lines)
        life = HUD_N * DT + 0.02
        for i, (txt, c) in enumerate(lines):
            self.world.debug.draw_string(base + carla.Location(z=0.38 * (n - 1 - i)), txt, draw_shadow=False,
                                         color=carla.Color(*c), life_time=life, persistent_lines=False)

    def draw_plan(self, ego_s):
        p = self.plan
        col = carla.Color(0, 200, 255) if p.target_speed > 0.5 else carla.Color(255, 60, 60)
        for k in range(1, 13):
            s = ego_s + 2.5 * k
            x, y, z, _ = self.route.point(s, self.planner.profile.value(s))
            self.world.debug.draw_point(carla.Location(x, y, z + 0.3), size=0.07, color=col,
                                        life_time=4 * DT + 0.02, persistent_lines=False)

    def print_status(self, v):
        thr, brk, st = self.applied
        p = self.plan
        a = self.arb or {}
        msg = (f"{self.scenario[:22]:22s} {self.mode:6s} {v * 3.6:5.1f}->{self.final_v * 3.6:5.1f} km/h "
               f"gap {min(p.lead_gap, 99):5.1f} thr {thr:.2f} brk {brk:.2f} str {st:+.2f} "
               f"{p.intent[:14]:14s} {a.get('source', '')[:16]:16s}")
        print("\r" + msg.ljust(130), end="", flush=True)


# =============================================================================
# main
# =============================================================================
def scenario_slice(args):
    names = list(SCENARIOS)
    if args.scenarios:
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        return [n for n in names if any(n == w or n.startswith(w + "_") or n.startswith(w) for w in wanted)]
    lo, hi = args.from_tc, args.to_tc

    def num(n):
        try:
            return int(n.split("_")[1])
        except Exception:
            return 0
    return [n for n in names if lo <= num(n) <= hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=str, default=None, help="start test case (name or TC number prefix)")
    ap.add_argument("--model", type=str, default=None, help="Ollama model, e.g. qwen2.5:1.5b")
    ap.add_argument("--mode", choices=["AI", "MANUAL"], default="AI")
    ap.add_argument("--llm", choices=["on", "off"], default="on", help="off = expert-only baseline")
    ap.add_argument("--authority", choices=["dagger", "advisory", "expert"],
                    default=getattr(config, "AD_LLM_AUTHORITY", "dagger"))
    ap.add_argument("--batch", action="store_true", help="run test cases automatically and write the report")
    ap.add_argument("--scenarios", type=str, default="", help="comma list for --batch, e.g. TC_02,TC_47")
    ap.add_argument("--from", dest="from_tc", type=int, default=1)
    ap.add_argument("--to", dest="to_tc", type=int, default=999)
    ap.add_argument("--repeat", type=int, default=1, help="batch repetitions per test case")
    ap.add_argument("--timeout", "--duration", dest="timeout", type=float, default=None,
                    help="seconds per test case (default 70, or 180 with --demo)")
    ap.add_argument("--demo", action="store_true",
                    help="showcase: every test case runs the full --duration, no early stop")
    ap.add_argument("--fast", action="store_true",
                    help="run faster than real time (quick metrics; the LLM then rarely keeps up)")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor, e.g. 0.5 = slow motion")
    ap.add_argument("--input", choices=["auto", "keyboard", "pygame", "none"], default="auto")
    ap.add_argument("--no-dagger-log", action="store_true", help="do not append dagger_dataset.jsonl")
    args = ap.parse_args()
    if args.timeout is None:
        args.timeout = 180.0 if args.demo else EPISODE_TIMEOUT_S

    print("[1] connecting to CARLA ...", flush=True)
    client = carla.Client(getattr(config, "CARLA_HOST", "127.0.0.1"), int(getattr(config, "CARLA_PORT", 2000)))
    client.set_timeout(float(getattr(config, "CARLA_TIMEOUT", 25.0)))
    world, original, tm = setup_world(client)
    sess = Session(world, client, tm, args)
    print(f"[2] LLM: {'off (expert only)' if not sess.llm_on else sess.model} | authority {sess.arbiter.authority}")

    keys = make_input(args.input, allow_window=not args.batch)
    print(CONTROLS)
    queue = scenario_slice(args) or list(SCENARIOS)

    def handle_keys(allow_nav=True):
        """Hot keys work in every mode.  Returns 'quit', 'next', 'prev', 'restart' or ''."""
        if not keys.update() or keys.once("esc"):
            return "quit"
        if keys.once("m") or keys.once("tab"):
            sess.set_mode("AI" if sess.mode == "MANUAL" else "MANUAL")
        if keys.once("l"):
            sess.cycle_model()
        if keys.once("b"):
            sess.assist = not sess.assist
            print(f"\n[Assist] AEB assist {'ON' if sess.assist else 'OFF'}")
        if keys.once("c"):
            sess.view = "driver" if sess.view == "chase" else "chase"
        if keys.once("q"):
            sess.toggle_reverse()
        if keys.once("t"):
            sess.cycle_route()
        if keys.once("g"):
            if args.demo and not args.batch:
                sess.auto_next = not sess.auto_next
                print(f"\n[Tour] automatic advance {'ON' if sess.auto_next else 'PAUSED'}")
        if keys.once("n"):
            return "next"
        if keys.once("p"):
            return "prev"
        if keys.once("r"):
            return "restart"
        return ""

    def queue_next(step=1):
        name = sess.scenario
        if name in queue:
            return SCENARIOS.index(queue[(queue.index(name) + step) % len(queue)])
        return SCENARIOS.index(queue[0])

    try:
        if args.batch:
            print(f"[3] batch: {len(queue)} test cases x {args.repeat}"
                  f"{' (demo: full duration each)' if args.demo else ''}  |  N = skip, ESC = stop")
            for name in queue:
                for _ in range(max(1, args.repeat)):
                    if not sess.load(SCENARIOS.index(name)):
                        continue
                    while True:
                        k = handle_keys()
                        if k == "quit":
                            raise KeyboardInterrupt
                        if k == "next":
                            sess.teardown("user_skip")
                            break
                        if k == "restart":
                            sess.load(sess.idx)
                            continue
                        end = sess.step(keys)
                        if end:
                            sess.teardown(end)
                            break
            print("\n[Batch] done -> LLM_Comparison_Report.md, drivegpt4_v2_paper_metrics_v2.csv")
        else:
            start = SCENARIOS.index(queue[0])
            if args.scenario:
                cands = [i for i, n in enumerate(SCENARIOS) if n == args.scenario or n.startswith(args.scenario)]
                start = cands[0] if cands else start
            if sess.auto_next:
                print(f"[3] TOUR: {len(queue)} test cases, {args.timeout:.0f} s each, loops until ESC")
            sess.load(start)
            while True:
                k = handle_keys()
                if k == "quit":
                    break
                if k == "next":
                    sess.load(queue_next(+1) if args.demo else sess.idx + 1)
                    continue
                if k == "prev":
                    sess.load(queue_next(-1) if args.demo else sess.idx - 1)
                    continue
                if k == "restart":
                    sess.load(sess.idx)
                    continue
                if sess.ego is None:
                    sess.load(queue_next(+1))
                    continue
                sess.step(keys)
                if sess.episode_done and sess.auto_next:
                    sess.load(queue_next(+1))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sess.teardown("quit")
        except Exception as e:
            print(f"[teardown] {e}")
        if sess.worker is not None:
            sess.worker.close()
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        print("\n[Done] metrics: drivegpt4_v2_paper_metrics_v2.csv | trace: drivegpt4_v2_decihead_trace_v2.csv"
              " | report: LLM_Comparison_Report.md | DAgger samples: dagger_dataset.jsonl")


if __name__ == "__main__":
    main()
