"""
drive_and_test.py

Interactive harness for the CARLA + Ollama closed-loop project.

 * Opens its OWN pygame window with a chase-camera (or driver-camera) view,
   so you do not depend on the CARLA window following the spectator.
 * Lets you DRIVE the car manually (WASD) or hand control to the LLM
   pipeline (LiDAR -> prompt -> Ollama -> SafetyFilter -> PID).
 * Lets you jump between the 20 test cases from run_test_cases.py
   (N / P / R) without restarting anything.
 * The LLM is started when AI mode is selected, avoiding model work during
   manual driving.

Run (CARLA and Ollama already running, put this file next to your others):
    pip install pygame
    python drive_and_test.py

Controls (click the pygame window first so it has keyboard focus):
    W / Up      throttle         S / Down   brake
    A / D       steer            Q          toggle reverse
    SPACE       handbrake
    TAB or M    switch MANUAL <-> AI mode
    B           toggle emergency-brake assist while in MANUAL mode
    C           switch chase camera <-> driver camera
    N / P       next / previous test case
    R           restart current test case
    ESC         quit
"""

import numpy as np
import pygame
import carla

import config

# Must be set BEFORE MultiSensorRig is created: we draw the camera ourselves.
config.SHOW_CV_WINDOW = False
config.SPECTATOR_FOLLOW = True   # best effort for the CARLA window as well

from main_closed_loop import (          # noqa: E402
    setup_simulation,
    restore_simulation,
    update_spectator,
    AsyncLLM,
)
from run_test_cases import setup_scenario   # noqa: E402
from sensor_rig import MultiSensorRig       # noqa: E402
from perception_processor import SensorProcessor   # noqa: E402
from prompt_builder import generate_multisensor_prompt   # noqa: E402
from llm_planner import LLMAgent            # noqa: E402
from safety_filter import SafetyFilter      # noqa: E402
from ambiguity_detector import AmbiguityDetector  # noqa: E402
from pid_controller import VehiclePIDController   # noqa: E402
from evaluation_logger import MetricsLogger       # noqa: E402


BASE_SCENARIOS = [
    "TC_01_CLEAR_LANE",
    "TC_02_STATIC_OBSTACLE",
    "TC_03_RAIN_AND_OBSTACLE",
    "TC_04_URBAN_JAYWALKER",
    "TC_05_SEMI_RURAL_ENCROACH",
    "TC_06_RURAL_SLOW_TRUCK",
    "TC_07_2WHEELER_CUT_IN",
    "TC_08_NIGHT_VISION_LIMIT",
    "TC_09_WRONG_WAY_ONCOMING",
    "TC_10_HEAVY_FOG_GRIDLOCK",
    "TC_11_ERRATIC_ANIMAL",
    "TC_12_SUN_GLARE",
    "TC_13_TIGHT_CORRIDOR",
    "TC_14_CROSS_TRAFFIC",
    "TC_15_CHAOTIC_MIXED",
    "TC_16_TRAFFIC_SIGNAL_STOP",
    "TC_17_PEDESTRIAN_CROWD",
    "TC_18_ANIMAL_CROSSING",
    "TC_19_DENSE_TRAFFIC_CONGESTION",
    "TC_20_ADVERSE_WEATHER",
    "TC_21_AMBIGUOUS_INTERSECTION",
    "TC_22_OCCLUDED_PEDESTRIAN",
    "TC_23_SIMULTANEOUS_CROSSING",
    "TC_24_CONFLICTING_SIGNALS",
    "TC_25_SUDDEN_LEAD_BRAKE",
    "TC_26_LOW_VISIBILITY_MIXED",
]

try:
    import extra_scenarios as extra
    EXTRA_NAMES = list(extra.EXTRA_NAMES)
except Exception:
    extra, EXTRA_NAMES = None, []

SCENARIOS = BASE_SCENARIOS + EXTRA_NAMES

ALLOWED_INTENTS = {
    "KEEP_LANE", "SLOW_DOWN", "STOP", "ACCELERATE",
    "MANEUVER_LEFT", "MANEUVER_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "OVERTAKE", "EMERGENCY_STOP"
}
WIN_W, WIN_H = 1280, 720


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sanitize_decision(d):
    """Make driving decision safe: valid intent, numeric speed, lateral maneuvers."""
    if isinstance(d, str):
        from llm_planner import _extract_and_sanitize
        parsed = _extract_and_sanitize(d)
        if parsed:
            return parsed
        d = {}
    elif not isinstance(d, dict):
        d = {}
    intent = str(d.get("intent", "KEEP_LANE")).upper().strip()
    if intent not in ALLOWED_INTENTS:
        if "RIGHT" in intent:
            intent = "MANEUVER_RIGHT"
        elif "LEFT" in intent:
            intent = "MANEUVER_LEFT"
        elif "STOP" in intent:
            intent = "STOP"
        elif "SLOW" in intent:
            intent = "SLOW_DOWN"
        elif "ACCEL" in intent:
            intent = "ACCELERATE"
        else:
            intent = "KEEP_LANE"
    try:
        speed = float(d.get("target_speed_kmh", 15.0))
    except (TypeError, ValueError):
        speed = 15.0
    if intent in ("STOP", "EMERGENCY_STOP"):
        speed = 0.0
    out = dict(d)
    out["intent"] = intent
    out["target_speed_kmh"] = round(speed, 1)
    return out


class ChaseCamera:
    """Third-person RGB camera on a spring arm, rendered in our window."""

    def __init__(self, world, vehicle, width, height):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")
        tf = carla.Transform(
            carla.Location(x=-7.0, z=3.2), carla.Rotation(pitch=-12.0)
        )
        self.frame = None   # RGB uint8 (H, W, 3)
        self.sensor = world.spawn_actor(
            bp, tf, attach_to=vehicle,
            attachment_type=carla.AttachmentType.SpringArm,
        )
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        self.frame = arr[:, :, :3][:, :, ::-1].copy()   # BGRA -> RGB

    def destroy(self):
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except Exception:
            pass


# ----------------------------------------------------------------------
# One live test-case session
# ----------------------------------------------------------------------
class Session:
    def __init__(self, world):
        self.world = world
        self.idx = 0
        self.mode = "MANUAL"          # "MANUAL" or "AI"
        self.assist = True            # AEB while driving manually
        self.view = "chase"           # "chase" or "driver"

        self.ego = None
        self.obstacles = []
        self.rig = None
        self.chase = None
        self.async_llm = None
        self.controller = None
        self.safety = SafetyFilter()
        self.ambiguity_detector = AmbiguityDetector()
        self.loggers = {}

        self.steer_cache = 0.0
        self.reverse = False
        self.advice = {"intent": "-", "target_speed_kmh": 0.0}
        self.override_reason = ""
        self.assist_active = False
        self.latency_ms = 0.0
        self.crash_flash = 0
        self.speed = 0.0
        self.lidar = 50.0
        self.last_intent = "-"
        self.error = ""
        self.ambiguity = {"active": False, "reason": ""}
        self._last_submit_t = 0.0
        self._frame = 0
        self._walker_hazard_latched = False

    # ---------------- scenario management ----------------
    def teardown(self):
        for res in (self.rig, self.chase):
            if res is not None:
                res.destroy()
        self.rig = None
        self.chase = None
        for actor in [self.ego] + list(self.obstacles):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.ego = None
        self.obstacles = []
        self.runtime = None

    def load(self, idx):
        self.teardown()
        self.idx = idx % len(SCENARIOS)
        name = SCENARIOS[self.idx]
        self.error = ""

        # Sweep anything left over
        for pattern in ("vehicle.*", "walker.*"):
            for a in self.world.get_actors().filter(pattern):
                try:
                    a.destroy()
                except Exception:
                    pass
        self._tick()

        try:
            self.ego, self.obstacles = setup_scenario(self.world, name)
            self.runtime = getattr(self.ego, "runtime", None)
            for _ in range(5):
                self._tick()
                update_spectator(self.world, self.ego)
            self.rig = MultiSensorRig(self.world, self.ego)
            self.chase = ChaseCamera(self.world, self.ego, WIN_W, WIN_H)
        except Exception as e:
            self.error = f"Could not set up {name}: {e}"
            print(f"[ERROR] {self.error}")
            return

        self.async_llm = AsyncLLM(LLMAgent()) if self.mode == "AI" else None
        self.controller = VehiclePIDController()
        self.steer_cache = 0.0
        self.reverse = False
        self.advice = {"intent": "-", "target_speed_kmh": 0.0}
        self.override_reason = ""
        self.latency_ms = 0.0
        self._last_submit_t = 0.0
        self._frame = 0
        self._walker_hazard_latched = False
        print(f"[Session] Loaded {name}")

    def _tick(self):
        if config.SYNCHRONOUS_MODE:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _logger(self):
        key = (SCENARIOS[self.idx], self.mode)
        if key not in self.loggers:
            self.loggers[key] = MetricsLogger(
                filename=f"result_{key[0]}_{key[1]}.csv"
            )
        return self.loggers[key]

    # ---------------- per-frame step ----------------
    def step(self, keys):
        if self.ego is None or not self.ego.is_alive:
            return

        self._tick()
        if hasattr(self, "runtime") and self.runtime is not None:
            try:
                self.runtime.tick(0.05)
            except Exception:
                pass
        if self._frame % 2 == 0:
            update_spectator(self.world, self.ego)

        self.speed = SensorProcessor.get_ego_velocity(self.ego)
        self.lidar = SensorProcessor.process_lidar_obstacle_distance(
            self.rig.latest_lidar_points
        )
        self.ambiguity = self.ambiguity_detector.assess(
            self.world, self.ego, self.lidar, self.speed
        )
        walkers = self.world.get_actors().filter("walker.pedestrian.*")
        ego_transform = self.ego.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        ego_right = ego_transform.get_right_vector()
        _, emergency_distance = self.safety.stopping_distances(self.speed)
        walker_lookahead = max(10.0, self.speed / 3.6 * 2.0 + emergency_distance)
        walker_hazard = False
        for actor in walkers:
            if not actor.is_alive:
                continue
            delta = actor.get_location() - ego_location
            longitudinal = delta.x * ego_forward.x + delta.y * ego_forward.y
            lateral = delta.x * ego_right.x + delta.y * ego_right.y
            if -2.0 <= longitudinal <= walker_lookahead and abs(lateral) <= 2.7:
                walker_hazard = True
                break
        if walker_hazard:
            self._walker_hazard_latched = True
        elif self._walker_hazard_latched and (
            any(
                actor.is_alive
                and (actor.get_location() - ego_location).x * ego_forward.x
                + (actor.get_location() - ego_location).y * ego_forward.y > -2.0
                for actor in walkers
            )
            or self.ambiguity.get("crossing_actor_count", 0) > 0
        ):
            walker_hazard = True
        else:
            self._walker_hazard_latched = False
        crashed = self.rig.collision_detected
        if crashed:
            self.crash_flash = 20
            self.rig.collision_detected = False

        if self.mode == "AI":
            if self.async_llm is None:
                self.async_llm = AsyncLLM(LLMAgent())
            raw, lat = self.async_llm.poll()
            if raw is not None:
                self.latency_ms = lat
                if lat <= getattr(config, "LLM_MAX_DECISION_AGE_S", 3.0) * 1000.0:
                    raw = sanitize_decision(raw)
                    self.advice = self.safety.validate_intent(
                        raw, self.lidar, self.speed, self.ambiguity
                    )
                    self.override_reason = self.advice.get("override_reason", "")
            now = time.time()
            if now - self._last_submit_t >= 1.0:
                self._last_submit_t = now
                self.async_llm.submit(
                    generate_multisensor_prompt(
                        self.speed, self.lidar, navigation_goal="Maintain Lane"
                    )
                )

        # --- control ---
        self.assist_active = False
        if self.mode == "AI":
            if walker_hazard or self.ambiguity.get("crossing_actor_count", 0) > 0:
                self.advice = {
                    "intent": "EMERGENCY_STOP",
                    "target_speed_kmh": 0.0,
                    "override_active": True,
                }
            throttle, brake = self.controller.calculate_longitudinal_control(
                self.speed, self.advice.get("target_speed_kmh", 0.0)
                if self.advice["intent"] != "-" else 20.0
            )
            ctrl = carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0)
            if self.advice["intent"] in ("STOP", "EMERGENCY_STOP"):
                self.controller.reset()
                ctrl.throttle = 0.0
                ctrl.brake = 1.0 if self.advice["intent"] == "EMERGENCY_STOP" else 0.7
            elif self.speed < 1.5 and ctrl.brake > 0.0:
                ctrl.throttle = 0.45
                ctrl.brake = 0.0
            self.last_intent = (
                "EMERGENCY_STOP"
                if walker_hazard or self.ambiguity.get("crossing_actor_count", 0) > 0
                else self.advice["intent"]
            )
        else:
            ctrl = self._manual_control(keys)
            self.last_intent = "MANUAL"
            if self.assist:
                chk = self.safety.validate_intent(
                    {"intent": "MANUAL", "target_speed_kmh": self.speed},
                    self.lidar, self.speed, self.ambiguity,
                )
                if walker_hazard or chk.get("override_active"):
                    ctrl.throttle = 0.0
                    ctrl.brake = (
                        1.0 if walker_hazard or chk["intent"] == "EMERGENCY_STOP"
                        else 0.5
                    )
                    self.assist_active = True
                    self.last_intent = "AEB_ASSIST"

        if self.lidar <= 3.8 and self.speed > 1.0:
            ctrl.throttle = 0.0
            ctrl.brake = 1.0
            self.assist_active = True
            self.last_intent = "AEB_EMERGENCY_STOP"
            self.override_reason = f"AEB Emergency Braking: Obstacle at {self.lidar:.1f}m"
        elif self.lidar <= 2.5:
            ctrl.throttle = 0.0
            ctrl.brake = max(ctrl.brake, 0.85)
            self.assist_active = True

        self.ego.apply_control(ctrl)
        self._frame += 1

        override = (
            self.advice.get("override_active", False)
            if self.mode == "AI" else self.assist_active
        )
        self._logger().log_cycle(
            self.speed, self.lidar, self.latency_ms,
            self.last_intent, override, crashed,
            self.ambiguity["active"],
        )

    def _manual_control(self, keys):
        throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        if left and not right:
            self.steer_cache = 0.0 if self.steer_cache > 0 else self.steer_cache - 0.05
        elif right and not left:
            self.steer_cache = 0.0 if self.steer_cache < 0 else self.steer_cache + 0.05
        else:
            self.steer_cache = 0.0
        self.steer_cache = max(-0.7, min(0.7, self.steer_cache))

        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=round(self.steer_cache, 2),
            hand_brake=bool(keys[pygame.K_SPACE]),
            reverse=self.reverse,
        )

    # ---------------- drawing ----------------
    def current_frame(self):
        if self.view == "chase":
            return self.chase.frame if self.chase else None
        if self.rig is not None and self.rig.latest_rgb is not None:
            return self.rig.latest_rgb[:, :, ::-1]   # BGR -> RGB
        return None


def draw(screen, font, small, sess):
    frame = sess.current_frame()
    if frame is not None:
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if surf.get_size() != screen.get_size():
            surf = pygame.transform.scale(surf, screen.get_size())
        screen.blit(surf, (0, 0))
    else:
        screen.fill((20, 20, 20))

    name = SCENARIOS[sess.idx]
    lines = [
        (f"{name}   ({sess.idx + 1}/{len(SCENARIOS)})", (255, 255, 255)),
        (
            f"Mode: {sess.mode}"
            + (f"   AEB assist: {'ON' if sess.assist else 'OFF'}"
               if sess.mode == "MANUAL" else ""),
            (120, 220, 255),
        ),
        (f"Speed: {sess.speed:5.1f} km/h    LiDAR: {sess.lidar:5.1f} m", (255, 255, 255)),
        (
            f"LLM {'decision' if sess.mode == 'AI' else 'advice'}: "
            f"{sess.advice['intent']} @ {sess.advice.get('target_speed_kmh', 0.0):.0f} km/h"
            f"   ({sess.latency_ms:.0f} ms)",
            (180, 255, 180),
        ),
    ]
    if sess.mode == "AI" and sess.override_reason:
        lines.append((f"SAFETY OVERRIDE: {sess.override_reason}", (255, 200, 80)))
    if sess.assist_active:
        lines.append(("AEB ASSIST BRAKING", (255, 120, 80)))
    if sess.ambiguity.get("active"):
        lines.append((
            f"AMBIGUITY: {sess.ambiguity['reason']}", (255, 170, 80)
        ))
    if sess.error:
        lines.append((sess.error, (255, 80, 80)))

    pad = 8
    box_h = len(lines) * 24 + pad * 2
    overlay = pygame.Surface((760, box_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (10, 10))
    for i, (txt, col) in enumerate(lines):
        screen.blit(font.render(txt, True, col), (10 + pad, 10 + pad + i * 24))

    if sess.crash_flash > 0:
        sess.crash_flash -= 1
        screen.blit(font.render("COLLISION!", True, (255, 60, 60)),
                    (WIN_W // 2 - 60, 40))

    help_txt = ("WASD drive | Q reverse | SPACE handbrake | TAB mode | B AEB | "
                "C camera | N/P/R test case | ESC quit")
    hs = small.render(help_txt, True, (230, 230, 230))
    bg = pygame.Surface((hs.get_width() + 16, hs.get_height() + 8), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 150))
    screen.blit(bg, (10, WIN_H - hs.get_height() - 18))
    screen.blit(hs, (18, WIN_H - hs.get_height() - 14))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    pygame.init()
    pygame.display.set_caption("CARLA + LLM Driving Test Harness")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT)
    world, original_settings = setup_simulation(
        client, map_name=config.CARLA_MAP, reload=config.RELOAD_WORLD_PER_TEST
    )

    sess = Session(world)
    sess.load(0)
    fps = int(round(1.0 / config.SIM_DELTA_SECONDS))

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k in (pygame.K_TAB, pygame.K_m):
                        sess.mode = "AI" if sess.mode == "MANUAL" else "MANUAL"
                    elif k == pygame.K_b:
                        sess.assist = not sess.assist
                    elif k == pygame.K_c:
                        sess.view = "driver" if sess.view == "chase" else "chase"
                    elif k == pygame.K_q:
                        sess.reverse = not sess.reverse
                    elif k == pygame.K_n:
                        sess.load(sess.idx + 1)
                    elif k == pygame.K_p:
                        sess.load(sess.idx - 1)
                    elif k == pygame.K_r:
                        sess.load(sess.idx)

            sess.step(pygame.key.get_pressed())
            draw(screen, font, small, sess)
            pygame.display.flip()
            clock.tick(fps)

    finally:
        sess.teardown()
        try:
            sess._tick()
        except Exception:
            pass
        restore_simulation(client, original_settings)
        pygame.quit()
        print("[Done] Session closed. CSVs saved as result_<scenario>_<mode>.csv")


if __name__ == "__main__":
    main()
