"""
offline_test_runner.py - Headless Off-Screen Test Runner for CARLA Scenarios.
Runs fully off-screen (no Pygame window, no CV spectator window).
Evaluates overtakes, lane changes, collision avoidance, and safety reasoning.
"""

import os
import sys
import time
import math
import subprocess
import argparse
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

# Force headless / off-screen settings in config
config.SHOW_CV_WINDOW = False
config.SPECTATOR_FOLLOW = False
config.SYNCHRONOUS_MODE = True

import carla
from main_closed_loop import setup_simulation, restore_simulation
from drive_in_carla import Session, SCENARIOS, DT


class HeadlessInput:
    """Dummy input provider that replaces Pygame/keyboard input for headless runs."""
    def __init__(self):
        self.event_count = 0

    def update(self):
        return True

    def down(self, key):
        return False

    def once(self, key):
        return False


def is_carla_running(host="127.0.0.1", port=2000, timeout=2.0):
    """Check if CARLA server is reachable."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def launch_carla_headless():
    """Launch CarlaUE4 with -RenderOffScreen -nosound."""
    carla_exe = os.path.join(BASE_DIR, "CarlaUE4.exe")
    if not os.path.exists(carla_exe):
        carla_exe = os.path.join(BASE_DIR, "CarlaUE4", "Binaries", "Win64", "CarlaUE4-Win64-Shipping.exe")
    if not os.path.exists(carla_exe):
        print(f"[Error] Could not find CarlaUE4.exe at {carla_exe}")
        return None

    cmd = [carla_exe, "-RenderOffScreen", "-nosound", "-quality-level=Low", "-benchmark", "-fps=20"]
    print(f"[Launcher] Starting headless CARLA: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("[Launcher] Waiting for CARLA to start on port 2000...")
    for attempt in range(40):
        time.sleep(1.0)
        if is_carla_running():
            print(f"[Launcher] CARLA connected successfully (attempt {attempt+1})!")
            return proc
    print("[Launcher] Failed to connect to CARLA within 40 seconds.")
    return proc


def run_scenario(session, scenario_name, max_frames=150):
    """Run a single scenario headlessly and record metrics."""
    print(f"\n============================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"============================================================")
    
    try:
        scen_idx = SCENARIOS.index(scenario_name)
    except ValueError:
        print(f"[Error] Unknown scenario: {scenario_name}")
        return None

    session.load(scen_idx)
    keys = HeadlessInput()

    stats = {
        "scenario": scenario_name,
        "frames": 0,
        "collisions": 0,
        "overtake_started": False,
        "overtake_completed": False,
        "overtake_phases": set(),
        "min_lidar_m": 999.0,
        "max_speed_kmh": 0.0,
        "avg_speed_kmh": 0.0,
        "reasons": set(),
        "passed": False,
        "score": 100.0,
    }

    speeds = []
    
    for frame in range(max_frames):
        try:
            session.step(keys)
        except Exception as e:
            print(f"\n[Frame {frame}] Exception in step: {e}")
            break

        stats["frames"] += 1
        speed = getattr(session, "speed", 0.0)
        speeds.append(speed)
        stats["max_speed_kmh"] = max(stats["max_speed_kmh"], speed)

        lidar = getattr(session, "lidar", 999.0)
        stats["min_lidar_m"] = min(stats["min_lidar_m"], lidar)

        if getattr(session, "crash_flash", 0) > 0:
            stats["collisions"] += 1

        ov_planner = getattr(session, "overtake_planner", None)
        if ov_planner is not None:
            if ov_planner.is_active():
                stats["overtake_started"] = True
                stats["overtake_phases"].add(ov_planner.state)
            if getattr(session, "_post_overtake_resume", False):
                stats["overtake_completed"] = True

        reason = getattr(session, "override_reason", "")
        if reason:
            stats["reasons"].add(reason)

    if speeds:
        stats["avg_speed_kmh"] = sum(speeds) / len(speeds)

    score = 100.0
    if stats["collisions"] > 0:
        score -= min(100.0, stats["collisions"] * 50.0)

    is_obstacle_tc = any(k in scenario_name for k in ("STATIC_OBSTACLE", "SLOW_TRUCK", "ENCROACH", "TIGHT_CORRIDOR"))
    if is_obstacle_tc:
        if stats["overtake_completed"]:
            score = max(score, 85.0)
        elif stats["overtake_started"]:
            score = max(score, 70.0)
        elif stats["collisions"] == 0 and stats["min_lidar_m"] > 1.0:
            score = max(score, 50.0)
        else:
            score = max(score, 20.0)

    stats["score"] = round(score, 1)
    stats["passed"] = (stats["collisions"] == 0 and (score >= 60.0 or not is_obstacle_tc))

    status_str = "PASSED" if stats["passed"] else "FAILED"
    print(f"\n[Result] {scenario_name}: {status_str} | Score: {stats['score']} | "
          f"Collisions: {stats['collisions']} | Overtake: started={stats['overtake_started']} "
          f"done={stats['overtake_completed']} phases={list(stats['overtake_phases'])}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Headless Off-Screen CARLA Scenario Runner")
    parser.add_argument("--scenario", type=str, default="TC_02_STATIC_OBSTACLE",
                        help="Scenario name or 'all' for all 26 scenarios")
    parser.add_argument("--frames", type=int, default=150,
                        help="Number of simulation frames per scenario (default: 150)")
    parser.add_argument("--launch-carla", action="store_true",
                        help="Automatically start CARLA server in headless mode if not running")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name (e.g. qwen2.5:1.5b)")
    args = parser.parse_args()

    if args.model:
        config.OLLAMA_MODEL = args.model

    carla_proc = None
    if not is_carla_running():
        if args.launch_carla:
            carla_proc = launch_carla_headless()
            if carla_proc is None or not is_carla_running():
                print("[Error] CARLA could not be launched. Exiting.")
                return 1
        else:
            print("[Error] CARLA is not running on port 2000.")
            print("Please start CARLA first, or run with --launch-carla flag.")
            return 1

    print("[1] Connecting to CARLA...")
    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT)

    print("[2] Setting up simulation world...")
    world, orig_settings = setup_simulation(client, map_name=config.CARLA_MAP, reload=False)

    session = Session(world, client=client)
    session.set_mode("AI")

    if args.scenario == "all":
        scenarios_to_run = SCENARIOS[:26]
    elif "," in args.scenario:
        scenarios_to_run = [s.strip() for s in args.scenario.split(",") if s.strip() in SCENARIOS]
    else:
        scenarios_to_run = [args.scenario]

    all_results = []
    try:
        for scen in scenarios_to_run:
            res = run_scenario(session, scen, max_frames=args.frames)
            if res:
                all_results.append(res)
    finally:
        print("\n[Teardown] Restoring simulation settings...")
        session.teardown()
        restore_simulation(client, orig_settings)
        if carla_proc is not None:
            print("[Launcher] Terminating headless CARLA server...")
            carla_proc.terminate()

    out_csv = os.path.join(BASE_DIR, "offline_test_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Passed", "Score", "Collisions", "OvertakeStarted",
                         "OvertakeCompleted", "OvertakePhases", "AvgSpeedKmh", "MinLidarM"])
        for r in all_results:
            writer.writerow([r["scenario"], r["passed"], r["score"], r["collisions"],
                             r["overtake_started"], r["overtake_completed"],
                             ";".join(r["overtake_phases"]), round(r["avg_speed_kmh"], 1),
                             round(r["min_lidar_m"], 1)])

    print(f"\n============================================================")
    print(f"OFFLINE TEST RUNNER SUMMARY (Saved to {out_csv})")
    print(f"============================================================")
    print(f"{'Scenario':<32} | {'Passed':<6} | {'Score':<5} | {'Collisions':<10} | {'Overtake':<12}")
    print("-" * 75)
    for r in all_results:
        ov_str = "DONE" if r["overtake_completed"] else ("ACTIVE" if r["overtake_started"] else "NONE")
        pass_str = "PASS" if r["passed"] else "FAIL"
        print(f"{r['scenario']:<32} | {pass_str:<6} | {r['score']:5.1f} | {r['collisions']:10d} | {ov_str:<12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
