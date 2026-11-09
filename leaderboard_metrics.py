"""
leaderboard_metrics.py - CARLA leaderboard metrics exactly as used in
Xu et al., "DriveGPT4-V2" (CVPR 2025), Sec. 4.3 / Tables 1-3.

Primary metrics
    RC  Route Completion (%)
    IS  Infraction Score = product of penalty factors (Table 1)
    DS  Driving Score    = RC x IS
Exception-event ratios (reported per km driven, lower is better)
    Ped   pedestrian collision          (penalty 0.50)
    Veh   vehicle collision             (penalty 0.60)
    Stat  static object collision       (penalty 0.65)
    Red   red light violation           (penalty 0.70)
    Stop  stop sign violation           (penalty 0.70)
    Dev   route deviation (> 30 m from the planned route)
    TO    scenario time-out (route not completed in the time budget)
    Block agent blocked (stationary for too long without a legitimate reason)
Efficiency
    FPS   LLM decision rate = 1000 / mean LLM latency (ms)   (paper Table 3)
    Step_FPS  wall-clock simulation-loop rate

One LeaderboardMetrics instance tracks ONE scenario / route. Use aggregate()
to combine scenarios into a paper-style table row.
"""

import csv
import math
import os

PENALTY = {"Ped": 0.50, "Veh": 0.60, "Stat": 0.65, "Red": 0.70, "Stop": 0.70}  # Table 1
EVENTS = ("Ped", "Veh", "Stat", "Red", "Stop")

# Reference row from Table 2 of the paper (CARLA Longest6) for the report
PAPER_REFERENCE = {
    "Model": "DriveGPT4-V2 (paper, Table 2)",
    "DS": 70.0, "RC": 91.0, "IS": 0.77, "Ped": 0.00, "Veh": 0.80, "Stat": 0.01,
    "Red": 0.04, "Stop": None, "Dev": 0.00, "TO": 0.07, "Block": 0.09, "FPS": 8.1,
}


def state_name(state):
    return str(state).split(".")[-1]


def build_route(world, ego, length_m, step_m=2.0):
    """Planned route: follow the ego lane straight (min yaw change) for length_m."""
    wmap = world.get_map()
    w = wmap.get_waypoint(ego.get_location(), project_to_road=True)
    pts = [(w.transform.location.x, w.transform.location.y)]
    travelled = 0.0
    while travelled < length_m:
        nxt = w.next(step_m)
        if not nxt:
            break
        yaw = w.transform.rotation.yaw
        w = min(nxt, key=lambda c: abs(((c.transform.rotation.yaw - yaw + 180.0) % 360.0) - 180.0))
        pts.append((w.transform.location.x, w.transform.location.y))
        travelled += step_m
    return pts


class LeaderboardMetrics:
    def __init__(self, world=None, ego=None, route_length_m=100.0, dt=0.05, block_s=8.0,
                 dev_dist_m=30.0, scenario="", model="", route_points=None):
        self.world = world
        self.dt = float(dt)
        self.scenario = scenario
        self.model = model
        self.block_s = float(block_s)
        self.dev_dist_m = float(dev_dist_m)

        if route_points is None:
            route_points = build_route(world, ego, route_length_m)
        self.route = list(route_points)
        self.route_s = [0.0]
        for i in range(1, len(self.route)):
            self.route_s.append(self.route_s[-1] + math.hypot(
                self.route[i][0] - self.route[i - 1][0], self.route[i][1] - self.route[i - 1][1]))
        self.route_len = max(self.route_s[-1], 1.0)
        self.progress_idx = 0
        self.max_progress_m = 0.0

        self.counts = {k: 0 for k in EVENTS}
        self.dev_events = 0
        self._dev_active = False
        self.block_events = 0
        self._stationary_s = 0.0
        self._block_counted = False
        self.events = []                 # human-readable infraction log

        self.frames = 0
        self.dist_m = 0.0
        self._prev_xy = None
        self._last_col_frame = {}
        self._prev_light = None
        self._last_red_frame = -10 ** 9
        self.latencies = []
        self.step_ms = []

        self.stop_signs = self._load_stop_signs(world) if world is not None else []
        self._stop_state = {}

    # ---------------- stop signs (OpenDRIVE landmark type 206) ----------------
    @staticmethod
    def _load_stop_signs(world):
        signs = []
        try:
            for lm in world.get_map().get_all_landmarks_of_type("206"):
                signs.append({"id": lm.id, "road": lm.road_id, "s": lm.s,
                              "orient": str(lm.orientation).split(".")[-1]})
        except Exception:
            pass
        return signs

    def _check_stop_signs(self, ego_wp, speed_kmh):
        if ego_wp is None or not self.stop_signs:
            return
        lane_sign = -1 if ego_wp.lane_id < 0 else 1
        for sg in self.stop_signs:
            if sg["road"] != ego_wp.road_id:
                continue
            if sg["orient"] == "Positive" and lane_sign > 0:
                continue
            if sg["orient"] == "Negative" and lane_sign < 0:
                continue
            remaining = (sg["s"] - ego_wp.s) * (1.0 if lane_sign < 0 else -1.0)
            st = self._stop_state.setdefault(sg["id"], {"active": False, "min": 1e9, "done": False})
            if 0.0 <= remaining <= 10.0 and not st["done"]:
                st["active"] = True
                st["min"] = min(st["min"], speed_kmh)
            elif remaining < 0.0 and st["active"] and not st["done"]:
                st["done"] = True
                if st["min"] > 1.0:
                    self.counts["Stop"] += 1
                    self.events.append((self.frames, "Stop", f"ran stop sign (min {st['min']:.1f} km/h)"))
            elif remaining > 12.0:
                st["active"], st["min"] = False, 1e9

    # ---------------- per-frame update ----------------
    def update(self, ego_loc, speed_kmh, ego_wp=None, crashed=False, collision=None,
               light=None, holds=False, latency_ms=0.0, step_ms=None):
        """Call once per simulation frame.

        collision: dict(other_id, other_type) of the latest impact.
        light:     (state, distance_m) of the light controlling the ego lane, or None.
        holds:     True while the car is stopped for a legitimate reason
                   (red light, zebra crossing, pedestrian).
        """
        self.frames += 1
        xy = (ego_loc.x, ego_loc.y)
        if self._prev_xy is not None:
            self.dist_m += math.hypot(xy[0] - self._prev_xy[0], xy[1] - self._prev_xy[1])
        self._prev_xy = xy
        if latency_ms and latency_ms > 0:
            self.latencies.append(float(latency_ms))
        if step_ms is not None:
            self.step_ms.append(float(step_ms))

        # route progress (RC) and deviation (Dev)
        best_i, best_d = self.progress_idx, 1e9
        for i in range(len(self.route)):
            d = math.hypot(xy[0] - self.route[i][0], xy[1] - self.route[i][1])
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= self.progress_idx and best_i - self.progress_idx <= 25 and best_d < self.dev_dist_m:
            self.progress_idx = best_i
            self.max_progress_m = max(self.max_progress_m, self.route_s[best_i])
        if best_d > self.dev_dist_m:
            if not self._dev_active:
                self._dev_active = True
                self.dev_events += 1
                self.events.append((self.frames, "Dev", f"{best_d:.0f} m off route"))
        else:
            self._dev_active = False

        # collisions (Ped / Veh / Stat) - debounced per actor for 1 s
        if crashed:
            info = collision or {}
            otype = str(info.get("other_type", "static.unknown"))
            key = info.get("other_id", otype)
            if self.frames - self._last_col_frame.get(key, -10 ** 9) > int(1.0 / self.dt):
                if otype.startswith("walker"):
                    cat = "Ped"
                elif otype.startswith("vehicle"):
                    cat = "Veh"
                else:
                    cat = "Stat"
                self.counts[cat] += 1
                self.events.append((self.frames, cat, f"collision with {otype}"))
            self._last_col_frame[key] = self.frames

        # red light: the controlling light disappears (bumper crossed the stop line)
        # while it was red and the car was still moving
        if light is not None:
            self._prev_light = (state_name(light[0]), float(light[1]), speed_kmh, self.frames)
        else:
            pl = self._prev_light
            if (pl is not None and pl[0] == "Red" and pl[1] < 4.5 and pl[2] > 3.0
                    and self.frames - pl[3] <= 2
                    and self.frames - self._last_red_frame > int(5.0 / self.dt)):
                self.counts["Red"] += 1
                self._last_red_frame = self.frames
                self.events.append((self.frames, "Red", f"ran red light at {pl[2]:.0f} km/h"))
            self._prev_light = None

        # stop sign
        try:
            self._check_stop_signs(ego_wp, speed_kmh)
        except Exception:
            pass

        # blocked: stationary too long without a legitimate reason
        if speed_kmh < 0.5 and not holds:
            self._stationary_s += self.dt
            if self._stationary_s >= self.block_s and not self._block_counted:
                self._block_counted = True
                self.block_events += 1
                self.events.append((self.frames, "Block", f"stationary > {self.block_s:.0f} s"))
        else:
            self._stationary_s = 0.0
            self._block_counted = False

    # ---------------- results ----------------
    def infraction_score(self):
        score = 1.0
        for k in EVENTS:
            score *= PENALTY[k] ** self.counts[k]
        return score

    def finalize(self, timed_out=False):
        rc = min(100.0, 100.0 * self.max_progress_m / self.route_len)
        is_ = self.infraction_score()
        km = self.dist_m / 1000.0
        mean_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        mean_step = sum(self.step_ms) / len(self.step_ms) if self.step_ms else 0.0
        row = {
            "Model": self.model, "Scenario": self.scenario,
            "DS": round(rc * is_, 2), "RC": round(rc, 1), "IS": round(is_, 3),
            **{k: self.counts[k] for k in EVENTS},
            "Dev": self.dev_events,
            "TO": 1 if (timed_out and rc < 99.5) else 0,
            "Block": self.block_events,
            "Km": round(km, 4), "Frames": self.frames,
            "Latency_ms": round(mean_lat, 1),
            "FPS": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
            "Step_FPS": round(1000.0 / mean_step, 1) if mean_step > 0 else 0.0,
        }
        return row

    def summary_text(self, timed_out=False):
        r = self.finalize(timed_out)
        ev = " ".join(f"{k}={r[k]}" for k in (*EVENTS, "Dev", "Block"))
        return (f"[Metrics] {self.scenario}: DS {r['DS']:.1f} | RC {r['RC']:.1f}% | "
                f"IS {r['IS']:.3f} | {ev} | {r['Km'] * 1000:.0f} m driven")

    def write_row(self, path, timed_out=False):
        row = self.finalize(timed_out)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)


def _mean_pos(rows, key, nd):
    vals = [r[key] for r in rows if r[key] > 0]
    return round(sum(vals) / len(vals), nd) if vals else 0.0


def aggregate(rows, model=""):
    """Paper-style row (Table 2) from per-scenario rows.

    DS / RC / IS: mean over scenarios (leaderboard convention).
    Ped, Veh, Stat, Red, Stop, Dev, Block: total events per km driven.
    TO: fraction of scenarios that timed out.
    """
    n = max(1, len(rows))
    km = sum(r["Km"] for r in rows)

    def per_km(key):
        total = sum(r[key] for r in rows)
        return round(total / km, 2) if km > 0.005 else 0.0

    return {
        "Model": model,
        "DS": round(sum(r["DS"] for r in rows) / n, 1),
        "RC": round(sum(r["RC"] for r in rows) / n, 1),
        "IS": round(sum(r["IS"] for r in rows) / n, 2),
        **{k: per_km(k) for k in EVENTS},
        "Dev": per_km("Dev"),
        "TO": round(sum(r["TO"] for r in rows) / n, 2),
        "Block": per_km("Block"),
        "Km": round(km, 3),
        "FPS": _mean_pos(rows, "FPS", 1),
        "Step_FPS": _mean_pos(rows, "Step_FPS", 1),
        "Latency_ms": _mean_pos(rows, "Latency_ms", 1),
        "Scenarios": len(rows),
    }