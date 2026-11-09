"""
ad_metrics.py - CARLA-leaderboard style evaluation exactly as in DriveGPT4-V2 (Sec. 4.3, Tab. 1):

    DS = RC x IS,   IS = prod(penalty ^ count)
    penalties: pedestrian 0.50, vehicle 0.60, static 0.65, red light 0.70, stop sign 0.70
    event ratios: Ped, Veh, Stat, Red, Dev (route deviation), TO (timeout), Block

Fixes w.r.t. the old tracker:
  * collisions are classified from the collision sensor's other_actor.type_id (not by
    guessing from distances) and de-bounced per actor (one contact = one event);
  * route completion = metres progressed along the planned route / route target
    (it was forced to >= 95 % before);
  * red-light violation = front bumper crosses the stop line while the light is red;
  * FPS = real simulation-loop rate; LLM latency / Hz reported separately.

Outputs (written next to drive_in_carla.py):
    drivegpt4_v2_paper_metrics_v2.csv      one row per episode
    drivegpt4_v2_decihead_trace_v2.csv     2 Hz trace of the four decision variables
    LLM_Comparison_Report.md               aggregated by model (regenerated each episode)
    results/result_<TC>_<MODE>.csv         2 Hz per-scenario log
"""
import csv
import json
import os
import statistics
import time

PENALTY = {"ped": 0.50, "veh": 0.60, "stat": 0.65, "red": 0.70, "stop": 0.70}

METRICS_CSV = "drivegpt4_v2_paper_metrics_v2.csv"
TRACE_CSV = "drivegpt4_v2_decihead_trace_v2.csv"
REPORT_MD = "LLM_Comparison_Report.md"
RESULTS_DIR = "results"

COLUMNS = [
    "Timestamp", "Test_Case", "Model", "Mode", "Authority", "Driving_Score_DS", "Route_Completion_RC_%",
    "Infraction_Score_IS", "Pedestrian_Collisions", "Vehicle_Collisions", "Static_Collisions",
    "Red_Light_Violations", "Route_Deviations", "Timeout_TO", "Agent_Block", "Expected_Hold_OK",
    "Sim_Time_s", "Distance_m", "Avg_Speed_kmh", "Sim_FPS", "LLM_Mean_Latency_ms", "LLM_Hz",
    "LLM_Decisions", "LLM_Control_%", "DAgger_Takeover_%", "L1_Speed_Error_kmh", "Primary_Failure_Mode",
    "End_Reason",
]


class EpisodeMetrics:
    def __init__(self, base_dir, scenario, model, mode, authority, route_target_m=150.0,
                 expect_hold=False):
        self.base = base_dir
        self.scenario, self.model, self.mode, self.authority = scenario, model, mode, authority
        self.route_target = route_target_m
        self.expect_hold = expect_hold
        self.t0_wall = time.time()
        self.sim_t = 0.0
        self.frames = 0
        self.progress = 0.0
        self.s_start = None
        self.dist = 0.0
        self.counts = {"ped": 0, "veh": 0, "stat": 0, "red": 0}
        self._last_hit = {}
        self.dev_events = 0
        self._dev_t = 0.0
        self._in_dev = False
        self.block_t = 0.0
        self.max_block = 0.0
        self.latencies = []
        self.l1 = []
        self.takeovers = 0
        self.llm_frames = 0
        self.arb_frames = 0
        self.failures = {}
        self.llm_count = 0
        self._prev_front = None
        self.s_front_last = 0.0
        self._trace_t = -1.0
        self.finished = False
        self.end_reason = ""
        self.hold_ok = False
        self.last_hold = ""
        self.loop_times = []
        os.makedirs(os.path.join(base_dir, RESULTS_DIR), exist_ok=True)
        self._log_path = os.path.join(base_dir, RESULTS_DIR, f"result_{scenario}_{mode}.csv")
        self._log = open(self._log_path, "w", newline="", encoding="utf-8")
        self._logw = csv.writer(self._log)
        self._logw.writerow(["sim_t", "speed_kmh", "target_kmh", "lead_gap_m", "intent", "source",
                             "llm_action", "llm_speed", "latency_ms", "hold", "reason", "collision",
                             "light", "lateral_offset_m"])
        tpath = os.path.join(base_dir, TRACE_CSV)
        new = not os.path.exists(tpath)
        self._trace = open(tpath, "a", newline="", encoding="utf-8")
        self._tw = csv.writer(self._trace)
        if new:
            self._tw.writerow(["Timestamp", "Model", "Scenario", "Sim_t", "Ego_Speed_kmh", "Fused_Front_Gap_m",
                               "LLM_Action", "LLM_Target_Speed_kmh", "LLM_Latency_ms", "Expert_Action",
                               "Expert_Target_Speed_kmh", "DeciHead_Target_Speed_kmh",
                               "DeciHead_Target_Angle_deg", "L1_Speed_Error_kmh", "DAgger_Takeover",
                               "Source", "Waypoints_8_Future_4s", "Route_Points_10_Ahead"])

    # ---------------------------------------------------------------- per tick
    def collision(self, events, sim_t):
        new = []
        for _, oid, tid, imp in events:
            key = oid if oid else tid
            if sim_t - self._last_hit.get(key, -99.0) < 2.0:
                self._last_hit[key] = sim_t
                continue
            self._last_hit[key] = sim_t
            cat = "ped" if tid.startswith("walker.") else ("veh" if tid.startswith("vehicle.") else "stat")
            if cat == "stat" and not tid.startswith("static.prop") and imp < 50.0:
                continue                         # brushing a kerb is not a crash (props always count)
            self.counts[cat] += 1
            new.append((cat, tid))
        return new

    def update(self, sim_t, dt, ego_s, ego_v, front_s, light, ego_d, lane_hw, lateral_mode,
               plan, arb_info, loop_dt, dist_step, collisions_now):
        self.frames += 1
        self.sim_t = sim_t
        self.loop_times.append(loop_dt)
        if len(self.loop_times) > 2000:
            self.loop_times = self.loop_times[-2000:]
        if self.s_start is None:
            self.s_start = ego_s
        self.progress = max(self.progress, ego_s - self.s_start)
        self.dist += dist_step
        # red light violation
        if light is not None and self._prev_front is not None:
            if light.state == "RED" and self._prev_front <= light.s_line < front_s and ego_v > 0.5:
                self.counts["red"] += 1
        self._prev_front = front_s
        self.s_front_last = front_s
        # route deviation (outside the lane while not deliberately bypassing)
        dev = abs(ego_d) > lane_hw + 1.2 and lateral_mode == "LANE"
        if dev:
            self._dev_t += dt
            if self._dev_t > 1.5 and not self._in_dev:
                self.dev_events += 1
                self._in_dev = True
        else:
            self._dev_t, self._in_dev = 0.0, False
        # block: stand-still without a legitimate reason (red light / yield / queue / closure)
        self.last_hold = plan.hold if plan is not None else ""
        legit = bool(plan is not None and plan.hold and any(
            k in plan.hold for k in ("light", "yield", "queue", "closed", "junction")))
        if ego_v < 0.3 and not legit:
            self.block_t += dt
            self.max_block = max(self.max_block, self.block_t)
        else:
            self.block_t = 0.0
        # LLM / DAgger
        if arb_info is not None:
            self.arb_frames += 1
            if arb_info.get("source", "").startswith("LLM"):
                self.llm_frames += 1
            if arb_info.get("takeover"):
                self.takeovers += 1
            if arb_info.get("l1") is not None:
                self.l1.append(arb_info["l1"])
            f = arb_info.get("failure", "NONE")
            self.failures[f] = self.failures.get(f, 0) + 1

    def llm_result(self, dec):
        if dec is not None:
            self.llm_count += 1
            self.latencies.append(dec.get("latency_ms", 0.0))

    def log_2hz(self, sim_t, ego_v, plan, final_v, arb, angle_deg, waypoints, route_pts, fused_gap,
                collided, light=None, ego_d=0.0):
        if sim_t - self._trace_t < 0.5:
            return
        self._trace_t = sim_t
        a = arb or {}
        try:
            self._logw.writerow([round(sim_t, 2), round(ego_v * 3.6, 1), round(final_v * 3.6, 1),
                                 round(plan.lead_gap, 1) if plan else "", plan.intent if plan else "MANUAL",
                                 a.get("source", ""), a.get("llm_action", ""),
                                 "" if a.get("llm_speed_kmh") is None else round(a["llm_speed_kmh"], 1),
                                 "" if a.get("latency_ms") is None else round(a["latency_ms"]),
                                 plan.hold if plan else "", (plan.reason if plan else "")[:220],
                                 int(bool(collided)),
                                 "" if light is None else f"{light.state}@{light.s_line - self.s_front_last:.0f}m",
                                 round(ego_d, 2)])
            self._tw.writerow([round(time.time(), 1), self.model, self.scenario, round(sim_t, 2),
                               round(ego_v * 3.6, 2), "" if fused_gap is None else round(fused_gap, 2),
                               a.get("llm_action", "-"),
                               "" if a.get("llm_speed_kmh") is None else round(a["llm_speed_kmh"], 1),
                               "" if a.get("latency_ms") is None else round(a["latency_ms"]),
                               a.get("expert_action", ""),
                               "" if a.get("expert_speed_kmh") is None else round(a["expert_speed_kmh"], 1),
                               round(final_v * 3.6, 2), round(angle_deg, 2),
                               "" if a.get("l1") is None else round(a["l1"], 2),
                               int(bool(a.get("takeover"))), a.get("source", ""),
                               json.dumps(waypoints), json.dumps(route_pts)])
        except Exception:
            pass

    # ---------------------------------------------------------------- finish
    def finish(self, end_reason="", final_speed=0.0):
        if self.finished:
            return None
        self.finished = True
        self.end_reason = end_reason
        for fh in (self._log, self._trace):
            try:
                fh.close()
            except Exception:
                pass
        if self.frames < 20:
            return None
        rc = min(100.0, 100.0 * self.progress / max(1.0, self.route_target))
        crashes = self.counts["ped"] + self.counts["veh"] + self.counts["stat"]
        if (self.expect_hold and crashes == 0 and self.counts["red"] == 0 and final_speed < 0.5
                and self.last_hold):
            self.hold_ok = True
            rc = 100.0
        is_ = 1.0
        for k, n in self.counts.items():
            is_ *= PENALTY[k] ** n
        ds = rc * is_
        lat = statistics.mean(self.latencies) if self.latencies else 0.0
        fps = (len(self.loop_times) / sum(self.loop_times)) if self.loop_times and sum(self.loop_times) > 0 else 0.0
        fails = {k: v for k, v in self.failures.items() if k not in ("NONE",)}
        primary = max(fails, key=fails.get) if fails else "NONE"
        row = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Test_Case": self.scenario, "Model": self.model, "Mode": self.mode, "Authority": self.authority,
            "Driving_Score_DS": round(ds, 2), "Route_Completion_RC_%": round(rc, 1),
            "Infraction_Score_IS": round(is_, 3),
            "Pedestrian_Collisions": self.counts["ped"], "Vehicle_Collisions": self.counts["veh"],
            "Static_Collisions": self.counts["stat"], "Red_Light_Violations": self.counts["red"],
            "Route_Deviations": self.dev_events,
            "Timeout_TO": int(end_reason == "timeout" and rc < 100.0 and not self.hold_ok),
            "Agent_Block": int(end_reason == "blocked"),
            "Expected_Hold_OK": int(self.hold_ok),
            "Sim_Time_s": round(self.sim_t, 1), "Distance_m": round(self.dist, 1),
            "Avg_Speed_kmh": round(3.6 * self.dist / max(1e-3, self.sim_t), 1),
            "Sim_FPS": round(fps, 1), "LLM_Mean_Latency_ms": round(lat, 1),
            "LLM_Hz": round(self.llm_count / max(1e-3, self.sim_t), 2),
            "LLM_Decisions": self.llm_count,
            "LLM_Control_%": round(100.0 * self.llm_frames / max(1, self.arb_frames), 1),
            "DAgger_Takeover_%": round(100.0 * self.takeovers / max(1, self.arb_frames), 1),
            "L1_Speed_Error_kmh": round(statistics.mean(self.l1), 2) if self.l1 else 0.0,
            "Primary_Failure_Mode": primary, "End_Reason": end_reason,
        }
        path = os.path.join(self.base, METRICS_CSV)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        write_report(self.base)
        return row


def write_report(base_dir):
    path = os.path.join(base_dir, METRICS_CSV)
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    # keep the latest run per (model, authority, test case)
    latest = {}
    for r in rows:
        latest[(r["Model"], r.get("Authority", ""), r["Test_Case"])] = r
    groups = {}
    for (m, a, _), r in latest.items():
        groups.setdefault((m, a), []).append(r)

    def f(r, k):
        try:
            return float(r.get(k, 0) or 0)
        except ValueError:
            return 0.0

    md = ["# DriveGPT4-V2 Closed-Loop Evaluation Report", "",
          "Metrics follow the CARLA leaderboard / DriveGPT4-V2 (CVPR 2025) Tab. 1-2: "
          "DS = RC x IS; penalties Ped 0.50, Veh 0.60, Stat 0.65, Red 0.70. "
          "Ped/Veh/Stat/Red/Dev are events per episode; TO/Block are episode ratios.", "",
          "## 1. Aggregated by model", "",
          "| Model | Authority | Episodes | DS | RC % | IS | Ped | Veh | Stat | Red | Dev | TO | Block | "
          "Sim FPS | LLM latency ms | LLM Hz | LLM control % | DAgger % | L1 km/h |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, a), rs in sorted(groups.items()):
        n = len(rs)

        def avg(k):
            return sum(f(r, k) for r in rs) / n
        md.append(f"| **{m}** | {a} | {n} | **{avg('Driving_Score_DS'):.1f}** | {avg('Route_Completion_RC_%'):.1f} | "
                  f"{avg('Infraction_Score_IS'):.2f} | {avg('Pedestrian_Collisions'):.2f} | "
                  f"{avg('Vehicle_Collisions'):.2f} | {avg('Static_Collisions'):.2f} | "
                  f"{avg('Red_Light_Violations'):.2f} | {avg('Route_Deviations'):.2f} | {avg('Timeout_TO'):.2f} | "
                  f"{avg('Agent_Block'):.2f} | {avg('Sim_FPS'):.1f} | {avg('LLM_Mean_Latency_ms'):.0f} | "
                  f"{avg('LLM_Hz'):.2f} | {avg('LLM_Control_%'):.1f} | {avg('DAgger_Takeover_%'):.1f} | "
                  f"{avg('L1_Speed_Error_kmh'):.2f} |")
    md += ["", "## 2. Per test case (latest run)", "",
           "| Test case | Model | Auth | DS | RC % | IS | Ped | Veh | Stat | Red | Hold OK | End | "
           "LLM ms | DAgger % | Failure |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def _tc_key(r):
        try:
            return int(r["Test_Case"].split("_")[1])
        except Exception:
            return 999
    for (m, a), rs in sorted(groups.items()):
        for r in sorted(rs, key=_tc_key):
            md.append(f"| `{r['Test_Case']}` | {m} | {a} | **{r['Driving_Score_DS']}** | {r['Route_Completion_RC_%']} | "
                      f"{r['Infraction_Score_IS']} | {r['Pedestrian_Collisions']} | {r['Vehicle_Collisions']} | "
                      f"{r['Static_Collisions']} | {r['Red_Light_Violations']} | {r['Expected_Hold_OK']} | "
                      f"{r['End_Reason']} | {r['LLM_Mean_Latency_ms']} | {r['DAgger_Takeover_%']} | "
                      f"`{r['Primary_Failure_Mode']}` |")
    with open(os.path.join(base_dir, REPORT_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
