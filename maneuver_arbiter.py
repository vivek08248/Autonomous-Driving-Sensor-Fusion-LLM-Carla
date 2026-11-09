"""
maneuver_arbiter.py  --  deterministic safety + maneuver layer for the CARLA/Ollama driving agent.

WHY THIS EXISTS (from the analysis of your result CSVs):
  * LLM answers arrive every ~2.5-4 s (0.4 FPS) while the sim ticks at 10 Hz -> the intent is 25-40 ticks
    stale, i.e. 15-20 m of travel at 20 km/h.  Time-critical decisions cannot live in the LLM loop.
  * Small models answer KEEP_LANE ~90 % of the time even when the obstacle is < 3 m away, and they
    essentially never emit an overtake.  The LLM must be ADVISORY; physics-based rules own the safety envelope.
  * Lane change / return was decided from the forward corridor LiDAR only, so returning to the lane
    next to a vehicle you are still passing (blind spot) produced the collisions.  Side/rear gap is now mandatory.
  * 40+ intent spellings (A1_/A2_/B1_/OVERTAKE_*/LANE_CHANGE_*) were produced by several parallel state machines.
    There is now exactly ONE overtake state machine and one canonical vocabulary.

USAGE (call once per sim tick, 10 Hz):
    arb = ManeuverArbiter()
    cmd = arb.step(Observation(...), llm=LlmAdvice(intent=..., target_speed_kmh=..., age_s=...))
    # cmd.target_speed_kmh -> your longitudinal PID
    # cmd.lateral_offset_m / cmd.lane_target -> your lateral controller (-1 left, 0 keep, +1 right)
    # cmd.intent           -> log this in the CSV (canonical vocabulary)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math

KMH = 1 / 3.6


# --------------------------------------------------------------------------- inputs / outputs
@dataclass
class Observation:
    t: float                      # sim time [s]
    ego_speed_kmh: float
    corridor_lidar_m: float       # forward corridor distance (50.0 = clear)
    # Side / rear clearance in the ADJACENT lane. Distances to nearest object along the lane,
    # measured from ego bumper: front gap = object ahead of ego, rear gap = object behind ego.
    # Use 50.0 when clear, None when the lane does not exist / is oncoming-only / unknown (treated as BLOCKED).
    left_front_gap_m: float | None = None
    left_rear_gap_m: float | None = None
    right_front_gap_m: float | None = None
    right_rear_gap_m: float | None = None
    side_beside_left: bool = False    # something physically alongside (blind spot) on that side
    side_beside_right: bool = False
    oncoming_ttc_s: float = 99.0      # time to collision with oncoming traffic in the target lane
    red_light_stop_dist_m: float | None = None   # distance to stop line if signal says stop
    cruise_kmh: float = 30.0          # speed limit / desired speed


@dataclass
class LlmAdvice:
    intent: str = "KEEP_LANE"
    target_speed_kmh: float | None = None
    age_s: float = 99.0               # time since this answer was produced


@dataclass
class Command:
    target_speed_kmh: float
    lane_target: int                  # -1 left, 0 own lane, +1 right
    intent: str                       # canonical intent for logging
    reason: str = ""
    hard_brake: bool = False


class Phase(Enum):
    FOLLOW = "FOLLOW"
    PREPARE = "PREPARE"
    OUT = "LANE_CHANGE_OUT"
    PASS = "PASSING"
    RETURN = "LANE_CHANGE_RETURN"
    ABORT = "ABORT_RETURN"


# --------------------------------------------------------------------------- canonical vocabulary
def canonical_intent(raw: str) -> str:
    """Collapse the ~50 spellings found in the logs into 6 advisory intents."""
    r = (raw or "").upper()
    if "EMERGENCY" in r or r.startswith("AEB") or "TTC_BRAKE" in r:
        return "EMERGENCY_STOP"
    if r in ("STOP", "RED_LIGHT_STOP", "HOLD_BRAKE_LEFT", "HOLD_BRAKE_RIGHT"):
        return "STOP"
    if "SLOW" in r or "GLIDE" in r:
        return "SLOW_DOWN"
    if any(k in r for k in ("OVERTAKE", "LANE_CHANGE", "PASSING", "MANEUVER", "PREPARE_GAP", "MERGE")):
        return "OVERTAKE_L" if "LEFT" in r else "OVERTAKE_R"
    if "ACCEL" in r:
        return "ACCELERATE"
    return "KEEP_LANE"


# --------------------------------------------------------------------------- tunables
@dataclass
class Params:
    a_comf: float = 3.0           # m/s^2 comfortable decel
    a_max: float = 6.5            # m/s^2 emergency decel
    t_react: float = 0.4          # s actuation delay (not LLM latency!)
    standstill_gap: float = 3.0   # m to keep from a stopped obstacle
    llm_fresh_s: float = 1.0      # LLM advice older than this is NOT used for lateral decisions
    min_dwell_s: float = 1.5      # hysteresis between phases
    # overtake triggers
    blocked_speed_ratio: float = 0.6    # lead slower than 60 % of cruise => worth passing
    blocked_dwell_s: float = 2.0        # lead must be slow for this long (avoids flapping)
    deadlock_s: float = 6.0             # ego ~stopped this long behind blocker => try passing
    # lane-change gap requirements (time-gap based so they scale with speed)
    front_gap_min_m: float = 12.0
    rear_gap_time_s: float = 2.0
    rear_gap_min_m: float = 8.0
    oncoming_ttc_min_s: float = 8.0
    # return-to-lane gate: ego must be this far AHEAD of the passed vehicle
    return_clear_m: float = 8.0
    out_timeout_s: float = 6.0
    pass_timeout_s: float = 20.0
    offset_m: float = 3.5


# --------------------------------------------------------------------------- the arbiter
class ManeuverArbiter:
    def __init__(self, p: Params | None = None):
        self.p = p or Params()
        self.phase = Phase.FOLLOW
        self.side = 0                         # -1 left / +1 right while overtaking
        self._t_phase = 0.0
        self._hist: deque[tuple[float, float]] = deque(maxlen=12)   # (t, lidar) -> closing speed
        self._slow_since: float | None = None
        self._stopped_since: float | None = None
        self._pass_gap_ref: float | None = None

    # ------------------------------------------------------------------ helpers
    def _closing_speed(self, o: Observation) -> float:
        """Positive = gap shrinking [m/s]. Uses LiDAR history, ignores the 50 m 'clear' clamp."""
        self._hist.append((o.t, o.corridor_lidar_m))
        pts = [(t, d) for t, d in self._hist if d < 49.5]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.2:
            return 0.0
        return (pts[0][1] - pts[-1][1]) / (pts[-1][0] - pts[0][0])

    def _safe_speed(self, dist: float, lead_v: float) -> float:
        """Max speed [m/s] such that we can still stop behind the obstacle with comfortable decel."""
        p = self.p
        free = max(dist - p.standstill_gap, 0.0)
        # v^2 = lead_v^2 + 2*a*free, minus reaction distance
        v = math.sqrt(max(lead_v ** 2 + 2 * p.a_comf * free, 0.0))
        return max(v - p.a_comf * p.t_react, 0.0) if free > 0 else 0.0

    def _gap_ok(self, o: Observation, side: int, v: float) -> bool:
        p = self.p
        fg, rg, beside = ((o.left_front_gap_m, o.left_rear_gap_m, o.side_beside_left) if side < 0
                          else (o.right_front_gap_m, o.right_rear_gap_m, o.side_beside_right))
        if fg is None or rg is None or beside:     # unknown lane / blind-spot occupied => NOT ok
            return False
        need_rear = max(p.rear_gap_min_m, v * p.rear_gap_time_s)
        return fg >= p.front_gap_min_m and rg >= need_rear and o.oncoming_ttc_s >= p.oncoming_ttc_min_s

    def _set(self, ph: Phase, t: float):
        self.phase, self._t_phase = ph, t

    def _dwell(self, t: float) -> bool:
        return (t - self._t_phase) >= self.p.min_dwell_s

    # ------------------------------------------------------------------ main entry
    def step(self, o: Observation, llm: LlmAdvice | None = None) -> Command:
        p, llm = self.p, llm or LlmAdvice()
        v = o.ego_speed_kmh * KMH
        closing = self._closing_speed(o)
        lead_v = max(v - closing, 0.0) if o.corridor_lidar_m < 49.5 else o.cruise_kmh * KMH
        advice = canonical_intent(llm.intent)
        llm_fresh = llm.age_s <= p.llm_fresh_s

        # ---- 1. physics envelope: ALWAYS evaluated, LLM cannot override it --------------------
        d = o.corridor_lidar_m
        stop_dist = v * v / (2 * p.a_max) + v * p.t_react          # absolute minimum to stop
        v_cap = o.cruise_kmh * KMH
        hard = False
        if d < 49.5:
            v_cap = min(v_cap, self._safe_speed(d, lead_v if closing > 0.3 else 0.0))
            if d < stop_dist + p.standstill_gap and v > 0.5:
                hard = True
        if o.red_light_stop_dist_m is not None:
            v_cap = min(v_cap, self._safe_speed(o.red_light_stop_dist_m, 0.0))
        if advice in ("STOP", "EMERGENCY_STOP") and llm_fresh:      # LLM may only make us MORE cautious
            v_cap = min(v_cap, 0.0 if advice == "EMERGENCY_STOP" else v_cap * 0.5)
        if llm.target_speed_kmh is not None and llm_fresh:
            v_cap = min(v_cap, max(llm.target_speed_kmh, 0.0) * KMH)

        # ---- 2. blocked-ahead bookkeeping ----------------------------------------------------
        blocked = d < 30.0 and lead_v < p.blocked_speed_ratio * o.cruise_kmh * KMH
        self._slow_since = (self._slow_since or o.t) if blocked else None
        self._stopped_since = (self._stopped_since or o.t) if (v < 0.3 and d < 15.0) else None
        want_pass = (
            (self._slow_since is not None and o.t - self._slow_since >= p.blocked_dwell_s)
            or (self._stopped_since is not None and o.t - self._stopped_since >= p.deadlock_s)
            or (advice.startswith("OVERTAKE") and llm_fresh and blocked)
        )

        # ---- 3. single overtake state machine -----------------------------------------------
        lane, intent, reason = 0, "KEEP_LANE", ""
        if self.phase == Phase.FOLLOW:
            if want_pass and self._dwell(o.t):
                pref = (+1, -1) if (advice == "OVERTAKE_R" or advice != "OVERTAKE_L") else (-1, +1)
                for s in pref:
                    if self._gap_ok(o, s, max(v, 3.0)):
                        self.side, self._pass_gap_ref = s, d
                        self._set(Phase.OUT, o.t)
                        break
                else:
                    reason = "pass wanted, no safe gap -> hold behind"
            intent = "SLOW_DOWN" if v_cap < v - 0.5 else "KEEP_LANE"

        elif self.phase == Phase.OUT:
            lane, intent = self.side, "LANE_CHANGE_OUT"
            if not self._gap_ok(o, self.side, max(v, 3.0)) and not o.side_beside_left and not o.side_beside_right:
                pass                                         # gap closed but nobody beside us: keep going
            if o.oncoming_ttc_s < p.oncoming_ttc_min_s * 0.6 or (o.t - self._t_phase) > p.out_timeout_s:
                reason = "abort: oncoming / timeout"
                self._set(Phase.ABORT, o.t)
            elif self._dwell(o.t):
                self._set(Phase.PASS, o.t)
            v_cap = max(v_cap, min(o.cruise_kmh * KMH, v + 1.0)) if d > 15 else v_cap

        elif self.phase == Phase.PASS:
            lane, intent = self.side, "PASSING"
            s = self.side
            beside = o.side_beside_left if s < 0 else o.side_beside_right
            # rear gap in the ORIGINAL lane is not measured here; we use the passed vehicle leaving
            # the blind-spot sensor (beside=False) AND having ego-front-gap in our lane clear of it.
            # Make sure your side sensor reports the vehicle we are passing as 'beside' until it is
            # >= return_clear_m behind the ego rear bumper.
            if o.oncoming_ttc_s < p.oncoming_ttc_min_s * 0.6:
                self._set(Phase.ABORT, o.t); reason = "abort: oncoming"
            elif not beside and self._dwell(o.t):
                self._set(Phase.RETURN, o.t)
            elif (o.t - self._t_phase) > p.pass_timeout_s:
                self._set(Phase.ABORT, o.t); reason = "abort: pass timeout"
            # accelerate gently to get past, but never beyond the cap of whatever is ahead in THIS lane
            v_cap = min(o.cruise_kmh * KMH, max(v_cap, lead_v + 3.0)) if d > 20 else v_cap

        elif self.phase in (Phase.RETURN, Phase.ABORT):
            lane, intent = 0, "LANE_CHANGE_RETURN" if self.phase == Phase.RETURN else "ABORT_RETURN"
            origin_side = -self.side            # the original lane is on the opposite side of where we are
            ok = (not (o.side_beside_left if self.side < 0 else o.side_beside_right)) or self.phase == Phase.ABORT
            if ok and self._dwell(o.t):
                self._set(Phase.FOLLOW, o.t)
                self._slow_since = self._stopped_since = None
            else:
                lane = self.side                # hold the passing lane until the original lane is clear
                intent = "PASSING"
                reason = "return gated: blind spot occupied"

        # ---- 4. final command ----------------------------------------------------------------
        if hard:
            return Command(0.0, lane, "EMERGENCY_STOP", reason or f"d={d:.1f}m < stop_dist", hard_brake=True)
        if v_cap <= 0.2 and intent in ("KEEP_LANE", "SLOW_DOWN"):
            intent = "STOP"
        return Command(v_cap / KMH, lane, intent, reason)
