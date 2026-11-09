"""
ad_planner.py - Deterministic behaviour planner (the privileged "expert" of DriveGPT4-V2).

The base paper (Xu et al., CVPR 2025) trains the LLM agent with an expert that has
privileged information (ground-truth objects + hazard information) and lets that
expert take over whenever the agent's decision deviates too much (DAgger).  This
module is that expert.  It is pure Python (no CARLA import) so it can be tested
offline, and it outputs exactly the paper's high-level decision variables:

    target speed  ->  speed PID      (longitudinal)
    path offset   ->  target angle   (lateral, pure-pursuit, see ad_control.py)

Design (one pass per tick, no latches that can freeze the car):
  1. Objects are given in the ROUTE frame (s along the lane, d across it), so a
     curve or a junction never creates phantom obstacles.
  2. In-path objects  -> car-following law (safe-stop speed + time headway).
  3. Moving objects   -> constant-velocity prediction in the route frame; if an
     object will occupy our corridor while we arrive there, a virtual stop point
     is placed in front of the conflict (pedestrians, cut-ins, cross traffic).
  4. Traffic lights   -> stop point at the stop line (red, or yellow if stoppable).
  5. Blocked by a static / very slow object -> lateral bypass (nudge, lane change
     or opposite lane) only if the whole manoeuvre corridor is drivable and free
     of traffic for the manoeuvre time.  Otherwise wait (queue, closure, oncoming).
  6. Dead-lock recovery: short reverse when too close to steer around, patience
     timers for parked queues / stationary VRUs.
"""
import math
from typing import Dict, List, Optional, Tuple

from ad_types import Obj, PlanOut, Scene

try:
    import config as _cfg
except Exception:            # offline tests without config.py
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


KMH = 1.0 / 3.6

# ---------------------------------------------------------------- parameters
CRUISE_KMH = float(_c("AD_CRUISE_KMH", 30.0))
JUNCTION_KMH = float(_c("AD_JUNCTION_KMH", 22.0))
WEATHER_CAP_KMH = dict(_c("AD_WEATHER_CAP_KMH", {
    "clear": 30.0, "wet": 26.0, "rain": 24.0, "heavy_rain": 20.0, "fog": 18.0,
    "night": 26.0, "night_rain": 20.0, "glare": 22.0, "storm_night": 18.0,
    "dusk": 26.0,
}))
A_LAT_MAX = float(_c("AD_LATERAL_ACCEL", 2.2))         # m/s^2 for curve speed
B_COMF = float(_c("AD_COMFORT_DECEL", 2.8))            # m/s^2
B_MAX = float(_c("AD_MAX_DECEL", 7.5))                 # m/s^2
A_START = 1.5                                           # assumed accel for arrival times
HEADWAY_S = float(_c("AD_HEADWAY_S", 1.2))
STANDSTILL = dict(_c("AD_STANDSTILL_GAP_M", {
    "vehicle": 4.0, "bike": 4.0, "walker": 4.5, "animal": 4.5,
    "prop": 3.0, "unknown": 3.0, "line": 1.0, "conflict": 2.5,
}))
CORRIDOR_MARGIN = float(_c("AD_CORRIDOR_MARGIN_M", 0.30))
BYPASS_MARGIN = dict(_c("AD_BYPASS_MARGIN_M", {
    "vehicle": 0.8, "bike": 1.1, "walker": 1.2, "animal": 1.2,
    "prop": 0.5, "unknown": 0.7,
}))
EDGE_MARGIN = 0.10                     # how close the ego body may get to the drivable edge
STATIC_CONFIRM_S = float(_c("AD_STATIC_CONFIRM_S", 1.2))        # parked / never seen moving
TRAFFIC_STOP_CONFIRM_S = float(_c("AD_TRAFFIC_STOP_CONFIRM_S", 4.0))  # was moving, then stopped
STATIC_STANDSTILL_M = float(_c("AD_STATIC_STANDSTILL_M", 10.0))  # wait far enough back to steer around
SLOW_LEAD_CONFIRM_S = 3.0
VRU_PATIENCE_S = float(_c("AD_VRU_PATIENCE_S", 10.0))
QUEUE_PATIENCE_S = float(_c("AD_QUEUE_PATIENCE_S", 30.0))
STUCK_REVERSE_S = float(_c("AD_STUCK_REVERSE_S", 4.0))
REVERSE_DIST_M = 3.0
PRED_HORIZON_S = 5.0
LIGHT_LOOKAHEAD_M = 80.0
BYPASS_MAX_S = 30.0


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def time_to_cover(x, v0, a=A_START, vmax=CRUISE_KMH * KMH):
    """Time to travel x metres from speed v0 with acceleration a, capped at vmax."""
    if x <= 0.0:
        return 0.0
    v0 = max(0.0, v0)
    vmax = max(vmax, v0, 0.5)
    t1 = (vmax - v0) / a
    x1 = v0 * t1 + 0.5 * a * t1 * t1
    if x <= x1:
        return (-v0 + math.sqrt(v0 * v0 + 2.0 * a * x)) / a
    return t1 + (x - x1) / vmax


class OffsetProfile:
    """Lateral offset d(s) along the route: smooth transition d0 -> d1 between s0 and s1."""

    def __init__(self):
        self.s0 = self.s1 = 0.0
        self.d0 = self.d1 = 0.0

    def value(self, s):
        if s <= self.s0:
            return self.d0
        if s >= self.s1 or self.s1 - self.s0 < 1e-3:
            return self.d1
        return self.d0 + (self.d1 - self.d0) * _smoothstep((s - self.s0) / (self.s1 - self.s0))

    @property
    def target(self):
        return self.d1

    def reset(self, s, d_now=0.0, length=8.0):
        self.s0, self.d0 = s, d_now
        self.s1, self.d1 = s + length, 0.0

    def set_target(self, s_now, d_target, length):
        if abs(d_target - self.d1) < 0.05:
            return
        cur = self.value(s_now)
        self.s0, self.d0 = s_now, cur
        self.s1, self.d1 = s_now + max(3.0, length), d_target


def transition_length(delta_d, v):
    """Distance used to shift laterally by delta_d at speed v (m/s)."""
    return max(6.0, min(28.0, 2.6 * abs(delta_d) + 1.3 * v))


class BehaviorPlanner:
    def __init__(self):
        self.profile = OffsetProfile()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, ego_s=0.0, ego_d=0.0):
        self.profile.reset(ego_s, ego_d)
        self.static_since: Dict[int, float] = {}
        self.slow_since: Dict[int, float] = {}
        self.ever_moved: Dict[int, bool] = {}
        self.mode = "LANE"                 # LANE | BYPASS | EVADE | REVERSE
        self.bypass_ids: List[int] = []
        self.bypass_side = ""
        self.bypass_end_s = 0.0
        self.bypass_t0 = 0.0
        self.bypass_uses_opp = False
        self.evade_id: Optional[int] = None
        self.stuck_t = 0.0
        self.queue_t = 0.0
        self.reverse_start_s = 0.0
        self.reverse_t0 = 0.0
        self.last_block_reason = ""
        self._t = 0.0
        self._too_close = False
        self.squeeze = False
        self.tight = False
        self.crawl = False
        self.bypass_rank = 0
        self.reverse_need = 0.0
        self.reverse_dist = REVERSE_DIST_M

    # -------------------------------------------------------------- helpers
    def _overlaps(self, o: Obj, off: float, hw_e: float, margin: float) -> bool:
        return (o.d_max > off - hw_e - margin) and (o.d_min < off + hw_e + margin)

    def _cruise_cap(self, sc: Scene) -> float:
        cap = WEATHER_CAP_KMH.get(sc.weather, CRUISE_KMH)
        return min(CRUISE_KMH, cap) * KMH

    def _curve_cap(self, sc: Scene) -> float:
        """Max speed now so that every curve / junction ahead can be taken comfortably."""
        geo = sc.geo
        b = B_COMF * 0.8 * sc.friction
        best = 99.0
        s = sc.ego_s
        while s < min(geo.s_end, sc.ego_s + 60.0):
            _, _, _, _, junc, kappa = geo.at(s)
            x = max(0.0, s - sc.ego_s - sc.ego_half_len)
            v_c = math.sqrt(A_LAT_MAX / max(kappa, 1e-4))
            if junc:
                v_c = min(v_c, JUNCTION_KMH * KMH)
            best = min(best, math.sqrt(v_c * v_c + 2.0 * b * x))
            s += geo.ds
        return best

    def _follow(self, v, gap, v_lead, s0, sc: Scene) -> Tuple[float, Optional[float]]:
        """Target speed and (optional) required acceleration to follow / stop behind something."""
        b = B_COMF * sc.friction
        T = HEADWAY_S * (1.0 / max(0.55, sc.friction))
        v_l = max(0.0, v_lead)
        free = gap - s0
        v_safe = math.sqrt(max(0.0, v_l * v_l + 2.0 * b * max(0.0, free)))
        g_star = s0 + v * T
        v_head = v_l + (gap - g_star) / 1.6
        v_t = max(0.0, min(v_safe, v_head if gap < g_star + 25.0 else 99.0))
        a_req = None
        if v > v_l + 0.1:
            room = max(0.15, gap - 0.5 * s0) if gap > 0.5 * s0 else 0.15
            need = (v * v - v_l * v_l) / (2.0 * room)
            if need > 0.6 * b:
                a_req = -min(B_MAX, need)
        if gap <= 0.6:
            a_req = -B_MAX if v > 0.2 else a_req
            v_t = 0.0
        return v_t, a_req

    # -------------------------------------------------------- prediction
    def _conflict(self, o: Obj, sc: Scene) -> Optional[Tuple[float, float, float]]:
        """(s_conflict, t_in, t_out) if a moving object is predicted to enter our corridor."""
        moving_lat = abs(o.vd) > (0.25 if o.is_vru else 0.4)
        if not moving_lat and abs(o.vs) < 0.3:
            return None
        hw_e = sc.ego_half_wid
        margin = 0.55 if o.is_vru else 0.35
        lat_cap = PRED_HORIZON_S if o.is_vru else 3.0
        t_in = t_out = None
        s_c = None
        t = 0.0
        while t <= PRED_HORIZON_S + 1e-6:
            s_t = o.s + o.vs * t
            d_t = o.d + o.vd * min(t, lat_cap)
            if s_t - o.half_len < sc.ego_s + sc.ego_half_len - 0.5:
                t += 0.25
                continue
            off = self.profile.value(s_t)
            if (d_t + o.half_wid > off - hw_e - margin) and (d_t - o.half_wid < off + hw_e + margin):
                if t_in is None:
                    t_in, s_c = t, s_t
                t_out = t
            elif t_in is not None:
                break
            t += 0.25
        if t_in is None:
            return None
        if t_out is not None and t_out >= PRED_HORIZON_S - 1e-6:
            t_out = PRED_HORIZON_S + 2.0
        return s_c, t_in, t_out

    # ----------------------------------------------------------- bypass
    def _bounds(self, sc: Scene, s: float, allow_opp: bool):
        """Drivable lateral limits: our lane + all same-direction lanes (+ first opposite lane)."""
        hw, ls, lo, rs, junc, _ = sc.geo.at(s)
        left = hw + ls + (lo if allow_opp else 0.0)
        right = hw + rs
        return -left, right, hw, ls, junc

    def _bypass_check(self, sc: Scene, cluster: List[Obj], d_b: float,
                      allow_opp: bool, s_end: float) -> Tuple[bool, str, bool]:
        """Is the corridor at offset d_b drivable and free for the whole manoeuvre?
        Returns (ok, reason, uses_opposite_lane).  Reasons starting with 'traffic:' are
        temporary (moving traffic) - the caller then waits instead of using the opposite lane."""
        hw_e, hl_e = sc.ego_half_wid, sc.ego_half_len
        uses_opp = False
        s = sc.ego_s
        while s <= s_end:
            lo_lim, hi_lim, hw, ls, junc = self._bounds(sc, s, allow_opp)
            if junc:
                return False, "junction in manoeuvre zone", uses_opp
            if d_b - hw_e < lo_lim + EDGE_MARGIN or d_b + hw_e > hi_lim - EDGE_MARGIN:
                return False, "no drivable space", uses_opp
            if d_b - hw_e < -(hw + ls) - 0.05:
                uses_opp = True
            s += 2.0
        ids = {o.id for o in cluster}
        v_pass = max(sc.ego_v, 4.5)
        t_man = (s_end - sc.ego_s) / v_pass + (4.0 if uses_opp else 2.0)
        for o in sc.objs:
            if o.id in ids:
                continue
            lat_m = max(0.3, BYPASS_MARGIN.get(o.kind, 0.6) - 0.1)
            lat_hit = abs(o.d - d_b) < o.half_wid + hw_e + lat_m
            if o.is_vru and abs(o.vd) > 0.25 and not lat_hit:
                d_t = o.d + o.vd * min(t_man, PRED_HORIZON_S)
                lat_hit = (min(o.d, d_t) - o.half_wid < d_b + hw_e + 0.6 and
                           max(o.d, d_t) + o.half_wid > d_b - hw_e - 0.6)
                if lat_hit and not (sc.ego_s - 5.0 < o.s < s_end + 5.0):
                    lat_hit = False
                if lat_hit:
                    return False, f"traffic: {o.kind} crossing the target lane", uses_opp
            if not lat_hit:
                continue
            rel = o.s - sc.ego_s
            if o.vs < -0.5:                                   # oncoming in that corridor
                margin = 30.0 if uses_opp else 10.0
                if rel + o.half_len > -hl_e and rel + o.vs * t_man - o.half_len < (s_end - sc.ego_s) + margin:
                    return False, f"traffic: oncoming {o.kind} {rel:.0f} m", uses_opp
            elif abs(o.vs) < 0.5:                              # static
                if o.s_max > sc.ego_s - hl_e - 2.0 and o.s_min < s_end + 4.0:
                    if o.is_vru:                               # people / animals move on: wait
                        return False, f"traffic: {o.kind} in target lane", uses_opp
                    if o.kind in ("vehicle", "bike") and self.ever_moved.get(o.id):
                        return False, f"traffic: stopped {o.kind} in target lane", uses_opp
                    return False, f"{o.kind} in bypass corridor", uses_opp
            else:                                              # same direction, moving
                if rel > 0:
                    if rel - o.half_len - hl_e < max(8.0, 1.5 * v_pass) and o.vs < v_pass + 0.5:
                        return False, f"traffic: slow {o.kind} ahead in target lane", uses_opp
                else:
                    gap_back = -rel - o.half_len - hl_e
                    closing = o.vs - sc.ego_v
                    if gap_back < 6.0 or (closing > 0.3 and gap_back / closing < 4.0):
                        return False, f"traffic: {o.kind} approaching from behind", uses_opp
        return True, "clear", uses_opp

    def _candidates(self, sc: Scene, cluster: List[Obj], s_ref: float, allow_opp: bool, relax: float = 0.0):
        """Lateral targets ranked by traffic rules:
             0 = stay inside our lane (nudge / squeeze),
             1 = centre of a same-direction lane,  2 = straddling same-direction lanes,
             3 = opposite lane (last resort)."""
        hw_e = sc.ego_half_wid
        margin = max(BYPASS_MARGIN.get(o.kind, 0.6) for o in cluster) - relax
        need_l = min(o.d_min for o in cluster) - hw_e - margin      # centre must be <= need_l
        need_r = max(o.d_max for o in cluster) + hw_e + margin      # centre must be >= need_r
        hw, ls, lo, rs, _, _ = sc.geo.at(s_ref)
        same, opp = sc.geo.lanes(s_ref)
        c = []
        off_now = self.profile.value(s_ref)
        lefts = [o.d_max for o in cluster if o.d < off_now]
        rights = [o.d_min for o in cluster if o.d >= off_now]
        if lefts and rights:
            L, R = max(lefts), min(rights)
            if R - L >= 2.0 * hw_e + 0.24:
                c.append((0, "CENTER", 0.5 * (L + R)))
        own = hw - hw_e - 0.05
        if need_l >= -own:
            c.append((0, "LEFT", need_l))
        if need_r <= own:
            c.append((0, "RIGHT", need_r))
        for lc in same:
            if lc < 0 and lc <= need_l:
                c.append((1, "LEFT", lc))
            if lc > 0 and lc >= need_r:
                c.append((1, "RIGHT", lc))
        if ls > 0 and -(hw + ls) + hw_e <= need_l < -own:
            c.append((2, "LEFT", need_l))
        if rs > 0 and own < need_r <= hw + rs - hw_e:
            c.append((2, "RIGHT", need_r))
        if allow_opp and opp is not None:
            if need_l < -(hw + ls) + hw_e:              # really has to leave our side of the road
                c.append((3, "LEFT", max(opp, need_l) if need_l > opp else need_l))
        seen, out = set(), []
        for rank, side, d in sorted(c, key=lambda x: (x[0], abs(x[2] - sc.ego_d))):
            key = round(d, 1)
            if key not in seen:
                seen.add(key)
                out.append((rank, side, d))
        return out

    def _plan_bypass(self, sc: Scene, lead: Obj, gap: float, allow_opp: bool, relax: float = 0.0):
        """Find the best legal lateral offset that clears `lead` (and objects right behind it).
        Returns (plan | None, reason, too_close)."""
        hw_e = sc.ego_half_wid
        off_now = self.profile.value(lead.s)
        # chain clustering: a cone taper / parked row is ONE obstacle (each member <= 14 m apart)
        cluster = [lead]
        cand = [o for o in sc.objs if o is not lead and abs(o.vs) < 1.0 and o.s_max > sc.ego_s
                and not o.is_vru and self._overlaps(o, off_now, hw_e, 1.2)]
        grew = True
        while grew:
            grew = False
            end = max(c.s_max for c in cluster)
            for o in list(cand):
                if lead.s_min - 1.0 <= o.s_min <= end + 14.0:
                    cluster.append(o)
                    cand.remove(o)
                    grew = True
        s_end_cluster = max(o.s_max for o in cluster)
        reasons, too_close, traffic_wait, own_too_close = [], False, False, False
        self.reverse_need = 0.0
        # Pass 1: our own lanes with normal clearance.  Pass 2: our own lanes with a tighter
        # (still safe) clearance at reduced speed.  Pass 3: only then the opposite lane.
        own1 = [c + (0.0,) for c in self._candidates(sc, cluster, lead.s, False) if c[0] < 3]
        own2 = [c + (0.3,) for c in self._candidates(sc, cluster, lead.s, False, relax=0.3)
                if c[0] < 3 and c[0] > 0]
        opp = [c + (0.0,) for c in self._candidates(sc, cluster, lead.s, allow_opp) if c[0] == 3]
        self.rejected_own = []
        for rank, side, d_b, rlx in own1 + own2 + opp:
            if rank == 3 and traffic_wait:
                reasons.append("opposite lane not used: own lanes only busy with traffic")
                break
            if rank == 3 and own_too_close:
                reasons.append("opposite lane not used: own lane reachable after backing up")
                break
            dd = abs(d_b - sc.ego_d)
            s_end = s_end_cluster + sc.ego_half_len + 3.0 + transition_length(abs(d_b), sc.ego_v)
            ok, why, uses_opp = self._bypass_check(sc, cluster, d_b, allow_opp and rank == 3, s_end)
            if not ok:
                if why.startswith("traffic:") and rank < 3:
                    traffic_wait = True
                reasons.append(f"{side}{'(opp)' if rank == 3 else ''}: {why}")
                if rank < 3:
                    self.rejected_own.append(f"{side.lower()} d={d_b:+.1f}: {why}")
                continue
            need_crawl = max(3.5, 1.3 * dd + 3.0)       # full-lock manoeuvre at walking pace
            need_normal = 1.6 * dd + 3.0
            if gap < need_crawl:
                if rank < 3:
                    own_too_close = True           # own lane possible after backing up a bit
                too_close = True
                self.reverse_need = max(self.reverse_need, min(4.0, need_crawl - gap + 0.6))
                reasons.append(f"{side}: too close to steer ({gap:.1f} m)")
                continue
            crawl = gap < need_normal
            # the offset must be reached when the ego CENTRE is ~2 m before the obstacle (tracking lag)
            L_in = max(3.0, gap - 1.0) if crawl else min(transition_length(dd, sc.ego_v),
                                                         max(4.0, gap - 2.0 - 0.15 * sc.ego_v))
            self.squeeze = side == "CENTER"
            self.crawl = crawl
            self.tight = rlx > 0.0
            return dict(side=side, d=d_b, L=L_in, cluster=cluster, rank=rank,
                        end_s=s_end_cluster + sc.ego_half_len + 3.0, uses_opp=uses_opp), "", False
        return None, "; ".join(reasons) or "no legal lateral space", too_close

    def _is_closure(self, sc: Scene, lead: Obj) -> bool:
        """Our lane completely barricaded by props (road works)."""
        hw, *_ = sc.geo.at(lead.s)
        props = [o for o in sc.objs if o.kind == "prop" and abs(o.s - lead.s) < 6.0]
        if len(props) < 2:
            return False
        lo = min(o.d_min for o in props)
        hi = max(o.d_max for o in props)
        return lo <= -hw + 0.6 and hi >= hw - 0.6

    def _queue_reason(self, sc: Scene, lead: Obj) -> str:
        if lead.kind not in ("vehicle", "bike"):
            return ""
        for o in sc.objs:
            if o.id == lead.id or o.kind not in ("vehicle", "bike"):
                continue
            if abs(o.d - lead.d) < 1.6 and lead.s_max - 1.0 < o.s_min < lead.s_max + 12.0 and o.speed < 1.0:
                return "queue of stopped vehicles"
        if sc.light is not None and sc.light.state in ("RED", "YELLOW") \
                and lead.s < sc.light.s_line < lead.s_max + 35.0:
            return f"vehicle waiting at {sc.light.state.lower()} light"
        s = lead.s_max
        while s < lead.s_max + 12.0:
            if sc.geo.at(s)[4]:
                return "vehicle waiting at junction"
            s += 2.0
        return ""

    # ------------------------------------------------------------------ main
    def step(self, sc: Scene) -> PlanOut:
        self._t = sc.t
        v = sc.ego_v
        hw_e, hl_e = sc.ego_half_wid, sc.ego_half_len
        front_s = sc.ego_s + hl_e
        dbg = {}

        # ---- static / slow timers
        seen = set()
        slow_thr = 0.55 * self._cruise_cap(sc)
        for o in sc.objs:
            seen.add(o.id)
            if o.speed > 1.0:
                self.ever_moved[o.id] = True
            if o.speed < 0.3:
                self.static_since.setdefault(o.id, sc.t)
            else:
                self.static_since.pop(o.id, None)
            if o.speed < slow_thr:
                self.slow_since.setdefault(o.id, sc.t)
            else:
                self.slow_since.pop(o.id, None)
        for k in list(self.static_since):
            if k not in seen:
                self.static_since.pop(k, None)
        for k in list(self.slow_since):
            if k not in seen:
                self.slow_since.pop(k, None)

        # ---- add confirmed LiDAR / radar return that no object explains
        objs = list(sc.objs)
        if sc.sensor_gap is not None and sc.sensor_gap < 30.0:
            s_hit = front_s + sc.sensor_gap
            off = self.profile.value(s_hit)
            explained = any(abs(o.s_min - s_hit) < 3.0 and self._overlaps(o, off, hw_e, 0.6) for o in objs)
            if not explained:
                objs.append(Obj(id=-1, kind="unknown", s=s_hit + 0.3, d=off,
                                half_len=0.3, half_wid=0.5, label="lidar"))
                dbg["sensor_only"] = round(sc.sensor_gap, 1)
        sc = Scene(**{**sc.__dict__, "objs": objs})

        # ---- REVERSE recovery (short, straight back)
        if self.mode == "REVERSE":
            travelled = self.reverse_start_s - sc.ego_s
            rear_ok, _ = self._rear_clear(sc, max(0.5, self.reverse_dist - travelled))
            if travelled >= self.reverse_dist or not rear_ok or sc.t - self.reverse_t0 > 8.0:
                self.mode = "LANE"
                self.stuck_t = sc.t
                self.profile.reset(sc.ego_s, sc.ego_d, 6.0)   # re-plan from where the car really is
            else:
                return PlanOut(target_speed=1.2, accel_req=None, offset_target=self.profile.target,
                               intent="REVERSE", reason=f"recovery reverse {travelled:.1f}/{self.reverse_dist:.1f} m",
                               reverse=True, safe_max_speed=1.2)

        caps = []           # (speed, reason, intent, hold, a_req, gap, lead_v, kind)
        cruise = self._cruise_cap(sc)
        caps.append((cruise, f"cruise ({sc.weather})", "KEEP_LANE", "", None, 999.0, 0.0, ""))
        curve = self._curve_cap(sc)
        if curve < cruise - 0.3:
            caps.append((curve, "curve / junction speed", "SLOW_DOWN", "", None, 999.0, 0.0, ""))

        # ---- in-path objects
        in_path = []
        for o in objs:
            if o.s_max < front_s - 2.0 * hl_e - 0.3:          # entirely behind our front-rear span
                continue
            if o.s_min > sc.ego_s + 120.0:
                continue
            off = self.profile.value(max(o.s_min, front_s))
            margin = CORRIDOR_MARGIN
            if self.mode == "BYPASS" and self.squeeze and o.id in self.bypass_ids:
                margin = 0.03
            if self.mode == "EVADE" and o.id == self.evade_id:
                off = self.profile.target          # judge the evasion by where we are going
            hit = self._overlaps(o, off, hw_e, margin)
            if not hit and o.s_min - front_s < 2.5 and self._overlaps(o, sc.ego_d, hw_e, 0.1):
                # the car itself has not moved over yet: creep while steering, stop only at 0.8 m
                gap_c = o.s_min - front_s
                if gap_c < 0.8:
                    hit = True
                else:
                    caps.append((2.0 * KMH, "creeping while steering past obstacle", "SLOW_DOWN",
                                 "", None, gap_c, 0.0, o.kind))
            if not hit:
                continue
            gap = o.s_min - front_s
            if gap < -2.0 * hl_e:
                continue
            in_path.append((max(gap, 0.0) if gap > -0.5 else 0.0, o))
        in_path.sort(key=lambda g: g[0])

        lead = None
        for gap, o in in_path:
            v_l = o.vs
            s0 = STANDSTILL.get(o.kind, 4.0)
            if not o.is_vru and self._is_static_obstacle(o, sc.t):
                s0 = max(s0, STATIC_STANDSTILL_M)
            oncoming = v_l < -0.5
            if oncoming:
                s0 += 3.0
                v_l = 0.0
            v_t, a_req = self._follow(v, gap, v_l, s0, sc)
            if oncoming:
                # vehicle coming at us in our lane: brake NOW - every km/h less reduces the impact
                ttc = gap / max(0.5, v - o.vs)
                v_t = 0.0
                a_req = -B_MAX if ttc < 3.0 else min(a_req or 0.0, -1.5 * B_COMF)
            hold = "lead stopped" if v_t < 0.3 else ""
            caps.append((v_t, f"following {o.kind} gap {gap:.1f} m", "FOLLOW" if v_t > 0.5 else "STOP",
                         hold, a_req, gap, o.vs, o.kind))
            if lead is None:
                lead = (gap, o)

        # ---- predicted conflicts (crossing pedestrians, cut-ins, cross traffic)
        for o in objs:
            if any(o is p for _, p in in_path):
                continue
            if o.s_max < front_s and abs(o.d - sc.ego_d) < o.half_wid + hw_e + 0.6:
                continue                      # follows us in our lane: not a conflict ahead
            c = self._conflict(o, sc)
            if c is None:
                continue
            s_c, t_in, t_out = c
            x = s_c - o.half_len - front_s
            if x < 0.0:
                continue
            t_e_in = time_to_cover(x, v, vmax=cruise)
            t_e_out = time_to_cover(x + 2.0 * hl_e + 2.0 * o.half_len, v, vmax=cruise)
            buf = 1.2 if o.is_vru else 0.8
            if t_e_in < t_out + buf and t_e_out > t_in - 1.0:
                s0 = STANDSTILL["conflict"] + (1.0 if o.is_vru else 0.0)
                v_t, a_req = self._follow(v, x, 0.0, s0, sc)
                caps.append((v_t, f"yield to {o.kind} (enters path in {t_in:.1f}s)", "YIELD",
                             f"yield {o.kind}", a_req, x, 0.0, o.kind))

        # ---- traffic light
        if sc.light is not None and not sc.in_junction:
            gap_l = sc.light.s_line - front_s
            if -0.5 < gap_l < LIGHT_LOOKAHEAD_M:
                must_stop = sc.light.state == "RED"
                if sc.light.state == "YELLOW":
                    must_stop = gap_l > (v * v) / (2.0 * 4.0) + 1.0
                if must_stop:
                    lc_note = ""
                    if self.mode in ("BYPASS", "EVADE") and abs(sc.ego_d - self.profile.target) > 0.35:
                        lc_note = self._lane_change_at_red(sc, gap_l)
                    v_t, a_req = self._follow(v, max(0.0, gap_l), 0.0, STANDSTILL["line"], sc)
                    if lc_note.startswith("finish"):
                        # keep rolling slowly until the car is fully in its new lane, then stop at the line
                        v_t = max(v_t, min(6.0 * KMH, 0.7 * math.sqrt(max(0.0, 2.0 * B_COMF * (gap_l - 1.0)))))
                    caps.append((v_t, f"{sc.light.state.lower()} light, line {gap_l:.1f} m"
                                 f"{' | ' + lc_note if lc_note else ''}", "STOP",
                                 f"{sc.light.state.lower()} light", a_req, gap_l, 0.0, "line"))

        if self.mode == "BYPASS" and sc.ego_s - hl_e < self.bypass_end_s:
            shifted = abs(self.profile.value(sc.ego_s) - self.profile.d0) / max(0.3, abs(self.profile.d1 - self.profile.d0))
            if self.squeeze:
                caps.append((8.0 * KMH, "squeezing through narrow gap", "SLOW_DOWN", "", None, 999.0, 0.0, ""))
            elif self.crawl and shifted < 0.85:
                caps.append((6.0 * KMH, "steering out slowly (short gap)", "SLOW_DOWN", "", None, 999.0, 0.0, ""))
            tg = [o for o in objs if o.id in self.bypass_ids]
            if tg and not self.squeeze:
                x_to = min(o.s_min for o in tg) - front_s
                lat_err = abs(sc.ego_d - self.profile.target)
                if 0.0 < x_to < 12.0 and lat_err > 0.5:
                    v_lat = 6.0 if x_to < 5.0 else 10.0
                    caps.append((v_lat * KMH, "waiting for the car to finish moving over", "SLOW_DOWN",
                                 "", None, 999.0, 0.0, ""))
            elif getattr(self, "tight", False):
                caps.append((12.0 * KMH, "tight clearance in own lane", "SLOW_DOWN", "", None, 999.0, 0.0, ""))
            elif self.bypass_rank in (0, 3):
                caps.append((20.0 * KMH, "passing obstacle", "SLOW_DOWN", "", None, 999.0, 0.0, ""))
        if self.mode == "EVADE":
            o_ev = next((x for x in objs if x.id == self.evade_id), None)
            if o_ev is not None and o_ev.s_max > sc.ego_s - hl_e:
                gap_ev = max(0.0, o_ev.s_min - front_s)
                ttc = gap_ev / max(0.5, v - o_ev.vs)
                still_in_line = self._overlaps(o_ev, sc.ego_d, hw_e, 0.3)
                # lateral distance still to cover and how long that takes at ~12 km/h
                if self.bypass_side == "RIGHT":
                    need = (o_ev.d_max + hw_e + 0.2) - sc.ego_d
                else:
                    need = sc.ego_d - (o_ev.d_min - hw_e - 0.2)
                v_ev = 12.0 * KMH
                t_clear = max(0.0, need) / (0.4 * max(v, v_ev))
                if still_in_line and t_clear > 0.9 * ttc:
                    # cannot get out of its way in time: full brake to minimise the impact speed
                    caps.append((0.0, f"brake to reduce impact (TTC {ttc:.1f}s)", "EMERGENCY_STOP",
                                 "", -B_MAX, gap_ev, o_ev.vs, o_ev.kind))
                elif still_in_line:
                    # brake hard down to ~12 km/h (impact energy -70 %) but keep rolling so the car
                    # can actually steer out of the oncoming vehicle's path
                    caps.append((v_ev, f"braking + steering away (clear in {t_clear:.1f}s, TTC {ttc:.1f}s)",
                                 "EVADE", "", -B_COMF * 1.5 if v > v_ev + 1.0 else None,
                                 gap_ev, o_ev.vs, o_ev.kind))
                else:
                    caps.append((12.0 * KMH, "out of its path, passing slowly", "SLOW_DOWN", "", None,
                                 999.0, 0.0, ""))

        # ---- decide lateral behaviour
        lat_reason = ""
        allow_bypass = not sc.in_junction
        if self.mode in ("BYPASS", "EVADE"):
            lat_reason = self._update_bypass(sc)
        if self.mode == "LANE" and lead is not None and allow_bypass:
            lat_reason = self._consider_bypass(sc, lead[1], lead[0])

        # ---- pick the most restrictive speed
        best = min(caps, key=lambda c: c[0])
        a_req = None
        for c in caps:
            if c[4] is not None and (a_req is None or c[4] < a_req):
                a_req = c[4]
        target = best[0]
        intent, reason, hold = best[2], best[1], best[3]

        # lateral intent label
        if self.mode in ("BYPASS", "EVADE") and target > 0.3:
            pre = ("EVADE" if self.mode == "EVADE" else "LANE_CHANGE" if self.bypass_rank in (1, 2)
                   else "OVERTAKE_OPP" if self.bypass_rank == 3 else "BYPASS")
            intent = f"{pre}_{self.bypass_side}"
        elif self.mode == "LANE" and abs(self.profile.value(sc.ego_s)) > 0.3 and target > 0.3:
            intent = "RETURN_LANE"
        if lat_reason:
            reason = f"{reason} | {lat_reason}"
        if hold == "lead stopped":
            if lat_reason.startswith("hold:"):
                hold = lat_reason[5:].strip()
            elif (sc.light is not None and sc.light.state in ("RED", "YELLOW")
                  and sc.light.s_line - front_s < best[5] + 45.0):
                hold = f"queued at {sc.light.state.lower()} light"
        if a_req is not None and a_req <= -0.9 * B_MAX and v > 1.0:
            intent = "EMERGENCY_STOP"

        # ---- stuck detection -> recovery
        self._update_stuck(sc, target, lead, hold)

        gap_l, lv, lk = best[5], best[6], best[7]
        if lead is not None and gap_l >= 999.0:
            gap_l, lv, lk = lead[0], lead[1].vs, lead[1].kind
        dbg["mode"] = self.mode
        return PlanOut(target_speed=max(0.0, target), accel_req=a_req,
                       offset_target=self.profile.target, intent=intent, reason=reason,
                       hold=hold, lead_gap=gap_l, lead_speed=lv, lead_kind=lk,
                       safe_max_speed=max(0.0, target),
                       bypass_side=self.bypass_side if self.mode in ("BYPASS", "EVADE") else "",
                       debug=dbg)

    # ------------------------------------------------------- lateral logic
    def _lane_change_at_red(self, sc: Scene, gap_l: float) -> str:
        """Light turned red while the car is between lanes: finish the lane change BEFORE the stop
        line (compress the manoeuvre if needed) or, if it has barely started, cancel it."""
        hl_e = sc.ego_half_len
        line_s = sc.light.s_line
        span = abs(self.profile.d1 - self.profile.d0)
        done = abs(sc.ego_d - self.profile.d0) / max(0.3, span)
        room = line_s - 3.0 - hl_e - sc.ego_s            # centre distance available before the line
        if self.profile.s1 <= line_s - hl_e - 2.0:
            return "finish lane change, then stop at line"
        if room >= 4.0:
            cur, d1 = self.profile.value(sc.ego_s), self.profile.d1
            self.profile.s0, self.profile.d0 = sc.ego_s, cur                   # re-anchor shorter
            self.profile.s1, self.profile.d1 = sc.ego_s + max(3.0, room), d1
            return "finish lane change (shortened), then stop at line"
        if done < 0.4 and self.mode == "BYPASS":
            self.mode = "LANE"
            self.bypass_ids = []
            self.profile.set_target(sc.ego_s, 0.0, max(4.0, room))
            return "lane change cancelled for red light"
        return "finish lane change, then stop at line"

    def _is_static_obstacle(self, o: Obj, t: float) -> bool:
        if abs(o.vs) >= 0.5:
            return False
        if o.kind in ("prop", "unknown"):
            return True
        static_for = t - self.static_since.get(o.id, t)
        need = TRAFFIC_STOP_CONFIRM_S if self.ever_moved.get(o.id) else STATIC_CONFIRM_S
        return static_for >= need

    def _consider_bypass(self, sc: Scene, o: Obj, gap: float) -> str:
        cruise = self._cruise_cap(sc)
        static_for = sc.t - self.static_since.get(o.id, sc.t)
        slow_for = sc.t - self.slow_since.get(o.id, sc.t)
        if gap > 40.0:
            return ""
        if o.vs < -1.0:
            if gap < 35.0:
                return self._start_evade(sc, o, gap)
            return ""
        is_static = self._is_static_obstacle(o, sc.t)
        is_slow = (not is_static and o.vs < max(2.5, 0.45 * cruise) and slow_for >= SLOW_LEAD_CONFIRM_S
                   and o.kind in ("vehicle", "bike"))
        if not (is_static or is_slow):
            return ""
        if is_slow:
            j = next((sc.geo.s0 + k * sc.geo.ds for k, jn in enumerate(sc.geo.junction)
                      if jn and sc.geo.s0 + k * sc.geo.ds > sc.ego_s), None)
            if j is not None and j - sc.ego_s < 60.0:
                return "no overtaking: junction ahead"
            if sc.light is not None and sc.light.state != "GREEN" and sc.light.s_line - sc.ego_s < 80.0:
                return "no overtaking: traffic light ahead"
        if o.is_vru and o.kind != "bike" and static_for < VRU_PATIENCE_S:
            return f"waiting for {o.kind} ({static_for:.0f}s)"
        q = self._queue_reason(sc, o)
        if q:
            if self.queue_t <= 0.0:
                self.queue_t = sc.t
            if sc.t - self.queue_t < QUEUE_PATIENCE_S or "light" in q or "junction" in q:
                if "light" in q or "junction" in q:
                    return f"hold: {q}"
                # a parked queue in OUR lane only blocks us if the other same-direction lanes are busy
                plan, _, _ = self._plan_bypass(sc, o, gap, False)
                if plan is None or plan["rank"] == 0:
                    return f"hold: {q}"
                return self._commit(sc, plan, "lane change around queue")
        else:
            self.queue_t = 0.0
        if sc.light is not None and sc.light.state in ("RED", "YELLOW"):
            to_line = sc.light.s_line - sc.ego_s - sc.ego_half_len
            man_len = (o.s_max - sc.ego_s) + transition_length(3.5, sc.ego_v) + 10.0
            if 0.0 < to_line < max(60.0, man_len):
                return f"hold: no lane change, {sc.light.state.lower()} light {to_line:.0f} m ahead"
        # Traffic rules: same-direction lanes first.  The opposite lane is only allowed for a
        # STATIC blockage (parked car, road works) - never to overtake moving traffic - and for
        # a closed lane (road works) only after stopping and checking the oncoming lane.
        closure = is_static and self._is_closure(sc, o)
        allow_opp = is_static and (not closure or (sc.ego_v < 1.0 and static_for > 2.0))
        relax = 0.15 if (sc.t - self.stuck_t > 10.0 and self.stuck_t > 0.0) else 0.0
        plan, why, too_close = self._plan_bypass(sc, o, gap, allow_opp, relax)
        if plan is None:
            self.last_block_reason = why
            self._too_close = too_close
            if closure and not allow_opp:
                return "hold: lane closed, checking oncoming lane"
            return f"waiting to pass ({why})"
        return self._commit(sc, plan, "")

    def _commit(self, sc: Scene, plan, note):
        self._too_close = False
        self.mode = "BYPASS"
        self.bypass_ids = [c.id for c in plan["cluster"]]
        self.bypass_side = plan["side"]
        self.bypass_end_s = plan["end_s"]
        self.bypass_t0 = sc.t
        self.bypass_uses_opp = plan["uses_opp"]
        self.bypass_rank = plan["rank"]
        self.profile.set_target(sc.ego_s, plan["d"], plan["L"])
        kind = ("opposite lane" if plan["rank"] == 3 else "lane change" if plan["rank"] in (1, 2)
                else "nudge in lane")
        hw, ls, lo, rs, *_ = sc.geo.at(sc.ego_s + 10.0)
        lanes = (f"own lanes L{len([c for c in sc.geo.lanes(sc.ego_s + 10.0)[0] if c < 0])}"
                 f"/R{len([c for c in sc.geo.lanes(sc.ego_s + 10.0)[0] if c > 0])}, opp {'yes' if lo > 0 else 'no'}")
        why_not = ("; own lanes rejected: " + " | ".join(getattr(self, "rejected_own", [])[:2])
                   if plan["rank"] == 3 else "")
        self.bypass_note = f"{kind} [{lanes}]{why_not}"
        return f"start {kind} {plan['side'].lower()} d={plan['d']:+.1f}{' ' + note if note else ''}"

    def _start_evade(self, sc: Scene, o: Obj, gap: float) -> str:
        """Wrong-way vehicle in our lane: move to a free same-direction lane (or the edge of ours)."""
        saved = dict(BYPASS_MARGIN)
        BYPASS_MARGIN["vehicle"] = 0.9
        try:
            cands = [c for c in self._candidates(sc, [o], max(o.s, sc.ego_s + 5.0), False) if c[1] != "CENTER"]
        finally:
            BYPASS_MARGIN.clear()
            BYPASS_MARGIN.update(saved)
        cands.sort(key=lambda c: (c[1] != "RIGHT", c[0], abs(c[2] - sc.ego_d)))   # keep right first
        for rank, side, d_b in cands:
            dd = abs(d_b - sc.ego_d)
            s_end = sc.ego_s + transition_length(dd, sc.ego_v) + 6.0
            ok, why, _ = self._bypass_check(sc, [o], d_b, False, s_end)
            if ok:
                self.mode = "EVADE"
                self.evade_id = o.id
                self.bypass_side = side
                self.bypass_t0 = sc.t
                self.bypass_uses_opp = False
                self.bypass_rank = rank
                self.profile.set_target(sc.ego_s, d_b, max(4.0, min(transition_length(dd, sc.ego_v), gap * 0.4)))
                return f"evading oncoming {o.kind} to the {side.lower()} (d={d_b:+.1f})"
        return "oncoming in lane: stopping (no evasion space)"

    def _return_clear(self, sc: Scene) -> bool:
        hw_e, hl_e = sc.ego_half_wid, sc.ego_half_len
        for o in sc.objs:
            if not self._overlaps(o, 0.0, hw_e, 0.5):
                continue
            if o.s_max > sc.ego_s - hl_e - 4.0 and o.s_min < sc.ego_s + hl_e + 14.0:
                return False
            if o.s_max <= sc.ego_s - hl_e - 4.0 and o.vs > sc.ego_v + 0.5:
                if (sc.ego_s - hl_e - o.s_max) / max(0.1, o.vs - sc.ego_v) < 3.0:
                    return False
        return True

    def _update_bypass(self, sc: Scene) -> str:
        hl_e = sc.ego_half_len
        if self.mode == "EVADE":
            o = next((x for x in sc.objs if x.id == self.evade_id), None)
            if (o is not None and o.s_max > sc.ego_s - hl_e
                    and self._overlaps(o, self.profile.target, sc.ego_half_wid, 0.3)):
                self._start_evade(sc, o, max(0.0, o.s_min - sc.ego_s - hl_e))   # it drifted: re-target
            if o is None or o.s_max < sc.ego_s - hl_e - 2.0 or sc.t - self.bypass_t0 > 20.0:
                if self._return_clear(sc):
                    self.mode = "LANE"
                    self.profile.set_target(sc.ego_s, 0.0, transition_length(self.profile.target, sc.ego_v))
                    return "evasion done, back to lane"
            return f"evading ({self.bypass_side.lower()})"

        # BYPASS: abort early if the corridor became unsafe before we really moved over
        shifted = abs(self.profile.value(sc.ego_s)) / max(0.3, abs(self.profile.target))
        targets = [o for o in sc.objs if o.id in self.bypass_ids]
        if targets:
            self.bypass_end_s = max(self.bypass_end_s, max(o.s_max for o in targets) + hl_e + 3.0)
        if shifted < 0.35 and sc.ego_s + 4.0 < self.bypass_end_s:
            ok, why, _ = self._bypass_check(sc, targets, self.profile.target,
                                            True, self.bypass_end_s + 6.0)
            if not ok:
                self.mode = "LANE"
                self.profile.set_target(sc.ego_s, 0.0, transition_length(self.profile.target, sc.ego_v))
                return f"bypass aborted ({why})"
        passed = sc.ego_s - hl_e > self.bypass_end_s
        if passed or sc.t - self.bypass_t0 > BYPASS_MAX_S:
            if self._return_clear(sc):
                self.mode = "LANE"
                self.bypass_ids = []
                self.profile.set_target(sc.ego_s, 0.0, transition_length(self.profile.target, sc.ego_v))
                return "passed, returning to lane"
            return "passed, waiting for gap to return"
        return f"bypassing {self.bypass_side.lower()}: {getattr(self, 'bypass_note', '')}"[:160]

    def _rear_space(self, sc: Scene) -> Tuple[float, str]:
        """Free distance behind the rear bumper (objects + LiDAR); 0 if a vehicle approaches."""
        hw_e, hl_e = sc.ego_half_wid, sc.ego_half_len
        rear = sc.ego_s - hl_e
        space, why = 15.0, "rear clear"
        if sc.rear_gap is not None and sc.rear_gap < space:
            space, why = sc.rear_gap, f"LiDAR: object {sc.rear_gap:.1f} m behind"
        for x in sc.objs:
            if not self._overlaps(x, sc.ego_d, hw_e, 0.6) or x.s_min > sc.ego_s:
                continue
            g = rear - x.s_max
            if x.vs > 0.5 and g < 25.0:
                return 0.0, f"{x.kind} approaching from behind"
            if -1.0 < g < space:
                space, why = max(0.0, g), f"{x.kind} {g:.1f} m behind"
        return space, why

    def _rear_clear(self, sc: Scene, dist: float) -> Tuple[bool, str]:
        space, why = self._rear_space(sc)
        return space >= dist + 1.0, why

    def _update_stuck(self, sc: Scene, target, lead, hold):
        legit = hold and ("light" in hold or "yield" in hold)
        if sc.ego_v < 0.3 and target < 0.3 and not legit and lead is not None:
            if self.stuck_t <= 0.0:
                self.stuck_t = sc.t
            stuck_for = sc.t - self.stuck_t
            gap, o = lead
            # reverse ONLY when the obstacle is very close (no room to steer out, see need_crawl),
            # or when a slow steer-out got stuck at the obstacle's corner
            stuck_lane = (self.mode == "LANE" and self._too_close and stuck_for > STUCK_REVERSE_S)
            stuck_bypass = (self.mode == "BYPASS" and o.id in self.bypass_ids and gap < 1.5
                            and stuck_for > 2.0)
            vru_block = (o.is_vru and gap < 2.0 and sc.t - self.static_since.get(o.id, sc.t) > VRU_PATIENCE_S
                         and stuck_for > VRU_PATIENCE_S)
            if vru_block:
                self.reverse_need = 2.5
            if (stuck_lane or stuck_bypass) and not o.is_vru or vru_block:
                want = max(1.5, self.reverse_need or REVERSE_DIST_M) if stuck_lane else 2.5
                space, why = self._rear_space(sc)
                dist = min(want, space - 1.0)          # keep >= 1 m to whatever is behind
                ok = dist >= 0.8
                if ok:
                    self.reverse_dist = dist
                    self.mode = "REVERSE"
                    self.reverse_start_s = sc.ego_s
                    self.reverse_t0 = sc.t
                    self._too_close = False
                else:
                    self.last_block_reason = f"cannot reverse: {why}"
        elif sc.ego_v > 1.0:
            self.stuck_t = 0.0
