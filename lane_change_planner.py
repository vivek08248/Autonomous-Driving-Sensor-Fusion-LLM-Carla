"""
lane_change_planner.py - Safe, deterministic same-direction lane-change overtake.

Why this exists
---------------
Overtaking is a high-stress decision. A 0.5B LLM (and the paper's own ablation
shows bigger LLMs do not help much: Table 3) should NOT decide it. The LLM keeps
proposing speed / intent; this module decides WHETHER and WHEN a lane change is
safe and executes it with a closed-loop controller.

Rules enforced (all verified every frame, not only at the start)
----------------------------------------------------------------
1. Only SAME-DIRECTION driving lanes (lane_id sign equals ego lane_id sign).
   The oncoming lane, sidewalks, shoulders and junctions are never used.
2. The target lane must be clear in front AND behind (closing-speed aware),
   with no pedestrian / prop / vehicle alongside, for the whole manoeuvre
   including the distance needed to pass the obstacle.
3. A lane change only starts if the gap to the obstacle is large enough to
   finish moving sideways before reaching it (optionally backs up first if the
   car stopped too close and nothing is behind).
4. During the change the planner re-checks the target lane; if it becomes
   unsafe early it returns to the origin lane, otherwise it slows down.
5. Merge back only when the origin lane is clear, otherwise it simply stays in
   the new (same-direction) lane.
6. While the car still overlaps the obstacle's footprint, speed is limited so
   the car can still stop before it.

Interface is a drop-in replacement for the old WaypointOvertakePlanner:
is_active(), plan(), start(), step(), reset(), state, side, target_speed_kmh,
target_actor, target_actor_id, initial_obs_loc, last_result.
"""

import math

import carla

import config

EGO_HALF_W = 0.93      # Tesla Model 3 half-width (m)
EGO_FRONT_M = 2.45     # ego centre -> front bumper
EGO_REAR_M = 2.40      # ego centre -> rear bumper
WHEELBASE_M = 2.85


def _cfg(name, default):
    return float(getattr(config, name, default))


# ----------------------------------------------------------------------
# Pure helpers (no CARLA calls except reading vectors) - unit-testable
# ----------------------------------------------------------------------
def blend(x):
    """Cosine ease 0..1 (zero slope at both ends -> smooth lane change)."""
    x = max(0.0, min(1.0, x))
    return 0.5 * (1.0 - math.cos(math.pi * x))


def max_steer_rad(ego):
    """Max front-wheel angle of the ego, so normalized steer is exact."""
    try:
        wheels = ego.get_physics_control().wheels
        ang = max(abs(w.max_steer_angle) for w in wheels)
        if ang > 5.0:
            return math.radians(ang)
    except Exception:
        pass
    return math.radians(70.0)


def pure_pursuit_steer(ego_tf, target, speed_mps, max_steer):
    """Normalized steer [-1, 1] toward `target` (pure pursuit, bicycle model).

    Curvature is limited so lateral acceleration stays below ~3 m/s^2.
    (The old planner fed radians straight into the normalized steer input,
    which over-steered by ~46% and caused oscillation.)
    """
    loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    dx, dy = target.x - loc.x, target.y - loc.y
    lx = dx * fwd.x + dy * fwd.y
    ly = dx * right.x + dy * right.y
    ld2 = max(lx * lx + ly * ly, 1.0)
    curvature = 2.0 * ly / ld2
    k_max = min(0.20, 3.0 / max(speed_mps * speed_mps, 4.0))
    curvature = max(-k_max, min(k_max, curvature))
    delta = math.atan(WHEELBASE_M * curvature)
    return delta / max_steer


def snapshot_actors(world, ego, ego_tf, exclude_ids=()):
    """Vehicles, walkers and props around the ego, in the ego frame."""
    loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    out = []
    for a in world.get_actors():
        try:
            if a.id == ego.id or a.id in exclude_ids or not a.is_alive:
                continue
            tid = a.type_id
            if tid.startswith("vehicle."):
                kind = "vehicle"
            elif tid.startswith("walker."):
                kind = "walker"
            elif tid.startswith("static.prop"):
                kind = "prop"
            else:
                continue
            d = a.get_location() - loc
            along = d.x * fwd.x + d.y * fwd.y
            lat = d.x * right.x + d.y * right.y
            if along < -40.0 or along > 80.0 or abs(lat) > 14.0:
                continue
            ext = a.bounding_box.extent
            v = a.get_velocity()
            af = a.get_transform().get_forward_vector()
            out.append({
                "id": a.id,
                "kind": kind,
                "along": along,
                "lat": lat,
                "half_len": max(ext.x, 0.2),
                "half_w": max(ext.y, 0.2),
                "v_along": v.x * fwd.x + v.y * fwd.y,
                "v_lat": v.x * right.x + v.y * right.y,
                "heading_dot": af.x * fwd.x + af.y * fwd.y,
            })
        except Exception:
            continue
    return out


def in_lane_band(a, lane_off, lane_w):
    """True if actor `a` overlaps the lane whose centre is `lane_off` m to the right."""
    return abs(a["lat"] - lane_off) < lane_w * 0.5 + a["half_w"] - 0.25


def lane_is_clear(actors, lane_off, lane_w, ego_speed, lc_time, pass_dist, relax=1.0,
                  ignore_ids=()):
    """Closing-speed aware front/rear/alongside check of one lane.

    pass_dist: distance (ego centre travel) the ego will occupy this lane.
    relax < 1 loosens the thresholds (used for in-progress hysteresis).
    Returns (is_clear, reason).
    """
    front_base = _cfg("LANE_CHANGE_FRONT_GAP_M", 8.0) * relax
    rear_base = _cfg("LANE_CHANGE_REAR_GAP_M", 7.0) * relax
    for a in actors:
        if a["id"] in ignore_ids or not in_lane_band(a, lane_off, lane_w):
            continue
        if a["kind"] != "vehicle":
            if -(EGO_REAR_M + a["half_len"] + 1.5) < a["along"] < pass_dist + 8.0:
                return False, f"{a['kind']} in target lane"
            continue
        gap_front = a["along"] - a["half_len"] - EGO_FRONT_M
        gap_rear = -a["along"] - a["half_len"] - EGO_REAR_M
        if gap_front < 0.0 and gap_rear < 0.0:
            return False, "vehicle alongside in target lane"
        if a["along"] >= 0.0:
            closing = max(0.0, ego_speed - a["v_along"])
            ratio = min(1.0, closing / max(ego_speed, 1.0))
            need = front_base + max(closing * lc_time, pass_dist * ratio)
            if a["heading_dot"] < 0.0:
                need *= 2.0
            if gap_front < need:
                return False, f"vehicle ahead in target lane ({gap_front:.0f}m < {need:.0f}m)"
        else:
            closing = max(0.0, a["v_along"] - ego_speed)
            need = rear_base + closing * (lc_time + 1.0)
            if gap_rear < need:
                return False, f"vehicle approaching from behind ({gap_rear:.0f}m < {need:.0f}m)"
    return True, "clear"


def front_limit_kmh(actors, lane_off, lane_w, base_kmh):
    """Speed cap so the ego can still stop behind whatever is ahead in a lane."""
    limit = base_kmh
    for a in actors:
        if a["along"] <= 0.0 or not in_lane_band(a, lane_off, lane_w):
            continue
        gap = a["along"] - a["half_len"] - EGO_FRONT_M
        if gap > 30.0:
            continue
        # conservative: 1.5 m/s^2 and a 4 m stand-off leave room for the speed
        # controller's tracking lag
        bound = math.sqrt(2.0 * 1.5 * max(0.0, gap - 4.0)) * 3.6
        v_a = max(0.0, a["v_along"]) * 3.6 if a["kind"] == "vehicle" else 0.0
        allowed = max(0.9 * v_a if gap > 6.0 else 0.0, bound)
        limit = min(limit, allowed)
    return max(0.0, limit)


def lateral_offset(wp_from, wp_to):
    """Signed lateral offset (m, + = right) of wp_to relative to wp_from."""
    r = wp_from.transform.get_right_vector()
    d = wp_to.transform.location - wp_from.transform.location
    return d.x * r.x + d.y * r.y


def no_junction_ahead(wp, dist, step=2.0):
    if wp is None or getattr(wp, 'is_junction', False):
        return False
    d = step
    while d <= dist:
        nxt = wp.next(d)
        if not nxt or nxt[0].is_junction:
            return False
        d += step
    return True


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------
class SafeLaneChangePlanner:
    DIRECTION = {"LANE_CHANGE_OUT": 1.0, "PASSING": 1.0, "MERGE_BACK": -1.0, "RETURN": -1.0}

    def __init__(self):
        self.last_result = None      # DONE / STAY / ABORT / TIMEOUT (read by Session)
        self.last_reason = ""
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.side = "LEFT"
        self.target_speed_kmh = 15.0
        self.pass_speed_kmh = 15.0
        self.target_actor = None
        self.target_actor_id = None
        self.initial_obs_loc = None
        self.origin_lane_id = None
        self.origin_road_id = None
        self.phase_timer = 0.0
        self.total_timer = 0.0
        self.lam = 0.0               # 0 = origin lane, 1 = target lane
        self.lc_dist = 18.0
        self.min_start_gap = 10.0
        self.last_steer = 0.0
        self.speed_cap = None
        self.is_opposite_lane = False
        self.max_steer = math.radians(70.0)

    def is_active(self):
        return self.state != "IDLE"

    @property
    def target_lane_label(self):
        return self.side

    def _finish(self, result, reason=""):
        self.last_result = result
        self.last_reason = reason
        if result in ("ABORT", "TIMEOUT"):
            print(f"\n[LaneChange] {result}: {reason}")
        else:
            print(f"\n[LaneChange] {result} ({self.side})")
        self.reset()
        return None

    # ---------------- geometry ----------------
    @staticmethod
    def _lc_dist(v_mps):
        return min(30.0, max(_cfg("LANE_CHANGE_MIN_DIST_M", 12.0),
                             _cfg("LANE_CHANGE_TIME_S", 2.6) * v_mps))

    @staticmethod
    def _clearance(v_mps):
        return 2.5 + 0.4 * v_mps

    @staticmethod
    def _lane_geometry(wmap, ego_loc, side, origin_lane_id=None, origin_road_id=None, allow_opposite=False):
        """(origin_wp, target_wp) or None. Target must be a driving lane."""
        wp = wmap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            return None
        if origin_lane_id is not None:
            if wp.road_id != origin_road_id:
                return None
            if wp.lane_id != origin_lane_id:
                cand = None
                for n in (wp.get_left_lane(), wp.get_right_lane()):
                    if n is not None and n.lane_id == origin_lane_id and n.road_id == wp.road_id:
                        cand = n
                if cand is None:
                    return None
                wp = cand
        tgt = wp.get_left_lane() if side == "LEFT" else wp.get_right_lane()
        if tgt is None or tgt.lane_type != carla.LaneType.Driving:
            return None
        is_opp = (tgt.lane_id * wp.lane_id <= 0)
        if is_opp:
            if not allow_opposite:
                return None
            # In opposite lane, verify road connection
            if tgt.road_id != wp.road_id and abs(tgt.road_id - wp.road_id) > 2:
                return None
        else:
            if tgt.road_id != wp.road_id:
                return None
        return wp, tgt

    def _obstacle(self, ego_tf):
        """Obstacle position in the ego frame."""
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        loc = ego_tf.location
        a = self.target_actor
        half_len, half_w, v_along = 2.2, 0.9, 0.0
        pos = None
        try:
            if a is not None and a.is_alive:
                pos = a.get_location()
                ext = a.bounding_box.extent
                half_len, half_w = max(ext.x, 0.2), max(ext.y, 0.2)
                v = a.get_velocity()
                v_along = v.x * fwd.x + v.y * fwd.y
        except Exception:
            pos = None
        if pos is None:
            pos = self.initial_obs_loc
        if pos is None:
            return {"along": -50.0, "lat": 0.0, "half_len": half_len, "half_w": half_w, "v_along": 0.0}
        d = pos - loc
        return {"along": d.x * fwd.x + d.y * fwd.y, "lat": d.x * right.x + d.y * right.y,
                "half_len": half_len, "half_w": half_w, "v_along": v_along}

    # ---------------- decision ----------------
    def plan(self, world, ego, target_actor, ego_speed_kmh, pass_speed_kmh=16.0):
        """Decide if a SAFE lane change around `target_actor` is possible right now.

        Returns dict(ok, side, reason, needs_reverse, min_gap, gap).
        """
        res = {"ok": False, "side": None, "reason": "", "needs_reverse": False,
               "min_gap": 0.0, "gap": 0.0}
        if ego is None or not ego.is_alive or target_actor is None:
            res["reason"] = "no ego/target"
            return res
        ego_tf = ego.get_transform()
        v = max(0.0, ego_speed_kmh) / 3.6
        wmap = world.get_map()
        lc_dist = self._lc_dist(max(v, pass_speed_kmh / 3.6 * 0.5))
        lc_time = lc_dist / max(v, 2.0)

        saved = (self.target_actor, self.initial_obs_loc)
        self.target_actor, self.initial_obs_loc = target_actor, None
        ob = self._obstacle(ego_tf)
        self.target_actor, self.initial_obs_loc = saved
        gap = ob["along"] - ob["half_len"] - EGO_FRONT_M
        res["gap"] = gap
        if ob["along"] < 0.5:
            res["reason"] = "obstacle not ahead"
            return res
        if ob["v_along"] < -0.2:
            res["reason"] = f"obstacle moving backward toward ego ({ob['v_along']*3.6:.1f} km/h)"
            return res

        _tgt_id = getattr(target_actor, 'id', None)
        _excl = {_tgt_id} if _tgt_id is not None else set()
        actors = snapshot_actors(world, ego, ego_tf, exclude_ids=_excl)
        pass_dist = max(0.0, ob["along"] + ob["half_len"] + EGO_REAR_M + self._clearance(v))
        junction_clear = _cfg("LANE_CHANGE_JUNCTION_CLEAR_M", 40.0)
        reasons = []
        right = ego_tf.get_right_vector()

        for side in ("LEFT", "RIGHT"):
            geo = self._lane_geometry(wmap, ego_tf.location, side)
            if geo is None:
                reasons.append(f"{side}: no same-direction lane")
                continue
            origin_wp, tgt_wp = geo
            # Veto side if the obstacle itself is inside or encroaching the target lane!
            d_tgt = tgt_wp.transform.location - ego_tf.location
            off_tgt = d_tgt.x * right.x + d_tgt.y * right.y
            obs_lat_in_tgt = abs(ob["lat"] - off_tgt)
            if obs_lat_in_tgt < (tgt_wp.lane_width * 0.5 + ob["half_w"] + 0.2):
                reasons.append(f"{side}: obstacle in target lane")
                continue
            if _cfg("LANE_CHANGE_RESPECT_MARKINGS", 0.0) > 0.5:
                lc_flag = origin_wp.lane_change
                allowed = (carla.LaneChange.Both,
                           carla.LaneChange.Left if side == "LEFT" else carla.LaneChange.Right)
                if lc_flag not in allowed:
                    reasons.append(f"{side}: solid line")
                    continue
            if not (no_junction_ahead(origin_wp, junction_clear)
                    and no_junction_ahead(tgt_wp, junction_clear)):
                reasons.append(f"{side}: junction ahead")
                continue
            d = tgt_wp.transform.location - ego_tf.location
            off_ego = d.x * right.x + d.y * right.y
            ok, why = lane_is_clear(actors, off_ego, tgt_wp.lane_width, v, lc_time, pass_dist)
            if not ok:
                reasons.append(f"{side}: {why}")
                continue

            # sideways distance needed before reaching the obstacle
            off_total = abs(lateral_offset(origin_wp, tgt_wp))
            y_clear = EGO_HALF_W + ob["half_w"] + 0.30 - abs(ob["lat"])
            if y_clear <= 0.0:
                s_clear = 0.0
            elif y_clear >= off_total:
                s_clear = lc_dist
            else:
                s_clear = lc_dist / math.pi * math.acos(1.0 - 2.0 * y_clear / off_total)
            min_gap = s_clear + 2.0
            res.update(ok=True, side=side, reason=f"{side} lane clear", min_gap=min_gap)

            if gap < min_gap:
                res["needs_reverse"] = True
                if not _cfg("LANE_CHANGE_ALLOW_REVERSE", 1.0) > 0.5:
                    res.update(ok=False, reason=f"gap {gap:.1f}m < {min_gap:.1f}m and reverse disabled")
                elif not self._rear_clear(actors):
                    res.update(ok=False, reason="gap too small and no room to back up")
            return res
        # Very rare scenario fallback: Jump to opposite lane (B1/B2)
        # ONLY considered when:
        # 1. No same-direction bypass lane is available/clear.
        # 2. Obstacle ahead is blocking ego (gap < 25.0m, obstacle is stopped/slow <= 5 km/h, or ego is crawling).
        # 3. No upcoming junction within junction_clear meters.
        # 4. Strict safety check: Opposite lane must have 0 oncoming vehicles within 65m.
        ego_speed_kmh = v * 3.6
        obs_spd_kmh = max(0.0, ob["v_along"]) * 3.6
        can_consider_rare_opposite = (
            gap < 25.0
            and (obs_spd_kmh < 6.0 or ego_speed_kmh < 12.0)
            and ob["along"] > 1.0
        )
        if can_consider_rare_opposite and not res["ok"] and not res.get("needs_reverse", False):
            for side in ("LEFT", "RIGHT"):
                geo = self._lane_geometry(wmap, ego_tf.location, side, allow_opposite=True)
                if geo is None:
                    continue
                origin_wp, tgt_wp = geo
                is_opposite_lane = (tgt_wp.lane_id * origin_wp.lane_id <= 0)
                if not is_opposite_lane:
                    continue  # already evaluated in same-direction loop

                # Junction guard
                if not (no_junction_ahead(origin_wp, junction_clear) and no_junction_ahead(tgt_wp, junction_clear)):
                    reasons.append(f"{side} opposite: junction ahead")
                    continue

                # Lateral clearance to target lane center
                d_tgt = tgt_wp.transform.location - ego_tf.location
                off_ego = d_tgt.x * right.x + d_tgt.y * right.y

                # Obstacle in target lane check
                obs_lat_in_tgt = abs(ob["lat"] - off_ego)
                if obs_lat_in_tgt < (tgt_wp.lane_width * 0.5 + ob["half_w"] + 0.2):
                    reasons.append(f"{side} opposite: obstacle blocks target lane")
                    continue

                # Standard clearance check with relaxed threshold
                ok, why = lane_is_clear(actors, off_ego, tgt_wp.lane_width, v, lc_time, pass_dist)
                if not ok:
                    reasons.append(f"{side} opposite: {why}")
                    continue

                # STRICT ONCOMING TRAFFIC GUARD for opposite lane:
                # Any vehicle in opposite lane within 65m ahead or closing toward ego is an immediate hard veto
                oncoming_hazard = False
                for a in actors:
                    if in_lane_band(a, off_ego, tgt_wp.lane_width):
                        if a["kind"] == "vehicle":
                            if -6.0 < a["along"] < 65.0:
                                oncoming_hazard = True
                                reasons.append(f"{side} opposite: oncoming vehicle within 65m ({a['along']:.1f}m)")
                                break
                            if a["v_along"] < -0.5 and a["along"] < 80.0:
                                oncoming_hazard = True
                                reasons.append(f"{side} opposite: fast oncoming closing ({a['v_along']*3.6:.1f} km/h)")
                                break
                if oncoming_hazard:
                    continue

                # Compute min required gap to clear the obstacle
                off_total = abs(lateral_offset(origin_wp, tgt_wp))
                y_clear = EGO_HALF_W + ob["half_w"] + 0.35 - abs(ob["lat"])
                if y_clear <= 0.0:
                    s_clear = 0.0
                elif y_clear >= off_total:
                    s_clear = lc_dist
                else:
                    s_clear = lc_dist / math.pi * math.acos(1.0 - 2.0 * y_clear / off_total)
                min_gap = s_clear + 2.0

                res.update(ok=True, side=side, is_opposite=True,
                           reason=f"Rare opposite lane bypass ({side}, 65m oncoming clear)", min_gap=min_gap)
                if gap < min_gap:
                    res["needs_reverse"] = True
                    if not _cfg("LANE_CHANGE_ALLOW_REVERSE", 1.0) > 0.5:
                        res.update(ok=False, reason=f"gap {gap:.1f}m < {min_gap:.1f}m and reverse disabled")
                    elif not self._rear_clear(actors):
                        res.update(ok=False, reason="gap too small and no room to back up")
                return res

        res["reason"] = "; ".join(reasons) or "no lane"
        return res

    @staticmethod
    def _rear_clear(actors, dist=12.0):
        for a in actors:
            if -dist < a["along"] < 0.0 and abs(a["lat"]) < 2.2:
                return False
        return True

    def start(self, world, ego, side, target_actor, target_speed_kmh=15.0, front_gap_m=None):
        """Begin a lane change if (and only if) it is safe. Returns True when started."""
        if self.is_active():
            return False
        ego_v = ego.get_velocity()
        v = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)
        plan = self.plan(world, ego, target_actor, v * 3.6, target_speed_kmh)
        if not plan["ok"]:
            self.last_reason = plan["reason"]
            return False
        self.side = plan["side"]
        self.target_actor = target_actor
        self.target_actor_id = getattr(target_actor, "id", None)
        try:
            self.initial_obs_loc = target_actor.get_location()
        except Exception:
            self.initial_obs_loc = None
        self.pass_speed_kmh = float(target_speed_kmh)
        self.target_speed_kmh = float(target_speed_kmh)
        self.lc_dist = self._lc_dist(max(v, target_speed_kmh / 3.6 * 0.5))
        self.min_start_gap = plan["min_gap"]
        self.max_steer = max_steer_rad(ego)
        self.speed_cap = None
        self.lam = 0.0
        self.last_steer = 0.0
        self.phase_timer = 0.0
        self.total_timer = 0.0
        self.is_opposite_lane = bool(plan.get("is_opposite", False))
        wp = world.get_map().get_waypoint(ego.get_location(), project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
        self.origin_lane_id = wp.lane_id if wp else None
        self.origin_road_id = wp.road_id if wp else None
        self._cached_geo = self._lane_geometry(world.get_map(), ego.get_location(), self.side,
                                               allow_opposite=self.is_opposite_lane)
        self.state = "PREPARE_REVERSE" if plan["needs_reverse"] else "LANE_CHANGE_OUT"
        opp_tag = " [OPPOSITE LANE BYPASS]" if self.is_opposite_lane else ""
        print(f"\n[LaneChange] {self.side}{opp_tag} lane change around obstacle "
              f"(gap {plan['gap']:.1f}m, need {plan['min_gap']:.1f}m, state {self.state}, "
              f"pass {self.pass_speed_kmh:.0f} km/h)")
        return True

    # ---------------- execution ----------------
    def step(self, world, ego, dt):
        """Returns control dict {reverse, steer, target_speed_kmh, phase, [throttle, brake]} or None when finished."""
        if self.state == "IDLE" or ego is None or not ego.is_alive:
            self.reset()
            return None

        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        vel = ego.get_velocity()
        v = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        v_signed = vel.x * fwd.x + vel.y * fwd.y

        # Freeze timers and timeouts when stopped (e.g. at red light / stop line)
        is_stopped = (v < 0.25) and (self.state != "PREPARE_REVERSE")
        if not is_stopped:
            self.phase_timer += dt
            self.total_timer += dt
        if self.total_timer > _cfg("OVERTAKE_TOTAL_TIMEOUT_S", 30.0):
            return self._finish("TIMEOUT", "total time exceeded")
        wmap = world.get_map()
        excl = {self.target_actor_id} if self.target_actor_id is not None else set()
        actors = snapshot_actors(world, ego, ego_tf, exclude_ids=excl)
        ob = self._obstacle(ego_tf)
        gap_obs = ob["along"] - ob["half_len"] - EGO_FRONT_M
        if ob["v_along"] < -0.2 and ob["along"] > 0.0 and self.state != "PREPARE_REVERSE":
            return self._finish("ABORT", f"obstacle moving backward toward ego ({ob['v_along']*3.6:.1f} km/h)")
        obs_kmh = max(0.0, ob["v_along"]) * 3.6
        if obs_kmh > 2.0:
            self.pass_speed_kmh = min(_cfg("MAX_ALLOWED_SPEED_KMH", 30.0),
                                      max(self.pass_speed_kmh, obs_kmh + 8.5))

        # ---------- optional back-up when the car stopped too close ----------
        if self.state == "PREPARE_REVERSE":
            if not self._rear_clear(actors, 10.0):
                return self._finish("ABORT", "vehicle behind while backing up")
            if gap_obs >= self.min_start_gap + 1.0:
                if self._side_ok(world, ego, ego_tf, actors, v):
                    self.state = "LANE_CHANGE_OUT"
                    self.phase_timer = 0.0
                    return {"reverse": False, "steer": 0.0, "target_speed_kmh": 0.0,
                            "phase": f"LANE_CHANGE_{self.side}_READY"}
                return self._finish("ABORT", "target lane no longer clear")
            if self.phase_timer > 6.0:
                return self._finish("ABORT", "could not back up far enough")
            return {"reverse": True, "throttle": 0.40, "brake": 0.0, "steer": 0.0,
                    "target_speed_kmh": 0.0, "phase": f"PREPARE_REVERSE_{self.side}"}

        # ---------- lane geometry ----------
        live_geo = self._lane_geometry(wmap, ego_loc, self.side, self.origin_lane_id, self.origin_road_id)
        if live_geo is not None:
            geo = live_geo
            self._cached_geo = geo
        else:
            geo = getattr(self, "_cached_geo", None)

        if geo is None:
            return self._finish("ABORT", "left the road section / junction")
        origin_wp, tgt_wp = geo

        def lat_to(wp):
            d = wp.transform.location - ego_loc
            return d.x * right.x + d.y * right.y

        off_origin, off_target = lat_to(origin_wp), lat_to(tgt_wp)
        err_origin, err_target = abs(off_origin), abs(off_target)
        yaw_err = abs(((ego_tf.rotation.yaw - origin_wp.transform.rotation.yaw) + 180.0) % 360.0 - 180.0)
        lc_time = self.lc_dist / max(v, 2.0)
        clearance = self._clearance(v)
        pass_dist = max(0.0, ob["along"] + ob["half_len"] + EGO_REAR_M + clearance)
        # Rate only advances lateral progress when moving; freezes while stopped
        rate = (max(v, 0.8) * dt / self.lc_dist) if not is_stopped else 0.0

        # If live road geometry reached junction boundary mid-action:
        if live_geo is None:
            if self.lam >= 0.7 and err_target < 0.45 and yaw_err < 6.0:
                return self._finish("STAY", "lane change completed before junction; staying aligned in target lane")
            elif self.lam < 0.4 and self.state != "RETURN":
                self.state = "RETURN"
                self.phase_timer = 0.0

        # ---------- state machine ----------
        if self.state == "LANE_CHANGE_OUT":
            ok, why = lane_is_clear(actors, off_target, tgt_wp.lane_width, v, lc_time,
                                    pass_dist, relax=0.7)
            if not ok:
                if self.lam < 0.4:
                    self.state, self.phase_timer = "RETURN", 0.0
                    print(f"\n[LaneChange] target lane became unsafe ({why}) -> returning")
                else:
                    self.speed_cap = 8.0
            self.lam = min(1.0, self.lam + rate)
            if self.state == "LANE_CHANGE_OUT":
                if self.lam >= 1.0 and err_target < 0.6:
                    self.state, self.phase_timer = "PASSING", 0.0
                elif self.phase_timer > 2.0 * lc_time + 4.0:
                    if err_target < 1.2:
                        self.state, self.phase_timer = "PASSING", 0.0
                    else:
                        self.state, self.phase_timer = "RETURN", 0.0

        elif self.state == "PASSING":
            self.lam = min(1.0, self.lam + rate)
            # Improved passed detection: car's front bumper past obstacle's rear
            passed = ob["along"] < -(ob["half_len"] + clearance)
            if passed:
                origin_w = origin_wp.lane_width
                ok, why = lane_is_clear(actors, off_origin, origin_w, v, lc_time, 0.0)
                if ok:
                    self.state, self.phase_timer = "MERGE_BACK", 0.0
                elif self.phase_timer > _cfg("LANE_CHANGE_MERGE_WAIT_S", 8.0):
                    return self._finish("STAY", f"origin lane not clear ({why}); staying in new lane")
            # Extended timeout for long obstacles (trucks, etc.)
            if self.phase_timer > _cfg("OVERTAKE_PASSING_TIMEOUT_S", 20.0):
                return self._finish("STAY", "pass taking too long; staying in new lane")

        elif self.state == "MERGE_BACK":
            ok, why = lane_is_clear(actors, off_origin, origin_wp.lane_width, v, lc_time, 0.0, relax=0.7)
            if not ok and self.lam > 0.6:
                self.state, self.phase_timer = "PASSING", 0.0     # abort the merge
            else:
                self.lam = max(0.0, self.lam - rate)
                if self.lam <= 0.0 and err_origin < 0.4 and yaw_err < 3.0:
                    return self._finish("DONE")
                if self.phase_timer > 2.0 * lc_time + 4.0:
                    return self._finish("DONE" if err_origin < 1.0 else "ABORT", "merge timeout")

        elif self.state == "RETURN":
            self.lam = max(0.0, self.lam - rate)
            if self.lam <= 0.0 and err_origin < 0.4 and yaw_err < 3.0:
                return self._finish("ABORT", "returned to origin lane")
            if self.phase_timer > 2.0 * lc_time + 4.0:
                return self._finish("ABORT", "return timeout")

        # ---------- speed command ----------
        speed = self.pass_speed_kmh
        speed = min(speed, front_limit_kmh(actors, off_target, tgt_wp.lane_width, speed))
        if self.speed_cap is not None:
            speed = min(speed, self.speed_cap)
        footprint_overlap = (abs(ob["lat"]) - (EGO_HALF_W + ob["half_w"])) < 0.25
        if footprint_overlap and ob["along"] > 0.0:
            bound = math.sqrt(2.0 * 2.0 * max(0.0, gap_obs - 2.5)) * 3.6
            speed = min(speed, obs_kmh + bound)
        if self.state == "RETURN":
            speed = min(speed, max(6.0, obs_kmh + math.sqrt(2.0 * 1.5 * max(0.0, gap_obs - 2.5)) * 3.6))
        speed = max(0.0, speed)
        # Only stop for gap_obs if there is actual lateral footprint overlap!
        if footprint_overlap and ob["along"] > 0.0 and (gap_obs < 3.0 or ob["v_along"] < -0.2):
            speed = 0.0
        self.target_speed_kmh = speed

        # ---------- steering ----------
        ld = max(4.5, min(10.0, 0.9 * v + 3.0))
        dirn = self.DIRECTION.get(self.state, 1.0)
        w = blend(self.lam + dirn * 0.5 * ld / self.lc_dist)
        o_next = origin_wp.next(ld)
        if o_next:
            o_loc = o_next[0].transform.location
        else:
            o_fwd = origin_wp.transform.get_forward_vector()
            o_loc = carla.Location(x=origin_wp.transform.location.x + o_fwd.x * ld,
                                  y=origin_wp.transform.location.y + o_fwd.y * ld,
                                  z=origin_wp.transform.location.z)
        t_next = tgt_wp.next(ld)
        if t_next:
            t_loc = t_next[0].transform.location
        else:
            orv = o_next[0].transform.get_right_vector()
            off = lateral_offset(origin_wp, tgt_wp)
            t_loc = carla.Location(x=o_loc.x + orv.x * off, y=o_loc.y + orv.y * off, z=o_loc.z)
        pt = carla.Location(x=o_loc.x * (1.0 - w) + t_loc.x * w,
                            y=o_loc.y * (1.0 - w) + t_loc.y * w,
                            z=o_loc.z)
        steer = pure_pursuit_steer(ego_tf, pt, max(v_signed, 0.0), self.max_steer)
        steer = self.last_steer + max(-0.06, min(0.06, steer - self.last_steer))
        self.last_steer = steer

        return {"reverse": False, "steer": round(steer, 3),
                "target_speed_kmh": round(speed, 1),
                "phase": f"LANE_CHANGE_{self.side}_{self.state}"}

    def _side_ok(self, world, ego, ego_tf, actors, v):
        geo = self._lane_geometry(world.get_map(), ego_tf.location, self.side)
        if geo is None:
            return False
        origin_wp, tgt_wp = geo
        right = ego_tf.get_right_vector()
        d = tgt_wp.transform.location - ego_tf.location
        off = d.x * right.x + d.y * right.y
        ob = self._obstacle(ego_tf)
        pass_dist = max(0.0, ob["along"] + ob["half_len"] + EGO_REAR_M + self._clearance(v))
        ok, _ = lane_is_clear(actors, off, tgt_wp.lane_width, v,
                              self.lc_dist / max(v, 2.0), pass_dist)
        return ok


# Backwards-compatible name used by drive_in_carla.py
WaypointOvertakePlanner = SafeLaneChangePlanner