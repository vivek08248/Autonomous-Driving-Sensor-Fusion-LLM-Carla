"""
ad_world.py - CARLA-facing perception for the driving stack.

  RouteManager   : dense route (1 m spacing) built from OpenDRIVE waypoints, with
                   junction branch selection (straight / left / right / random),
                   Frenet projection, lane geometry for the planner and the
                   paper's "route points".
  LightIndex     : traffic lights indexed by the (road, lane) of their stop line, so
                   the light that governs OUR route lane is found in O(route).
  Perception     : ONE world.get_actors() per tick -> objects in the route frame,
                   weather classification, fused front gap (object list + LiDAR + radar).
  SensorRig      : light sensor suite for a 6 GB GPU (LiDAR, front radar, IMU, GNSS,
                   collision).  No cameras -> zero render-target VRAM.
"""
import math
import random
import threading
import time
from collections import deque

import numpy as np

import carla

from ad_types import LaneGeo, LightInfo, Obj

try:
    import config as _cfg
except Exception:
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


ANIMAL_KEYS = ("deer", "dog", "cat", "animal", "cow", "horse")


# =============================================================================
# Route
# =============================================================================
class RouteManager:
    SPACING = 1.0
    AHEAD_M = 150.0
    BEHIND_M = 25.0

    def __init__(self, wmap, ego_tf, pref="straight"):
        self.map = wmap
        self.pref = pref or "straight"
        self.junctions_taken = 0
        self.wps = []
        self.s = []
        self._lane_cache = {}
        self._arr_dirty = True
        start = self._start_wp(ego_tf)
        self.wps.append(start)
        self.s.append(0.0)
        self.idx = 0
        self.ego_s = 0.0
        self.extend()

    # ---------------------------------------------------------- building
    def _start_wp(self, ego_tf):
        wp = self.map.get_waypoint(ego_tf.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        ego_yaw = ego_tf.rotation.yaw
        if wp is not None and abs(_wrap_deg(wp.transform.rotation.yaw - ego_yaw)) > 90.0:
            for getter in ("get_left_lane", "get_right_lane"):
                cand = getattr(wp, getter)()
                for _ in range(4):
                    if cand is None:
                        break
                    if (cand.lane_type == carla.LaneType.Driving and
                            abs(_wrap_deg(cand.transform.rotation.yaw - ego_yaw)) < 90.0):
                        return cand
                    cand = getattr(cand, getter)()
        return wp

    def _choose(self, cur, cands):
        if len(cands) == 1:
            return cands[0]
        pref = self.pref if self.junctions_taken == 0 else "straight"
        if pref == "random":
            pref = random.choice(["straight", "left", "right"])
        scored = []
        for c in cands:
            probe = c.next(14.0)
            p = probe[0] if probe else c
            scored.append((_wrap_deg(p.transform.rotation.yaw - cur.transform.rotation.yaw), c))
        if pref == "left":
            d, c = min(scored, key=lambda x: x[0])
            if d < -30.0:
                return c
        elif pref == "right":
            d, c = max(scored, key=lambda x: x[0])
            if d > 30.0:
                return c
        return min(scored, key=lambda x: abs(x[0]))[1]

    def extend(self):
        while self.s[-1] - self.ego_s < self.AHEAD_M:
            cur = self.wps[-1]
            cands = cur.next(self.SPACING)
            if not cands:
                break
            entering = (not cur.is_junction) and any(c.is_junction for c in cands)
            nxt = self._choose(cur, cands)
            if entering or (len(cands) > 1):
                if nxt.is_junction and not cur.is_junction:
                    self.junctions_taken += 1
            d = cur.transform.location.distance(nxt.transform.location)
            if d < 0.05:
                break
            self.wps.append(nxt)
            self.s.append(self.s[-1] + d)
            self._arr_dirty = True

    def _arrays(self):
        if self._arr_dirty:
            locs = [w.transform.location for w in self.wps]
            self.xs = np.array([p.x for p in locs])
            self.ys = np.array([p.y for p in locs])
            self.zs = np.array([p.z for p in locs])
            self.yaws = np.radians(np.array([w.transform.rotation.yaw for w in self.wps]))
            self.ss = np.array(self.s)
            self.ux, self.uy = np.cos(self.yaws), np.sin(self.yaws)
            self._arr_dirty = False

    # --------------------------------------------------------- projection
    def project(self, x, y, lo=0, hi=None):
        """(s, d, yaw_rad, i) of point (x, y) on the route (+d = right)."""
        self._arrays()
        hi = len(self.xs) if hi is None else min(hi, len(self.xs))
        lo = max(0, lo)
        dx = self.xs[lo:hi] - x
        dy = self.ys[lo:hi] - y
        i = lo + int(np.argmin(dx * dx + dy * dy))
        px, py = x - self.xs[i], y - self.ys[i]
        along = px * self.ux[i] + py * self.uy[i]
        if along < 0 and i > 0:
            i -= 1
            px, py = x - self.xs[i], y - self.ys[i]
            along = px * self.ux[i] + py * self.uy[i]
        d = -px * self.uy[i] + py * self.ux[i]
        return float(self.ss[i] + along), float(d), float(self.yaws[i]), i

    def update(self, ego_loc):
        """Project the ego, drop old points, extend ahead.  Returns (ego_s, ego_d, yaw)."""
        self._arrays()
        lo = max(0, self.idx - 10)
        hi = min(len(self.xs), self.idx + 80)
        s, d, yaw, i = self.project(ego_loc.x, ego_loc.y, lo, hi)
        if abs(d) > 8.0:                           # lost (reset / teleport) -> global search
            s, d, yaw, i = self.project(ego_loc.x, ego_loc.y)
        self.idx, self.ego_s = i, s
        cut = 0
        while cut < len(self.s) - 2 and self.s[cut] < s - self.BEHIND_M:
            cut += 1
        if cut > 0:
            del self.wps[:cut]
            del self.s[:cut]
            self.idx -= cut
            self._arr_dirty = True
        self.extend()
        return s, d, yaw

    def point(self, s, offset=0.0):
        """World (x, y, z, yaw_rad) of station s, laterally shifted by offset (+right)."""
        self._arrays()
        ss = self.ss
        if s <= ss[0]:
            i, t = 0, 0.0
        elif s >= ss[-1]:
            i, t = len(ss) - 2, 1.0
        else:
            i = int(np.searchsorted(ss, s) - 1)
            seg = ss[i + 1] - ss[i]
            t = (s - ss[i]) / seg if seg > 1e-6 else 0.0
        x = self.xs[i] + (self.xs[i + 1] - self.xs[i]) * t
        y = self.ys[i] + (self.ys[i + 1] - self.ys[i]) * t
        z = self.zs[i] + (self.zs[i + 1] - self.zs[i]) * t
        yaw = math.atan2(self.ys[i + 1] - self.ys[i], self.xs[i + 1] - self.xs[i])
        if s > ss[-1]:
            x += math.cos(yaw) * (s - ss[-1])
            y += math.sin(yaw) * (s - ss[-1])
        x += -math.sin(yaw) * offset
        y += math.cos(yaw) * offset
        return x, y, z, yaw

    def wp_at(self, s):
        self._arrays()
        i = int(np.clip(np.searchsorted(self.ss, s), 0, len(self.wps) - 1))
        return self.wps[i]

    def length_ahead(self):
        return self.s[-1] - self.ego_s

    # ------------------------------------------------------- lane geometry
    def _lane_info(self, wp):
        """(half_width, left_same_total, left_opp, right_same_total, junction, n_left_same, n_right_same)
        Walks over ALL same-direction driving lanes on each side (4/6-lane roads), then the first
        opposite-direction driving lane on the left."""
        key = wp.id
        hit = self._lane_cache.get(key)
        if hit is not None:
            return hit
        hw = 0.5 * max(2.6, wp.lane_width)
        ls = lo = rs = 0.0
        nl = nr = 0
        try:
            cur = wp
            for _ in range(4):
                nxt = cur.get_left_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving:
                    break
                if nxt.lane_id * wp.lane_id > 0:
                    ls += nxt.lane_width
                    nl += 1
                    cur = nxt
                else:
                    lo = nxt.lane_width
                    break
            cur = wp
            for _ in range(4):
                nxt = cur.get_right_lane()
                if (nxt is None or nxt.lane_type != carla.LaneType.Driving
                        or nxt.lane_id * wp.lane_id <= 0):
                    break
                rs += nxt.lane_width
                nr += 1
                cur = nxt
        except Exception:
            pass
        info = (hw, ls, lo, rs, bool(wp.is_junction), nl, nr)
        if len(self._lane_cache) > 20000:
            self._lane_cache.clear()
        self._lane_cache[key] = info
        return info

    def lane_geo(self, ds=2.0, behind=10.0, ahead=120.0):
        self._arrays()
        s0 = max(self.ss[0], self.ego_s - behind)
        n = int((min(self.ss[-1], self.ego_s + ahead) - s0) / ds) + 1
        n = max(n, 2)
        hw, ls, lo, rs, jn, kap, nls, nrs = [], [], [], [], [], [], [], []
        for k in range(n):
            s = s0 + k * ds
            i = int(np.clip(np.searchsorted(self.ss, s), 0, len(self.wps) - 1))
            h, a, b, c, j, nl, nr = self._lane_info(self.wps[i])
            hw.append(h); ls.append(a); lo.append(b); rs.append(c); jn.append(j)
            nls.append(nl); nrs.append(nr)
            i0 = max(0, i - 3)
            i1 = min(len(self.wps) - 1, i + 3)
            dyaw = abs(math.atan2(math.sin(self.yaws[i1] - self.yaws[i0]),
                                  math.cos(self.yaws[i1] - self.yaws[i0])))
            dist = max(1.0, self.ss[i1] - self.ss[i0])
            kap.append(dyaw / dist)
        return LaneGeo(s0=s0, ds=ds, half_width=hw, left_same=ls, left_opp=lo,
                       right_same=rs, junction=jn, curvature=kap,
                       left_same_n=nls, right_same_n=nrs)

    def lane_position(self):
        """'lane k of n' for our driving direction (1 = leftmost, next to the centre line)."""
        try:
            h, ls, lo, rs, j, nl, nr = self._lane_info(self.wps[self.idx])
            return nl + 1, nl + nr + 1
        except Exception:
            return 1, 1

    def junction_ahead(self, max_d=80.0):
        self._arrays()
        for i in range(self.idx, len(self.wps)):
            if self.ss[i] - self.ego_s > max_d:
                break
            if self.wps[i].is_junction:
                return self.ss[i] - self.ego_s
        return None

    def describe_ahead(self, dist=40.0):
        """Short text of the route shape ahead (for the LLM prompt)."""
        self._arrays()
        j = self.junction_ahead(60.0)
        i_end = int(np.clip(np.searchsorted(self.ss, self.ego_s + dist), 0, len(self.wps) - 1))
        dyaw = math.degrees(math.atan2(math.sin(self.yaws[i_end] - self.yaws[self.idx]),
                                       math.cos(self.yaws[i_end] - self.yaws[self.idx])))
        shape = "straight" if abs(dyaw) < 25 else ("right turn" if dyaw > 0 else "left turn")
        return shape, j


# =============================================================================
# Traffic lights
# =============================================================================
class LightIndex:
    def __init__(self, world):
        self.by_lane = {}
        self.all_stops = []          # (tl, location, yaw_rad) for the geometric fallback
        try:
            for tl in world.get_actors().filter("traffic.traffic_light*"):
                try:
                    for sw in tl.get_stop_waypoints():
                        key = (sw.road_id, sw.lane_id)
                        self.by_lane.setdefault(key, []).append((tl, sw.transform.location))
                        self.all_stops.append((tl, sw.transform.location,
                                               math.radians(sw.transform.rotation.yaw)))
                except Exception:
                    continue
        except Exception:
            pass

    @staticmethod
    def state_name(st):
        try:
            if st == carla.TrafficLightState.Red:
                return "RED"
            if st == carla.TrafficLightState.Yellow:
                return "YELLOW"
            if st == carla.TrafficLightState.Green:
                return "GREEN"
        except Exception:
            pass
        return "OFF"

    def find(self, route, max_d=80.0):
        """First traffic light whose stop line lies on our route lane ahead."""
        if not self.by_lane:
            return None, None
        route._arrays()
        seen = set()
        for i in range(max(0, route.idx - 3), len(route.wps)):
            if route.ss[i] - route.ego_s > max_d:
                break
            w = route.wps[i]
            key = (w.road_id, w.lane_id)
            if key in seen or key not in self.by_lane:
                continue
            seen.add(key)
            best = None
            for tl, loc in self.by_lane[key]:
                s, d, _, _ = route.project(loc.x, loc.y, max(0, i - 40), i + 120)
                if abs(d) < 3.0 and s > route.ego_s - 3.0 and (best is None or s < best[1]):
                    best = (tl, s)
            if best is not None:
                tl, s = best
                return LightInfo(state=self.state_name(tl.state), s_line=s), tl
        return self._find_geometric(route, max_d)

    def _find_geometric(self, route, max_d):
        """Fallback: any stop line lying across OUR lane ahead with the same heading
        (robust when the road/lane ids of the stop waypoint differ from the route's)."""
        ex, ey = route.xs[route.idx], route.ys[route.idx]
        best = None
        for tl, loc, yaw in self.all_stops:
            dx, dy = loc.x - ex, loc.y - ey
            if dx * dx + dy * dy > (max_d + 10.0) ** 2:
                continue
            s, d, ryaw, _ = route.project(loc.x, loc.y)
            if abs(d) > 1.8 or not (route.ego_s - 3.0 < s < route.ego_s + max_d):
                continue
            if math.cos(yaw - ryaw) < 0.75:
                continue
            if best is None or s < best[1]:
                best = (tl, s)
        if best is None:
            return None, None
        return LightInfo(state=self.state_name(best[0].state), s_line=best[1]), best[0]


# =============================================================================
# Sensors
# =============================================================================
class SensorRig:
    """LiDAR + front radar + IMU + GNSS + collision.  Zero camera render targets."""

    def __init__(self, world, ego):
        self.world, self.ego = world, ego
        self.sensors = []
        self._lock = threading.Lock()
        self.lidar_pts = None
        self.radar_pts = None
        self.imu = None
        self.gnss = None
        self.collisions = deque(maxlen=64)
        self.rear_gap = None
        self.mount_z = float(_c("LIDAR_MOUNT_Z_M", 2.40))
        lib = world.get_blueprint_library()

        if _c("AD_ENABLE_LIDAR", True):
            bp = lib.find("sensor.lidar.ray_cast")
            bp.set_attribute("channels", str(int(_c("LIDAR_CHANNELS", 32))))
            bp.set_attribute("range", "50.0")
            bp.set_attribute("rotation_frequency", str(int(round(1.0 / float(_c("SIM_DELTA_SECONDS", 0.05))))))
            bp.set_attribute("points_per_second", str(int(_c("LIDAR_POINTS_PER_SECOND", 50000))))
            bp.set_attribute("upper_fov", "10.0")
            bp.set_attribute("lower_fov", "-25.0")
            for attr, val in (("dropoff_general_rate", "0.0"), ("dropoff_intensity_limit", "1.0"),
                              ("dropoff_zero_intensity", "0.0")):
                if bp.has_attribute(attr):
                    bp.set_attribute(attr, val)
            s = world.spawn_actor(bp, carla.Transform(carla.Location(x=0.0, z=self.mount_z)), attach_to=ego)
            s.listen(self._on_lidar)
            self.sensors.append(s)

        if _c("AD_ENABLE_RADAR", True):
            bp = lib.find("sensor.other.radar")
            bp.set_attribute("horizontal_fov", "30")
            bp.set_attribute("vertical_fov", "8")
            bp.set_attribute("range", "80")
            bp.set_attribute("points_per_second", "1500")
            s = world.spawn_actor(bp, carla.Transform(carla.Location(x=2.4, z=0.7)), attach_to=ego)
            s.listen(self._on_radar)
            self.sensors.append(s)

        try:
            s = world.spawn_actor(lib.find("sensor.other.imu"), carla.Transform(), attach_to=ego)
            s.listen(lambda m: setattr(self, "imu", (m.accelerometer.x, m.accelerometer.y, m.gyroscope.z)))
            self.sensors.append(s)
            s = world.spawn_actor(lib.find("sensor.other.gnss"), carla.Transform(), attach_to=ego)
            s.listen(lambda m: setattr(self, "gnss", (m.latitude, m.longitude)))
            self.sensors.append(s)
        except Exception:
            pass

        s = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        s.listen(self._on_collision)
        self.sensors.append(s)

    def _on_lidar(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
        with self._lock:
            self.lidar_pts = pts

    def _on_radar(self, data):
        arr = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        # columns: velocity, azimuth, altitude, depth
        with self._lock:
            self.radar_pts = arr

    def _on_collision(self, ev):
        other = ev.other_actor
        tid = other.type_id if other is not None else "static.world"
        oid = other.id if other is not None else 0
        imp = ev.normal_impulse
        self.collisions.append((time.time(), oid, tid,
                                math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)))

    def pop_collisions(self):
        out = []
        while self.collisions:
            out.append(self.collisions.popleft())
        return out

    def destroy(self):
        for s in self.sensors:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass
        self.sensors = []

    # ------------------------------------------------------------ fusion
    def corridor_gaps(self, ego_tf, route, profile, hw_e, hl_e):
        """Front gaps [m from front bumper] measured by LiDAR and radar along the planned path."""
        with self._lock:
            lid = self.lidar_pts
            rad = self.radar_pts
        yaw = math.radians(ego_tf.rotation.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex, ey, ez = ego_tf.location.x, ego_tf.location.y, ego_tf.location.z
        # path centre line in the ego frame, 0..30 m
        xs, ys, gz = [], [], []
        for k in range(0, 32, 2):
            s = route.ego_s + k
            px, py, pz, _ = route.point(s, profile.value(s))
            dx, dy = px - ex, py - ey
            xs.append(dx * cy + dy * sy)
            ys.append(-dx * sy + dy * cy)
            gz.append(pz - ez)
        xs, ys, gz = np.array(xs), np.array(ys), np.array(gz)
        order = np.argsort(xs)
        xs, ys, gz = xs[order], ys[order], gz[order]

        def _gap(px, py, pz, z_base, min_pts):
            if px.size == 0:
                return None
            m = (px > hl_e + 0.15) & (px < 30.0)
            px, py, pz = px[m], py[m], pz[m]
            if px.size == 0:
                return None
            yc = np.interp(px, xs, ys)
            g = np.interp(px, xs, gz)
            h = pz + z_base - g
            m = (np.abs(py - yc) < hw_e + 0.15) & (h > 0.35) & (h < 2.2)
            if np.count_nonzero(m) < min_pts:
                return None
            hits = np.sort(px[m])
            # need min_pts returns within 1 m of the closest one (rejects single rain drops)
            for k in range(len(hits) - min_pts + 1):
                if hits[k + min_pts - 1] - hits[k] < 1.0:
                    return float(hits[k] - hl_e)
            return None

        self.rear_gap = None
        if lid is not None and len(lid):
            m = ((lid[:, 0] < -(hl_e + 0.15)) & (lid[:, 0] > -(hl_e + 12.0))
                 & (np.abs(lid[:, 1]) < hw_e + 0.2))
            h = lid[m, 2] + self.mount_z
            xs_r = lid[m, 0][(h > 0.35) & (h < 2.2)]
            if xs_r.size >= 4:
                self.rear_gap = float(-np.max(xs_r) - hl_e)
        lidar_gap = radar_gap = None
        if lid is not None and len(lid):
            lidar_gap = _gap(lid[:, 0], lid[:, 1], lid[:, 2], self.mount_z, 4)
        if rad is not None and len(rad):
            vel, az, alt, dep = rad[:, 0], rad[:, 1], rad[:, 2], rad[:, 3]
            px = dep * np.cos(alt) * np.cos(az) + 2.4
            py = dep * np.cos(alt) * np.sin(az)
            pz = dep * np.sin(alt)
            radar_gap = _gap(px, py, pz, 0.7, 2)
        return lidar_gap, radar_gap


# =============================================================================
# Perception / world model
# =============================================================================
class Perception:
    def __init__(self, world, ego):
        self.world = world
        self.ego = ego
        self._ext = {}
        self._kind = {}
        self._gap_hist = deque(maxlen=3)
        self.weather = "clear"
        self.friction = 1.0
        self._weather_t = -99.0
        self.fusion = {}

    # --------------------------------------------------------------- kinds
    def _classify(self, a):
        k = self._kind.get(a.id)
        if k is not None:
            return k
        tid = a.type_id
        if tid.startswith("vehicle."):
            try:
                k = "bike" if int(a.attributes.get("number_of_wheels", 4)) == 2 else "vehicle"
            except Exception:
                k = "vehicle"
        elif tid.startswith("walker.pedestrian"):
            k = "walker"
        elif tid.startswith("static.prop"):
            k = "animal" if any(x in tid for x in ANIMAL_KEYS) else "prop"
        else:
            k = ""
        self._kind[a.id] = k
        return k

    def _extent(self, a, kind):
        e = self._ext.get(a.id)
        if e is None:
            try:
                ext = a.bounding_box.extent
                e = (max(0.25, ext.x), max(0.25, ext.y))
            except Exception:
                e = (0.3, 0.3)
            if kind == "walker":
                e = (max(0.3, e[0]), max(0.3, e[1]))
            self._ext[a.id] = e
        return e

    # ------------------------------------------------------------- weather
    def update_weather(self, t):
        if t - self._weather_t < 1.0:
            return
        self._weather_t = t
        try:
            w = self.world.get_weather()
        except Exception:
            return
        night = w.sun_altitude_angle < -8.0
        dusk = -8.0 <= w.sun_altitude_angle < 2.0
        glare = 0.0 <= w.sun_altitude_angle < 12.0 and w.cloudiness < 35.0
        fog = w.fog_density > 40.0
        heavy = w.precipitation > 75.0
        rain = w.precipitation > 20.0
        wet = w.wetness > 50.0 or getattr(w, "precipitation_deposits", 0.0) > 50.0
        if night and (heavy or fog) and rain:
            name = "storm_night"
        elif night and rain:
            name = "night_rain"
        elif fog:
            name = "fog"
        elif heavy:
            name = "heavy_rain"
        elif rain:
            name = "rain"
        elif night:
            name = "night"
        elif glare:
            name = "glare"
        elif dusk:
            name = "dusk"
        elif wet:
            name = "wet"
        else:
            name = "clear"
        self.weather = name
        self.friction = {"heavy_rain": 0.7, "storm_night": 0.7, "rain": 0.8, "night_rain": 0.78,
                         "wet": 0.85}.get(name, 1.0)
        self.raw_weather = w

    # ------------------------------------------------------------- objects
    def objects(self, route, ego_tf, radius=75.0):
        ego_id = self.ego.id
        ex, ey = ego_tf.location.x, ego_tf.location.y
        out = []
        route._arrays()
        r2 = radius * radius
        for a in self.world.get_actors():
            if a.id == ego_id:
                continue
            kind = self._classify(a)
            if not kind:
                continue
            tf = a.get_transform()
            loc = tf.location
            dx, dy = loc.x - ex, loc.y - ey
            if dx * dx + dy * dy > r2:
                continue
            s, d, ryaw, _ = route.project(loc.x, loc.y)
            hl, hw = self._extent(a, kind)
            dyaw = math.radians(tf.rotation.yaw) - ryaw
            c, sn = abs(math.cos(dyaw)), abs(math.sin(dyaw))
            half_len = hl * c + hw * sn
            half_wid = hl * sn + hw * c
            if kind in ("prop", "animal"):
                vx = vy = 0.0
            else:
                v = a.get_velocity()
                vx, vy = v.x, v.y
            ux, uy = math.cos(ryaw), math.sin(ryaw)
            vs = vx * ux + vy * uy
            vd = -vx * uy + vy * ux
            speed = math.hypot(vx, vy)
            wp_j = False
            out.append(Obj(id=a.id, kind=kind, s=s, d=d, vs=vs, vd=vd,
                           half_len=half_len, half_wid=half_wid, speed=speed,
                           oncoming=math.cos(dyaw) < -0.5, in_junction=wp_j,
                           label=a.type_id))
        return out

    # -------------------------------------------------------------- fusion
    def fused_sensor_gap(self, rig, ego_tf, route, profile, hw_e, hl_e, objs):
        """Confirmed sensor-only obstacle gap (LiDAR primary, radar in fog / rain)."""
        if rig is None:
            return None
        try:
            lidar_gap, radar_gap = rig.corridor_gaps(ego_tf, route, profile, hw_e, hl_e)
        except Exception:
            lidar_gap = radar_gap = None
        weights = dict(_c("SENSOR_FUSION_WEIGHTS", {}).get(
            {"heavy_rain": "rain", "storm_night": "night_rain", "glare": "clear",
             "dusk": "night", "wet": "rain"}.get(self.weather, self.weather), {}))
        w_l = weights.get("lidar", 0.45)
        w_r = weights.get("radar", 0.35)
        cand = None
        if lidar_gap is not None and radar_gap is not None and abs(lidar_gap - radar_gap) < 3.0:
            cand = (w_l * lidar_gap + w_r * radar_gap) / max(1e-3, w_l + w_r)
        elif lidar_gap is not None:
            cand = lidar_gap
        elif radar_gap is not None and w_r >= w_l:
            cand = radar_gap
        self.fusion = {"lidar_gap": lidar_gap, "radar_gap": radar_gap,
                       "w_lidar": w_l, "w_radar": w_r}
        self._gap_hist.append(cand)
        if len(self._gap_hist) < 3 or any(g is None for g in self._gap_hist):
            return None
        if max(self._gap_hist) - min(self._gap_hist) > 3.0:
            return None
        return max(self._gap_hist)
