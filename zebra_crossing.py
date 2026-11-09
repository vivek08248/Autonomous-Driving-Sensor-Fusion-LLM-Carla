"""
zebra_crossing.py - Robust zebra-crossing / crosswalk yielding.

Fixes:
- Returns distance to the NEAREST EDGE of the crossing (not the polygon centre).
- Returns 'stop_line_gap_m' (bumper-to-stop-line) using ego front overhang,
  the 1.5 m pre-crossing buffer and the actual polygon geometry.
- Returns 'brake_demand' driven by bumper-to-stop-line, never by centre distance.
"""
import math
import carla
from shapely.geometry import Polygon, Point

EGO_FRONT_OVERHANG_M = 2.48   # Tesla Model 3 centre -> front bumper
STOP_LINE_BUFFER_M  = 1.5     # white line sits this far before the near edge


class ZebraCrossingWatch:
    def __init__(self, world):
        self.world = world
        self.crosswalks = []
        self._load_crosswalks()

    def _load_crosswalks(self):
        try:
            cw_points = self.world.get_map().get_crosswalks()
            if not cw_points:
                return
            for i in range(0, len(cw_points), 5):
                pts = cw_points[i:i + 4]
                if len(pts) < 3:
                    continue
                coords = [(p.x, p.y) for p in pts]
                try:
                    poly = Polygon(coords)
                    cx = sum(p.x for p in pts) / float(len(pts))
                    cy = sum(p.y for p in pts) / float(len(pts))
                    cz = sum(p.z for p in pts) / float(len(pts))
                    self.crosswalks.append({
                        "poly": poly,
                        "center": carla.Location(x=cx, y=cy, z=cz),
                        "pts": pts,
                        "corners_xy": coords,
                    })
                except Exception:
                    pass
            print(f"[ZebraCrossing] Loaded {len(self.crosswalks)} zebra crossings.")
        except Exception as e:
            print(f"[ZebraCrossing] Error loading crosswalks: {e}")

    def assess(self, ego, walkers=None, max_distance=35.0, current_speed_kmh=0.0):
        if ego is None or not ego.is_alive or not self.crosswalks:
            return self._empty()

        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        speed_mps = max(0.0, current_speed_kmh) / 3.6

        # --- Nearest crossing by NEAREST CORNER (near edge) ---
        best_cw = None
        best_near_along = float("inf")
        best_near_lat = float("inf")

        for cw in self.crosswalks:
            # First rough check: crossing centre must be ahead / near lane
            c = cw["center"]
            dxc = c.x - ego_loc.x
            dyc = c.y - ego_loc.y
            c_along = dxc * fwd.x + dyc * fwd.y
            c_lat = abs(dxc * right.x + dyc * right.y)
            if c_along < -3.0 or c_along > max_distance + 8.0 or c_lat > 6.0:
                continue

            # Precise check: project each corner; take the nearest one ahead
            for (px, py) in cw["corners_xy"]:
                dx = px - ego_loc.x
                dy = py - ego_loc.y
                along = dx * fwd.x + dy * fwd.y
                lat = abs(dx * right.x + dy * right.y)
                if along <= 0.3 or lat > 6.0:
                    continue
                if along < best_near_along:
                    best_near_along = along
                    best_near_lat = lat
                    best_cw = cw

        if best_cw is None:
            return self._empty()

        # Distance (ego centre) to near edge of crossing
        near_edge_along = best_near_along

        # Bumper -> stop-line (white line 1.5 m before near edge)
        bumper_to_stop_line = max(0.0, near_edge_along - STOP_LINE_BUFFER_M - EGO_FRONT_OVERHANG_M)

        # --- Occupancy ---
        poly = best_cw["poly"]
        pedestrian_count = 0
        if walkers:
            for w in walkers:
                if not w.is_alive:
                    continue
                wl = w.get_location()
                pt = Point(wl.x, wl.y)
                try:
                    dist_to_cw = poly.distance(pt)
                except Exception:
                    continue
                if dist_to_cw < 2.5:
                    pedestrian_count += 1

        occupied = pedestrian_count > 0

        # --- Brake demand based on bumper-to-stop-line ---
        brake_demand = 0.0
        if occupied:
            if speed_mps < 0.3:
                brake_demand = 0.80
            elif bumper_to_stop_line < 15.0:
                # Comfortable 1.5 m/s^2 deceleration, stop 0.35 m before white line
                d = max(0.2, bumper_to_stop_line - 0.35)
                required = (speed_mps ** 2) / (2.0 * d)
                brake_demand = max(0.20, min(0.92, required / 3.0))
            status_text = f"ZEBRA: PEDESTRIANS ({pedestrian_count}) @ {bumper_to_stop_line:.1f}m to line"
        else:
            status_text = f"ZEBRA: CLEAR @ {bumper_to_stop_line:.1f}m to line"

        return {
            "detected": True,
            "distance": near_edge_along,             # ego-centre -> near edge (informational)
            "stop_line_gap_m": bumper_to_stop_line,  # <-- authoritative bumper-to-line
            "occupied": occupied,
            "pedestrian_count": pedestrian_count,
            "crossing_obj": best_cw,
            "brake_demand": brake_demand,
            "status_text": status_text,
        }

    @staticmethod
    def _empty():
        return {
            "detected": False,
            "occupied": False,
            "distance": 999.0,
            "stop_line_gap_m": 999.0,
            "pedestrian_count": 0,
            "brake_demand": 0.0,
            "status_text": "",
        }

    def draw_debug(self, crossing_obj, occupied=False, life_time=0.1):
        if not crossing_obj or "pts" not in crossing_obj:
            return
        pts = crossing_obj["pts"]
        color = carla.Color(255, 30, 30) if occupied else carla.Color(30, 240, 100)
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            self.world.debug.draw_line(
                carla.Location(p1.x, p1.y, p1.z + 0.15),
                carla.Location(p2.x, p2.y, p2.z + 0.15),
                thickness=0.12, color=color, life_time=life_time,
            )