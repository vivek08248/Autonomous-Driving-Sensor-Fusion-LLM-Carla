"""
ambiguity_detector.py – Precision Conflict & Ambiguity Detector.
Eliminates false ambiguity triggers from side-parked cars or normal background traffic.
"""

import math
import carla


class AmbiguityDetector:
    def __init__(self, actor_radius=25.0, signal_radius=25.0):
        self.actor_radius = actor_radius
        self.signal_radius = signal_radius
        self._stop_cache = {}

    def _stop_locs(self, light):
        """Stop-line locations for a light (cached): ties a light to the lane it controls."""
        if light.id not in self._stop_cache:
            try:
                self._stop_cache[light.id] = [w.transform.location for w in light.get_stop_waypoints()]
            except Exception:
                self._stop_cache[light.id] = []
        return self._stop_cache[light.id]

    def assess(self, world, ego, lidar_distance, speed_kmh):
        if ego is None or not ego.is_alive:
            return self._result(False)

        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        nearby = []
        crossing = 0

        for actor in world.get_actors():
            if actor.id == ego.id or not actor.is_alive:
                continue
            if not (actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.")):
                continue

            delta = actor.get_location() - ego_location
            distance = delta.length()
            if distance > self.actor_radius:
                continue

            longitudinal = delta.x * forward.x + delta.y * forward.y
            lateral = delta.x * right.x + delta.y * right.y
            velocity = actor.get_velocity()
            actor_speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            lateral_speed = velocity.x * right.x + velocity.y * right.y

            if longitudinal > -1.0 and abs(lateral) < 3.5:
                nearby.append((distance, longitudinal, lateral, actor))

            # 1. Pedestrian actively in or stepping into ego lane (< 16m ahead)
            if actor.type_id.startswith("walker.") and 0.0 < longitudinal < 16.0:
                moving_toward_lane = (lateral * lateral_speed) < -0.15
                stationary = actor_speed_mps < 0.3
                # A stationary pedestrian counts only if it is in the lane and close
                if abs(lateral) <= 1.55 and (not stationary or longitudinal < 8.0):
                    crossing += 1
                elif abs(lateral) <= 2.8 and moving_toward_lane:
                    crossing += 1

            # 2. Vehicle Cross-Traffic / Cut-In / Head-On Conflict (< 18m ahead)
            elif actor.type_id.startswith("vehicle.") and 0.5 < longitudinal < 18.0:
                actor_fwd = actor.get_transform().get_forward_vector()
                fwd_dot = actor_fwd.x * forward.x + actor_fwd.y * forward.y
                # Only count if the vehicle is moving toward or across ego centerline
                is_converging = (lateral * lateral_speed) < -0.30 or abs(lateral) < 1.4

                # Case A: Perpendicular Cross-Traffic converging on ego path
                if abs(fwd_dot) < 0.55 and actor_speed_mps > 1.2 and is_converging:
                    if abs(lateral) <= 2.2:
                        crossing += 1
                    elif abs(lateral) < 8.5 and abs(lateral_speed) > 0.5:
                        ttc = (abs(lateral) - 1.5) / abs(lateral_speed)
                        if 0.0 < ttc <= 2.2:
                            crossing += 1

                # Case B: Aggressive Cut-In from adjacent lane
                elif 1.35 < abs(lateral) < 4.2 and (lateral * lateral_speed) < -0.45 and actor_speed_mps > 1.0:
                    ttc = (abs(lateral) - 1.35) / max(0.3, abs(lateral_speed))
                    if ttc <= 2.0:
                        crossing += 1

        reasons = []
        if crossing > 0:
            reasons.append("active crossing/cut-in conflict")

        # Conflicting signals directly ahead in ego lane
        # Only lights whose stop line lies in the ego lane can be 'conflicting'
        ahead_lights = []
        for light in world.get_actors().filter("traffic.traffic_light*"):
            for p in self._stop_locs(light):
                dx, dy = p.x - ego_location.x, p.y - ego_location.y
                along = dx * forward.x + dy * forward.y
                lat = abs(dx * right.x + dy * right.y)
                if 1.0 < along < 30.0 and lat < 2.8:
                    ahead_lights.append(light)
                    break

        if len(ahead_lights) > 1:
            states = {light.state for light in ahead_lights}
            if len(states) > 1:
                reasons.append("conflicting traffic signals")

        return {
            "active": bool(reasons),
            "reasons": reasons,
            "reason": "; ".join(reasons),
            "nearby_actor_count": len(nearby),
            "crossing_actor_count": crossing,
        }

    @staticmethod
    def _result(active):
        return {
            "active": active,
            "reasons": [],
            "reason": "",
            "nearby_actor_count": 0,
            "crossing_actor_count": 0,
        }