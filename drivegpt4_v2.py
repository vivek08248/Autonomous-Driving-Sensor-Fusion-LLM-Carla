"""
drivegpt4_v2.py - Lightweight Multi-Sensor Fusion & Control.

Calibrated for 50,000 pts/s LiDAR (RTX 3050 6GB), 4D Radar TTC, Thermal LWIR,
Night Vision, IMU, GNSS. Reduced throttle aggression to prevent the 25 km/h
overshoot seen in TC_04; smoother human-like stopping distances.
"""

import math
import numpy as np
import carla


class MultiViewSpatialTokenizer:
    """8-Sensor Spatial Tokenizer calibrated for 50,000 pts/s LiDAR."""

    @staticmethod
    def tokenize(
        point_cloud,
        ego_transform=None,
        nearby_actors=None,
        overtake_side=None,
        sensor_summary=None,
    ):
        sectors = {
            "FRONT_CENTER": 50.0,
            "FRONT_LEFT":   40.0,
            "FRONT_RIGHT":  40.0,
            "SIDE_LEFT":    15.0,
            "SIDE_RIGHT":   15.0,
            "REAR":         35.0,
        }

        encroachment = {
            "has_encroachment": False,
            "side": "NONE",
            "distance_m": 50.0,
            "encroachment_depth_m": 0.0,
            "bypass_bias_m": 0.0,
            "can_bypass": True,
        }

        tube_shift_y = 0.0
        if overtake_side == "RIGHT":
            tube_shift_y = 1.65
        elif overtake_side == "LEFT":
            tube_shift_y = -1.65

        if point_cloud is not None and len(point_cloud) > 0:
            valid_mask = (
                (point_cloud[:, 2] > -1.05)
                & (point_cloud[:, 2] < 1.45)
                & (point_cloud[:, 3] >= 0.30)
            )
            pts = point_cloud[valid_mask]
            if len(pts) > 0:
                fc_mask = (
                    (pts[:, 0] > 2.55)
                    & (pts[:, 0] < 50.0)
                    & (np.abs(pts[:, 1] - tube_shift_y) <= 0.82)
                )
                if np.sum(fc_mask) >= 2:
                    sectors["FRONT_CENTER"] = round(
                        float(np.percentile(pts[fc_mask, 0], 5) - 2.48), 2)

                fl_mask = ((pts[:, 0] > 2.55) & (pts[:, 0] < 38.0)
                           & (pts[:, 1] >= -3.6) & (pts[:, 1] < -0.95))
                if np.sum(fl_mask) >= 2:
                    sectors["FRONT_LEFT"] = round(
                        float(np.percentile(pts[fl_mask, 0], 5) - 2.48), 2)

                fr_mask = ((pts[:, 0] > 2.55) & (pts[:, 0] < 38.0)
                           & (pts[:, 1] > 0.95) & (pts[:, 1] <= 3.6))
                if np.sum(fr_mask) >= 2:
                    sectors["FRONT_RIGHT"] = round(
                        float(np.percentile(pts[fr_mask, 0], 5) - 2.48), 2)

                sl_mask = ((pts[:, 0] >= -2.8) & (pts[:, 0] <= 2.6)
                           & (pts[:, 1] >= -3.5) & (pts[:, 1] < -1.05))
                if np.sum(sl_mask) >= 2:
                    sectors["SIDE_LEFT"] = round(
                        float(np.min(np.abs(pts[sl_mask, 1])) - 1.02), 2)

                sr_mask = ((pts[:, 0] >= -2.8) & (pts[:, 0] <= 2.6)
                           & (pts[:, 1] > 1.05) & (pts[:, 1] <= 3.5))
                if np.sum(sr_mask) >= 2:
                    sectors["SIDE_RIGHT"] = round(
                        float(np.min(np.abs(pts[sr_mask, 1])) - 1.02), 2)

                rear_mask = ((pts[:, 0] < -2.55) & (pts[:, 0] >= -35.0)
                             & (np.abs(pts[:, 1]) <= 1.20))
                if np.sum(rear_mask) >= 2:
                    sectors["REAR"] = round(
                        float(np.min(np.abs(pts[rear_mask, 0])) - 2.48), 2)

        if sensor_summary:
            r_left = float(sensor_summary.get("radar_left_m", 45.0))
            r_right = float(sensor_summary.get("radar_right_m", 45.0))
            if r_left < 2.8:
                sectors["SIDE_LEFT"] = min(
                    sectors["SIDE_LEFT"], max(0.15, round(r_left - 0.85, 2)))
            if r_right < 2.8:
                sectors["SIDE_RIGHT"] = min(
                    sectors["SIDE_RIGHT"], max(0.15, round(r_right - 0.85, 2)))

            thermal_info = sensor_summary.get("thermal", {})
            nv_info = sensor_summary.get("night_vision", {})
            fog_veil = float(nv_info.get("fog_veil_index", 0.0))
            lum = float(nv_info.get("mean_luminance", 128.0))

            if overtake_side is None and (fog_veil > 0.25 or lum < 55.0):
                t_depth = float(thermal_info.get("thermal_min_depth_m", 50.0))
                if 1.5 < t_depth < sectors["FRONT_CENTER"]:
                    sectors["FRONT_CENTER"] = round(t_depth, 2)

        if nearby_actors:
            for actor in nearby_actors:
                if not isinstance(actor, dict):
                    continue
                bumper_gap = actor.get("obb_bumper_gap_m", 50.0)
                in_corridor = actor.get("in_ego_corridor", False)
                long_m = actor.get("longitudinal_m", 50.0)
                lat_m = actor.get("lateral_m", 0.0)
                abs_lat = abs(lat_m)

                if overtake_side == "RIGHT" and lat_m < 0.85 and bumper_gap > 1.8:
                    in_corridor = False
                elif overtake_side == "LEFT" and lat_m > -0.85 and bumper_gap > 1.8:
                    in_corridor = False

                if in_corridor and bumper_gap < sectors["FRONT_CENTER"]:
                    sectors["FRONT_CENTER"] = max(0.05, round(bumper_gap, 2))

                if long_m < -1.0 and abs_lat <= 1.65:
                    rear_d = max(0.05, abs(long_m) - 2.45)
                    sectors["REAR"] = min(sectors["REAR"], round(rear_d, 2))

                if -3.0 <= long_m <= 3.8:
                    if -3.8 <= lat_m < -0.95:
                        sectors["SIDE_LEFT"] = min(
                            sectors["SIDE_LEFT"],
                            max(0.05, round(abs_lat - 1.95, 2)))
                    elif 0.95 < lat_m <= 3.8:
                        sectors["SIDE_RIGHT"] = min(
                            sectors["SIDE_RIGHT"],
                            max(0.05, round(abs_lat - 1.95, 2)))

                atype = actor.get("type", "vehicle")
                if atype in ("walker", "animal", "bicycle", "pedestrian"):
                    continue

                if overtake_side is None and 0.5 < long_m < 24.0 \
                        and 0.80 <= abs_lat <= 2.35:
                    depth = round(2.40 - abs_lat, 2)
                    if depth > encroachment["encroachment_depth_m"]:
                        encroachment["has_encroachment"] = True
                        encroachment["distance_m"] = round(long_m, 1)
                        encroachment["encroachment_depth_m"] = depth
                        if lat_m < 0:
                            encroachment["side"] = "LEFT"
                            encroachment["bypass_bias_m"] = min(0.90, depth * 0.75)
                        else:
                            encroachment["side"] = "RIGHT"
                            encroachment["bypass_bias_m"] = -min(0.90, depth * 0.75)

        return {
            "sectors": sectors,
            "encroachment": encroachment,
            "min_front_m": max(0.05, sectors["FRONT_CENTER"]),
        }


class DriveGPT4DeciHead:
    def __init__(self):
        # Reduced gains to prevent 25 km/h overshoot in TC_04
        self.kp_speed = 0.28
        self.ki_speed = 0.03
        self.kd_speed = 0.04
        self.speed_err_int = 0.0
        self.prev_speed_err = 0.0
        self.prev_steer = 0.0
        self.lateral_offset_filter = 0.0
        self.ramped_target_kmh = 0.0

    def reset(self):
        self.speed_err_int = 0.0
        self.prev_speed_err = 0.0
        self.prev_steer = 0.0
        self.lateral_offset_filter = 0.0
        self.ramped_target_kmh = 0.0

    def compute_longitudinal_control(self, current_speed_kmh,
                                     raw_target_speed_kmh,
                                     front_m=50.0, dt=0.05):
        if raw_target_speed_kmh <= 0.4 and (front_m < 4.0 or current_speed_kmh < 2.2):
            self.speed_err_int = 0.0
            self.prev_speed_err = 0.0
            self.ramped_target_kmh = 0.0
            return 0.0, 0.92

        if self.ramped_target_kmh <= 0.1 and current_speed_kmh > 1.0:
            self.ramped_target_kmh = current_speed_kmh

        max_up_kmh = (2.4 * 3.6) * dt
        max_down_kmh = (3.0 * 3.6) * dt if front_m > 10.0 else (5.2 * 3.6) * dt
        diff = raw_target_speed_kmh - self.ramped_target_kmh
        if diff > 0:
            self.ramped_target_kmh += min(max_up_kmh, diff)
        else:
            self.ramped_target_kmh += max(-max_down_kmh, diff)

        target_speed_kmh = max(0.0, self.ramped_target_kmh)
        speed_err = target_speed_kmh - current_speed_kmh
        self.speed_err_int = max(-6.0, min(6.0,
                                           self.speed_err_int + speed_err * dt))
        de = (speed_err - self.prev_speed_err) / max(dt, 1e-3)
        self.prev_speed_err = speed_err

        ff = min(0.30, 0.10 + 0.006 * target_speed_kmh)
        raw_cmd = (ff + self.kp_speed * speed_err
                   + self.ki_speed * self.speed_err_int
                   + self.kd_speed * de)

        if target_speed_kmh <= 0.5:
            throttle, brake = 0.0, 0.95
        elif speed_err >= 0.0:
            throttle = max(0.10 if target_speed_kmh > 4.0 else 0.0,
                           min(0.42, raw_cmd))
            brake = 0.0
        elif speed_err >= -1.0:
            throttle = max(0.0, min(0.22, raw_cmd))
            brake = 0.0
        else:
            throttle = 0.0
            urgency = 5.0 if front_m < 10.0 else 9.0
            brake = min(0.95, max(0.18, abs(speed_err) / urgency))

        return round(throttle, 3), round(brake, 3)

    def compute_lateral_control(self, base_steer, target_lateral_bias_m=0.0,
                                wing_repulsion_steer=0.0, is_maneuvering=False,
                                imu_yaw_rate=0.0, dt=0.05):
        alpha = min(1.0, dt / 0.16 if is_maneuvering else dt / 0.22)
        self.lateral_offset_filter += alpha * (
            target_lateral_bias_m - self.lateral_offset_filter)
        steer_offset = max(-0.24, min(0.24, self.lateral_offset_filter * 0.25))

        yaw_damping = (-0.09 * imu_yaw_rate
                       if abs(imu_yaw_rate) > 0.40 else 0.0)
        target_steer = max(-0.40, min(0.40,
                                      base_steer + steer_offset
                                      + wing_repulsion_steer + yaw_damping))

        steer_rate = (1.40 if (is_maneuvering or abs(base_steer) > 0.12)
                      else 0.95)
        max_d_steer = steer_rate * dt
        d_steer = target_steer - self.prev_steer
        filtered_steer = self.prev_steer + max(-max_d_steer,
                                               min(max_d_steer, d_steer))
        self.prev_steer = filtered_steer
        return round(filtered_steer, 3)

    @staticmethod
    def extract_paper_representations(world, ego, target_speed_kmh, steer_cmd):
        target_angle_deg = round(float(steer_cmd) * 70.0, 2)
        waypoints_8 = []
        route_points_10 = []

        if ego is None or not ego.is_alive or world is None:
            return {
                "target_speed_kmh": round(float(target_speed_kmh), 2),
                "target_angle_deg": target_angle_deg,
                "waypoints_8": [(0.0, 0.0)] * 8,
                "route_points_10": [(0.0, 0.0)] * 10,
            }

        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        wmap = world.get_map()
        base_wp = wmap.get_waypoint(ego_loc, project_to_road=True)

        speed_mps = max(0.0, float(target_speed_kmh)) / 3.6
        for i in range(1, 9):
            dist_ahead = max(0.5, speed_mps * (0.5 * i))
            if base_wp is not None:
                nxt = base_wp.next(dist_ahead)
                wp = nxt[0] if nxt else base_wp
                w_loc = wp.transform.location
                dx, dy = w_loc.x - ego_loc.x, w_loc.y - ego_loc.y
                lx = round(dx * fwd.x + dy * fwd.y, 2)
                ly = round(dx * right.x + dy * right.y, 2)
                waypoints_8.append((lx, ly))
            else:
                waypoints_8.append((round(dist_ahead, 2), 0.0))

        for j in range(1, 11):
            r_dist = 3.0 * j
            if base_wp is not None:
                nxt = base_wp.next(r_dist)
                wp = nxt[0] if nxt else base_wp
                w_loc = wp.transform.location
                dx, dy = w_loc.x - ego_loc.x, w_loc.y - ego_loc.y
                rx = round(dx * fwd.x + dy * fwd.y, 2)
                ry = round(dx * right.x + dy * right.y, 2)
                route_points_10.append((rx, ry))
            else:
                route_points_10.append((round(r_dist, 2), 0.0))

        return {
            "target_speed_kmh": round(float(target_speed_kmh), 2),
            "target_angle_deg": target_angle_deg,
            "waypoints_8": waypoints_8,
            "route_points_10": route_points_10,
        }


class PrivilegedExpertSupervisor:
    def __init__(self):
        self.tokenizer = MultiViewSpatialTokenizer()
        self.decihead = DriveGPT4DeciHead()
        self.last_supervision_reason = "Nominal cruise"
        self.last_human_thought = "Scanning ahead -> clear road, cruising smoothly."
        self.last_paper_outputs = {}
        self.last_dagger_metrics = {}
        self._prev_front_m = 50.0

    def reset(self):
        self.decihead.reset()
        self.last_supervision_reason = "Reset"
        self.last_human_thought = "Resetting cognitive state."
        self.last_paper_outputs = {}
        self.last_dagger_metrics = {}
        self._prev_front_m = 50.0

    def supervise_and_control(
        self, ego, point_cloud, base_agent_ctrl, llm_decision,
        current_speed_kmh, traffic_light=None, is_in_junction=False,
        zebra_crossing=None, nearby_actors=None, raw_llm_decision=None,
        overtake_side=None, weather_info=None, ambiguity=None,
        sensor_summary=None, dt=0.05,
    ):
        if ego is None or not ego.is_alive:
            return (carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0),
                    True, "Ego not alive")

        ego_tf = ego.get_transform()
        spatial = self.tokenizer.tokenize(
            point_cloud, ego_tf, nearby_actors,
            overtake_side=overtake_side, sensor_summary=sensor_summary,
        )
        sectors = spatial["sectors"]
        encroach = spatial["encroachment"]
        front_m = sectors["FRONT_CENTER"]

        s_sum = sensor_summary or {}
        radar_ttc = float(s_sum.get("radar_front_ttc_s", 99.0))
        radar_closing = float(s_sum.get("radar_closing_mps", 0.0))
        closing_rate_mps = max(
            radar_closing,
            max(0.0, (self._prev_front_m - front_m) / max(dt, 1e-3))
            if front_m < 35.0 else 0.0,
        )
        self._prev_front_m = front_m

        llm_speed = (float(llm_decision.get("target_speed_kmh", 19.5))
                     if llm_decision else 19.5)
        target_speed = (19.5 if (front_m > 18.0 and llm_speed >= 14.0
                                  and overtake_side is None)
                        else min(21.0, llm_speed))
        target_lat_bias = 0.0
        wing_repulsion = 0.0
        supervision_override = False
        reason = "Balanced Human Cruise"
        human_thought = (f"Observe: {front_m:.1f}m clear | "
                         f"Decide: Cruise smoothly at {target_speed:.1f} km/h.")
        expert_intent = "KEEP_LANE" if overtake_side is None else "OVERTAKE"

        base_steer = (base_agent_ctrl.steer
                      if base_agent_ctrl is not None else 0.0)
        imu_yaw_rate = float(s_sum.get("imu_yaw_rate_rads", 0.0))
        imu_lat_accel = abs(float(s_sum.get("imu_lat_accel_mps2", 0.0)))

        if sectors["SIDE_LEFT"] < 0.55 and overtake_side != "LEFT":
            wing_repulsion = min(0.14, (0.60 - sectors["SIDE_LEFT"]) * 0.25)
            if base_steer < 0.0:
                base_steer = 0.0
        elif sectors["SIDE_RIGHT"] < 0.55 and overtake_side != "RIGHT":
            wing_repulsion = -min(0.14, (0.60 - sectors["SIDE_RIGHT"]) * 0.25)
            if base_steer > 0.0:
                base_steer = 0.0

        w_desc = str((weather_info or {}).get("description", "clear")).lower()
        fog_val = float((weather_info or {}).get("fog", 0.0))
        rain_val = float((weather_info or {}).get("precipitation", 0.0))
        nv_info = s_sum.get("night_vision", {})

        is_night = ("night" in w_desc or "low_sun" in w_desc
                    or float(nv_info.get("mean_luminance", 128.0)) < 55.0)
        is_heavy_fog = (fog_val >= 35.0 or "dense_fog" in w_desc
                        or float(nv_info.get("fog_veil_index", 0.0)) > 0.35)
        is_bad_weather = (is_heavy_fog or rain_val >= 20.0
                          or "rain" in w_desc or "fog" in w_desc)
        decel_limit = 4.2 if is_heavy_fog else (4.6 if is_bad_weather else 5.2)

        if is_heavy_fog:
            target_speed = min(target_speed, 16.5)
        elif is_bad_weather:
            target_speed = min(target_speed, 17.5)
        elif is_night:
            target_speed = min(target_speed, 18.5)

        if imu_lat_accel > 3.2:
            target_speed = min(target_speed, 15.0)

        if ambiguity and ambiguity.get("active") and front_m < 18.0 \
                and overtake_side is None:
            target_speed = min(target_speed, 15.5 if front_m > 8.0 else 11.5)
            supervision_override = True
            reason = (f"Ambiguous zone ({ambiguity.get('reason', 'caution')}) "
                      f"-> {target_speed:.1f} km/h")

        speed_mps = max(0.0, current_speed_kmh) / 3.6
        standstill_buf = 2.2 if is_bad_weather else 1.9
        dynamic_aeb_dist = max(
            2.2, speed_mps * 0.20
            + (speed_mps ** 2) / (2.0 * decel_limit) + standstill_buf)

        lead_speed_chk_kmh = 0.0
        if nearby_actors:
            for _a in nearby_actors:
                if isinstance(_a, dict) and _a.get("position") == "front" \
                        and _a.get("in_ego_corridor"):
                    lead_speed_chk_kmh = float(_a.get("speed_kmh", 0.0))
                    break
        closing_mps_chk = max(0.0,
                              (current_speed_kmh - lead_speed_chk_kmh) / 3.6)

        if (front_m <= dynamic_aeb_dist and current_speed_kmh > 1.2
                and (closing_mps_chk > 1.6 or front_m <= 2.8)):
            self.last_supervision_reason = (
                f"AEB Emergency Stop: Obstacle at {front_m:.1f}m")
            steer_cmd = self.decihead.compute_lateral_control(
                base_steer, 0.0, wing_repulsion,
                is_maneuvering=(overtake_side is not None),
                imu_yaw_rate=imu_yaw_rate, dt=dt)
            ctrl = carla.VehicleControl(throttle=0.0, brake=1.0,
                                        steer=steer_cmd)
            self._record_dagger(ego, raw_llm_decision, 0.0, steer_cmd,
                                "EMERGENCY_STOP", front_m)
            return ctrl, True, self.last_supervision_reason
        elif front_m <= standstill_buf:
            self.last_supervision_reason = (
                f"Standstill Safety Hold: {front_m:.1f}m gap")
            steer_cmd = self.decihead.compute_lateral_control(
                base_steer, 0.0, wing_repulsion,
                is_maneuvering=(overtake_side is not None),
                imu_yaw_rate=imu_yaw_rate, dt=dt)
            ctrl = carla.VehicleControl(throttle=0.0, brake=0.95,
                                        steer=steer_cmd)
            self._record_dagger(ego, raw_llm_decision, 0.0, steer_cmd,
                                "STOP", front_m)
            return ctrl, True, self.last_supervision_reason

        # Guarded Traffic Light & Zebra Crossing check
        stop_line_gap_m = None
        stop_source = ""
        stop_hold_dist = 0.35
        if zebra_crossing and zebra_crossing.get("detected") \
                and zebra_crossing.get("occupied"):
            z_gap = float(zebra_crossing.get("stop_line_gap_m",
                                             max(0.0, float(
                                                 zebra_crossing.get(
                                                     "distance", 999.0)) - 2.48)))
            stop_line_gap_m = z_gap
            stop_source = "Zebra Crossing"

        if traffic_light and not is_in_junction:
            tl_state, tl_dist = traffic_light
            state_str = str(tl_state).upper()
            if ("RED" in state_str or "YELLOW" in state_str) \
                    and float(tl_dist) >= 1.85:
                tl_bumper_gap = max(0.0, float(tl_dist) - 2.48)
                if stop_line_gap_m is None or tl_bumper_gap < stop_line_gap_m:
                    stop_line_gap_m = tl_bumper_gap
                    stop_source = f"{state_str} Light"

        if ambiguity and not is_in_junction \
                and "conflicting traffic signals" in str(ambiguity.get("reason", "")):
            if traffic_light and float(traffic_light[1]) < 18.0:
                stop_line_gap_m = max(0.0, float(traffic_light[1]) - 2.48)
                stop_source = "Conflicting Signals"

        if overtake_side is None and stop_line_gap_m is not None:
            if stop_line_gap_m <= stop_hold_dist:
                target_speed = 0.0
                expert_intent = "STOP"
                supervision_override = True
                reason = f"Stopped at {stop_source} ({stop_line_gap_m:.2f}m gap)"
            elif stop_line_gap_m <= 20.0:
                glide_mps = math.sqrt(2.0 * 1.50 * max(
                    0.05, stop_line_gap_m - stop_hold_dist))
                target_speed = min(target_speed, max(4.0, round(
                    glide_mps * 3.6, 1)))
                expert_intent = "SLOW_DOWN"
                supervision_override = True
                reason = (f"Gliding to {stop_source} "
                          f"({stop_line_gap_m:.1f}m @ {target_speed:.1f} km/h)")
        elif overtake_side is not None and stop_line_gap_m is not None:
            # Do NOT abort a committed lane change for a red light
            target_speed = min(target_speed, 12.0)
            supervision_override = True
            reason = (f"Pass in progress - red-light speed cap "
                      f"({stop_line_gap_m:.1f}m)")

        if overtake_side is None and encroach["has_encroachment"] \
                and encroach["distance_m"] < 22.0:
            target_lat_bias = encroach["bypass_bias_m"]
            target_speed = min(target_speed, 15.5)
            supervision_override = True
            reason = (f"Corridor slalom {encroach['side']} "
                      f"(offset {target_lat_bias:+.2f}m)")

        if overtake_side is None and front_m < 32.0 \
                and (stop_line_gap_m is None or front_m < stop_line_gap_m):
            d_buffer = max(0.0, front_m - (3.5 if is_bad_weather else 3.2))
            rss_decel = 1.15 if is_heavy_fog else (
                1.25 if is_bad_weather else 1.35)
            rss_bound_kmh = (0.0 if front_m <= 3.5 else
                             math.sqrt(2.0 * rss_decel * d_buffer) * 3.6)
            lead_speed_kmh = None
            if nearby_actors:
                for a in nearby_actors:
                    if isinstance(a, dict) and a.get("in_ego_corridor"):
                        if a.get("type") in ("walker", "animal", "bicycle"):
                            rss_bound_kmh = (0.0 if front_m < 20.0
                                             else min(rss_bound_kmh, 3.5))
                        elif a.get("type") == "vehicle":
                            lead_speed_kmh = a.get("speed_kmh", 0.0)
                        break

            if (closing_rate_mps > 3.5 or radar_ttc < 2.8) and front_m < 24.0:
                rss_bound_kmh = min(rss_bound_kmh,
                                    max(0.0, (lead_speed_kmh or 0.0) - 2.0))

            if lead_speed_kmh is not None and lead_speed_kmh > 2.0 \
                    and front_m > 6.0:
                acc_speed = min(target_speed,
                                max(lead_speed_kmh, rss_bound_kmh))
            else:
                acc_speed = rss_bound_kmh

            if target_speed > acc_speed:
                target_speed = round(acc_speed, 1)
                expert_intent = "SLOW_DOWN" if target_speed > 1.0 else "STOP"
                supervision_override = True
                reason = (f"RSS headway bound ({front_m:.1f}m gap "
                          f"@ {target_speed:.1f} km/h)")

        throttle, brake = self.decihead.compute_longitudinal_control(
            current_speed_kmh, target_speed, front_m=front_m, dt=dt)
        steer = self.decihead.compute_lateral_control(
            base_steer, target_lat_bias, wing_repulsion,
            is_maneuvering=(overtake_side is not None),
            imu_yaw_rate=imu_yaw_rate, dt=dt)

        if overtake_side is None and target_speed <= 0.4 \
                and current_speed_kmh < 2.0:
            throttle, brake = 0.0, 0.92

        self._record_dagger(ego, raw_llm_decision, target_speed, steer,
                            expert_intent, front_m)
        self.last_supervision_reason = reason
        self.last_human_thought = human_thought

        ctrl = carla.VehicleControl(throttle=throttle, brake=brake,
                                    steer=steer, hand_brake=False)
        return ctrl, supervision_override, reason

    def _record_dagger(self, ego, raw_llm_decision, expert_speed_kmh,
                       expert_steer, expert_intent, front_m):
        world = ego.get_world() if (ego and ego.is_alive) else None
        self.last_paper_outputs = self.decihead.extract_paper_representations(
            world, ego, expert_speed_kmh, expert_steer)

        llm_speed = (float(raw_llm_decision.get("target_speed_kmh",
                                                expert_speed_kmh))
                     if isinstance(raw_llm_decision, dict) else expert_speed_kmh)
        llm_intent = (str(raw_llm_decision.get("intent", "KEEP_LANE")).upper()
                      if isinstance(raw_llm_decision, dict) else "KEEP_LANE")

        speed_l1_err = round(abs(llm_speed - expert_speed_kmh), 2)
        dagger_takeover = (speed_l1_err > 6.0
                           or (expert_intent == "STOP" and llm_speed > 5.0))

        failure_mode = "NONE"
        if raw_llm_decision is None:
            failure_mode = "LLM_NOT_READY"
        elif llm_speed <= 0.5 and expert_speed_kmh >= 12.0 and front_m > 12.0:
            failure_mode = "FAIL_FALSE_STOP"
        elif llm_speed > expert_speed_kmh + 8.0 and front_m < 10.0:
            failure_mode = "FAIL_UNSAFE_SPEED"

        self.last_dagger_metrics = {
            "llm_intent": llm_intent,
            "llm_speed_kmh": round(llm_speed, 1),
            "expert_intent": expert_intent,
            "expert_speed_kmh": round(expert_speed_kmh, 1),
            "speed_l1_error": speed_l1_err,
            "dagger_takeover": dagger_takeover,
            "failure_mode": failure_mode,
        }