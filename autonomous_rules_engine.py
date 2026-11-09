"""
autonomous_rules_engine.py - Research-Grade Autonomous Driving Rules Engine.

References:
- ISO 21448:2022 (SOTIF)
- UNECE Regulation No. 157: ALKS
- Euro NCAP Assessment Protocol - VRU Protection (2023+)
- NHTSA DOT HS 812 748
- Mobileye RSS & NVIDIA SFF

KEY PRINCIPLES:
1. NEVER use opposite lane (B1) for overtaking - ALKS compliant
2. VRU protection is absolute priority
3. Human-like stopping distances (0.35 m)
4. Smooth deceleration curves
"""

import math
from rss_safety_shield import RSSSafetyShield


class AutonomousRulesEngine:
    def __init__(self):
        self.rss = RSSSafetyShield()

    def evaluate_scene(self, context):
        speed_kmh = float(context.get("speed_kmh", 0.0))
        front_m = float(context.get("front_obstacle_m", 50.0))
        tl = context.get("traffic_light")
        is_in_junction = bool(context.get("is_in_junction", False))
        zebra = context.get("zebra_crossing") or {}
        actors = context.get("nearby_actors") or context.get("actors") or []
        weather = context.get("weather") or {}

        matched_rules = []
        action_intent = "KEEP_LANE"
        target_speed = min(speed_kmh, 25.0) if speed_kmh > 5 else 22.0
        override_needed = False
        primary_reason = ""

        # =====================================================================
        # CATEGORY 1: VRU PROTECTION - ABSOLUTE PRIORITY
        # =====================================================================
        if zebra.get("detected") and zebra.get("occupied"):
            stop_gap = float(zebra.get("stop_line_gap_m",
                                        zebra.get("distance", 999.0)))
            if stop_gap <= 25.0:
                matched_rules.append("VRU_ZEBRA_CROSSING_OCCUPIED_YIELD")
                if stop_gap <= 0.35:
                    action_intent = "STOP"
                    target_speed = 0.0
                    primary_reason = f"Holding at zebra stop-line ({stop_gap:.2f} m)"
                else:
                    creep = math.sqrt(2.0 * 1.5 * max(0.4, stop_gap - 0.35)) * 3.6
                    action_intent = "SLOW_DOWN"
                    target_speed = round(min(22.0, max(4.0, creep)), 1)
                    primary_reason = (f"Gliding to zebra stop-line "
                                      f"({stop_gap:.1f} m @ {target_speed:.1f} km/h)")
                override_needed = True
                return self._build_result(
                    action_intent, target_speed, override_needed,
                    primary_reason, matched_rules)

        # Rule 1.2: VRU in travel corridor - progressive human-like yield.
        vru_gap = 999.0
        vru_actor = None
        for a in actors:
            if a.get("type") in ("walker", "pedestrian", "animal", "bicycle"):
                gap = float(a.get("obb_bumper_gap_m",
                                  a.get("distance_m", 999.0)))
                if a.get("in_ego_corridor") and gap < 28.0 and gap < vru_gap:
                    vru_gap = gap
                    vru_actor = a
        if vru_actor is not None:
            if vru_gap <= 7.5:
                matched_rules.append("VRU_COLLISION_AVOIDANCE_IN_PATH")
                action_intent = "EMERGENCY_STOP"
                target_speed = 0.0
                primary_reason = (f"VRU ({vru_actor.get('type')}) "
                                  f"critical at {vru_gap:.1f}m")
            else:
                matched_rules.append("VRU_SMOOTH_YIELD_IN_PATH")
                creep = math.sqrt(2.0 * 1.5 * max(0.4, vru_gap - 2.5)) * 3.6
                action_intent = "SLOW_DOWN"
                target_speed = round(min(target_speed, max(4.0, creep)), 1)
                primary_reason = (f"VRU ({vru_actor.get('type')}) in corridor "
                                  f"at {vru_gap:.1f}m - smooth yield")
            override_needed = True
            return self._build_result(action_intent, target_speed,
                                      override_needed, primary_reason,
                                      matched_rules)

        # Rule 1.3: VRU approaching corridor laterally
        for a in actors:
            if a.get("type") in ("walker", "pedestrian", "animal", "bicycle"):
                gap = float(a.get("obb_bumper_gap_m",
                                  a.get("distance_m", 999.0)))
                lat = abs(a.get("lateral_m", 0.0))
                if gap < 15.0 and lat < 4.5:
                    v_lat = a.get("lateral_speed", 0.0)
                    if v_lat * a.get("lateral_m", 0.0) < -0.15 and abs(v_lat) > 0.25:
                        matched_rules.append("VRU_LATERAL_CONVERGENCE")
                        action_intent = "STOP" if gap < 10.0 else "SLOW_DOWN"
                        target_speed = (0.0 if gap < 10.0 else
                                        round(min(target_speed, max(
                                            4.0, math.sqrt(2.0 * 1.5 * max(
                                                0.4, gap - 2.5)) * 3.6)), 1))
                        override_needed = True
                        primary_reason = (f"VRU ({a.get('type')}) converging "
                                          f"into path ({gap:.1f}m)")
                        return self._build_result(action_intent, target_speed,
                                                  override_needed,
                                                  primary_reason,
                                                  matched_rules)

        # Rule 1.4: Cross-traffic / intersecting vehicle yielding
        for a in actors:
            if a.get("type") in ("vehicle", "car") and a.get("distance_m", 999.0) < 20.0:
                pos = a.get("position", "")
                if (pos in ("front", "crossing", "lateral") and a.get("crossing")) \
                        or a.get("ttc_s", 999.0) <= 2.5:
                    matched_rules.append("CROSS_TRAFFIC_INTERSECTION_YIELD")
                    action_intent = "STOP"
                    target_speed = 0.0
                    override_needed = True
                    primary_reason = (f"Yielding to crossing vehicle "
                                      f"({a['distance_m']:.1f}m)")
                    return self._build_result(action_intent, target_speed,
                                              override_needed,
                                              primary_reason, matched_rules)

        # =====================================================================
        # CATEGORY 2: INTERSECTION & TRAFFIC SIGNAL
        # =====================================================================
        if is_in_junction:
            matched_rules.append("JUNCTION_INTERSECTION_CLEARANCE_ACTIVE")
            target_speed = max(15.0, min(speed_kmh, 22.0))
            action_intent = "KEEP_LANE"
            primary_reason = "Clearing intersection"
            return self._build_result(action_intent, target_speed, False,
                                      primary_reason, matched_rules)

        if tl and not is_in_junction:
            state, dist = tl
            if state in ("RED", "YELLOW") and dist > 1.5:
                buffer_dist = max(0.0, float(dist) - 2.48)
                if buffer_dist <= 0.35:
                    matched_rules.append("SIGNAL_RED_LIGHT_HOLD_AT_LINE")
                    action_intent = "STOP"
                    target_speed = 0.0
                    override_needed = True
                    primary_reason = (f"Holding at stop line for {state} light "
                                      f"({buffer_dist:.2f}m)")
                    return self._build_result(action_intent, target_speed,
                                              override_needed,
                                              primary_reason, matched_rules)
                elif buffer_dist <= 30.0:
                    matched_rules.append("SIGNAL_RED_LIGHT_SMOOTH_APPROACH")
                    action_intent = "SLOW_DOWN"
                    comfort_mps = math.sqrt(2.0 * 1.5 * max(0.5, buffer_dist - 0.35))
                    target_speed = round(min(22.0, max(4.0, comfort_mps * 3.6)), 1)
                    override_needed = True
                    primary_reason = (f"Smooth decel to stop line "
                                      f"({buffer_dist:.1f}m left)")
                    return self._build_result(action_intent, target_speed,
                                              override_needed,
                                              primary_reason, matched_rules)

        # =====================================================================
        # CATEGORY 3: MOBILEYE RSS LONGITUDINAL SAFETY SHIELD
        # =====================================================================
        friction = 0.7 if weather.get("precipitation", 0) > 40 else 1.0
        lead_speed = 0.0
        for a in actors:
            if a.get("position") == "front" and a.get("distance_m", 999) < 25.0:
                lead_speed = a.get("speed_kmh", 0.0)
                break

        rss_eval = self.rss.evaluate_rss_state(
            front_m, speed_kmh, lead_speed, friction)
        if not rss_eval["is_safe"]:
            matched_rules.append(f"RSS_VIOLATION_{rss_eval['severity']}")
            if rss_eval["severity"] in ("EMERGENCY", "STANDSTILL"):
                action_intent = "STOP" if front_m <= 3.0 else "EMERGENCY_STOP"
                target_speed = 0.0
                override_needed = True
                primary_reason = (f"RSS safety distance violated "
                                  f"(gap {front_m:.1f}m < d_safe {rss_eval['d_safe']}m)")
                return self._build_result(action_intent, target_speed,
                                          override_needed,
                                          primary_reason, matched_rules)
            elif rss_eval["severity"] in ("WARNING", "CAUTION"):
                action_intent = "SLOW_DOWN"
                if lead_speed > 0:
                    target_speed = max(5.0, lead_speed - 1.5)
                else:
                    d_buf = max(0.0, front_m - 2.5)
                    target_speed = (round(math.sqrt(2.0 * 1.5 * d_buf) * 3.6, 1)
                                    if d_buf > 0.3 else 0.0)
                override_needed = True
                primary_reason = f"RSS headway recovery ({rss_eval['severity']})"

        # =====================================================================
        # CATEGORY 4: ADVERSE WEATHER & SOTIF
        # =====================================================================
        precip = weather.get("precipitation", 0.0)
        fog = weather.get("fog", 0.0)
        if precip > 50.0 or fog > 40.0:
            matched_rules.append("SOTIF_ADVERSE_WEATHER_SPEED_RESTRICTION")
            target_speed = min(target_speed, 16.0)
            if action_intent == "KEEP_LANE":
                action_intent = "SLOW_DOWN"
                primary_reason = "Weather sight distance restriction"

        # =====================================================================
        # CATEGORY 5: CLEAR OPEN CRUISE
        # =====================================================================
        if front_m > 45.0 and (not tl or tl[0] == "GREEN") \
                and action_intent == "KEEP_LANE":
            matched_rules.append("CRUISE_EFFICIENCY_OPTIMAL")
            action_intent = "ACCELERATE" if speed_kmh < 20.0 else "KEEP_LANE"
            target_speed = 24.0
            primary_reason = "Clear corridor open ahead"

        # =====================================================================
        # CATEGORY 6: STATIC OBSTACLE HANDLING
        # =====================================================================
        for a in actors:
            if a.get("type") == "vehicle" and a.get("speed_kmh", 0.0) < 1.0 \
                    and a.get("in_ego_corridor"):
                if a.get("obb_bumper_gap_m", 999) < 25.0:
                    matched_rules.append("STATIC_OBSTACLE_DETECTED")
                    if action_intent == "KEEP_LANE":
                        action_intent = "SLOW_DOWN"
                    target_speed = min(target_speed, 15.0)
                    primary_reason = (f"Static obstacle ahead "
                                      f"({a['obb_bumper_gap_m']:.1f}m)")

        return self._build_result(action_intent, target_speed,
                                  override_needed, primary_reason, matched_rules)

    def _build_result(self, intent, speed, override, reason, rules):
        return {
            "intent": intent,
            "target_speed_kmh": round(speed, 1),
            "override_active": override,
            "reason": reason,
            "matched_rules": rules,
        }