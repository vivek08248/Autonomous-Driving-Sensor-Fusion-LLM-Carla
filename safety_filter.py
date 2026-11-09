"""
safety_filter.py - Deterministic safety barrier (Mobileye RSS + SOTIF).
Key fixes:
  - Always 0.35 m bumper standstill (never 1.2 m early).
  - Progressive VRU creep, hard stop only < 7.5 m.
  - RSS buffer intervenes early but caps speed drop to 5 km/h / cycle.
"""
import math
import config
from rss_safety_shield import RSSSafetyShield
from autonomous_rules_engine import AutonomousRulesEngine


class SafetyFilter:
    def __init__(
        self,
        emergency_dist=1.8,
        standstill_margin_m=0.5,
        response_time_s=0.45,
        max_emergency_decel_mps2=7.0,
    ):
        self.emergency_dist = float(emergency_dist)
        self.standstill_margin_m = float(standstill_margin_m)
        self.rss = RSSSafetyShield(
            response_time_rho=response_time_s,
            standstill_margin=standstill_margin_m,
            max_emergency_brake=max_emergency_decel_mps2,
        )
        self.rules_engine = AutonomousRulesEngine()

    # ---------------- helpers ----------------
    def stopping_distances(self, current_speed_kmh):
        v = max(0.0, current_speed_kmh) / 3.6
        reaction = v * self.rss.rho
        comfortable = reaction + (v ** 2) / (2.0 * self.rss.b_min) + self.standstill_margin_m
        emergency = reaction + (v ** 2) / (2.0 * self.rss.b_max) + 0.8
        return comfortable, emergency

    # ---------------- main API ----------------
    def validate_intent(
        self, llm_decision, raw_lidar_dist_m, current_speed_kmh,
        ambiguity=None, traffic_light=None, zebra_crossing=None,
        is_in_junction=False, nearby_actors=None, weather=None,
        overtake_active=False,
    ):
        """
        overtake_active=True  ->  safety filter must NOT issue a STOP for
        red-light / zebra / static obstacles that would abort the manoeuvre.
        It may still cap the SPEED.
        """
        validated = dict(llm_decision)
        proposed_speed = float(llm_decision.get("target_speed_kmh", 20.0))
        speed_mps = max(0.0, current_speed_kmh) / 3.6

        # ---------- 1. Hard imminent-collision guard ----------
        if raw_lidar_dist_m <= self.emergency_dist and current_speed_kmh > 2.0:
            validated.update(
                intent="EMERGENCY_STOP",
                target_speed_kmh=0.0,
                override_active=True,
                override_reason=f"Emergency stop: obstacle {raw_lidar_dist_m:.1f} m",
            )
            return validated

        # ---------- 2. Realistic standstill hold (0.35 m bumper) ----------
        if raw_lidar_dist_m <= 0.35 and current_speed_kmh <= 2.0:
            validated.update(
                intent="STOP",
                target_speed_kmh=0.0,
                override_active=True,
                override_reason=f"Standstill hold at {raw_lidar_dist_m:.2f} m",
            )
            return validated

        # ---------- 3. Progressive VRU protection ----------
        if nearby_actors:
            for a in nearby_actors:
                if a.get("type") not in ("walker", "pedestrian", "animal", "bicycle"):
                    continue
                gap = float(a.get("obb_bumper_gap_m", 999.0))
                if not (a.get("in_ego_corridor") and gap < 28.0):
                    continue
                if overtake_active:
                    # Don't abort a committed lane change; just cap speed.
                    creep = math.sqrt(2.0 * 1.2 * max(0.4, gap - 2.5)) * 3.6
                    validated["intent"] = "SLOW_DOWN"
                    validated["target_speed_kmh"] = round(
                        min(proposed_speed, max(4.0, creep)), 1)
                    validated["override_active"] = True
                    validated["override_reason"] = (
                        f"VRU {a.get('type')} {gap:.1f} m (pass in progress)")
                    return validated
                if gap <= 7.5:
                    validated.update(
                        intent="EMERGENCY_STOP", target_speed_kmh=0.0,
                        override_active=True,
                        override_reason=f"VRU {a.get('type')} critical at {gap:.1f} m",
                    )
                else:
                    creep = math.sqrt(2.0 * 1.5 * max(0.4, gap - 2.5)) * 3.6
                    validated.update(
                        intent="SLOW_DOWN",
                        target_speed_kmh=round(min(proposed_speed, max(4.0, creep)), 1),
                        override_active=True,
                        override_reason=f"VRU {a.get('type')} in corridor at {gap:.1f} m â€” smooth yield",
                    )
                return validated

        # ---------- 4. Rule engine (red light, zebra, junction) ----------
        rule = self.rules_engine.evaluate_scene({
            "speed_kmh": current_speed_kmh,
            "front_obstacle_m": raw_lidar_dist_m,
            "traffic_light": traffic_light,
            "is_in_junction": is_in_junction,
            "zebra_crossing": zebra_crossing,
            "nearby_actors": nearby_actors,
            "weather": weather,
        })
        if rule.get("override_active"):
            if overtake_active and rule["intent"] in ("STOP", "EMERGENCY_STOP"):
                # Only cap speed â€” let the planner finish the pass
                validated["target_speed_kmh"] = min(
                    proposed_speed, max(6.0, rule["target_speed_kmh"]))
                validated["override_active"] = True
                validated["override_reason"] = (
                    f"{rule['reason']} â€” pass in progress, capping speed")
            elif rule["intent"] in ("STOP", "EMERGENCY_STOP"):
                validated.update(
                    intent=rule["intent"],
                    target_speed_kmh=rule["target_speed_kmh"],
                    override_active=True,
                    override_reason=rule["reason"],
                )
            else:
                if validated.get("intent") in (
                        "MANEUVER_RIGHT", "MANEUVER_LEFT", "OVERTAKE",
                        "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"):
                    validated["target_speed_kmh"] = min(
                        proposed_speed, rule["target_speed_kmh"])
                else:
                    validated["intent"] = rule["intent"]
                    validated["target_speed_kmh"] = rule["target_speed_kmh"]
                validated["override_active"] = True
                validated["override_reason"] = rule["reason"]
            return validated

        # ---------- 5. RSS headway (smooth) ----------
        lead_speed = 0.0
        if nearby_actors:
            for a in nearby_actors:
                if a.get("position") == "front" and a.get("distance_m", 999) < 35.0:
                    lead_speed = float(a.get("speed_kmh", 0.0))
                    break

        rss = self.rss.evaluate_rss_state(
            raw_lidar_dist_m, current_speed_kmh, lead_speed)
        if not rss["is_safe"] or raw_lidar_dist_m < rss["d_safe"] + 2.0:
            if lead_speed > 2.0:
                cap = max(8.0, lead_speed - 2.0)
            else:
                d = max(0.0, raw_lidar_dist_m - 2.5)
                cap = math.sqrt(2.0 * 1.3 * d) * 3.6 if d > 1.0 else 0.0
            if proposed_speed > cap:
                target = max(cap, proposed_speed - 5.0)
                if overtake_active:
                    validated["target_speed_kmh"] = round(
                        max(cap, 8.0), 1)
                else:
                    validated["intent"] = "SLOW_DOWN"
                    validated["target_speed_kmh"] = round(target, 1)
                validated["override_active"] = True
                validated["override_reason"] = (
                    f"RSS headway bound (gap {raw_lidar_dist_m:.1f} m)")
                return validated

        # ---------- 6. Global speed limit ----------
        max_speed = float(getattr(config, "MAX_ALLOWED_SPEED_KMH", 22.0))
        if proposed_speed > max_speed:
            validated["target_speed_kmh"] = max_speed
            validated["override_active"] = True
            validated["override_reason"] = f"Capped to {max_speed:.0f} km/h limit"
            return validated

        validated["override_active"] = False
        return validated