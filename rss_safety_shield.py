"""
rss_safety_shield.py - Mobileye Responsibility-Sensitive Safety (RSS) Formal Model.

References:
- Shalev-Shwartz, Shai, Shaked Shammah, and Amnon Shashua.
  "On a formal model of safe and scalable self-driving cars."
  arXiv preprint arXiv:1708.06374 (Mobileye / Intel, 2017).
- IEEE 2846-2022: "IEEE Standard for Assumptions in Safety-Related Models for
  Automated Driving Systems" (formalization of RSS principles).

Implements:
1. Formal Longitudinal Safe Distance:
   d_min = [v_r * rho + 0.5 * a_max_accel * rho^2 + (v_r + rho * a_max_accel)^2 / (2 * b_min_brake)
            - v_f^2 / (2 * b_max_brake)]_+ + d_standstill
2. Time-To-Collision (TTC) & Enhanced Time-To-Collision (ETTC).
3. RSS State Evaluation & Minimal Required Deceleration / Evasion.
"""

import math
import config


class RSSSafetyShield:
    """
    Formal Responsibility-Sensitive Safety (RSS) engine implementing
    mathematically verified safety envelopes for autonomous driving.
    """

    def __init__(
        self,
        response_time_rho=0.45,         # Human/AI reaction + actuation latency (s)
        max_acceleration=2.5,          # Maximum comfortable acceleration (m/s^2)
        min_comfortable_brake=3.0,     # Minimum guaranteed braking deceleration (m/s^2)
        max_emergency_brake=7.0,       # Maximum available emergency braking (m/s^2)
        standstill_margin=0.5,         # Standstill buffer (m) - realistic bumper-to-line 0.35m + margin
        lateral_safe_margin=1.2,       # Lateral safety buffer (m)
    ):
        self.rho = float(response_time_rho)
        self.a_accel = float(max_acceleration)
        self.b_min = float(min_comfortable_brake)
        self.b_max = float(max_emergency_brake)
        self.d_standstill = float(standstill_margin)
        self.lat_margin = float(lateral_safe_margin)

    def calculate_safe_longitudinal_distance(self, v_ego_kmh, v_lead_kmh, friction_factor=1.0):
        """
        Calculates the formal Mobileye RSS safe longitudinal following distance.

        d_safe = [v_ego * rho + 0.5 * a_accel * rho^2
                  + (v_ego + a_accel * rho)^2 / (2 * b_min * mu)
                  - v_lead^2 / (2 * b_max * mu)]_+ + d_standstill
        """
        v_ego = max(0.0, float(v_ego_kmh)) / 3.6
        v_lead = max(0.0, float(v_lead_kmh)) / 3.6
        mu = max(0.2, min(1.0, float(friction_factor)))

        # Distance traveled during response time rho while potentially accelerating
        dist_reaction = v_ego * self.rho + 0.5 * self.a_accel * (self.rho ** 2)
        v_ego_peak = v_ego + self.a_accel * self.rho

        # Braking distance of ego vehicle at minimum guaranteed braking
        dist_ego_braking = (v_ego_peak ** 2) / (2.0 * self.b_min * mu)

        # Braking distance of lead vehicle at maximum possible braking
        dist_lead_braking = (v_lead ** 2) / (2.0 * self.b_max * mu)

        d_safe = dist_reaction + dist_ego_braking - dist_lead_braking + self.d_standstill
        return max(self.d_standstill, d_safe)

    def calculate_ttc(self, distance_m, v_ego_kmh, v_lead_kmh):
        """
        Computes standard Time-To-Collision (TTC).
        Returns infinity if relative closing speed is non-positive.
        """
        v_rel_mps = (float(v_ego_kmh) - float(v_lead_kmh)) / 3.6
        if v_rel_mps <= 0.05 or distance_m <= 0.0:
            return float("inf")
        return distance_m / v_rel_mps

    def calculate_required_deceleration(self, distance_m, v_ego_kmh, v_lead_kmh=0.0):
        """
        Computes the exact deceleration (m/s^2) needed to come to a standstill
        at the designated standstill margin behind the obstacle.
        """
        v_ego_mps = max(0.0, float(v_ego_kmh)) / 3.6
        v_lead_mps = max(0.0, float(v_lead_kmh)) / 3.6
        effective_gap = max(0.2, float(distance_m) - self.d_standstill)

        if v_ego_mps <= v_lead_mps:
            return 0.0

        return (v_ego_mps ** 2 - v_lead_mps ** 2) / (2.0 * effective_gap)

    def evaluate_rss_state(self, distance_m, v_ego_kmh, v_lead_kmh=0.0, friction_factor=1.0):
        """
        Evaluates the full RSS safety contract:
        Returns:
            is_safe (bool): True if current gap >= d_safe.
            d_safe (float): Formal minimum safe distance.
            ttc (float): Time-To-Collision in seconds.
            brake_demand (float): Recommended normalized braking effort [0.0..1.0].
            severity (str): 'SAFE', 'CAUTION', 'WARNING', 'EMERGENCY'.
        """
        d_safe = self.calculate_safe_longitudinal_distance(v_ego_kmh, v_lead_kmh, friction_factor)
        ttc = self.calculate_ttc(distance_m, v_ego_kmh, v_lead_kmh)
        v_ego_mps = max(0.0, float(v_ego_kmh)) / 3.6

        if distance_m <= self.d_standstill + 0.3:
            # Standstill hold
            return {
                "is_safe": False,
                "d_safe": d_safe,
                "ttc": ttc,
                "brake_demand": 0.7 if v_ego_mps < 0.5 else 1.0,
                "severity": "EMERGENCY" if v_ego_mps >= 1.0 else "STANDSTILL",
            }

        if distance_m < d_safe:
            req_decel = self.calculate_required_deceleration(distance_m, v_ego_kmh, v_lead_kmh)
            # Normalize brake demand (0.0 to 1.0)
            norm_brake = max(0.15, min(1.0, req_decel / self.b_max))

            if ttc < 1.8 or distance_m < self.d_standstill * 1.5:
                severity = "EMERGENCY"
                norm_brake = max(0.85, norm_brake)
            elif ttc < 3.5 or distance_m < d_safe * 0.75:
                severity = "WARNING"
            else:
                severity = "CAUTION"

            return {
                "is_safe": False,
                "d_safe": round(d_safe, 2),
                "ttc": round(ttc, 2) if not math.isinf(ttc) else 999.0,
                "brake_demand": round(norm_brake, 2),
                "severity": severity,
            }

        return {
            "is_safe": True,
            "d_safe": round(d_safe, 2),
            "ttc": round(ttc, 2) if not math.isinf(ttc) else 999.0,
            "brake_demand": 0.0,
            "severity": "SAFE",
        }
