"""
prompt_builder.py – Full Multi-Sensor Analytical & Cognitive Chain-of-Thought Prompt Builder.
Enhanced with proactive decision framework, traffic rule compliance, collision avoidance,
road condition detection, lane confinement logic, and road surface speed adaptation.
"""

import json


def generate_multisensor_prompt(
    speed_kmh,
    lidar_front_dist_m,
    navigation_goal,
    nearby_actors=None,
    traffic_light=None,
    weather_info=None,
    spatial_sectors=None,
    sensor_summary=None,
    human_thought="",
    **kwargs,
):
    w_desc = (weather_info or {}).get("description", "clear")
    sectors = spatial_sectors or {}
    s_sum = sensor_summary or {}

    tl_str = f"{traffic_light[0]} at {traffic_light[1]:.1f}m" if traffic_light else "CLEAR/GREEN"
    radar_f = s_sum.get("radar_front_m", lidar_front_dist_m)
    radar_ttc = s_sum.get("radar_front_ttc_s", 99.0)
    radar_v = s_sum.get("radar_closing_mps", 0.0)

    th = s_sum.get("thermal", {})
    nv = s_sum.get("night_vision", {})
    lid = s_sum.get("lidar_stats", {})
    fw = s_sum.get("fusion_weights", {"w_lidar": 1.0, "w_radar": 0.85, "w_thermal": 0.85})

    # =====================================================================
    # ROAD CONDITION & LANE AVAILABILITY DETECTION
    # =====================================================================
    
    # --- Detect road conditions from weather description and sensor summary ---
    road_condition = "clear"
    road_condition_details = []
    road_surface_condition = "good"  # good, patchy, bad, uneven, slippery, unpaved
    speed_reduction_factor = 1.0  # Multiplier for speed caps
    
    # Parse weather description for road surface conditions
    w_desc_lower = w_desc.lower()
    road_condition = "clear"
    road_condition_details = []
    road_surface_condition = "good"  # good, patchy, bad, uneven, slippery, unpaved, potholed, low_traction
    speed_reduction_factor = 1.0  # Multiplier for speed caps
    
    # Parse weather description for road surface conditions
    w_desc_lower = w_desc.lower()
    road_condition = "clear"
    road_condition_details = []
    road_surface_condition = "good"  # good, patchy, bad, uneven, slippery, unpaved, potholed, low_traction
    speed_reduction_factor = 1.0  # Multiplier for speed caps
    
    # Parse weather description for road surface conditions - USE PRIORITY ORDER (most restrictive first)
    w_desc_lower = w_desc.lower()
    # Priority 1: Road work (highest priority - affects lane availability)
    if any(kw in w_desc_lower for kw in ["road work", "construction", "roadwork", "road works"]):
        road_condition = "road_work"
        road_condition_details.append("Road works detected")
    # Priority 2: Slippery conditions (highest safety priority)
    elif any(kw in w_desc_lower for kw in ["wet", "slippery", "icy", "frost", "skiddy", "flooded", "standing water", "waterlogged", "hydroplane", "aquaplane"]):
        road_surface_condition = "slippery"
        road_condition_details.append(f"Road surface: {road_surface_condition}")
    # Priority 3: Low traction (oil, grease, etc.)
    elif any(kw in w_desc_lower for kw in ["oily", "oil", "greasy", "debris", "sand", "loose gravel"]):
        road_surface_condition = "low_traction"
        road_condition_details.append(f"Road surface: {road_surface_condition}")
    # Priority 4: Potholed
    elif any(kw in w_desc_lower for kw in ["pothole", "potholes", "crater", "depression"]):
        road_surface_condition = "potholed"
        road_condition_details.append(f"Road surface: potholed")
    # Priority 4: Patchy/bad/uneven
    elif any(kw in w_desc_lower for kw in ["patchy", "uneven", "pothole", "rough", "damaged"]):
        road_surface_condition = "patchy" if "patchy" in w_desc_lower else "bad"
        road_condition_details.append(f"Road surface: {road_surface_condition}")
    # Priority 5: Unpaved
    elif any(kw in w_desc_lower for kw in ["gravel", "dirt", "unpaved"]):
        road_surface_condition = "unpaved"
        road_condition_details.append(f"Road surface: unpaved")
    
    # Speed reduction factors based on road conditions - apply most restrictive
    # Road work takes priority, then surface condition (slippery has highest priority for safety)
    if road_condition == "road_work":
        speed_reduction_factor = 0.6  # 40% speed reduction
    elif road_surface_condition == "slippery":
        speed_reduction_factor = 0.5  # 50% speed reduction (highest priority for safety)
    elif road_surface_condition == "low_traction":
        speed_reduction_factor = 0.65  # 35% speed reduction
    elif road_surface_condition == "potholed":
        speed_reduction_factor = 0.65  # 35% speed reduction
    elif road_surface_condition in ("patchy", "bad", "uneven"):
        speed_reduction_factor = 0.7  # 30% speed reduction
    elif road_surface_condition == "unpaved":
        speed_reduction_factor = 0.6  # 40% speed reduction
    else:
        speed_reduction_factor = 1.0  # No reduction
    
    lane_restrictions = {
        "left_blocked": False,
        "right_blocked": False,
        "left_lanes_available": 2,
        "right_lanes_available": 2,
        "left_cones": 0,
        "right_cones": 0,
        "barriers_left": 0,
        "barriers_right": 0,
        "construction_zone": False,
        "lane_shift": None,
    }
    
    if sensor_summary:
        # Check for road work indicators from thermal/camera/radar
        th = sensor_summary.get("thermal", {})
        nv = sensor_summary.get("night_vision", {})
        
        # Construction/road work detection
        if s_sum.get("construction_detected", False):
            road_condition = "road_work"
            road_condition_details.append("Construction zone detected")
            lane_restrictions["construction_zone"] = True
        
        # Cone/barrier detection from thermal
        warm_pixels = th.get("warm_actor_pixels", 0)
        if warm_pixels > 100:  # Significant thermal activity could indicate work zone
            road_condition_details.append(f"Thermal activity: {warm_pixels} warm pixels")
    
    # --- Detect lane restrictions from nearby actors (cones, barriers, construction props) ---
    if nearby_actors:
        for a in nearby_actors:
            a_type = a.get("type", "").lower()
            dist = a.get("longitudinal_m", 999)
            lat = a.get("lateral_m", 0)
            
            # Only check objects within reasonable range
            if dist < 0 or dist > 80:
                continue
            
            # Static props: cones, barriers, construction signs
            if a_type in ("static.prop", "prop", "constructioncone", "trafficcone", 
                          "barrier", "constructionbarrier", "warning_sign",
                          "static.prop.constructioncone", "static.prop.barrier"):
                if lat < -1.5:  # Left side
                    lane_restrictions["left_cones"] += 1
                    if dist < 40:
                        lane_restrictions["left_blocked"] = True
                elif lat > 1.5:  # Right side
                    lane_restrictions["right_cones"] += 1
                    if dist < 40:
                        lane_restrictions["right_blocked"] = True
            
            # Construction vehicles/equipment
            if a_type in ("construction", "excavator", "dumper", "roller", "crane"):
                if lat < 0:
                    lane_restrictions["left_blocked"] = True
                else:
                    lane_restrictions["right_blocked"] = True
                road_condition = "road_work"
                road_condition_details.append(f"Construction vehicle at {dist:.0f}m lat {lat:+.1f}m")
            
            # Barriers
            if "barrier" in a_type:
                if lat < -1.0:
                    lane_restrictions["barriers_left"] += 1
                    if dist < 50:
                        lane_restrictions["left_blocked"] = True
                elif lat > 1.0:
                    lane_restrictions["barriers_right"] += 1
                    if dist < 50:
                        lane_restrictions["right_blocked"] = True
    
    # Determine available lanes
    if lane_restrictions["left_blocked"]:
        lane_restrictions["left_lanes_available"] = max(0, lane_restrictions["left_lanes_available"] - 1)
    if lane_restrictions["right_blocked"]:
        lane_restrictions["right_lanes_available"] = max(0, lane_restrictions["right_lanes_available"] - 1)
    
    # Detect lane shift from navigation goal or traffic
    if "merge" in navigation_goal.lower() or "shift" in navigation_goal.lower():
        if "left" in navigation_goal.lower():
            lane_restrictions["lane_shift"] = "left"
        elif "right" in navigation_goal.lower():
            lane_restrictions["lane_shift"] = "right"
    
    # Build road condition description
    road_condition_str = road_condition
    if road_condition_details:
        road_condition_str += f" ({'; '.join(road_condition_details)})"
    
    # =====================================================================
    # PROACTIVE DECISION FRAMEWORK - Build comprehensive context
    # =====================================================================
    
    # --- Analyze surrounding traffic for proactive decisions ---
    traffic_analysis = {}
    if nearby_actors:
        # Categorize all nearby actors by threat/opportunity level
        front_slow = []
        front_fast = []
        side_left = []
        side_right = []
        rear_closing = []
        cross_traffic = []
        
        for a in nearby_actors:
            if not a.get("in_ego_corridor", False):
                continue
            dist = a.get("longitudinal_m", 999)
            speed = a.get("speed_kmh", 0)
            lat = a.get("lateral_m", 0)
            a_type = a.get("type", "vehicle")
            
            if dist > 0 and dist < 50:
                # Front vehicles
                if speed < max(5.0, speed_kmh - 10.0):
                    front_slow.append(f"{a_type} at {dist:.0f}m @ {speed:.0f} km/h (lat {lat:+.1f}m)")
                elif speed > speed_kmh + 5.0:
                    front_fast.append(f"{a_type} at {dist:.0f}m @ {speed:.0f} km/h")
            elif dist < -10 and dist > -40:
                # Rear vehicles closing fast
                closing = speed_kmh - speed
                if closing > 10:
                    rear_closing.append(f"{a_type} behind {abs(dist):.0f}m closing {closing:.0f} km/h")
            
            # Lateral traffic (potential cut-ins)
            if abs(lat) < 4.0 and dist > -10 and dist < 30:
                if lat < -1.5:
                    side_left.append(f"{a_type} left {abs(lat):.1f}m @ {dist:.0f}m")
                elif lat > 1.5:
                    side_right.append(f"{a_type} right {lat:.1f}m @ {dist:.0f}m")
            
            # Cross traffic at intersections
            if dist > 10 and dist < 30 and abs(lat) > 3.0:
                cross_traffic.append(f"{a_type} crossing {lat:+.1f}m @ {dist:.0f}m")
        
        traffic_analysis = {
            "front_slow": front_slow,
            "front_fast": front_fast,
            "side_left": side_left,
            "side_right": side_right,
            "rear_closing": rear_closing,
            "cross_traffic": cross_traffic,
        }
    
    # --- Overtake / Lane Change Decision Context ---
    overtake_context = ""
    lane_change_context = ""
    # Initialize lane change thresholds at function scope (always defined)
    lead_dist_threshold = 35.0 * speed_reduction_factor
    speed_advantage_threshold = 8.0
    if nearby_actors:
        slow_ahead = [a for a in nearby_actors if a.get("in_ego_corridor", False) 
                      and a.get("longitudinal_m", 0) > 3.0
                      and a.get("longitudinal_m", 0) < 40.0
                      and a.get("speed_kmh", 0) < max(5.0, speed_kmh - 8.0)]
        
        # Lane change feasibility
        left_clear = sectors.get("SIDE_LEFT", 0)
        right_clear = sectors.get("SIDE_RIGHT", 0)
        rear_clear = sectors.get("REAR", 0)
        
        # Check if lane change is feasible and beneficial
        lane_change_feasible = False
        lc_reason = ""
        lc_side = None
        
        # lead_dist_threshold and speed_advantage_threshold already initialized above
        
        if slow_ahead:
            lead = min(slow_ahead, key=lambda x: x.get("longitudinal_m", 999))
            lead_dist = lead.get("longitudinal_m", 999)
            lead_speed = lead.get("speed_kmh", 0)
            
            # Apply lane restrictions and speed reduction factor
            speed_with_reduction = speed_kmh * speed_reduction_factor
            # lead_dist_threshold already initialized above
            
            # Apply lane restrictions
            left_ok = (left_clear > 3.0 and rear_clear > 15 
                       and lane_restrictions["left_lanes_available"] > 0 
                       and not lane_restrictions["left_blocked"])
            right_ok = (right_clear > 3.0 and rear_clear > 15 
                        and lane_restrictions["right_lanes_available"] > 0 
                        and not lane_restrictions["right_blocked"])
            
            if lead_dist < lead_dist_threshold and (speed_kmh - lead.get("speed_kmh", 0)) > speed_advantage_threshold:
                if left_ok and (not right_ok or left_clear > right_clear):
                    lane_change_feasible = True
                    lc_reason = (f"Lead at {lead_dist:.0f}m @ {lead_speed} km/h; "
                                 f"LEFT clear {left_clear:.1f}m, REAR {rear_clear:.1f}m "
                                 f"(surface: {road_surface_condition}, speed_factor: {speed_reduction_factor:.2f})")
                    lc_side = "LEFT"
                elif right_ok:
                    lane_change_feasible = True
                    lc_reason = (f"Lead at {lead_dist:.0f}m @ {lead_speed} km/h; "
                                 f"RIGHT clear {right_clear:.1f}m, REAR {rear_clear:.1f}m "
                                 f"(surface: {road_surface_condition}, speed_factor: {speed_reduction_factor:.2f})")
                    lc_side = "RIGHT"
        
        if lane_change_feasible:
            overtake_context = (
                f"\n- OVERTAKE OPPORTUNITY: {lc_reason}. "
                f"ACTION: Initiate LANE_CHANGE_{lc_side} at {min(max(lc_side == 'LEFT' and sectors.get('SIDE_LEFT', 0) * 10, 25), 35):.0f}m gap "
                f"with target speed +8-12 km/h (adjusted for surface: {int(speed_kmh * speed_reduction_factor)} km/h max). "
                f"VERIFY: No cross-traffic, no oncoming, REAR > 15m clear."
            )
        
        # Anti-unnecessary-maneuver constraints with lane restrictions and speed adaptation
        lane_change_context = (
            f"\n- LANE CHANGE CONSTRAINTS: "
            f"Only change if lead < {lead_dist_threshold:.0f}m AND speed advantage > {speed_advantage_threshold} km/h "
            f"AND target lane clear > 2m + REAR > 15m. "
            f"DO NOT change for minor speed differences. "
            f"NO U-turns. NO opposite-lane (B1) except WRONG-WAY emergency. "
            f"NO lane change in junction/intersection or near red light. "
            f"NO change if cross-traffic present. "
            f"LEFT: clearance={sectors.get('SIDE_LEFT', 0):.1f}m, cones={lane_restrictions['left_cones']}, "
            f"blocked={lane_restrictions['left_blocked']}, lanes_avail={lane_restrictions['left_lanes_available']}. "
            f"RIGHT: clearance={sectors.get('SIDE_RIGHT', 0):.1f}m, cones={lane_restrictions['right_cones']}, "
            f"blocked={lane_restrictions['right_blocked']}, lanes_avail={lane_restrictions['right_lanes_available']}. "
            f"REAR={sectors.get('REAR', 0):.1f}m. "
            f"ROAD CONDITION: {road_condition_str}. "
            f"SURFACE: {road_surface_condition} (speed factor: {speed_reduction_factor:.2f}). "
            f"LANE RESTRICTIONS: Left_avail={lane_restrictions['left_lanes_available']}, "
            f"Right_avail={lane_restrictions['right_lanes_available']}, "
            f"Shift={lane_restrictions['lane_shift']}, Construction={lane_restrictions['construction_zone']}. "
            f"CONFINE TO AVAILABLE LANES ONLY. "
            f"SPEED CAP: {int(speed_kmh * speed_reduction_factor)} km/h on this surface."
        )
    else:
        # Ensure variables are defined even when nearby_actors is empty
        lead_dist_threshold = 35.0 * speed_reduction_factor
        speed_advantage_threshold = 8.0

    # =====================================================================
    # MASTER DECISION FRAMEWORK - Structured reasoning for every output
    # =====================================================================
    # Local copies to avoid Python 3.11+ LOAD_FAST_CHECK issues
    _ldt = lead_dist_threshold
    _sat = speed_advantage_threshold
    _srf = speed_reduction_factor
    _rsc = road_surface_condition
    _rsc2 = road_surface_condition  # Alias for _road_surface_condition usage
    decision_framework = (
        f"\n=== MANDATORY DECISION FRAMEWORK (apply to EVERY output) ===\n"
        f"1. SAFETY FIRST: If ANY collision risk (TTC < 2.5s, gap < 3m, VRU in path) -> EMERGENCY_STOP.\n"
        f"2. TRAFFIC RULES: Red light at line -> STOP. Yellow < 25m -> SLOW_DOWN. Junction -> KEEP_LANE ≤ 22 km/h.\n"
        f"3. VRU PRIORITY: Pedestrian/animal/cyclist in corridor < 25m -> EMERGENCY_STOP/SLOW_DOWN.\n"
        f"4. LANE DISCIPLINE: KEEP_LANE default. Change ONLY if: lead < {lead_dist_threshold:.0f}m + "
        f"speed advantage > {speed_advantage_threshold} km/h + target lane clear > 2m + REAR > 15m.\n"
        f"5. NO UNNECESSARY MANEUVERS: No lane change for < {speed_advantage_threshold} km/h advantage. "
        f"NO U-turns. NO B1 lane unless WRONG-WAY emergency.\n"
        f"5. PROACTIVE TIMING: At 25 km/h (7 m/s), 30m = 4.3s. Decide lane change at 30m, not 10m.\n"
        f"6. BLIND SPOT CHECK: Rear > 15m, target side > 2m, no cross-traffic, no oncoming in target lane.\n"
        f"6. WEATHER ADAPTATION: Fog/Rain -> speed cap 18/20 km/h, expand gaps 1.15x, trust radar > LiDAR.\n"
        f"6. ROAD CONDITION ADAPTATION: Road works/patchy/bad/slippery -> apply speed factor {speed_reduction_factor:.2f}, "
        f"increase following distance 2x, avoid sudden maneuvers, expect uneven surfaces.\n"
        f"7. INTERSECTION: No lane change in/near junction. Clear intersection before lane change.\n"
        f"7. LANE CONFINEMENT: Stay in available lanes only. Respect cones/barriers/construction. Shift only if directed.\n"
        f"8. DECISION VALIDATION: Before every output, verify: (a) No rule violation, (b) No collision risk, (c) Maneuver necessary.\n"
    )

    # Build the full prompt
    tl_str = f"{traffic_light[0]} at {traffic_light[1]:.1f}m" if traffic_light else "CLEAR/GREEN"
    radar_f = s_sum.get("radar_front_m", lidar_front_dist_m)
    radar_ttc = s_sum.get("radar_front_ttc_s", 99.0)
    radar_v = s_sum.get("radar_closing_mps", 0.0)

    th = s_sum.get("thermal", {})
    nv = s_sum.get("night_vision", {})
    lid = s_sum.get("lidar_stats", {})
    fw = s_sum.get("fusion_weights", {"w_lidar": 1.0, "w_radar": 0.85, "w_thermal": 0.85})

    # Build traffic summary for prompt
    traffic_summary = ""
    if traffic_analysis:
        parts = []
        if traffic_analysis.get("front_slow"):
            parts.append(f"SLOW AHEAD: {'; '.join(traffic_analysis['front_slow'])}")
        if traffic_analysis.get("front_fast"):
            parts.append(f"FAST APPROACHING: {'; '.join(traffic_analysis['front_fast'])}")
        if traffic_analysis.get("side_left"):
            parts.append(f"LEFT SIDE: {'; '.join(traffic_analysis['side_left'])}")
        if traffic_analysis.get("side_right"):
            parts.append(f"RIGHT SIDE: {'; '.join(traffic_analysis['side_right'])}")
        if traffic_analysis.get("rear_closing"):
            parts.append(f"REAR CLOSING: {'; '.join(traffic_analysis['rear_closing'])}")
        if traffic_analysis.get("cross_traffic"):
            parts.append(f"CROSS TRAFFIC: {'; '.join(traffic_analysis['cross_traffic'])}")
        if parts:
            traffic_summary = "\n- TRAFFIC ANALYSIS:\n  " + "\n  ".join(parts)

    # Calculate effective speed cap
    effective_speed_cap = int(max(15.0, min(22.0, 20.0 * speed_reduction_factor)))

    return (
        f"[8-SENSOR FULL-FIDELITY AUTONOMOUS ANALYTICS & HUMAN CoT]\n"
        f"- Weather: {w_desc} | Speed: {speed_kmh:.1f} km/h | Fused Front Gap: {lidar_front_dist_m:.1f}m\n"
        f"- 3D LiDAR Analytics: Pts={lid.get('total_points', 0)}, MeanInt={lid.get('mean_intensity', 0.85):.2f}, BackscatterNoise={lid.get('backscatter_noise_ratio', 0.0):.4f}\n"
        f"- 4D Doppler Radar: Front={radar_f:.1f}m, ClosingVel={radar_v:.1f}m/s, TTC={radar_ttc:.1f}s | Left={s_sum.get('radar_left_m', 45.0):.1f}m, Right={s_sum.get('radar_right_m', 45.0):.1f}m\n"
        f"- Thermal LWIR Camera: WarmPixels={th.get('warm_actor_pixels', 0)}, ThermalDepth={th.get('thermal_min_depth_m', 50.0):.1f}m, PedHeat={th.get('pedestrian_heat_ratio', 0.0):.4f}\n"
        f"- Night-Vision Camera: Luminance={nv.get('mean_luminance', 128.0):.1f}, Contrast={nv.get('contrast_rms', 45.0):.1f}, FogVeil={nv.get('fog_veil_index', 0.0):.2f}, Glare={nv.get('glare_saturation_pct', 0.0):.1f}%, Gain={nv.get('nv_gain_db', 0.0):.1f}dB\n"
        f"- IMU & RTK-GNSS: YawRate={s_sum.get('imu_yaw_rate_rads', 0.0):.3f}rad/s, LatAccel={s_sum.get('imu_lat_accel_mps2', 0.0):.2f}m/s^2, GPS={s_sum.get('gps', (0,0,0))}\n"
        f"- Sensor Confidence Weights: LiDAR={fw['w_lidar']}, Radar={fw['w_radar']}, Thermal={fw['w_thermal']}\n"
        f"- Surround Clearances (m): FL={sectors.get('FRONT_LEFT', 40.0):.1f}, FR={sectors.get('FRONT_RIGHT', 40.0):.1f}, "
        f"SL={sectors.get('SIDE_LEFT', 15.0):.1f}, SR={sectors.get('SIDE_RIGHT', 15.0):.1f}, REAR={sectors.get('REAR', 35.0):.1f}\n"
        f"- Signal: {tl_str} | Goal: {navigation_goal}"
        f"{traffic_summary}"
        f"{overtake_context}"
        f"{lane_change_context}"
        f"{decision_framework}"
        f"- Pro Driver CoT: {human_thought}\n"
        f"Output ONLY valid JSON:\n"
        f'{{"intent": "KEEP_LANE|LANE_CHANGE_LEFT|LANE_CHANGE_RIGHT|SLOW_DOWN|STOP|OVERTAKE|EMERGENCY_STOP", "target_speed_kmh": {effective_speed_cap}, "reasoning": "Step-by-step: 1) Threat assessment 2) Rule check 3) Maneuver necessity 4) Clearance validation 5) Lane availability check 6) Surface condition check 7) Final decision with speed"}}'
    )


if __name__ == "__main__":
    # Quick test
    test_prompt = generate_multisensor_prompt(
        speed_kmh=30.0,
        lidar_front_dist_m=40.0,
        navigation_goal="Highway with road works, right 2 lanes closed, merge left",
        nearby_actors=[
            {"type": "vehicle", "in_ego_corridor": True, "longitudinal_m": 50, "speed_kmh": 25, "lateral_m": 0.0},
            {"type": "static.prop.constructioncone", "in_ego_corridor": False, "longitudinal_m": 30, "speed_kmh": 0, "lateral_m": 2.5},
            {"type": "static.prop.constructioncone", "in_ego_corridor": False, "longitudinal_m": 35, "speed_kmh": 0, "lateral_m": 2.8},
            {"type": "static.prop.barrier", "in_ego_corridor": False, "longitudinal_m": 45, "speed_kmh": 0, "lateral_m": 3.0},
            {"type": "construction", "in_ego_corridor": False, "longitudinal_m": 55, "speed_kmh": 0, "lateral_m": 3.5},
        ],
        traffic_light=("GREEN", 200.0),
        weather_info={"description": "clear, road works, patchy surface"},
        spatial_sectors={
            "FRONT_LEFT": 40.0, "FRONT_RIGHT": 40.0,
            "SIDE_LEFT": 5.0, "SIDE_RIGHT": 0.5,
            "REAR": 40.0, "FRONT_CENTER": 50.0
        },
        sensor_summary={
            "radar_front_m": 50.0, "radar_front_ttc_s": 10.0, "radar_closing_mps": 0.0,
            "radar_left_m": 50.0, "radar_right_m": 50.0,
            "thermal": {"warm_actor_pixels": 0, "thermal_min_depth_m": 50.0},
            "night_vision": {"mean_luminance": 150, "contrast_rms": 50, "fog_veil_index": 0.0, "glare_saturation_pct": 0.0, "nv_gain_db": 0.0},
            "lidar_stats": {"total_points": 50000, "mean_intensity": 0.85, "backscatter_noise_ratio": 0.05},
            "fusion_weights": {"w_lidar": 1.0, "w_radar": 0.85, "w_thermal": 0.85},
            "construction_detected": True
        }
    )
    print(test_prompt)