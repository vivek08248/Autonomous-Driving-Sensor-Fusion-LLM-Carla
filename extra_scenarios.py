"""
extra_scenarios.py  -  52 extra test cases (TC_27 ... TC_78) for drive_in_carla.py

Put this file NEXT TO drive_in_carla.py. Each case is one data row in CASES, so
adding your own case is a single line. A generic builder spawns the ego car,
the other actors (relative to the ego lane) and optionally forces a traffic
light; a small Runtime drives the NPCs (lead cars, cut-ins, braking, walkers).

Actor helpers
    V(d, lane, kmh, ...)   vehicle,  d = metres ahead of ego along the lane
                           lane = same | left | right | oncoming
                           beh  = static | drive | cut_in
                           brake_at / resume_at = seconds (lead braking, stop-and-go)
    W(d, lat, ...)         walker,   lat = metres from ego-lane centre (+ = right)
                           beh = cross | static ; trigger = start when ego is
                           within this many metres ; speed in m/s
    P(d, lat, bp)          static prop (cone, barrier, warning sign ...)
    d may be the string "j+30" = 30 m beyond the junction ahead.

Road types:  straight | multi (has a right lane, same direction) |
             oncoming (has an opposite lane) | junction (junction 55-90 m ahead)
"""
import math

import carla

try:
    import config
    DT = float(getattr(config, "SIM_DELTA_SECONDS", 0.05))
except Exception:
    config, DT = None, 0.05

EGO_BP = getattr(config, "EGO_VEHICLE_BLUEPRINT", "vehicle.tesla.model3") if config else "vehicle.tesla.model3"
CAR = "vehicle.audi.a2"
TRUCK = "vehicle.carlamotors.carlacola"
BIKE = "vehicle.diamondback.century"
MOTO = "vehicle.yamaha.yzf"

WEATHER = {
    "clear":      dict(cloudiness=10.0, sun_altitude_angle=65.0),
    "glare":      dict(cloudiness=0.0, sun_altitude_angle=6.0, sun_azimuth_angle=0.0),
    "rain":       dict(cloudiness=80.0, precipitation=70.0, precipitation_deposits=60.0,
                       wetness=70.0, sun_altitude_angle=40.0),
    "heavy_rain": dict(cloudiness=100.0, precipitation=100.0, precipitation_deposits=100.0,
                       wetness=100.0, wind_intensity=60.0, fog_density=15.0,
                       fog_distance=30.0, sun_altitude_angle=30.0),
    "fog":        dict(cloudiness=60.0, fog_density=80.0, fog_distance=10.0,
                       sun_altitude_angle=40.0),
    "night":      dict(cloudiness=20.0, sun_altitude_angle=-60.0),
    "night_rain": dict(cloudiness=90.0, precipitation=70.0, precipitation_deposits=70.0,
                       wetness=80.0, sun_altitude_angle=-60.0),
    "dusk":       dict(cloudiness=30.0, sun_altitude_angle=-3.0),
    "wet":        dict(cloudiness=40.0, precipitation_deposits=80.0, wetness=90.0,
                       sun_altitude_angle=50.0),
    "storm_night": dict(cloudiness=100.0, precipitation=100.0, precipitation_deposits=100.0,
                        wetness=100.0, fog_density=40.0, fog_distance=20.0,
                        sun_altitude_angle=-60.0),
}


# ----------------------------------------------------------------------
# Case table
# ----------------------------------------------------------------------
def V(d, lane="same", kmh=0.0, beh=None, bp=CAR, lat=0.0, **kw):
    return dict(kind="vehicle", d=d, lane=lane, kmh=kmh, lat=lat, bp=bp,
                beh=beh or ("drive" if kmh > 0 else "static"), **kw)


def W(d, lat=4.0, beh="cross", lane="same", bp="walker.pedestrian.*", **kw):
    return dict(kind="walker", d=d, lane=lane, lat=lat, beh=beh, bp=bp, **kw)


def A(d, lat=4.0, **kw):
    return dict(kind="animal", d=d, lane="same", lat=lat, beh="cross",
                bp="static.prop.deer", **kw)


def P(d, lat=0.0, bp="static.prop.constructioncone", lane="same"):
    return dict(kind="prop", d=d, lane=lane, lat=lat, bp=bp, beh="static")


def case(desc, expect, road="straight", weather="clear", ego_kmh=0.0, actors=(), **kw):
    return dict(desc=desc, expect=expect, road=road, weather=weather,
                ego_kmh=ego_kmh, actors=list(actors), **kw)


CASES = {
    # ---------------- A. lead-vehicle behaviour ----------------
    "TC_27_LEAD_SLOW_CRUISE": case(
        "Slow lead car at 15 km/h", "Match lead speed, keep a safe gap, no collision",
        ego_kmh=25, actors=[V(45, kmh=15)]),
    "TC_28_LEAD_HARD_BRAKE_CLOSE": case(
        "Lead car 25 m ahead brakes hard after 4 s", "Brake in time, no collision",
        ego_kmh=30, actors=[V(25, kmh=30, brake_at=4.0)]),
    "TC_29_LEAD_STOP_AND_GO": case(
        "Lead stops at 3 s and moves again at 9 s", "Stop behind it, resume when it moves",
        ego_kmh=20, actors=[V(30, kmh=20, brake_at=3.0, resume_at=9.0)]),
    "TC_30_PLATOON_FOLLOW": case(
        "Two cars ahead at 25 km/h", "Follow the platoon smoothly",
        ego_kmh=25, actors=[V(30, kmh=25), V(55, kmh=25)]),
    "TC_31_STATIONARY_QUEUE": case(
        "Three stopped cars in lane", "Stop with a gap behind the last car",
        ego_kmh=25, actors=[V(40), V(47), V(54)]),
    "TC_32_STOPPED_TRUCK_IN_LANE": case(
        "Truck stopped in the lane 50 m ahead", "Stop (or pass safely), no collision",
        ego_kmh=25, actors=[V(50, bp=TRUCK)]),

    # ---------------- B. cut-ins and lane situations ----------------
    "TC_33_CUT_IN_NEAR": case(
        "Car in right lane cuts in 20 m ahead", "Brake, keep gap, no collision",
        road="multi", ego_kmh=30,
        actors=[V(20, "right", kmh=20, beh="cut_in", trigger=22)]),
    "TC_34_CUT_IN_AT_SPEED": case(
        "Faster car cuts in from right lane", "Smooth reaction, no hard collision",
        road="multi", ego_kmh=30,
        actors=[V(15, "right", kmh=32, beh="cut_in", trigger=20)]),
    "TC_35_BICYCLE_CUT_IN": case(
        "Cyclist from right lane moves into ego lane", "Slow down and give clearance",
        road="multi", ego_kmh=25,
        actors=[V(22, "right", kmh=12, beh="cut_in", bp=BIKE, trigger=18)]),
    "TC_36_MOTORCYCLE_CUT_IN": case(
        "Motorcycle cuts in at short distance", "Brake, no collision",
        road="multi", ego_kmh=30,
        actors=[V(18, "right", kmh=22, beh="cut_in", bp=MOTO, trigger=20)]),
    "TC_37_PARKED_ROW_RIGHT_LANE": case(
        "Row of parked cars in the right lane", "Pass without an unnecessary stop",
        road="multi", ego_kmh=25,
        actors=[V(30, "right"), V(42, "right"), V(54, "right")]),
    "TC_38_DOUBLE_PARKED_TRUCK": case(
        "Truck double-parked, half blocking the lane", "Slow and stop or pass with clearance",
        ego_kmh=20, actors=[V(40, lat=1.2, bp=TRUCK)]),

    # ---------------- C. pedestrians ----------------
    "TC_39_JAYWALK_WALKING": case(
        "Pedestrian walks across the road", "Stop or yield, no collision",
        ego_kmh=25, actors=[W(40, 4.0, trigger=28)]),
    "TC_40_JAYWALK_RUNNING": case(
        "Pedestrian runs across the road", "Emergency braking, no collision",
        ego_kmh=25, actors=[W(35, 4.0, trigger=22, speed=3.5)]),
    "TC_41_CHILD_DART_BETWEEN_PARKED": case(
        "Pedestrian darts out between parked cars", "Late detection: hard brake, no collision",
        road="multi", ego_kmh=20,
        actors=[V(32, "right"), V(38, "right"),
                W(35, 5.8, trigger=15, speed=2.8, walk_s=4.0)]),
    "TC_42_PED_STANDING_ROADSIDE": case(
        "Pedestrian stands at the kerb, not crossing", "No emergency stop (false-positive test)",
        ego_kmh=25, actors=[W(40, 3.0, beh="static")]),
    "TC_43_PED_GROUP_CROSSING": case(
        "Group of four pedestrians crosses", "Yield until the group has cleared",
        ego_kmh=25,
        actors=[W(40, 4.0, trigger=30), W(41.5, 4.6, trigger=30),
                W(43, 3.6, trigger=30), W(44.5, 4.2, trigger=30)]),
    "TC_44_PED_STOPS_IN_LANE": case(
        "Pedestrian stops in the middle of the lane for 4 s", "Wait, then resume after it leaves",
        ego_kmh=25,
        actors=[W(40, 4.0, trigger=28, walk_s=6.0, pause_at=2.5, pause_for=4.0)]),
    "TC_45_PED_NIGHT_CROSSING": case(
        "Pedestrian crossing at night", "Detect via LiDAR, stop",
        weather="night", ego_kmh=25, actors=[W(40, 4.0, trigger=28)]),
    "TC_46_PED_RAIN_CROSSING": case(
        "Pedestrian crossing in rain", "Stop, longer braking distance on wet road",
        weather="rain", ego_kmh=25, actors=[W(40, 4.0, trigger=28)]),

    # ---------------- D. traffic signals ----------------
    "TC_47_RED_LIGHT_STOP": case(
        "Approach a red light", "Stop before the stop line and wait",
        road="junction", ego_kmh=25, light="red"),
    "TC_48_GREEN_LIGHT_PASS": case(
        "Approach a green light", "Pass without stopping (false-positive test)",
        road="junction", ego_kmh=25, light="green", route="straight"),
    "TC_49_YELLOW_STOPPABLE": case(
        "Yellow light while still far away", "Stop (enough distance to brake)",
        road="junction", ego_kmh=25, light="yellow"),
    "TC_50_GREEN_TO_RED": case(
        "Light turns red 3 s into the run", "Brake and stop at the line",
        road="junction", ego_kmh=25, light="green", light_to="red", switch_s=3.0),
    "TC_51_RED_TO_GREEN_RESUME": case(
        "Wait at red, light turns green at 8 s", "Stopped at the line, then move off",
        road="junction", ego_kmh=25, light="red", light_to="green", switch_s=8.0,
        route="straight"),
    "TC_52_RED_WITH_QUEUE": case(
        "Red light with two cars queued", "Stop behind the queue, not at the line",
        road="junction", ego_kmh=25, light="red", actors=[V(32), V(39)]),
    "TC_53_GREEN_PED_ON_APPROACH": case(
        "Green light but pedestrian crossing before the junction", "Yield to the pedestrian",
        road="junction", ego_kmh=25, light="green", actors=[W(35, 4.0, trigger=26)]),
    "TC_54_RED_NIGHT_RAIN": case(
        "Red light at night in rain", "Stop at the line",
        road="junction", weather="night_rain", ego_kmh=20, light="red"),

    # ---------------- E. turns and junctions ----------------
    "TC_55_RIGHT_TURN_CLEAR": case(
        "Right turn at a clear junction", "Slow down, turn right, stay in lane",
        road="junction", ego_kmh=20, light="green", route="right"),
    "TC_56_LEFT_TURN_CLEAR": case(
        "Left turn at a clear junction", "Slow down, turn left, stay in lane",
        road="junction", ego_kmh=20, light="green", route="left"),
    "TC_57_LEFT_TURN_ONCOMING": case(
        "Left turn with oncoming car through the junction", "Yield to oncoming traffic",
        road="oncoming", ego_kmh=20, route="left",
        actors=[V("j+30", "oncoming", kmh=25)]),
    "TC_58_RIGHT_TURN_SLOW_LEAD": case(
        "Right turn behind a slow car", "Follow it through the turn, no collision",
        road="junction", ego_kmh=20, light="green", route="right",
        actors=[V(25, kmh=15)]),
    "TC_59_STRAIGHT_THROUGH_JUNCTION": case(
        "Drive straight through a junction", "Keep lane through the junction",
        road="junction", ego_kmh=25, light="green", route="straight"),

    # ---------------- F. oncoming traffic ----------------
    "TC_60_ONCOMING_NORMAL_PASS": case(
        "Oncoming car in the opposite lane", "Stay in lane, no panic braking",
        road="oncoming", ego_kmh=25, actors=[V(80, "oncoming", kmh=30)]),
    "TC_61_ONCOMING_WITH_PARKED_CAR": case(
        "Parked car intrudes into lane, oncoming car approaching", "Wait for oncoming, then pass",
        road="oncoming", ego_kmh=20,
        actors=[V(35, lat=1.4), V(60, "oncoming", kmh=25)]),
    "TC_62_ONCOMING_TRUCK": case(
        "Oncoming truck", "Stay in lane, keep clearance",
        road="oncoming", ego_kmh=25, actors=[V(80, "oncoming", kmh=30, bp=TRUCK)]),

    # ---------------- G. obstacles and props ----------------
    "TC_63_CONE_CHICANE": case(
        "Cones forming a chicane", "Weave through or stop, no cone hit",
        ego_kmh=15, actors=[P(30, -1.5), P(40, 1.5), P(50, -1.5)]),
    "TC_64_BARRIER_FULL_BLOCK": case(
        "Barriers block the entire lane", "Stop before the barrier",
        ego_kmh=20,
        actors=[P(40, -1.6, "static.prop.streetbarrier"),
                P(40, 0.0, "static.prop.streetbarrier"),
                P(40, 1.6, "static.prop.streetbarrier")]),
    "TC_65_SMALL_DEBRIS": case(
        "Small box on the road", "Slow down and avoid or stop",
        ego_kmh=20, actors=[P(35, 0.0, "static.prop.box*")]),
    "TC_66_ROADWORK_ZONE": case(
        "Warning sign and cones tapering into the lane", "Slow down, pass with clearance",
        ego_kmh=20,
        actors=[P(24, 1.7, "static.prop.trafficwarning"),
                P(30, 1.2), P(36, 0.7), P(42, 0.2)]),

    # ---------------- H. weather and lighting ----------------
    "TC_67_NIGHT_STATIC_CAR": case(
        "Stopped car at night", "Detect with LiDAR, stop",
        weather="night", ego_kmh=25, actors=[V(45)]),
    "TC_68_FOG_STATIC_CAR": case(
        "Stopped car in dense fog", "Reduce speed, stop in time",
        weather="fog", ego_kmh=20, actors=[V(40)]),
    "TC_69_HEAVY_RAIN_LEAD_BRAKE": case(
        "Lead brakes hard in heavy rain", "Keep larger gap, no collision",
        weather="heavy_rain", ego_kmh=25, actors=[V(30, kmh=25, brake_at=4.0)]),
    "TC_70_GLARE_SLOW_LEAD": case(
        "Low sun glare, slow lead car", "Camera degraded: LiDAR keeps safe gap",
        weather="glare", ego_kmh=25, actors=[V(40, kmh=12)]),
    "TC_71_WET_ROAD_PED_CROSSING": case(
        "Pedestrian crossing on wet road", "Brake early, no collision",
        weather="wet", ego_kmh=25, actors=[W(40, 4.0, trigger=28)]),
    "TC_72_DUSK_CYCLIST_AHEAD": case(
        "Slow cyclist ahead at dusk", "Slow to cyclist speed, keep clearance",
        weather="dusk", ego_kmh=25, actors=[V(40, kmh=10, bp=BIKE)]),

    # ---------------- I. speed and start-up ----------------
    "TC_73_HIGH_SPEED_CLOSING": case(
        "Ego at 50 km/h closing on a stopped car", "Brake early, no collision",
        ego_kmh=50, actors=[V(80)]),
    "TC_74_SPEED_LIMIT_CRUISE": case(
        "Empty road, start from rest", "Accelerate smoothly, respect the speed cap",
        ego_kmh=0),
    "TC_75_NO_ROOM_TO_MOVE": case(
        "Stopped car only 10 m ahead at start", "Do not drive into it",
        ego_kmh=0, actors=[V(10)]),

    # ---------------- J. stress cases ----------------
    "TC_76_RUSH_HOUR_MIXED": case(
        "Lead car, cut-in, parked car and pedestrian", "Handle all, no collision",
        road="multi", ego_kmh=25,
        actors=[V(25, kmh=18), V(18, "right", kmh=22, beh="cut_in", trigger=20),
                V(40, "right"), W(60, 4.5, trigger=30)]),
    "TC_77_LEAD_BRAKE_PLUS_PED": case(
        "Lead stops, then pedestrian crosses", "Stop for both, no collision",
        ego_kmh=25,
        actors=[V(30, kmh=25, brake_at=3.5, resume_at=10.0), W(50, 4.0, trigger=25)]),
    "TC_78_STORM_NIGHT_CROSSING": case(
        "Storm at night: slow lead and pedestrian", "Worst case: slow, stop, no collision",
        weather="storm_night", ego_kmh=20,
        actors=[V(25, kmh=15, brake_at=4.0), W(45, 4.0, trigger=24)]),
    "TC_79_MIXED_RULES_VALIDATION": case(
        "Red-to-green junction with lead car, cyclist, pedestrian, animal and oncoming traffic",
        "Stop at the line for red, yield to hazards, then complete the selected maneuver",
        road="junction", ego_kmh=25, light="red", light_to="green", switch_s=8.0,
        actors=[V(30, kmh=15),
                V(24, "right", kmh=12, beh="cut_in", bp=BIKE, trigger=22),
                W(42, 4.0, trigger=30, speed=1.6),
                A(56, 4.5, trigger=28, speed=1.2),
                V("j+30", "oncoming", kmh=25),
                P(36, 2.8, "static.prop.trafficwarning")]),

    # ---------------- K. Advanced Autonomy & Research Edge Cases ----------------
    "TC_80_EMERGENCY_VEHICLE_APPROACH": case(
        "Emergency police vehicle rapidly closing from rear at 45 km/h",
        "Recognize approaching emergency vehicle and hold position / yield lane",
        road="multi", ego_kmh=25,
        actors=[V(-30, "right", kmh=45, beh="drive")]),

    "TC_81_BLIND_SPOT_MERGE_VEHICLE": case(
        "Vehicle pacing in left rear blind spot (-8m, left lane)",
        "Maintain current lane, do not swerve or lane-change into occupied blind spot",
        road="multi", ego_kmh=25,
        actors=[V(-8, "left", kmh=25, beh="drive")]),

    "TC_82_CONSTRUCTION_LANE_DROP_MERGE": case(
        "Right travel lane closed by construction taper 35m ahead",
        "Detect barricade cones, plan smooth 3-phase maneuver into open left lane",
        road="multi", ego_kmh=25,
        actors=[P(35, lat=0.0), P(42, lat=0.4), P(49, lat=0.8), V(60, "left", kmh=20)]),

    "TC_83_DOORING_BICYCLIST_HAZARD": case(
        "Parked vehicle on shoulder with cyclist overtaking alongside in lane",
        "Decelerate smoothly, give lateral clearance, do not collide",
        road="multi", ego_kmh=25,
        actors=[V(28, "right", lat=1.8), V(22, "right", kmh=14, bp=BIKE, beh="drive")]),

    "TC_84_CROSSWALK_Z_WALKER_RETREAT": case(
        "Pedestrian steps onto crosswalk, pauses, then retreats to curb",
        "Hold Point B stop line until pedestrian completely clears crosswalk",
        ego_kmh=25,
        actors=[W(30, lat=3.5, beh="cross", speed=1.4, trigger=25, pause_at=2.0, pause_for=2.5)]),

    "TC_85_SCHOOL_BUS_EXTENDED_STOP": case(
        "Large bus stopped ahead with children boarding near roadside",
        "Stop at safe headway distance behind bus, yield to pedestrian activity",
        ego_kmh=25,
        actors=[V(35, bp=TRUCK), W(40, lat=4.0, speed=1.8, trigger=28)]),

    "TC_86_WHEELCHAIR_CROSSING": case(
        "Low-profile mobility device crossing slowly at 1.0 m/s",
        "Detect VRU early, decelerate smoothly to zero before crosswalk",
        ego_kmh=25,
        actors=[W(28, lat=4.2, speed=1.0, trigger=25)]),

    "TC_87_MOTORCYCLE_FILTERING_SPLITTING": case(
        "Motorcycle lane-splitting closely at 28 km/h",
        "Maintain lane centerline, smooth headway tracking without erratic twitching",
        road="multi", ego_kmh=22,
        actors=[V(18, lat=1.3, kmh=28, bp=MOTO, beh="drive")]),

    "TC_88_CARRIER_TRUCK_OVERHANG_CARGO": case(
        "Heavy cargo truck stopped with long rear profile 40m ahead",
        "Multi-modal LiDAR fusion detects setback, stops with full buffer",
        ego_kmh=25,
        actors=[V(40, bp=TRUCK)]),

    "TC_89_FALLEN_TREE_BRANCH_OBSTACLE": case(
        "Debris box obstacle partially obstructing travel path",
        "Detect obstacle in travel corridor, initiate lateral bypass maneuver",
        ego_kmh=25,
        actors=[P(30, lat=0.4, bp="static.prop.box01")]),

    "TC_90_SLIPPERY_OIL_SLICK_AQUAPLANING": case(
        "Heavy rain storm with sudden lead vehicle deceleration",
        "RSS safety distance automatically compensates for wet friction, stops safely",
        weather="heavy_rain", ego_kmh=25,
        actors=[V(32, kmh=25, brake_at=3.0)]),

    "TC_91_TUNNEL_LIGHT_TRANSITION": case(
        "Stalled car in dark illumination transition",
        "LiDAR bumper detection recognizes unlit vehicle ahead, halts with safe gap",
        weather="night", ego_kmh=25,
        actors=[V(36)]),

    "TC_92_FREEWAY_ON_RAMP_MERGE": case(
        "Vehicle merging from on-ramp into ego corridor at 26 km/h",
        "Adjust cruise speed to negotiate safe merging gap",
        road="multi", ego_kmh=28,
        actors=[V(16, "right", kmh=26, beh="cut_in", trigger=22)]),

    "TC_93_UNPROTECTED_LEFT_TURN_GAP": case(
        "Unprotected intersection with oncoming vehicle approaching at 35 km/h",
        "Yield at intersection entrance until oncoming vehicle clears corridor",
        road="junction", ego_kmh=22,
        actors=[V("j+25", "oncoming", kmh=35)]),

    "TC_94_REVERSING_OUT_OF_DRIVEWAY": case(
        "Vehicle slowly encroaching into travel lane from roadside alley",
        "Recognize perpendicular intrusion, slow down and yield right-of-way",
        road="multi", ego_kmh=25,
        actors=[V(22, "right", kmh=6, beh="cut_in", trigger=24)]),

    "TC_95_DOUBLE_PARKED_DELIVERY_VAN": case(
        "Delivery van stopped in travel lane blocking route",
        "AutonomousManeuverPlanner conducts smooth 3-phase shift-out, pass, and re-entry",
        road="multi", ego_kmh=22,
        actors=[V(30, bp=TRUCK)]),

    "TC_96_HIGHWAY_GHOST_DRIVER_DODGE": case(
        "Wrong-way driver oncoming in ego lane at 25 km/h",
        "Detect imminent collision, trigger emergency stop / lateral evasion",
        road="oncoming", ego_kmh=25,
        actors=[V(45, "same", kmh=25, beh="wrong_way", yaw_offset=180.0)]),

    "TC_97_CROSS_TRAFFIC_INTERSECTION_RUNNER": case(
        "Cross-traffic vehicle crosses intersection perpendicularly at 35 km/h",
        "Cross-traffic cut-in detector yields before intersection collision point",
        road="junction", ego_kmh=22,
        actors=[V("j+15", "oncoming", kmh=35)]),

    "TC_98_MULTI_PEDESTRIAN_ZEBRA_SURGE": case(
        "Group of pedestrians crossing at zebra crossing simultaneously",
        "Point A to Point B smooth glide, line hold until all pedestrians cross",
        ego_kmh=25,
        actors=[W(26, lat=4.0, trigger=25, speed=1.3),
                W(29, lat=5.2, trigger=25, speed=1.5),
                W(32, lat=3.0, trigger=25, speed=1.2)]),

    "TC_99_CONE_SLALOM_WORKZONE": case(
        "Work zone chicane with alternating cones",
        "Follow clear corridor through slalom without striking cones",
        ego_kmh=20,
        actors=[P(18, lat=-0.8), P(26, lat=0.8), P(34, lat=-0.8)]),

    "TC_100_TOTAL_AUTONOMY_GRAND_CHALLENGE": case(
        "Grand Challenge: Adverse dusk, red-to-green light, lead vehicle, VRU cross, and cutting vehicle",
        "Execute end-to-end autonomous driving stack: zero collisions, smooth glide, full pass",
        weather="dusk", road="junction", ego_kmh=25, light="red", light_to="green", switch_s=7.0,
        actors=[V(32, kmh=18, brake_at=3.5),
                W(45, 4.0, trigger=28, speed=1.4),
                V(18, "right", kmh=16, beh="cut_in", trigger=22)]),
}

EXTRA_NAMES = list(CASES)


def describe(name):
    c = CASES[name]
    return c["desc"], c["expect"]


# ----------------------------------------------------------------------
# Map helpers
# ----------------------------------------------------------------------
_START_CACHE = {}


def _junction_ahead(wp, max_d):
    d, w = 0.0, wp
    while d < max_d:
        nxt = w.next(10.0)
        if not nxt:
            return None
        w = nxt[0]
        d += 10.0
        if w.is_junction:
            return d
    return None


def _oncoming_lane(wp):
    w = wp.get_left_lane()
    while w is not None and w.lane_id * wp.lane_id > 0:
        w = w.get_left_lane()
    if w is not None and w.lane_type == carla.LaneType.Driving:
        return w
    return None


def _find_start(wmap, road):
    key = (wmap.name, road)
    if key in _START_CACHE:
        return wmap.get_waypoint(_START_CACHE[key])
    spawns = sorted(wmap.get_spawn_points(),
                    key=lambda t: (round(t.location.x), round(t.location.y)))
    for sp in spawns:
        wp = wmap.get_waypoint(sp.location)
        if wp is None or wp.is_junction:
            continue
        jd = _junction_ahead(wp, 170.0)
        if road == "junction":
            if jd is None or not (55.0 <= jd <= 90.0):
                continue
        else:
            if jd is not None:
                continue
            if road == "multi":
                r = wp.get_right_lane()
                if (r is None or r.lane_type != carla.LaneType.Driving
                        or r.lane_id * wp.lane_id < 0):
                    continue
            elif road == "oncoming" and _oncoming_lane(wp) is None:
                continue
        _START_CACHE[key] = wp.transform.location
        return wp
    print(f"[Scenario] No ideal start found for road type '{road}'; using first spawn point.")
    if not spawns:
        return None
    wp = wmap.get_waypoint(spawns[0].location)
    _START_CACHE[key] = wp.transform.location
    return wp



def _signal_on_path(world, wp, max_d=110.0):
    """Traffic light whose stop line lies on THIS lane before the next junction (or None)."""
    pts, w, d = [], wp, 0.0
    while d < max_d:
        pts.append(w.transform.location)
        if w.is_junction:
            break
        nxt = w.next(2.0)
        if not nxt:
            break
        w = nxt[0]
        d += 2.0
    best = None
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            stops = tl.get_stop_waypoints()
        except Exception:
            continue
        for sw in stops:
            p = sw.transform.location
            for k, q in enumerate(pts):
                if p.distance(q) < 2.0:
                    if best is None or k < best[0]:
                        best = (k, tl)
                    break
    return best[1] if best else None


def _find_signal_start(world, wmap):
    """Junction start (55-90 m) whose approach is controlled by a traffic light."""
    key = (wmap.name, "junction_signal")
    if key in _START_CACHE:
        wp = wmap.get_waypoint(_START_CACHE[key])
        return wp, _signal_on_path(world, wp)
    spawns = sorted(wmap.get_spawn_points(), key=lambda t: (round(t.location.x), round(t.location.y)))
    for sp in spawns:
        wp = wmap.get_waypoint(sp.location)
        if wp is None or wp.is_junction:
            continue
        jd = _junction_ahead(wp, 170.0)
        if jd is None or not (55.0 <= jd <= 90.0):
            continue
        tl = _signal_on_path(world, wp)
        if tl is not None:
            _START_CACHE[key] = wp.transform.location
            return wp, tl
    print("[Scenario] no signalised junction start found; falling back to any junction.")
    return _find_start(wmap, "junction"), None

def _advance(wp, d):
    if d < 0:                                   # behind the ego (TC_80, TC_81)
        prev = wp.previous(max(-float(d), 0.1))
        return prev[0] if prev else wp
    nxt = wp.next(max(float(d), 0.1))
    return nxt[0] if nxt else wp


def _lane_wp(wp, lane):
    if lane == "right":
        return wp.get_right_lane() or wp
    if lane == "left":
        return wp.get_left_lane() or wp
    if lane == "oncoming":
        return _oncoming_lane(wp) or wp
    return wp


def _ego_light(world, wp0):
    tf = wp0.transform
    loc, f = tf.location, tf.get_forward_vector()
    best = None
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            stops = tl.get_stop_waypoints()
        except Exception:
            continue
        for sw in stops:
            p = sw.transform.location
            dx, dy = p.x - loc.x, p.y - loc.y
            along = dx * f.x + dy * f.y
            lat = abs(dx * f.y - dy * f.x)
            if 5.0 < along < 160.0 and lat < 2.5 and (best is None or along < best[0]):
                best = (along, tl)
    return best[1] if best else None


def _bp(lib, pattern, kind):
    found = list(lib.filter(pattern))
    if not found:
        fallback = {"vehicle": "vehicle.audi.a2", "walker": "walker.pedestrian.*",
                    "animal": "walker.pedestrian.*",
                    "prop": "static.prop.constructioncone"}.get(kind, "vehicle.*")
        found = list(lib.filter(fallback)) or list(lib.filter(f"{kind}.*")) or list(lib.filter("vehicle.*"))
    return found[0] if found else None


# ----------------------------------------------------------------------
# NPC driving
# ----------------------------------------------------------------------
def _pursuit(actor, target, wheelbase=2.8, max_steer_rad=1.22):
    tf = actor.get_transform()
    loc = tf.location
    yaw = math.radians(tf.rotation.yaw)
    dx, dy = target.x - loc.x, target.y - loc.y
    ang = math.atan2(dy, dx) - yaw
    ang = math.atan2(math.sin(ang), math.cos(ang))
    ld = max(math.hypot(dx, dy), 1.0)
    steer = math.atan2(2.0 * wheelbase * math.sin(ang), ld) / max_steer_rad
    return max(-1.0, min(1.0, steer))


def _toward_lane(cur, dest_id):
    for side in (cur.get_left_lane(), cur.get_right_lane()):
        if side is not None and side.lane_id == dest_id:
            return side
    return cur


def _drive(actor, wmap, kmh, cut_lane=None):
    v = actor.get_velocity()
    speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
    wp = wmap.get_waypoint(actor.get_location())
    if cut_lane is not None and wp.lane_id != cut_lane:
        wp = _toward_lane(wp, cut_lane)
    nxt = wp.next(max(6.0, speed / 3.6 * 0.8))
    tgt = nxt[0].transform.location if nxt else wp.transform.location
    err = kmh - speed
    throttle = max(0.0, min(0.7, 0.08 * err))
    brake = max(0.0, min(1.0, -0.08 * err)) if err < -3.0 else 0.0
    actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake,
                                             steer=_pursuit(actor, tgt)))


class Runtime:
    """Per-frame behaviour of the NPCs and timed traffic-light changes."""

    def __init__(self, world, ego, items, tl, spec, ego_lane):
        self.world, self.ego, self.items, self.tl = world, ego, items, tl
        self.spec, self.ego_lane = spec, ego_lane
        self.map = world.get_map()
        self.route_pref = spec.get("route")
        self.desc, self.expect = spec["desc"], spec["expect"]
        self.t = 0.0
        self._switched = False

    def tick(self):
        self.t += DT
        sp = self.spec
        if (self.tl is not None and sp.get("switch_s") is not None
                and not self._switched and self.t >= sp["switch_s"]):
            self.tl.set_state(_light_state(sp["light_to"]))
            self._switched = True
        eloc = self.ego.get_location()
        for it in self.items:
            a, s = it["actor"], it["spec"]
            if not a.is_alive:
                continue
            if s["kind"] == "vehicle":
                self._vehicle(it, a, s, eloc)
            elif s["kind"] == "walker":
                self._walker(it, a, s, eloc)

    def _vehicle(self, it, a, s, eloc):
        beh = s["beh"]
        if beh == "static":
            return
        if beh == "wrong_way":                  # drive straight along its own heading
            v = a.get_velocity()
            spd = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
            a.apply_control(carla.VehicleControl(
                throttle=0.5 if spd < s["kmh"] else 0.0, steer=0.0))
            return
        cut = None
        if beh == "cut_in":
            if it["t0"] is None and eloc.distance(a.get_location()) < s.get("trigger", 25.0):
                it["t0"] = self.t
            if it["t0"] is not None:
                cut = self.ego_lane
        ba, ra = s.get("brake_at"), s.get("resume_at", 1e9)
        if ba is not None and ba <= self.t < ra:
            a.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        _drive(a, self.map, s["kmh"], cut)

    def _walker(self, it, a, s, eloc):
        if s["beh"] != "cross":
            return
        if it["t0"] is None:
            if eloc.distance(a.get_location()) < s.get("trigger", 25.0):
                it["t0"] = self.t
            else:
                return
        e = self.t - it["t0"]
        pa, pf = s.get("pause_at"), s.get("pause_for", 0.0)
        paused = pa is not None and pa <= e < pa + pf
        end = s.get("walk_s", 7.0) + (pf if pa is not None else 0.0)
        speed = s.get("speed", 1.5) if (e < end and not paused) else 0.0
        a.apply_control(carla.WalkerControl(direction=it["dir"], speed=speed))


def _light_state(name):
    return {"red": carla.TrafficLightState.Red,
            "yellow": carla.TrafficLightState.Yellow,
            "green": carla.TrafficLightState.Green}[name]


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------
def build(world, name):
    """Returns (ego, actors_to_destroy, runtime)."""
    spec = CASES[name]
    wmap = world.get_map()
    lib = world.get_blueprint_library()
    world.set_weather(carla.WeatherParameters(**WEATHER[spec["weather"]]))

    sig_tl = None
    if spec.get("light"):
        wp0, sig_tl = _find_signal_start(world, wmap)      # the red/green light must be ON our route
    else:
        wp0 = _find_start(wmap, spec["road"])
    tf0 = wp0.transform
    ego_matches = list(lib.filter(EGO_BP)) or list(lib.filter("vehicle.tesla.*")) or list(lib.filter("vehicle.*"))
    ego_bp = ego_matches[0]
    ego_bp.set_attribute("role_name", "hero")
    ego = world.try_spawn_actor(
        ego_bp, carla.Transform(tf0.location + carla.Location(z=0.4), tf0.rotation))
    if ego is None:
        raise RuntimeError("could not spawn ego for " + name)
    if spec["ego_kmh"] > 0:
        ego.set_target_velocity(tf0.get_forward_vector() * (spec["ego_kmh"] / 3.6))

    jd = _junction_ahead(wp0, 160.0)
    actors, items = [], []
    for a in spec["actors"]:
        d = a["d"]
        if isinstance(d, str):                       # "j+30"
            d = (jd if jd is not None else 70.0) + float(d[2:])
        w = _lane_wp(_advance(wp0, d), a["lane"])
        tf = w.transform
        rv = tf.get_right_vector()
        loc = tf.location + rv * a["lat"] + carla.Location(z=0.3)
        bp = _bp(lib, a["bp"], a["kind"])
        if bp is None:
            continue
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "npc")
        actor_kind = "walker" if a["kind"] == "animal" and bp.id.startswith("walker.") else a["kind"]
        if actor_kind == "walker" and bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        rot = carla.Rotation(pitch=tf.rotation.pitch,
                             yaw=tf.rotation.yaw + a.get("yaw_offset", 0.0),
                             roll=tf.rotation.roll)
        spawn_tf = carla.Transform(loc, rot)
        actor = world.try_spawn_actor(bp, spawn_tf)
        if actor is None:
            print(f"[Scenario] {name}: could not spawn {a['bp']} at {d:.0f} m (blocked).")
            continue
        sign = 1.0 if a["lat"] > 0 else -1.0
        direction = carla.Vector3D(-rv.x * sign, -rv.y * sign, 0.0)
        if actor_kind == "vehicle":
            if a["beh"] == "static":
                actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
            elif a["kmh"] > 0:
                actor.set_target_velocity(spawn_tf.get_forward_vector() * (a["kmh"] / 3.6))
        actors.append(actor)
        runtime_spec = dict(a, kind=actor_kind)
        items.append(dict(actor=actor, spec=runtime_spec, t0=None, dir=direction))

    tl = None
    if spec.get("light"):
        tl = sig_tl or _ego_light(world, wp0)
        if tl is None:
            print(f"[Scenario] {name}: no traffic light found ahead of the start point.")
        else:
            tl.set_state(_light_state(spec["light"]))
            tl.freeze(True)

    return ego, actors, Runtime(world, ego, items, tl, spec, wp0.lane_id)