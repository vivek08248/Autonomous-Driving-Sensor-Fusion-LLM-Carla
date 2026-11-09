"""
offline_planner_tests.py - Closed-loop regression tests for the driving stack WITHOUT CARLA.

A small kinematic simulator (bicycle model + simple drive-train) runs the real
BehaviorPlanner + SpeedPID + SteeringPID on archetypes of the project test cases
(static obstacle, oncoming traffic, lead braking, cut-in, pedestrians, red light,
barrier closure, queue, wrong-way driver ...).  Each test checks: no collision,
correct stop/go behaviour and that the car is not dead-locked.

Run:  python offline_planner_tests.py          (prints a table, exit code 1 on failure)
"""
import math
import sys

from ad_types import LaneGeo, LightInfo, Obj, Scene
from ad_planner import BehaviorPlanner
from ad_control import SpeedPID, SteeringPID

DT = 0.05
LANE_W = 3.5
EGO_HL, EGO_HW = 2.4, 1.0


class Actor:
    def __init__(self, aid, kind, x, y, vx=0.0, vy=0.0, hl=2.2, hw=0.9, script=None):
        self.id, self.kind = aid, kind
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.hl, self.hw = hl, hw
        self.script = script

    def step(self, t, dt, ego):
        if self.script:
            self.script(self, t, ego)
        self.x += self.vx * dt
        self.y += self.vy * dt


class World:
    """Straight road along +x.  Ego lane centre y=0.  y is + to the RIGHT (CARLA style)."""

    def __init__(self, left_same=0.0, left_opp=LANE_W, right_same=0.0, junction_at=None,
                 weather="clear", friction=1.0):
        self.left_same, self.left_opp, self.right_same = left_same, left_opp, right_same
        self.junction_at = junction_at
        self.weather, self.friction = weather, friction
        self.actors = []
        self.light = None          # (state_fn(t), x_line)
        self.ex, self.ey, self.eyaw, self.ev = 0.0, 0.0, 0.0, 0.0
        self.t = 0.0
        self.collision = None
        self.min_gap = 99.0

    def geo(self):
        n = 90
        s0 = self.ex - 20.0
        junc = []
        for i in range(n):
            s = s0 + 2.0 * i
            junc.append(self.junction_at is not None and self.junction_at[0] <= s <= self.junction_at[1])
        return LaneGeo(s0=s0, ds=2.0, half_width=[LANE_W / 2] * n,
                       left_same=[self.left_same] * n,
                       left_opp=[0.0 if self.left_same else self.left_opp] * n,
                       right_same=[self.right_same] * n, junction=junc,
                       curvature=[0.0] * n)

    def scene(self):
        objs = []
        for a in self.actors:
            if abs(a.x - self.ex) > 90:
                continue
            sp = math.hypot(a.vx, a.vy)
            objs.append(Obj(id=a.id, kind=a.kind, s=a.x, d=a.y, vs=a.vx, vd=a.vy,
                            half_len=a.hl, half_wid=a.hw, speed=sp, oncoming=a.vx < -0.5))
        light = None
        if self.light is not None:
            st = self.light[0](self.t)
            light = LightInfo(state=st, s_line=self.light[1])
        in_j = self.junction_at is not None and self.junction_at[0] <= self.ex <= self.junction_at[1]
        return Scene(t=self.t, dt=DT, ego_s=self.ex, ego_d=self.ey, ego_v=max(0.0, self.ev),
                     ego_half_len=EGO_HL, ego_half_wid=EGO_HW, objs=objs, geo=self.geo(),
                     light=light, in_junction=in_j, weather=self.weather, friction=self.friction)

    def check_collision(self):
        for a in self.actors:
            # oriented boxes are axis aligned enough on a straight road (+ ego yaw small)
            dx = abs(a.x - self.ex) - (a.hl + EGO_HL)
            dy = abs(a.y - self.ey) - (a.hw + EGO_HW)
            gap = max(dx, dy)
            self.min_gap = min(self.min_gap, gap)
            if dx < 0 and dy < 0 and self.collision is None:
                self.collision = (round(self.t, 2), a.kind, a.id)


def run(world, seconds, v0=0.0, red_violation_x=None):
    planner, spid, steer = BehaviorPlanner(), SpeedPID(), SteeringPID()
    world.ev = v0
    planner.reset(world.ex, world.ey)
    log = []
    red_viol = False
    stopped_time = 0.0
    max_stop = 0.0
    for k in range(int(seconds / DT)):
        sc = world.scene()
        out = planner.step(sc)
        thr, brk, _ = spid.step(world.ev, out.target_speed, out.accel_req, DT, reverse=out.reverse)
        # lateral: look-ahead point on offset path
        ld = SteeringPID.lookahead(abs(world.ev))
        tx = world.ex + ld
        ty = planner.profile.value(tx)
        if out.reverse:
            st = 0.0
        else:
            alpha = steer.target_angle(world.ex, world.ey, world.eyaw, tx, ty)
            st = steer.step(alpha, ld, DT, world.ev)
        # vehicle model
        if out.reverse:
            acc = -(thr * 3.0) + (brk * 8.0 if world.ev < 0 else 0.0)
            world.ev = max(-2.0, world.ev + acc * DT)
        else:
            if world.ev < -0.01:          # still rolling backwards: throttle/brake both stop it
                world.ev = min(0.0, world.ev + (thr * 4.0 + brk * 9.0 + 0.3) * DT)
            else:
                acc = thr * 4.0 - brk * 9.0 - 0.02 * world.ev * world.ev - 0.15
                world.ev = max(0.0, world.ev + acc * DT)
        delta = st * math.radians(70.0)
        world.eyaw += world.ev / 2.875 * math.tan(delta) * DT
        world.ex += world.ev * math.cos(world.eyaw) * DT
        world.ey += world.ev * math.sin(world.eyaw) * DT
        world.t += DT
        for a in world.actors:
            a.step(world.t, DT, world)
        world.check_collision()
        if world.light is not None and red_violation_x is not None:
            if world.light[0](world.t) == "RED" and world.ex + EGO_HL > world.light[1] + 0.5 \
                    and world.ex + EGO_HL - world.ev * DT <= world.light[1] + 0.5:
                red_viol = True
        if world.ev < 0.2:
            stopped_time += DT
            max_stop = max(max_stop, stopped_time)
        else:
            stopped_time = 0.0
        if k % 20 == 0:
            log.append((round(world.t, 1), round(world.ex, 1), round(world.ey, 2), round(world.ev * 3.6, 1),
                        out.intent, out.reason[:70]))
    return dict(x=world.ex, y=world.ey, v=world.ev, collision=world.collision, min_gap=world.min_gap,
                red_viol=red_viol, max_stop=max_stop, log=log)


# ----------------------------------------------------------------- scenarios
def tc_clear():
    w = World()
    r = run(w, 25)
    return r, r["collision"] is None and r["x"] > 150 and abs(r["y"]) < 0.4


def tc_static_obstacle():
    w = World()
    w.actors.append(Actor(1, "vehicle", 20, 0.0))
    r = run(w, 30)
    return r, r["collision"] is None and r["x"] > 80 and abs(r["y"]) < 0.5


def tc_static_obstacle_oncoming():
    w = World()
    w.actors.append(Actor(1, "vehicle", 35, 0.3))
    w.actors.append(Actor(2, "vehicle", 110, -3.5, vx=-7.0))
    r = run(w, 40, v0=5.5)
    return r, r["collision"] is None and r["x"] > 70


def tc_lead_slow():
    w = World()
    w.actors.append(Actor(1, "vehicle", 45, 0.0, vx=15 / 3.6))
    r = run(w, 40, v0=25 / 3.6)
    return r, r["collision"] is None and r["min_gap"] > 2.0


def tc_lead_hard_brake():
    def brake(a, t, e):
        if t > 4.0:
            a.vx = max(0.0, a.vx - 8.0 * DT)
    w = World(left_opp=0.0)
    w.actors.append(Actor(1, "vehicle", 25, 0.0, vx=30 / 3.6, script=brake))
    r = run(w, 20, v0=30 / 3.6)
    return r, r["collision"] is None


def tc_stop_and_go():
    def sg(a, t, e):
        if 3.0 < t < 9.0:
            a.vx = max(0.0, a.vx - 6.0 * DT)
        elif t >= 9.0:
            a.vx = min(20 / 3.6, a.vx + 2.0 * DT)
    w = World(left_opp=0.0)
    w.actors.append(Actor(1, "vehicle", 30, 0.0, vx=20 / 3.6, script=sg))
    r = run(w, 30, v0=20 / 3.6)
    return r, r["collision"] is None and r["x"] > 120


def tc_cut_in():
    def cut(a, t, e):
        if a.x - e.ex < 22 and a.y > 0.2:
            a.vy = -1.2
        elif a.y <= 0.2:
            a.vy, a.y = 0.0, 0.0
    w = World(right_same=LANE_W)
    w.actors.append(Actor(1, "vehicle", 20, 3.5, vx=20 / 3.6, script=cut))
    r = run(w, 25, v0=30 / 3.6)
    return r, r["collision"] is None


def tc_ped_crossing():
    def walk(a, t, e):
        a.vy = -1.5 if (a.x - e.ex < 28 and a.y > -5.0) else 0.0
    w = World()
    w.actors.append(Actor(1, "walker", 40, 4.0, hl=0.3, hw=0.3, script=walk))
    r = run(w, 30, v0=25 / 3.6)
    return r, r["collision"] is None and r["x"] > 100


def tc_ped_running():
    def run_(a, t, e):
        a.vy = -3.5 if (a.x - e.ex < 22 and a.y > -5.0) else 0.0
    w = World()
    w.actors.append(Actor(1, "walker", 35, 4.0, hl=0.3, hw=0.3, script=run_))
    r = run(w, 30, v0=25 / 3.6)
    return r, r["collision"] is None and r["x"] > 100


def tc_ped_standing():
    w = World()
    w.actors.append(Actor(1, "walker", 40, 3.0, hl=0.3, hw=0.3))
    r = run(w, 20, v0=25 / 3.6)
    return r, r["collision"] is None and r["max_stop"] < 0.5 and r["x"] > 100


def tc_ped_stops_in_lane():
    def walk(a, t, e):
        if not hasattr(a, "t0"):
            a.t0 = None
        if a.t0 is None and a.x - e.ex < 28:
            a.t0 = t
        if a.t0 is None:
            a.vy = 0.0
            return
        el = t - a.t0
        a.vy = 0.0 if 2.5 <= el < 6.5 else (-1.5 if a.y > -5 else 0.0)
    w = World()
    w.actors.append(Actor(1, "walker", 40, 4.0, hl=0.3, hw=0.3, script=walk))
    r = run(w, 35, v0=25 / 3.6)
    return r, r["collision"] is None and r["x"] > 100


def tc_red_light():
    w = World(junction_at=(62, 80))
    w.light = (lambda t: "RED", 60.0)
    r = run(w, 25, v0=25 / 3.6, red_violation_x=60)
    ok = r["collision"] is None and not r["red_viol"] and 50.0 < r["x"] + EGO_HL <= 60.3 and r["v"] < 0.3
    return r, ok


def tc_red_to_green():
    w = World(junction_at=(62, 80))
    w.light = (lambda t: "RED" if t < 12 else "GREEN", 60.0)
    r = run(w, 30, v0=25 / 3.6, red_violation_x=60)
    return r, r["collision"] is None and not r["red_viol"] and r["x"] > 110


def tc_green_to_red():
    w = World(junction_at=(62, 80))
    w.light = (lambda t: "GREEN" if t < 3 else "RED", 60.0)
    r = run(w, 25, v0=25 / 3.6, red_violation_x=60)
    return r, r["collision"] is None and not r["red_viol"]


def tc_red_with_queue():
    w = World(junction_at=(62, 80))
    w.light = (lambda t: "RED", 60.0)
    w.actors += [Actor(1, "vehicle", 50, 0.0), Actor(2, "vehicle", 57, 0.0)]
    r = run(w, 30, v0=25 / 3.6, red_violation_x=60)
    return r, r["collision"] is None and r["x"] < 45 and abs(r["y"]) < 0.5


def tc_queue():
    w = World()
    w.actors += [Actor(1, "vehicle", 40, 0.0), Actor(2, "vehicle", 47, 0.0), Actor(3, "vehicle", 54, 0.0)]
    r = run(w, 25, v0=25 / 3.6)
    return r, r["collision"] is None and r["x"] < 35


def tc_barrier_closure():
    w = World()
    w.actors += [Actor(1, "prop", 40, -1.6, hl=0.3, hw=0.9), Actor(2, "prop", 40, 0.0, hl=0.3, hw=0.9),
                 Actor(3, "prop", 40, 1.6, hl=0.3, hw=0.9)]
    r = run(w, 30, v0=20 / 3.6)
    # road works close our lane: stop, check the oncoming lane, then pass through it
    return r, r["collision"] is None and r["x"] > 80


def tc_lane_drop_merge():
    w = World(left_same=LANE_W)
    w.actors += [Actor(1, "prop", 35, 0.0, hl=0.25, hw=0.25), Actor(2, "prop", 42, 0.4, hl=0.25, hw=0.25),
                 Actor(3, "prop", 49, 0.8, hl=0.25, hw=0.25)]
    r = run(w, 30, v0=25 / 3.6)
    return r, r["collision"] is None and r["x"] > 100


def tc_wrong_way():
    def drive(a, t, e):
        a.vx = -5.0
    w = World(right_same=LANE_W)
    w.actors.append(Actor(1, "vehicle", 45, 0.0, vx=-5.0, script=drive))
    r = run(w, 20, v0=25 / 3.6)
    return r, r["collision"] is None


def tc_blind_spot():
    def pace(a, t, e):
        a.vx = e.ev
    w = World(left_same=LANE_W)
    w.actors.append(Actor(1, "vehicle", -8, -3.5, vx=25 / 3.6, script=pace))
    r = run(w, 20, v0=25 / 3.6)
    return r, r["collision"] is None and abs(r["y"]) < 0.6


def tc_no_room():
    w = World()
    w.actors.append(Actor(1, "vehicle", 10, 0.0))
    r = run(w, 30)
    return r, r["collision"] is None


def tc_slow_truck_multilane():
    w = World(left_same=LANE_W)
    w.actors.append(Actor(1, "vehicle", 22, 0.0, vx=2.7, hl=4.0, hw=1.2))
    r = run(w, 40, v0=0.0)
    return r, r["collision"] is None and r["x"] > 120


def tc_cone_chicane():
    w = World()
    w.actors += [Actor(1, "prop", 30, -1.5, hl=0.2, hw=0.2), Actor(2, "prop", 40, 1.5, hl=0.2, hw=0.2),
                 Actor(3, "prop", 50, -1.5, hl=0.2, hw=0.2)]
    r = run(w, 20, v0=15 / 3.6)
    return r, r["collision"] is None and r["x"] > 80


def tc_heavy_rain_lead_brake():
    def brake(a, t, e):
        if t > 4.0:
            a.vx = max(0.0, a.vx - 8.0 * DT)
    w = World(weather="heavy_rain", friction=0.7, left_opp=0.0)
    w.actors.append(Actor(1, "vehicle", 30, 0.0, vx=25 / 3.6, script=brake))
    r = run(w, 20, v0=25 / 3.6)
    return r, r["collision"] is None


def tc_tight_corridor():
    w = World()
    w.actors += [Actor(1, "vehicle", 15, -2.0, hw=0.8), Actor(2, "vehicle", 18, 2.0, hw=0.8)]
    r = run(w, 30)
    return r, r["collision"] is None and r["x"] > 60


def tc_parked_close_partial():
    w = World(right_same=LANE_W)
    w.actors += [Actor(1, "vehicle", 12, -1.8)]
    r = run(w, 25)
    return r, r["collision"] is None and r["x"] > 80


def tc_barrier_closure_oncoming():
    w = World()
    w.actors += [Actor(1, "prop", 40, -1.6, hl=0.3, hw=0.9), Actor(2, "prop", 40, 0.0, hl=0.3, hw=0.9),
                 Actor(3, "prop", 40, 1.6, hl=0.3, hw=0.9),
                 Actor(4, "vehicle", 140, -3.5, vx=-8.0), Actor(5, "vehicle", 175, -3.5, vx=-8.0)]
    r = run(w, 40, v0=20 / 3.6)
    ok = r["collision"] is None and r["x"] > 80
    # must not have entered the opposite lane before the oncoming cars passed (~t=13 s)
    early = [row for row in r["log"] if row[0] < 12.0 and row[2] < -1.5]
    return r, ok and not early


def tc_multilane_prefers_own_side():
    """Our side has 2 lanes (we are in the left one, next to the opposite lane):
    a stopped car must be passed on the RIGHT (same direction), never via the opposite lane."""
    w = World(right_same=LANE_W, left_opp=LANE_W)
    w.actors.append(Actor(1, "vehicle", 30, 0.0))
    r = run(w, 30, v0=25 / 3.6)
    went_opp = any(row[2] < -1.2 for row in r["log"])
    return r, r["collision"] is None and r["x"] > 100 and not went_opp


def tc_multilane_busy_waits():
    """Right lane has moving traffic alongside: wait for the gap instead of using the opposite lane."""
    def pace(a, t, e):
        a.vx = 6.0
    w = World(right_same=LANE_W, left_opp=LANE_W)
    w.actors += [Actor(1, "vehicle", 30, 0.0), Actor(2, "vehicle", -5, 3.5, vx=6.0, script=pace)]
    r = run(w, 35, v0=20 / 3.6)
    went_opp = any(row[2] < -1.2 for row in r["log"])
    return r, r["collision"] is None and not went_opp and r["x"] > 60


def tc_reverse_rear_blocked():
    """Too close to steer out AND a car parked right behind us: must not reverse into it."""
    w = World()
    w.actors += [Actor(1, "vehicle", 7.6, 0.0), Actor(2, "vehicle", -6.6, 0.0)]
    r = run(w, 20)
    return r, r["collision"] is None


def tc_bus_with_ped_in_next_lane():
    """Stopped bus in our lane, pedestrian standing/crossing in the free same-direction lane:
    wait for the pedestrian, then change to OUR other lane - never use the opposite lane."""
    def walk(a, t, e):
        a.vy = -1.2 if 4.0 < t < 9.0 else 0.0
    w = World(right_same=LANE_W, left_opp=LANE_W)
    w.actors += [Actor(1, "vehicle", 35, 0.0, hl=5.0, hw=1.25), Actor(2, "walker", 40, 6.5, hl=0.3, hw=0.3, script=walk)]
    r = run(w, 40, v0=25 / 3.6)
    went_opp = any(row[2] < -1.2 for row in r["log"])
    return r, r["collision"] is None and not went_opp and r["x"] > 80


def tc_roadwork_taper():
    w = World()
    w.actors += [Actor(1, "prop", 24, 1.7, hl=0.4, hw=0.4), Actor(2, "prop", 30, 1.2, hl=0.25, hw=0.25),
                 Actor(3, "prop", 36, 0.7, hl=0.25, hw=0.25), Actor(4, "prop", 42, 0.2, hl=0.25, hw=0.25)]
    r = run(w, 30, v0=20 / 3.6)
    return r, r["collision"] is None and r["x"] > 90


def tc_wrong_way_no_escape():
    """Wrong-way car in our lane, no free lane to dodge into (oncoming lane busy):
    brake at once and be (almost) stopped before any contact to minimise the impact."""
    def drive(a, t, e):
        a.vx = -5.0
    w = World(left_opp=LANE_W)
    w.actors += [Actor(1, "vehicle", 45, 0.0, vx=-5.0, script=drive),
                 Actor(2, "vehicle", 40, -3.5, vx=-6.0, script=lambda a, t, e: None)]
    r = run(w, 12, v0=25 / 3.6)
    if r["collision"] is None:
        return r, True
    t_hit = r["collision"][0]
    before = [row for row in r["log"] if row[0] <= t_hit]
    return r, bool(before) and before[-1][3] <= 3.0


def tc_truck_protruding_right_lane_free():
    """Truck stopped in our lane, sticking a bit into the free right lane; opposite lane also free:
    must use OUR right lane (tighter clearance, slower), never the opposite lane."""
    w = World(right_same=LANE_W, left_opp=LANE_W)
    w.actors.append(Actor(1, "vehicle", 40, 1.2, hl=5.0, hw=1.25))
    r = run(w, 35, v0=20 / 3.6)
    went_opp = any(row[2] < -1.2 for row in r["log"])
    return r, r["collision"] is None and not went_opp and r["x"] > 100


def tc_bus_centre_right_lane_free():
    w = World(right_same=LANE_W, left_opp=LANE_W)
    w.actors.append(Actor(1, "vehicle", 35, 0.0, hl=5.5, hw=1.3))
    r = run(w, 35, v0=25 / 3.6)
    went_opp = any(row[2] < -1.2 for row in r["log"])
    return r, r["collision"] is None and not went_opp and r["x"] > 100


def tc_red_blocks_lane_change():
    """Parked car ahead AND a red light just behind it: must not start a bypass; stop in own lane."""
    w = World(right_same=LANE_W, junction_at=(75, 95))
    w.light = (lambda t: "RED", 70.0)
    w.actors.append(Actor(1, "vehicle", 55, 0.0))
    r = run(w, 25, v0=25 / 3.6, red_violation_x=70)
    went_side = any(abs(row[2]) > 1.0 for row in r["log"])
    return r, r["collision"] is None and not r["red_viol"] and not went_side and r["x"] < 55


def tc_red_during_lane_change():
    """Bypass of a parked car starts on green; light turns red mid-manoeuvre: finish the move,
    never cross the line on red, then proceed after green."""
    w = World(right_same=LANE_W, junction_at=(95, 115))
    w.light = (lambda t: "GREEN" if t < 3.0 else ("RED" if t < 16.0 else "GREEN"), 90.0)
    w.actors.append(Actor(1, "vehicle", 45, 0.0))
    r = run(w, 40, v0=25 / 3.6, red_violation_x=90)
    return r, r["collision"] is None and not r["red_viol"] and r["x"] > 100


TESTS = [
    ("red light blocks lane change", tc_red_blocks_lane_change),
    ("red during lane change", tc_red_during_lane_change),
    ("truck sticks into free right lane", tc_truck_protruding_right_lane_free),
    ("bus in lane, right lane free", tc_bus_centre_right_lane_free),
    ("wrong-way, no escape: brake", tc_wrong_way_no_escape),
    ("bus + pedestrian in next lane", tc_bus_with_ped_in_next_lane),
    ("road-work cone taper", tc_roadwork_taper),
    ("barrier closure + oncoming", tc_barrier_closure_oncoming),
    ("multi-lane: overtake on own side", tc_multilane_prefers_own_side),
    ("multi-lane busy: wait, no opp", tc_multilane_busy_waits),
    ("reverse with rear blocked", tc_reverse_rear_blocked),
    ("tight corridor squeeze", tc_tight_corridor),
    ("parked car close, partial", tc_parked_close_partial),
    ("clear lane cruise", tc_clear),
    ("static obstacle bypass", tc_static_obstacle),
    ("static obstacle + oncoming", tc_static_obstacle_oncoming),
    ("slow lead follow", tc_lead_slow),
    ("lead hard brake", tc_lead_hard_brake),
    ("lead stop-and-go", tc_stop_and_go),
    ("cut-in", tc_cut_in),
    ("pedestrian crossing", tc_ped_crossing),
    ("pedestrian running", tc_ped_running),
    ("pedestrian standing (no FP)", tc_ped_standing),
    ("pedestrian pauses in lane", tc_ped_stops_in_lane),
    ("red light stop", tc_red_light),
    ("red -> green resume", tc_red_to_green),
    ("green -> red", tc_green_to_red),
    ("red light with queue", tc_red_with_queue),
    ("stationary queue", tc_queue),
    ("barrier closure", tc_barrier_closure),
    ("lane-drop merge", tc_lane_drop_merge),
    ("wrong-way driver", tc_wrong_way),
    ("blind-spot vehicle", tc_blind_spot),
    ("no room to move", tc_no_room),
    ("slow truck lane change", tc_slow_truck_multilane),
    ("cone chicane", tc_cone_chicane),
    ("heavy rain lead brake", tc_heavy_rain_lead_brake),
]


def main(verbose=False):
    fails = 0
    for name, fn in TESTS:
        r, ok = fn()
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:30s} x={r['x']:6.1f} y={r['y']:+5.2f} "
              f"v={r['v'] * 3.6:5.1f} km/h  min_gap={r['min_gap']:5.2f}  coll={r['collision']}  "
              f"max_stop={r['max_stop']:.1f}s")
        if verbose or not ok:
            for row in r["log"]:
                print("      ", row)
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main("-v" in sys.argv) else 0)
