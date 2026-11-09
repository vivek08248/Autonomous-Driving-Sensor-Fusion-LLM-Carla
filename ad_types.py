"""
ad_types.py - Plain-Python data structures shared by the driving stack.

Nothing in here imports CARLA, so the planner / controller / LLM arbiter can be
unit-tested offline (see offline_planner_tests.py).

Coordinate convention (route / Frenet frame, same as CARLA's left-handed frame):
    s  = metres along the planned route (absolute, grows while driving)
    d  = lateral offset from the route-lane centre, + = RIGHT, - = LEFT
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Obj:
    """One surrounding object expressed in the route frame."""
    id: int
    kind: str                 # vehicle | bike | walker | animal | prop | unknown
    s: float                  # route station of the object centre [m]
    d: float                  # lateral offset of the object centre [m] (+ right)
    vs: float = 0.0           # velocity along the route [m/s]
    vd: float = 0.0           # velocity across the route [m/s] (+ = moving right)
    half_len: float = 2.2     # half extent along the route [m]
    half_wid: float = 0.9     # half extent across the route [m]
    speed: float = 0.0        # absolute speed [m/s]
    oncoming: bool = False    # heading opposite to the route direction
    in_junction: bool = False
    label: str = ""           # blueprint id (for logs / LLM prompt)

    @property
    def is_vru(self) -> bool:
        return self.kind in ("walker", "animal", "bike")

    @property
    def d_min(self) -> float:
        return self.d - self.half_wid

    @property
    def d_max(self) -> float:
        return self.d + self.half_wid

    @property
    def s_min(self) -> float:
        return self.s - self.half_len

    @property
    def s_max(self) -> float:
        return self.s + self.half_len


@dataclass
class LaneGeo:
    """Lane geometry sampled every `ds` metres starting at absolute station `s0`."""
    s0: float
    ds: float
    half_width: List[float]          # half width of the route lane
    left_same: List[float]           # width of a same-direction driving lane on the left (0 = none)
    left_opp: List[float]            # width of an opposite-direction driving lane on the left (0 = none)
    right_same: List[float]          # width of a same-direction driving lane on the right (0 = none)
    junction: List[bool]
    curvature: List[float]           # |1/R| [1/m]
    left_same_n: Optional[List[int]] = None    # number of same-direction lanes on the left
    right_same_n: Optional[List[int]] = None   # number of same-direction lanes on the right

    def idx(self, s: float) -> int:
        i = int(round((s - self.s0) / self.ds))
        return max(0, min(len(self.half_width) - 1, i))

    def at(self, s: float):
        i = self.idx(s)
        return (self.half_width[i], self.left_same[i], self.left_opp[i],
                self.right_same[i], self.junction[i], self.curvature[i])

    def lanes(self, s: float):
        """Lane centres reachable from the route lane at s.
        Returns (same_dir_centres, opposite_centre or None); left = negative d."""
        i = self.idx(s)
        hw = self.half_width[i]
        same = []
        nl = (self.left_same_n[i] if self.left_same_n else (1 if self.left_same[i] > 0 else 0))
        nr = (self.right_same_n[i] if self.right_same_n else (1 if self.right_same[i] > 0 else 0))
        if nl > 0:
            w = self.left_same[i] / nl
            same += [-(hw + w * (k + 0.5)) for k in range(nl)]
        if nr > 0:
            w = self.right_same[i] / nr
            same += [hw + w * (k + 0.5) for k in range(nr)]
        opp = None
        if self.left_opp[i] > 0:
            opp = -(hw + self.left_same[i] + 0.5 * self.left_opp[i])
        return same, opp

    @property
    def s_end(self) -> float:
        return self.s0 + self.ds * (len(self.half_width) - 1)


@dataclass
class LightInfo:
    state: str                # RED | YELLOW | GREEN | OFF
    s_line: float             # absolute route station of the stop line


@dataclass
class Scene:
    """Everything the behaviour planner needs for one tick."""
    t: float
    dt: float
    ego_s: float
    ego_d: float              # actual lateral position of the ego centre w.r.t. the route lane
    ego_v: float              # m/s (>= 0, forward)
    ego_half_len: float
    ego_half_wid: float
    objs: List[Obj]
    geo: LaneGeo
    light: Optional[LightInfo] = None
    in_junction: bool = False
    weather: str = "clear"    # clear | rain | heavy_rain | fog | night | night_rain | glare | wet
    friction: float = 1.0     # 1 = dry, <1 = wet
    sensor_gap: Optional[float] = None   # confirmed LiDAR/radar gap [m] from the front bumper (fusion)
    route_pref: str = "straight"
    ego_reversing: bool = False
    rear_gap: Optional[float] = None     # LiDAR free distance behind the rear bumper [m] (None = clear)


@dataclass
class PlanOut:
    target_speed: float           # m/s (>= 0)
    accel_req: Optional[float]    # m/s^2 (negative = required deceleration) or None
    offset_target: float          # lateral offset the path is steering towards [m]
    intent: str                   # expert meta-action (paper "decision")
    reason: str
    hold: str = ""                # why the car is allowed to stand still ("" = not holding)
    reverse: bool = False
    lead_gap: float = 999.0       # gap to the governing in-path object (front bumper -> object) [m]
    lead_speed: float = 0.0
    lead_kind: str = ""
    safe_max_speed: float = 0.0   # envelope used by the LLM arbiter [m/s]
    bypass_side: str = ""         # LEFT / RIGHT while a bypass / lane change is active
    debug: dict = field(default_factory=dict)
