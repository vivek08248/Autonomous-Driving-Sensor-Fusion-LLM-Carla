"""
ad_control.py - Low-level controllers (paper Sec. 4.2: "two PID controllers convert the
predicted decision variables into throttle, steer and brake").

  * SpeedPID      : target speed (+ optional required deceleration) -> throttle / brake
  * SteeringPID   : target angle (pure-pursuit heading to a look-ahead point on the
                    planned, laterally-offset path) -> steer, with rate limiting so the
                    car never zig-zags.

Pure Python, no CARLA import.
"""
import math

try:
    import config as _cfg
except Exception:
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class SpeedPID:
    """Acceleration-shaped speed controller with feed-forward and anti-windup."""

    def __init__(self):
        self.kp_a = float(_c("AD_SPEED_KP", 1.1))      # (m/s^2) per (m/s) error
        self.ki = float(_c("AD_SPEED_KI", 0.06))       # throttle per (m/s * s)
        self.k_thr = float(_c("AD_THROTTLE_PER_ACCEL", 0.16))
        self.k_brk = float(_c("AD_BRAKE_PER_DECEL", 1.0 / 7.5))
        self.thr_max = float(_c("AD_THROTTLE_MAX", 0.80))
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev_thr = 0.0
        self.prev_brk = 0.0

    def step(self, v, v_target, a_req, dt, reverse=False):
        """Return (throttle, brake, a_des)."""
        if reverse:
            err = v_target - abs(v)
            thr = _clip(0.35 + 0.15 * err, 0.0, 0.5)
            brk = 0.0 if err > -0.3 else _clip(-0.2 * err, 0.0, 0.5)
            return thr, brk, 0.0
        err = v_target - v
        a_des = _clip(self.kp_a * err, -7.5, 2.5)
        if a_req is not None:
            a_des = min(a_des, a_req)
        # stand-still hold: no creep, no roll-back
        if v_target < 0.25 and v < 0.6:
            self.i = 0.0
            self.prev_thr, self.prev_brk = 0.0, 0.6
            return 0.0, 0.6, a_des
        if a_des > -0.25:
            self.i = _clip(self.i + err * dt, -2.0, 6.0)
            ff = 0.20 + 0.012 * v if v_target > 0.3 else 0.0
            thr = _clip(ff + self.k_thr * max(a_des, 0.0) + self.ki * self.i, 0.0, self.thr_max)
            if a_des < 0.0:
                thr *= 0.5
            brk = 0.0
            # smooth throttle onset (avoids wheel spin / jerky starts)
            thr = min(thr, self.prev_thr + 1.5 * dt)
        else:
            self.i = max(0.0, self.i - 2.0 * dt)
            thr = 0.0
            brk = _clip(self.k_brk * (-a_des) + 0.05, 0.0, 1.0)
        self.prev_thr, self.prev_brk = thr, brk
        return thr, brk, a_des


class SteeringPID:
    """Pure-pursuit target angle + PD shaping + rate limit."""

    def __init__(self, wheelbase=2.875, max_steer_deg=70.0):
        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer_deg)
        self.kd = float(_c("AD_STEER_KD", 0.08))
        self.rate = float(_c("AD_STEER_RATE", 2.2))    # normalised steer per second
        self.reset()

    def reset(self):
        self.prev_alpha = 0.0
        self.prev_steer = 0.0

    @staticmethod
    def lookahead(v):
        return _clip(2.5 + 0.55 * v, 3.0, 13.0)

    def target_angle(self, ego_x, ego_y, ego_yaw_rad, tx, ty):
        dx, dy = tx - ego_x, ty - ego_y
        ang = math.atan2(dy, dx) - ego_yaw_rad
        return math.atan2(math.sin(ang), math.cos(ang))

    def step(self, alpha, ld, dt, v=0.0):
        """alpha = target angle [rad] to look-ahead point at distance ld -> normalised steer."""
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), max(ld, 1.0))
        d_alpha = (alpha - self.prev_alpha) / max(dt, 1e-3)
        self.prev_alpha = alpha
        cmd = delta / self.max_steer + self.kd * d_alpha * 0.1
        # at speed the usable steer range shrinks (stability)
        lim = _clip(1.0 - 0.025 * v, 0.45, 1.0)
        cmd = _clip(cmd, -lim, lim)
        step_max = self.rate * dt
        cmd = _clip(cmd, self.prev_steer - step_max, self.prev_steer + step_max)
        self.prev_steer = cmd
        return cmd
