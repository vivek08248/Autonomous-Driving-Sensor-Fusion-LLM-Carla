from collections import deque
import config

class VehiclePIDController:
    def __init__(self):
        self.kp = config.PID_KP_SPEED
        self.kd = config.PID_KD_SPEED
        self.ki = config.PID_KI_SPEED
        self.error_buffer = deque(maxlen=10)

    def reset(self):
        self.error_buffer.clear()

    def calculate_longitudinal_control(self, current_speed_kmh, target_speed_kmh):
        if target_speed_kmh <= 0.0:
            self.reset()
            return 0.0, 1.0

        error = target_speed_kmh - current_speed_kmh
        self.error_buffer.append(error)

        de = (
            (error - self.error_buffer[-2]) / max(config.SIM_DELTA_SECONDS, 1e-3)
            if len(self.error_buffer) > 1 else 0.0
        )
        ie = sum(self.error_buffer) * config.SIM_DELTA_SECONDS

        output = (self.kp * error) + (self.kd * de) + (self.ki * ie)

        throttle = min(max(output, 0.0), 1.0)
        brake = min(max(-output, 0.0), 1.0)

        return throttle, brake
