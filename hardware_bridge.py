import cv2
import serial
import time
import config

class PhysicalCamera:
    def __init__(self, camera_index=config.CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open physical camera at index {camera_index}")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.resize(frame, (800, 600))
        return None

    def release(self):
        self.cap.release()

class PhysicalActuator:
    def __init__(self, port=config.SERIAL_PORT, baudrate=config.BAUD_RATE):
        self.mock_mode = False
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2.0)
            print(f"[Actuator] Connected to ESP32 on {port}")
        except Exception as e:
            print(f"[Actuator Warning] ESP32 not found on {port}. Entering MOCK MODE.")
            self.ser = None
            self.mock_mode = True

    def send_control(self, throttle_pct, steer_deg):
        command_str = f"T:{throttle_pct:.1f},S:{steer_deg:.1f}\n"
        if self.mock_mode:
            # Instead of sending electrical signals, just print what it WOULD do
            print(f"[Mock Motors] -> {command_str.strip()}")
        elif self.ser and self.ser.is_open:
            self.ser.write(command_str.encode('utf-8'))

    def stop(self):
        self.send_control(0.0, 0.0)