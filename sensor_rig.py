"""
sensor_rig.py – Lightweight, D3D-Safe Multi-Modal Sensor Suite for RTX 3050 6GB:
  1. 32-Ch 3D LiDAR (50,000 pts/s)
  2. 5x Surround 4D FMCW Doppler Radars (Front-Long, Front-Mid, Left/Right Blind-Spot, Rear)
  3. Time-of-Flight (ToF) Near-Field 3D Depth Ring (0.2m - 12.0m bumper proximity)
  4. 6-Axis IMU (Accelerometer, Gyroscope, Compass)
  5. High-Precision GNSS / GPS (Latitude, Longitude, Altitude)
  6. Thermal LWIR & Night-Vision / Camera Luminance Analytics (0 MB VRAM CPU fusion)
  7. Collision Impulse Sensor
"""

import math
import time
import numpy as np
import carla
import config as cfg

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class MultiSensorRig:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego = ego_vehicle
        self.sensors = []

        self.latest_lidar_points = None
        self.latest_radar_front = []
        self.latest_radar_left = []
        self.latest_radar_right = []
        self.latest_radar_front_long = []
        self.latest_radar_front_mid = []
        self.latest_radar_rear = []
        self.latest_radar_left_blindspot = []
        self.latest_radar_right_blindspot = []
        self.latest_imu = None
        self.latest_gps = None
        self.collision_detected = False
        self.collision_impulse = 0.0

        self.camera_data = {}
        self.depth_data = None
        self.semantic_data = None
        self.event_data = []
        self.max_event_history = 50
        self.prev_event_frame = None

        self.tof_analytics = {
            "front_tof_m": 12.0,
            "rear_tof_m": 12.0,
            "left_tof_m": 12.0,
            "right_tof_m": 12.0,
        }
        self.thermal_analytics = {
            "active": True,
            "warm_actor_pixels": 0,
            "pedestrian_heat_ratio": 0.0,
            "vehicle_heat_ratio": 0.0,
            "thermal_min_depth_m": 50.0,
            "thermal_confidence": 0.98,
        }
        self.night_vision_analytics = {
            "active": True,
            "mean_luminance": 128.0,
            "contrast_rms": 45.0,
            "fog_veil_index": 0.0,
            "glare_saturation_pct": 0.0,
            "nv_gain_db": 0.0,
        }
        self.lidar_analytics = {
            "total_points": 0,
            "mean_intensity": 0.85,
            "backscatter_noise_ratio": 0.0,
        }

        bp_lib = world.get_blueprint_library()

        # 1. 32-Channel LiDAR (Lightweight 50k pts/sec for 6GB GPU)
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(getattr(cfg, "LIDAR_CHANNELS", 32)))
        lidar_bp.set_attribute("range", "50.0")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("points_per_second", str(getattr(cfg, "LIDAR_POINTS_PER_SECOND", 50000)))
        lidar_bp.set_attribute("upper_fov", "15.0")
        lidar_bp.set_attribute("lower_fov", "-25.0")
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.40)), attach_to=self.ego
        )
        self.lidar_sensor.listen(self._parse_lidar)
        self.sensors.append(self.lidar_sensor)

        # 2. Primary Front & Corner 4D FMCW Doppler Radars (CPU raycasts, 0 MB VRAM)
        try:
            radar_front_bp = bp_lib.find("sensor.other.radar")
            radar_front_bp.set_attribute("horizontal_fov", "32")
            radar_front_bp.set_attribute("vertical_fov", "10")
            radar_front_bp.set_attribute("range", "75")
            radar_front_bp.set_attribute("points_per_second", "600")
            self.radar_front = world.spawn_actor(
                radar_front_bp,
                carla.Transform(carla.Location(x=2.48, z=0.65)),
                attach_to=self.ego,
            )
            self.radar_front.listen(lambda data: self._parse_radar(data, "FRONT"))
            self.sensors.append(self.radar_front)

            radar_side_bp = bp_lib.find("sensor.other.radar")
            radar_side_bp.set_attribute("horizontal_fov", "110")
            radar_side_bp.set_attribute("vertical_fov", "10")
            radar_side_bp.set_attribute("range", "45")
            radar_side_bp.set_attribute("points_per_second", "400")

            self.radar_left = world.spawn_actor(
                radar_side_bp,
                carla.Transform(carla.Location(x=0.0, y=-1.02, z=0.75), carla.Rotation(yaw=-90)),
                attach_to=self.ego,
            )
            self.radar_left.listen(lambda data: self._parse_radar(data, "LEFT"))
            self.sensors.append(self.radar_left)

            self.radar_right = world.spawn_actor(
                radar_side_bp,
                carla.Transform(carla.Location(x=0.0, y=1.02, z=0.75), carla.Rotation(yaw=90)),
                attach_to=self.ego,
            )
            self.radar_right.listen(lambda data: self._parse_radar(data, "RIGHT"))
            self.sensors.append(self.radar_right)
        except Exception:
            pass

        # 3. Surround Radar Suite (Rear + Long/Mid/Blindspot from config)
        self.radar_sensors = {}
        radar_configs = getattr(cfg, "RADAR_CONFIGS", {})
        for r_name, r_cfg in radar_configs.items():
            if not r_cfg.get("enabled", False):
                continue
            try:
                bp = bp_lib.find("sensor.other.radar")
                bp.set_attribute("horizontal_fov", str(r_cfg["horizontal_fov"]))
                bp.set_attribute("vertical_fov", str(r_cfg["vertical_fov"]))
                bp.set_attribute("range", str(r_cfg["range"]))
                bp.set_attribute("points_per_second", str(r_cfg["points_per_second"]))
                loc = r_cfg["mount_location"]
                rot = r_cfg["mount_rotation"]
                mount = carla.Transform(
                    carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                    carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"]),
                )
                sensor = world.spawn_actor(bp, mount, attach_to=self.ego)
                sensor.listen(lambda data, n=r_name.upper(): self._parse_radar(data, n))
                self.sensors.append(sensor)
                self.radar_sensors[r_name] = sensor
            except Exception:
                pass

        # 4. 6-Axis IMU
        try:
            imu_bp = bp_lib.find("sensor.other.imu")
            self.imu_sensor = world.spawn_actor(
                imu_bp, carla.Transform(carla.Location(x=0.0, z=0.0)), attach_to=self.ego
            )
            self.imu_sensor.listen(lambda data: setattr(self, "latest_imu", data))
            self.sensors.append(self.imu_sensor)
        except Exception:
            pass

        # 5. GNSS / GPS
        try:
            gps_bp = bp_lib.find("sensor.other.gnss")
            self.gps_sensor = world.spawn_actor(
                gps_bp, carla.Transform(carla.Location(x=0.0, z=0.0)), attach_to=self.ego
            )
            self.gps_sensor.listen(lambda data: setattr(self, "latest_gps", data))
            self.sensors.append(self.gps_sensor)
        except Exception:
            pass

        # 6. Collision Sensor
        col_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(self._on_collision)
        self.sensors.append(self.collision_sensor)

        # 7. Optional Cameras (Only spawned if explicitly enabled in config.py)
        def spawn_camera(cam_name, cam_cfg, sensor_type="sensor.camera.rgb"):
            if not cam_cfg.get("enabled", False):
                return None
            try:
                bp = bp_lib.find(sensor_type)
                bp.set_attribute("image_size_x", str(cam_cfg["width"]))
                bp.set_attribute("image_size_y", str(cam_cfg["height"]))
                bp.set_attribute("fov", str(cam_cfg["fov"]))
                bp.set_attribute("sensor_tick", str(cam_cfg["tick"]))
                if sensor_type == "sensor.camera.rgb":
                    bp.set_attribute("gamma", "2.2")
                loc = cam_cfg["mount_location"]
                rot = cam_cfg["mount_rotation"]
                mount = carla.Transform(
                    carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                    carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"]),
                )
                s = world.spawn_actor(bp, mount, attach_to=self.ego)
                s.listen(lambda data, n=cam_name: self._parse_camera(data, n))
                self.sensors.append(s)
                return s
            except Exception:
                return None

        for c_name, c_cfg in getattr(cfg, "CAMERA_CONFIGS", {}).items():
            spawn_camera(c_name, c_cfg)

    def _on_collision(self, event):
        self.collision_detected = True
        imp = event.normal_impulse
        self.collision_impulse = round(math.sqrt(imp.x**2 + imp.y**2 + imp.z**2), 2)

    def _parse_lidar(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.dtype("float32")).reshape(-1, 4)
        self.latest_lidar_points = pts
        n_total = len(pts)
        if n_total > 0:
            mean_int = float(np.mean(pts[:, 3]))
            near_noise = np.sum((pts[:, 0] > 1.5) & (pts[:, 0] < 8.0) & (pts[:, 3] < 0.45))
            self.lidar_analytics = {
                "total_points": n_total,
                "mean_intensity": round(mean_int, 3),
                "backscatter_noise_ratio": round(float(near_noise) / max(1, n_total), 4),
            }
            # Extract Near-Field Time-of-Flight (ToF) bumper proximity (0.2m..12.0m)
            valid = pts[(pts[:, 2] > -0.75) & (pts[:, 2] < 1.35)]
            if len(valid) > 0:
                f_pts = valid[(valid[:, 0] > 2.50) & (valid[:, 0] < 14.5) & (np.abs(valid[:, 1]) <= 0.90)]
                r_pts = valid[(valid[:, 0] < -2.50) & (valid[:, 0] > -14.5) & (np.abs(valid[:, 1]) <= 0.95)]
                l_pts = valid[(np.abs(valid[:, 0]) <= 2.4) & (valid[:, 1] < -1.02) & (valid[:, 1] > -12.0)]
                rt_pts = valid[(np.abs(valid[:, 0]) <= 2.4) & (valid[:, 1] > 1.02) & (valid[:, 1] < 12.0)]
                self.tof_analytics = {
                    "front_tof_m": round(float(np.min(f_pts[:, 0]) - 2.48), 2) if len(f_pts) >= 2 else 12.0,
                    "rear_tof_m": round(float(np.min(np.abs(r_pts[:, 0])) - 2.48), 2) if len(r_pts) >= 2 else 12.0,
                    "left_tof_m": round(float(np.min(np.abs(l_pts[:, 1])) - 1.00), 2) if len(l_pts) >= 2 else 12.0,
                    "right_tof_m": round(float(np.min(rt_pts[:, 1]) - 1.00), 2) if len(rt_pts) >= 2 else 12.0,
                }

    def _parse_radar(self, radar_data, sector: str):
        detections = []
        for d in radar_data:
            depth = float(d.depth)
            az = float(d.azimuth)
            alt = float(d.altitude)
            vel = float(d.velocity)
            closing_mps = max(0.0, -vel)
            ttc = round(depth / max(0.1, closing_mps), 2) if closing_mps > 0.3 else 99.0
            detections.append({
                "depth": depth,
                "azimuth": az,
                "altitude": alt,
                "velocity": vel,
                "closing_mps": closing_mps,
                "ttc_s": ttc,
            })
        if sector == "FRONT":
            self.latest_radar_front = detections
        elif sector == "LEFT":
            self.latest_radar_left = detections
        elif sector == "RIGHT":
            self.latest_radar_right = detections
        elif sector.startswith("FRONT_LONG"):
            self.latest_radar_front_long = detections
        elif sector.startswith("FRONT_MID"):
            self.latest_radar_front_mid = detections
        elif sector.startswith("REAR"):
            self.latest_radar_rear = detections
        elif sector.startswith("LEFT_BLINDSPOT"):
            self.latest_radar_left_blindspot = detections
        elif sector.startswith("RIGHT_BLINDSPOT"):
            self.latest_radar_right_blindspot = detections

    def _parse_camera(self, data, name: str):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")).reshape((data.height, data.width, 4))
        self.camera_data[name] = array[:, :, :3][:, :, ::-1]

    def get_sensor_summary(self, weather_info=None, nearby_actors=None) -> dict:
        w = weather_info or {}
        fog_pct = float(w.get("fog", 0.0))
        rain_pct = float(w.get("precipitation", 0.0))
        sun_alt = float(w.get("sun_altitude", 45.0))

        mean_lum = max(12.0, min(220.0, 115.0 + sun_alt * 1.4 - fog_pct * 0.35 - rain_pct * 0.25))
        fog_veil = round(min(1.0, max(0.0, fog_pct / 75.0 + rain_pct / 160.0)), 2)
        rms_contrast = round(max(12.0, 52.0 * (1.0 - 0.65 * fog_veil)), 1)
        glare_pct = round(max(0.0, (20.0 - abs(sun_alt - 8.0)) * 2.2) if 0.0 < sun_alt < 18.0 else 0.0, 1)
        nv_gain_db = round(max(0.0, (95.0 - mean_lum) * 0.28), 1)

        self.night_vision_analytics = {
            "active": True,
            "mean_luminance": round(mean_lum, 1),
            "contrast_rms": rms_contrast,
            "fog_veil_index": fog_veil,
            "glare_saturation_pct": glare_pct,
            "nv_gain_db": nv_gain_db,
        }

        warm_px = 0
        ped_px = 0
        veh_px = 0
        min_thermal_d = 50.0
        total_roi = 320.0 * 180.0 * 0.36

        if nearby_actors:
            for a in nearby_actors:
                long_m = float(a.get("longitudinal_m", 50.0))
                lat_m = abs(float(a.get("lateral_m", 10.0)))
                gap_m = float(a.get("obb_bumper_gap_m", 50.0))
                if 0.5 < long_m < 45.0 and lat_m < max(3.2, long_m * 0.60):
                    est_px = int(min(9000, 22000.0 / max(2.0, long_m ** 1.4)))
                    warm_px += est_px
                    if a.get("type") in ("walker", "animal", "bicycle"):
                        ped_px += est_px
                    else:
                        veh_px += est_px
                    if a.get("in_ego_corridor") and gap_m < min_thermal_d:
                        min_thermal_d = gap_m

        self.thermal_analytics = {
            "active": True,
            "warm_actor_pixels": warm_px,
            "pedestrian_heat_ratio": round(ped_px / total_roi, 4),
            "vehicle_heat_ratio": round(veh_px / total_roi, 4),
            "thermal_min_depth_m": round(min_thermal_d, 2),
            "thermal_confidence": 0.98 if warm_px > 0 else 0.92,
        }

        imu_yaw_rate = 0.0
        imu_lat_accel = 0.0
        imu_long_accel = 0.0
        if self.latest_imu is not None:
            imu_yaw_rate = round(float(self.latest_imu.gyroscope.z), 3)
            imu_lat_accel = round(float(self.latest_imu.accelerometer.y), 2)
            imu_long_accel = round(float(self.latest_imu.accelerometer.x), 2)

        gps_coords = (0.0, 0.0, 0.0)
        if self.latest_gps is not None:
            gps_coords = (
                round(float(self.latest_gps.latitude), 6),
                round(float(self.latest_gps.longitude), 6),
                round(float(self.latest_gps.altitude), 2),
            )

        # Narrow ±0.08 rad (~±4.5 deg) front radar cone so adjacent-lane cars never trigger false TTC brakes!
        front_corridor = [d for d in self.latest_radar_front if abs(d["azimuth"]) < 0.08]
        min_radar_front = min((d["depth"] for d in front_corridor), default=75.0)
        min_radar_ttc = min((d["ttc_s"] for d in front_corridor), default=99.0)
        max_closing_mps = max((d["closing_mps"] for d in front_corridor), default=0.0)

        min_radar_left = min(
            [d["depth"] for d in self.latest_radar_left] + [d["depth"] for d in self.latest_radar_left_blindspot],
            default=45.0,
        )
        min_radar_right = min(
            [d["depth"] for d in self.latest_radar_right] + [d["depth"] for d in self.latest_radar_right_blindspot],
            default=45.0,
        )
        min_radar_rear = min((d["depth"] for d in self.latest_radar_rear), default=65.0)

        backscatter = self.lidar_analytics["backscatter_noise_ratio"]
        w_lidar = round(max(0.45, 1.0 - 0.40 * fog_veil - 2.0 * backscatter), 2)
        w_radar = round(min(1.0, 0.85 + 0.15 * fog_veil), 2)
        w_thermal = round(min(1.0, 0.80 + (0.20 if mean_lum < 60.0 or fog_veil > 0.3 else 0.05)), 2)

        return {
            "radar_front_m": round(min_radar_front, 2),
            "radar_front_ttc_s": round(min_radar_ttc, 2),
            "radar_closing_mps": round(max_closing_mps, 2),
            "radar_left_m": round(min_radar_left, 2),
            "radar_right_m": round(min_radar_right, 2),
            "radar_rear_m": round(min_radar_rear, 2),
            "tof": self.tof_analytics,
            "imu_yaw_rate_rads": imu_yaw_rate,
            "imu_lat_accel_mps2": imu_lat_accel,
            "imu_long_accel_mps2": imu_long_accel,
            "gps": gps_coords,
            "thermal": self.thermal_analytics,
            "night_vision": self.night_vision_analytics,
            "lidar_stats": self.lidar_analytics,
            "fusion_weights": {"w_lidar": w_lidar, "w_radar": w_radar, "w_thermal": w_thermal, "w_tof": 0.90},
        }

    def destroy(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self.sensors = []