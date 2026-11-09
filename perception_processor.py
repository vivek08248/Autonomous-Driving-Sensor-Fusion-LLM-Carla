import numpy as np
import math

class SensorProcessor:
    @staticmethod
    def process_lidar_obstacle_distance(point_cloud):
        if point_cloud is None or len(point_cloud) == 0:
            return 50.0  # Return default clear distance in meters

        # Filter points in the ego-vehicle travel corridor:
        # Forward X: 0.85m to 50.0m (bumper is at +0.85m from sensor)
        # Lateral Y: -1.9m to +1.9m (covers full lane width & encroachment)
        # Vertical Z: -1.8m to +1.8m (LiDAR at z=2.1 -> sees objects taller than ~0.3m) (above asphalt, captures vehicle hulls & pedestrians)
        in_lane_mask = (
            (point_cloud[:, 0] > 0.85) & 
            (point_cloud[:, 0] < 50.0) & 
            (np.abs(point_cloud[:, 1]) < 1.90) & 
            (point_cloud[:, 2] > (-1.75 + 0.012 * np.maximum(point_cloud[:, 0], 0.0))) & 
            (point_cloud[:, 2] < 1.80)
        )
        
        in_lane_points = point_cloud[in_lane_mask]
        if len(in_lane_points) == 0:
            return 50.0

        # Distance from front bumper to closest obstacle point
        sensor_x = float(np.min(in_lane_points[:, 0]))
        bumper_dist = max(0.0, sensor_x - 0.85)
        return round(bumper_dist, 2)

    @staticmethod
    def get_ego_velocity(vehicle):
        if vehicle is None or not vehicle.is_alive:
            return 0.0
        vel = vehicle.get_velocity()
        speed_m_s = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed_m_s * 3.6  # Convert to km/h