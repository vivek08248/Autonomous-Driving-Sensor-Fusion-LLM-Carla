# System Parameters and Environment Configuration

# CARLA Simulator Parameters
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 25.0
SIM_DELTA_SECONDS = 0.05   # 20 FPS in synchronous mode
SYNCHRONOUS_MODE = True
TRAFFIC_MANAGER_PORT = 8000
TRAFFIC_MANAGER_SEED = 42
TRAFFIC_MANAGER_HYBRID_PHYSICS = True
TRAFFIC_MANAGER_HYBRID_RADIUS = 70.0

# Map to load
CARLA_MAP = "Town03"

# Local LLM (Ollama) Parameters â€” Locked to qwen2.5:1.5b
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"
LLM_MODEL = "qwen2.5:1.5b"

# Available LLM Models for L-key switching (RTX 3050 6GB optimized)
AVAILABLE_LLM_MODELS = [
    "qwen2.5:1.5b",       # Primary - Best structured JSON & control reasoning (~1.2GB VRAM)
    "llama3.2:1b",        # Ultra-light fallback (~0.8GB VRAM)
    "qwen2.5:0.5b",       # Ultra-fast fallback (~0.5GB VRAM)
    "gemma2:2b",          # Alternative (~1.6GB VRAM)
    "llama3.2:3b",        # High capacity (~2.0GB VRAM)
]
DEFAULT_LLM_INDEX = 0

LLM_TIMEOUT_S = 8.0
LLM_MAX_DECISION_AGE_S = 6.0   # Increased: LLM latency ~2.8s; must exceed submit period
LLM_SUBMIT_PERIOD_S = 5.0      # Increased: gives LLM time to respond
LLM_KEEP_ALIVE = "30m"
LLM_NUM_CTX = 1024
LLM_NUM_PREDICT = 128
LLM_MAX_RESPONSE_BYTES = 16384

# Deterministic Safety Thresholds (Meters & km/h)
EMERGENCY_BRAKE_DIST = 3.2
WARNING_DIST = 14.0
WALKER_MIN_LOOKAHEAD_M = 10.0
WALKER_LOOKAHEAD_S = 2.0
WALKER_LANE_HALF_WIDTH_M = 1.7
WALKER_CLEARANCE_M = 1.0
WALKER_PREDICTION_MARGIN_M = 4.0
WALKER_REAR_MARGIN_M = 2.0
WALKER_CLEAR_DISTANCE_M = 4.2
WALKER_CROSSING_TTC_S = 3.5
RESTART_SPEED_KMH = 1.5
RESTART_THROTTLE = 0.45

# Balanced Speed Profile (Meters & km/h)
MAX_ALLOWED_SPEED_KMH = 22.0
AI_MAX_KMH = 20.5
AI_CRUISE_KMH = 19.0

# PID Controller Coefficients
PID_KP_SPEED = 0.5
PID_KD_SPEED = 0.1
PID_KI_SPEED = 0.05

# Physical Hardware Peripherals
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
CAMERA_INDEX = 0

# Visualization & Recording
SHOW_CV_WINDOW = False
SPECTATOR_FOLLOW = True
RECORD_VIDEO = False
VIDEO_OUTPUT_DIR = "recordings"
VIDEO_FPS = 20
VIDEO_CODEC = "mp4v"
RELOAD_WORLD_PER_TEST = False

# Ego vehicle blueprint
EGO_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

# Sensor Rig Parameters (RTX 3050 6GB D3D-Safe)
CAMERA_MOUNT_X = 1.6
CAMERA_MOUNT_Z = 1.7
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90
ENABLE_RGB_SENSOR = False
CAMERA_SENSOR_TICK = 0.1
LIDAR_SENSOR_TICK = 0.05
LIDAR_CHANNELS = 32
LIDAR_POINTS_PER_SECOND = 50000   # 50k pts/s for 6GB GPU stability
LIDAR_MOUNT_Z_M = 2.40            # Sensor height above ground (sensor_rig z=2.40)
HUD_UPDATE_EVERY_N_FRAMES = 5
STATUS_UPDATE_EVERY_N_FRAMES = 5

# ============================================================
# SENSOR CONFIGURATIONS (Lightweight for RTX 3050 6GB)
# ============================================================
# GPU render-target cameras are disabled by default to prevent UE4 D3D11 VRAM crashes on 6GB GPUs.
# Analytical ToF/Depth, Thermal LWIR, and Night-Vision run at 20Hz on CPU with 0 MB VRAM!
CAMERA_CONFIGS = {
    "front_wide": {
        "enabled": False,
        "width": 640,
        "height": 360,
        "fov": 100,
        "mount_location": {"x": 1.6, "y": 0.0, "z": 1.7},
        "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "tick": 0.2,
    },
}

DEPTH_CAMERA_CONFIG = {
    "enabled": False,
    "width": 640,
    "height": 480,
    "fov": 90,
    "mount_location": {"x": 1.6, "y": 0.0, "z": 1.7},
    "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "tick": 0.2,
}

SEMANTIC_CAMERA_CONFIG = {
    "enabled": False,
    "width": 640,
    "height": 480,
    "fov": 90,
    "mount_location": {"x": 1.6, "y": 0.0, "z": 1.7},
    "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "tick": 0.2,
}

EVENT_CAMERA_CONFIG = {
    "enabled": False,
    "width": 640,
    "height": 480,
    "fov": 90,
    "mount_location": {"x": 1.6, "y": 0.0, "z": 1.7},
    "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "tick": 0.1,
}

HIGH_RES_LIDAR_CONFIG = {
    "enabled": False,
    "channels": 64,
    "range": 100.0,
    "points_per_second": 100000,
    "rotation_frequency": 20,
    "upper_fov": 10.0,
    "lower_fov": -30.0,
    "mount_location": {"x": 0.0, "y": 0.0, "z": 2.5},
    "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
}

# Surround 5-Radar Suite (CPU raycasts in CARLA = 0 MB VRAM load!)
RADAR_CONFIGS = {
    "front_long": {
        "enabled": True,
        "range": 120.0,
        "horizontal_fov": 28,
        "vertical_fov": 10,
        "points_per_second": 600,
        "mount_location": {"x": 2.5, "y": 0.0, "z": 0.7},
        "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    },
    "front_mid": {
        "enabled": True,
        "range": 75.0,
        "horizontal_fov": 55,
        "vertical_fov": 12,
        "points_per_second": 500,
        "mount_location": {"x": 2.48, "y": 0.0, "z": 0.65},
        "mount_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    },
    "rear": {
        "enabled": True,
        "range": 65.0,
        "horizontal_fov": 60,
        "vertical_fov": 12,
        "points_per_second": 400,
        "mount_location": {"x": -2.45, "y": 0.0, "z": 0.65},
        "mount_rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
    },
    "left_blindspot": {
        "enabled": True,
        "range": 45.0,
        "horizontal_fov": 115,
        "vertical_fov": 12,
        "points_per_second": 400,
        "mount_location": {"x": 0.0, "y": -1.02, "z": 0.72},
        "mount_rotation": {"pitch": 0.0, "yaw": -90.0, "roll": 0.0},
    },
    "right_blindspot": {
        "enabled": True,
        "range": 45.0,
        "horizontal_fov": 115,
        "vertical_fov": 12,
        "points_per_second": 400,
        "mount_location": {"x": 0.0, "y": 1.02, "z": 0.72},
        "mount_rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
    },
}

SENSOR_FUSION_WEIGHTS = {
    "clear": {"lidar": 0.45, "radar": 0.35, "camera": 0.15, "thermal": 0.05},
    "rain": {"lidar": 0.30, "radar": 0.50, "camera": 0.10, "thermal": 0.10},
    "fog": {"lidar": 0.20, "radar": 0.60, "camera": 0.05, "thermal": 0.15},
    "night": {"lidar": 0.35, "radar": 0.35, "camera": 0.15, "thermal": 0.15},
    "night_rain": {"lidar": 0.25, "radar": 0.50, "camera": 0.10, "thermal": 0.15},
}

# Overtake & Safe Lane-Change Parameters
OVERTAKE_PASSING_TIMEOUT_S = 14.0
OVERTAKE_TOTAL_TIMEOUT_S = 28.0
BACKGROUND_VEHICLES = 0
BACKGROUND_WALKERS = 0

LANE_CHANGE_FRONT_GAP_M = 10.0  # Reduced: was too conservative for urban overtakes
LANE_CHANGE_REAR_GAP_M = 8.0   # Reduced: blocked valid overtakes on slow roads
LANE_CHANGE_TIME_S = 3.0       # Slightly faster LC for urban speeds
LANE_CHANGE_MIN_DIST_M = 14.0  # Minimum LC travel distance
LANE_CHANGE_JUNCTION_CLEAR_M = 30.0
LANE_CHANGE_RESPECT_MARKINGS = 0
LANE_CHANGE_ALLOW_REVERSE = 1
LANE_CHANGE_MERGE_WAIT_S = 6.0
LANE_CHANGE_COOLDOWN_S = 3.0    # Reduced cooldown after successful completion
OVERTAKE_MAX_LEAD_KMH = 18.0   # Increased: allow overtaking vehicles up to 18 km/h

# Leaderboard Metrics
ROUTE_LENGTH_M = 100.0
METRICS_CSV = "leaderboard_metrics.csv"
BLOCK_SECONDS = 8.0
OVERTAKE_DETECT_RANGE_M = 36.0
PREFER_MULTILANE_START = 1

# =====================================================================
# v20 DriveGPT4-V2 stack (ad_*.py) - values below override nothing above
# =====================================================================
# --- Expert planner (privileged "expert" of the paper) ---
AD_CRUISE_KMH = 30.0              # urban speed limit used by the expert
AD_JUNCTION_KMH = 22.0
AD_WEATHER_CAP_KMH = {"clear": 30.0, "wet": 26.0, "rain": 24.0, "heavy_rain": 20.0, "fog": 18.0,
                      "night": 26.0, "night_rain": 20.0, "glare": 22.0, "storm_night": 18.0, "dusk": 26.0}
AD_HEADWAY_S = 1.2                # time gap behind moving vehicles (x1/friction when wet)
AD_COMFORT_DECEL = 2.8            # m/s^2
AD_MAX_DECEL = 7.5                # m/s^2 (AEB)
AD_STATIC_CONFIRM_S = 1.2         # parked object -> may bypass after this
AD_TRAFFIC_STOP_CONFIRM_S = 4.0   # vehicle that was moving and stopped (traffic) -> wait longer
AD_STATIC_STANDSTILL_M = 10.0     # wait this far behind a static obstacle (room to steer around)
AD_VRU_PATIENCE_S = 10.0          # pedestrian / animal standing in lane -> bypass after this
AD_QUEUE_PATIENCE_S = 30.0

# --- LLM agent + DAgger arbitration (paper Sec. 3.2) ---
AD_LLM_AUTHORITY = "dagger"       # dagger | advisory | expert
AD_DAGGER_THRESHOLD_KMH = 6.0     # |LLM - expert| above this -> expert takes over (+ sample logged)
AD_LLM_PERIOD_S = 0.5             # new prompt every 0.5 s (single request in flight)
AD_LLM_MAX_AGE_S = 2.0            # older LLM decisions are ignored
AD_LLM_TIMEOUT_S = 6.0
AD_LLM_NUM_PREDICT = 40           # ~25 tokens are needed for the JSON reply
AD_LLM_NUM_CTX = 1024             # prompt is ~250 tokens -> never truncated again
LLM_DISABLE_THINK = True
AD_DAGGER_DATASET = "dagger_dataset.jsonl"

# --- Episodes / evaluation ---
AD_ROUTE_TARGET_M = 150.0
AD_EPISODE_TIMEOUT_S = 70.0
AD_BLOCK_LIMIT_S = 30.0
AD_HOLD_DONE_S = 15.0

# --- Performance (RTX 3050 6 GB) ---
AD_DRAW_PLAN = True               # planned path dots in the CARLA window
AD_LOAD_MAP = False               # True -> load CARLA_MAP at start (scenarios were tuned on the current map)
AD_UNLOAD_MAP_LAYERS = ["ParkedVehicles", "Particles", "Foliage"]   # _Opt maps only
AD_ENABLE_LIDAR = True
AD_ENABLE_RADAR = True
