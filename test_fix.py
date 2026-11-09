import sys
sys.path.insert(0, 'C:/Users/vivek/Documents/CARLA_0.9.16')

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    source = f.read()

# Replace the decision_framework tuple creation with a local variable copy
old_code = '''    # =====================================================================
    # MASTER DECISION FRAMEWORK - Structured reasoning for every output
    # =====================================================================
    decision_framework = ('''

new_code = '''    # =====================================================================
    # MASTER DECISION FRAMEWORK - Structured reasoning for every output
    # =====================================================================
    # Ensure variables are accessible in f-strings (Python 3.11+ LOAD_FAST_CHECK workaround)
    _lead_dist_threshold = lead_dist_threshold
    _speed_advantage_threshold = speed_advantage_threshold
    _speed_reduction_factor = speed_reduction_factor
    _road_surface_condition = road_surface_condition
    decision_framework = ('''

new_source = source.replace(old_code, new_code)

# Also update the f-strings to use the local copies
new_source = new_source.replace(
    'lead_dist_threshold:.0f}m',
    '_lead_dist_threshold:.0f}m'
).replace(
    'speed_advantage_threshold} km/h',
    '_speed_advantage_threshold} km/h'
).replace(
    'speed_reduction_factor:.2f}',
    '_speed_reduction_factor:.2f}'
).replace(
    'road_surface_condition}',
    '_road_surface_condition}'
)

namespace = {}
exec(new_source, namespace)
prompt_builder = namespace['generate_multisensor_prompt']

prompt = prompt_builder(
    speed_kmh=30.0,
    lidar_front_dist_m=40.0,
    navigation_goal='Highway',
    nearby_actors=[],
    traffic_light=('GREEN', 200.0),
    weather_info={'description': 'wet skiddy'},
    spatial_sectors={'SIDE_LEFT': 5.0, 'SIDE_RIGHT': 0.5, 'REAR': 40.0},
    sensor_summary={}
)
print('SUCCESS')
print(f'Length: {len(prompt)}')