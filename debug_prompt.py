import sys
sys.path.insert(0, 'C:/Users/vivek/Documents/CARLA_0.9.16')

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    source = f.read()

# Add debug prints after the variable assignment
lines = source.split('\n')
new_lines = []
for line in source.split('\n'):
    new_lines.append(line)
    # Add debug after the variable assignment
    if 'lead_dist_threshold = 35.0 * speed_reduction_factor' in line and 'if' not in line and 'for' not in line:
        new_lines.append('    print(f"DEBUG: lead_dist_threshold = {lead_dist_threshold}")')
    if 'speed_advantage_threshold = 8.0' in line:
        new_lines.append('    print(f"DEBUG: speed_advantage_threshold = {speed_advantage_threshold}")')
    # Add debug before the decision_framework f-string
    if 'decision_framework = (' in line:
        new_lines.append('    print(f"DEBUG: lead_dist_threshold = {lead_dist_threshold}, speed_advantage_threshold = {speed_advantage_threshold}")')

new_source = '\n'.join(new_lines)
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