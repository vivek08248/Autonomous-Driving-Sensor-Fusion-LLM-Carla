import sys
sys.path.insert(0, 'C:/Users/vivek/Documents/CARLA_0.9.16')

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    source = f.read()

lines = source.split('\n')
new_lines = []
for line in source.split('\n'):
    new_lines.append(line)
    if '):\n' == line.rstrip():  # The ): line that ends function signature
        new_lines.append('    print("FUNCTION_START")')

new_source = '\n'.join(new_lines)
namespace = {}
exec(new_source, namespace)
func = namespace['generate_multisensor_prompt']

try:
    result = func(
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
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()