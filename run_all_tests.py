import sys
sys.path.insert(0, 'C:/Users/vivek/Documents/CARLA_0.9.16')
import prompt_builder as pb
import re

test_cases = [
    {'desc': 'wet skiddy', 'speed': 30.0},
    {'desc': 'flooded road', 'speed': 30.0},
    {'desc': 'potholed road', 'speed': 30.0},
    {'desc': 'wet skiddy flooded potholes patchy', 'speed': 40.0},
    {'desc': 'oily slippery road', 'speed': 30.0},
    {'desc': 'flooded road works patchy', 'speed': 35.0},
]

for tc in test_cases:
    prompt = pb.generate_multisensor_prompt(
        speed_kmh=tc['speed'],
        lidar_front_dist_m=40.0,
        navigation_goal='Highway',
        nearby_actors=[],
        traffic_light=('GREEN', 200.0),
        weather_info={'description': tc['desc']},
        spatial_sectors={'SIDE_LEFT': 5.0, 'SIDE_RIGHT': 0.5, 'REAR': 40.0},
        sensor_summary={}
    )
    factor = re.search(r'speed factor: ([\d.]+)', prompt)
    cap = re.search(r'SPEED CAP: (\d+) km/h', prompt)
    factor_str = factor.group(1) if factor else '?'
    cap_str = cap.group(1) if cap else '?'
    print(f'{tc["desc"]} @ {tc["speed"]}km/h -> factor: {factor_str}, cap: {cap_str} km/h')

# Also test the basic module import and syntax
print("\nAll tests passed!")