import re

exec(open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py').read())

test_cases = [
    {'desc': 'wet skiddy', 'speed': 30.0},
    {'desc': 'flooded road', 'speed': 30.0},
    {'desc': 'potholed road', 'speed': 30.0},
    {'desc': 'wet skiddy flooded potholes patchy', 'speed': 40.0},
    {'desc': 'oily slippery road', 'speed': 30.0},
    {'desc': 'flooded road works patchy', 'speed': 35.0},
]

for tc in test_cases:
    prompt = generate_multisensor_prompt(
        speed_kmh=tc['speed'],
        lidar_front_dist_m=40.0,
        navigation_goal='Highway',
        nearby_actors=[],
        traffic_light=('GREEN', 200.0),
        weather_info={'description': tc['desc']},
        spatial_sectors={'SIDE_LEFT': 5.0, 'SIDE_RIGHT': 0.5, 'REAR': 40.0},
        sensor_summary={}
    )
    factor_match = re.search(r'speed factor: ([\d.]+)', prompt)
    speed_cap = re.search(r'SPEED CAP: (\d+) km/h', prompt)
    condition = re.search(r'ROAD CONDITION: ([^(]+)', prompt)
    print(f'"{tc["desc"]}" @ {tc["speed"]}km/h -> factor: {factor_match.group(1) if factor_match else "?"}, cap: {speed_cap.group(1) if speed_cap else "?"} km/h')