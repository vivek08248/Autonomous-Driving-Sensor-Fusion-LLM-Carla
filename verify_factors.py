import sys
sys.path.insert(0, 'C:/Users/vivek/Documents/CARLA_0.9.16')
import prompt_builder as pb

test_cases = [
    ('wet skiddy', 30.0, 0.50),
    ('flooded road', 30.0, 0.50),
    ('potholed road', 30.0, 0.65),
    ('wet skiddy flooded potholes patchy', 40.0, 0.50),
    ('oily slippery road', 30.0, 0.50),
    ('flooded road works patchy', 35.0, 0.60),
]

print('=== SPEED FACTOR VERIFICATION ===')
for desc, speed, expected in test_cases:
    prompt = pb.generate_multisensor_prompt(
        speed_kmh=speed,
        lidar_front_dist_m=40.0,
        navigation_goal='Highway',
        nearby_actors=[],
        traffic_light=('GREEN', 200.0),
        weather_info={'description': desc},
        spatial_sectors={'SIDE_LEFT': 5.0, 'SIDE_RIGHT': 0.5, 'REAR': 40.0},
        sensor_summary={}
    )
    for line in prompt.split('\n'):
        if 'speed factor' in line.lower():
            factor = float(line.split('speed factor ')[1].split(',')[0])
            status = 'OK' if abs(factor - expected) < 0.01 else 'FAIL'
            print(f'{desc}: factor={factor} (expected {expected}) {status}')
            break