import re

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    content = f.read()

# Find all f-strings
fstrings = re.findall(r'f"[^"]*"', content)
for fs in fstrings:
    if '_road_surface_condition' in fs or '_road_surface_condition' in fs:
        print(f'Found: {fs[:200]}')