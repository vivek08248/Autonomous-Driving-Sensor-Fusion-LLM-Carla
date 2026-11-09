import sys

with open('C:/Users/vivek/Documents/CARLA_0.9.16/lane_change_planner.py', 'r') as f:
    lines = f.readlines()

# Check the PASSING state section
for i, line in enumerate(lines):
    if 'elif self.state == "PASSING":' in line:
        print(f'Line {i+1}: {line.strip()}')
        for j in range(i, min(i+30, len(lines))):
            print(f'  {i+j+1}: {lines[i+j].rstrip()}')
        break