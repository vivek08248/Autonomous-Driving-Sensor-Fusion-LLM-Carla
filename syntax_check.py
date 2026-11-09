import py_compile
import sys

files = [
    'C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py',
    'C:/Users/vivek/Documents/CARLA_0.9.16/drive_in_carla.py',
    'C:/Users/vivek/Documents/CARLA_0.9.16/sensor_rig.py',
    'C:/Users/vivek/Documents/CARLA_0.9.16/config.py',
]

for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f.split('\\')[-1] + ': Syntax OK')
    except Exception as e:
        print(f.split('\\')[-1] + ': ERROR - ' + str(e))
        sys.exit(1)

print('All files: Syntax OK')