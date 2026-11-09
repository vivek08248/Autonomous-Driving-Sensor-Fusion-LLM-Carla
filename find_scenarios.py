with open("run_test_cases.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
cases = re.findall(r'elif scenario_name == "(.*?)":', text)
for c in cases:
    pos = text.find(f'scenario_name == "{c}":')
    block = text[pos:pos+800]
    n_v = block.count('_spawn_vehicle') + block.count('_safe_spawn_obstacle')
    print(f"{c}: {n_v} actors")
