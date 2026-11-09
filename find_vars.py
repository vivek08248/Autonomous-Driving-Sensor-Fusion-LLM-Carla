import re

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    content = f.read()

# Find all f-strings
fstrings = re.findall(r'f"[^"]*"', content)
for fs in fstrings:
    vars_found = re.findall(r'\{([^}:]+)', fs)
    for var in vars_found:
        print(var.strip())