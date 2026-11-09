import re

with open('C:/Users/vivek/Documents/CARLA_0.9.16/prompt_builder.py', 'r') as f:
    content = f.read()

# Find all f-strings and extract variables
fstrings = re.findall(r'f\"[^\"]*\{([^}]+)\}', content)
for fs in fstrings:
    for var in re.findall(r'\{([^}:]+)', fs):
        print(var.strip())