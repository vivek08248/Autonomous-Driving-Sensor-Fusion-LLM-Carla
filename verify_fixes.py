with open('C:/Users/vivek/Documents/CARLA_0.9.16/drive_in_carla.py', 'r') as f:
    content = f.read()

checks = {
    'LLM_PERIOD_S = 3.0': 'LLM_PERIOD_S = float(getattr(config, "LLM_SUBMIT_PERIOD_S", 3.0))' in content,
    'LLM_MAX_DECISION_AGE_S = 2.5': 'LLM_MAX_DECISION_AGE_S = float(getattr(config, "LLM_MAX_DECISION_AGE_S", 2.5))' in content,
    'Removed STOP->KEEP_LANE override': 'Intersection clear -> cruise' not in content,
    'Cached result age check': 'cached but stale' in content,
    'Debug LLM decision logging': 'print(f"[LLM] intent=' in content,
    'Debug LLM submit logging': 'print(f"[LLM Submit]' in content,
    'TC_09 head-on brake': 'HEAD_ON_BRAKE' in content,
    'TC_12 sun glare': 'sun_glare_active' in content,
    'TC_18 animal detection': 'animal_hazard' in content,
    'TC_21 ambiguous intersection': 'Ambiguous intersection' in content,
    'TC_31 stationary queue': 'stationary_queue' in content,
    'TC_33 cut-in detection': 'cutin_threat' in content,
}

print('=== VERIFICATION OF FIXES ===')
all_pass = True
for check, result in checks.items():
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'  {status}: {check}')

print('Overall: ' + ('ALL PASS' if all_pass else 'SOME FAIL'))