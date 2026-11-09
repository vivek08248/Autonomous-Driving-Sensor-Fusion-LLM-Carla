#!/usr/bin/env python3
"""Comprehensive CSV analysis for all test cases"""
import csv, os, glob, json
from collections import defaultdict

base_path = 'C:/Users/vivek/Documents/CARLA_0.9.16'
csv_files = glob.glob(os.path.join(base_path, '*.csv'))

print(f'=== COMPREHENSIVE CSV ANALYSIS ===')
print(f'Total CSV files: {len(csv_files)}')

# ============================================================
# 1. PAPER METRICS - drivegpt4_v2_paper_metrics.csv
# ============================================================
paper_metrics_path = os.path.join(base_path, 'drivegpt4_v2_paper_metrics.csv')
print('\n=== PAPER METRICS (drivegpt4_v2_paper_metrics.csv) ===')
if os.path.exists(paper_metrics_path):
    with open(paper_metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f'Total rows: {len(rows)}')
    models = set(r['Model'] for r in rows)
    print(f'Models: {sorted(models)}')
    
    # Summary per test case
    tc_summary = defaultdict(list)
    for r in rows:
        tc_summary[r['Test_Case']].append(r)
    
    print(f'\nTest Cases: {len(tc_summary)}')
    for tc in sorted(tc_summary.keys()):
        entries = tc_summary[tc]
        best_ds = max(float(e['Driving_Score_DS']) for e in entries)
        worst_ds = min(float(e['Driving_Score_DS']) for e in entries)
        models_used = set(e['Model'] for e in entries)
        print(f'  {tc}: DS range [{worst_ds:.1f}-{best_ds:.1f}], Models: {len(models_used)}')

# ============================================================
# 2. LLM COMPARISON REPORT
# ============================================================
report_path = os.path.join(base_path, 'LLM_Comparison_Report.md')
print('\n=== LLM COMPARISON REPORT ===')
if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        content = f.read()
    print(f'Report length: {len(content)} chars')
    # Extract table 1
    if 'Aggregated CARLA Leaderboard Metrics' in content:
        print('  Found: Aggregated metrics table')
    if 'Per-Scenario Breakdown' in content:
        print('  Found: Per-scenario breakdown')

# ============================================================
# 3. DECIHEAD TRACE ANALYSIS
# ============================================================
decihead_path = os.path.join(base_path, 'drivegpt4_v2_decihead_trace.csv')
print('\n=== DECIHEAD TRACE ===')
if os.path.exists(decihead_path):
    with open(decihead_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Total frames: {len(rows)}')
    scenarios = set(r['Scenario'] for r in rows if r['Scenario'])
    print(f'Scenarios: {len(scenarios)}')
    
    # Failure modes
    failures = [r for r in rows if r['Failure_Mode'] != 'NONE']
    print(f'Failure frames: {len(failures)}')
    failure_modes = defaultdict(int)
    for r in failures:
        failure_modes[r['Failure_Mode']] += 1
    for fm, cnt in sorted(failure_modes.items(), key=lambda x: -x[1]):
        print(f'  {fm}: {cnt}')

# ============================================================
# 4. PER-TC RESULT ANALYSIS
# ============================================================
print('\n=== PER-TC RESULT ANALYSIS ===')
ai_files = [f for f in csv_files if f.endswith('_AI.csv')]
print(f'AI result files: {len(ai_files)}')

# Focus on the 6 failing test cases
failing_tcs = ['TC_09', 'TC_12', 'TC_18', 'TC_21', 'TC_31', 'TC_33']

for tc in failing_tcs:
    ai_file = os.path.join(base_path, f'result_{tc}_AI.csv')
    if os.path.exists(ai_file):
        with open(ai_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f'\n{tc} ({len(rows)} frames):')
        
        intents = defaultdict(int)
        collisions = 0
        safety_interventions = 0
        for r in rows:
            intents[r.get('LLM_Intent', 'UNKNOWN')] += 1
            if r.get('Collision_Event', 'False') == 'True':
                collisions += 1
            if r.get('Safety_Intervention_Active', 'False') == 'True':
                safety_interventions += 1
        
        print(f'  Collisions: {collisions}')
        print(f'  Safety interventions: {safety_interventions}')
        print(f'  Intent distribution:')
        for intent, cnt in sorted(intents.items(), key=lambda x: -x[1])[:5]:
            print(f'    {intent}: {cnt}')

# ============================================================
# 5. STATE TRANSITIONS
# ============================================================
state_path = os.path.join(base_path, 'state_transitions_log.csv')
print('\n=== STATE TRANSITIONS ===')
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Transitions logged: {len(rows)}')
    if rows:
        transitions = defaultdict(int)
        for r in rows:
            key = f"{r.get('from_state', '?')}->{r.get('to_state', '?')}"
            transitions[key] += 1
        for t, cnt in sorted(transitions.items(), key=lambda x: -x[1]):
            print(f'  {t}: {cnt}')

# ============================================================
# 6. LEADERBOARD METRICS
# ============================================================
lb_path = os.path.join(base_path, 'leaderboard_metrics.csv')
print('\n=== LEADERBOARD METRICS ===')
if os.path.exists(lb_path):
    with open(lb_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Total entries: {len(rows)}')
    if rows:
        print(f'Sample columns: {list(rows[0].keys())[:10]}')

print('\n=== ANALYSIS COMPLETE ===')