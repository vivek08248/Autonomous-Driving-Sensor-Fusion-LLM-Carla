# Autonomous Driving with Sensor Fusion and LLM Decision-Making (CARLA 0.9.16 + Ollama)

Author: **Vivek Chakali**, B.E. Electronics and Communication Engineering, Vasavi College of Engineering, Hyderabad.

Closed-loop driving framework that combines LiDAR, radar, IMU and GNSS sensor fusion with a local LLM (Ollama) for high-level decisions, a deterministic safety layer, and DriveGPT4-V2 style evaluation (DS / RC / IS).

## Contents
- `start_drive.ps1` - launcher (starts Ollama, then CARLA and the driver).
- `drive_in_carla.py` - interactive / batch test runner for the 100 test cases.
- `ad_*.py`, `lane_change_planner.py`, `safety_filter.py`, `rss_safety_shield.py`, `sensor_rig.py` - perception, planning, LLM and safety modules.
- `offline_planner_tests.py` - 37 closed-loop planner tests (no CARLA needed).
- `drivegpt4_v2_paper_metrics_v2.csv`, `LLM_Comparison_Report.md` - latest results.
- `result_TC_*.csv` - per-test-case logs (`_AI` = LLM-assisted run, `_MANUAL` = manual driving).
- `PROJECT_FIXES_v20.md` - change log and parameter notes.

Requires CARLA 0.9.16 installed separately (not included).
