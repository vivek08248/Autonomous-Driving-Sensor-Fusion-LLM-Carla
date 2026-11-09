# Autonomous Driving with Sensor Fusion and LLM Decision-Making (CARLA 0.9.16 + Ollama)

**Author:** Vivek Chakali, B.E. Electronics and Communication Engineering

A closed-loop autonomous driving framework that fuses LiDAR, radar, IMU and GNSS data, asks a local LLM (served by Ollama) for high-level driving decisions, and passes every decision through a deterministic safety layer before it reaches the vehicle controllers. Evaluation follows the DriveGPT4-V2 (CVPR 2025) metrics: Driving Score DS = RC x IS, where RC is route completion and IS the infraction score.

## System block diagram

![System block diagram: sensing, fusion, LLM reasoning, safety validation, trajectory and control, actuation](docs/images/block_diagram.png)

The diagram is the one embedded in slide 8 of the seminar: sensor inputs, synchronisation, fusion, scene representation, LLM reasoning, safety validation, waypoint generation, trajectory control, embedded actuation, and the vehicle in CARLA or on the prototype.

## Architecture

| Layer | Files | Role |
|---|---|---|
| Sensing and fusion | `sensor_rig.py`, `perception_processor.py`, `ad_world.py` | LiDAR (32 channels), radar, IMU, GNSS and collision sensors; objects are expressed in the route (Frenet) frame and fused into a single gap estimate |
| LLM decision | `ad_llm.py`, `prompt_builder.py`, `llm_planner.py`, `drivegpt4_v2.py` | Compact prompt, JSON reply with target speed, target angle and waypoints; keep-alive HTTP to Ollama; newest prompt wins |
| Behaviour planning | `ad_planner.py`, `lane_change_planner.py`, `maneuver_arbiter.py` | Car following, crossing and cut-in prediction, traffic-light stops, own-lane-first overtakes, opposite lane only when needed, reverse recovery |
| Safety | `safety_filter.py`, `rss_safety_shield.py`, `autonomous_rules_engine.py`, `ambiguity_detector.py` | Rejects or modifies unsafe commands using collision checks, lane limits, speed limits and minimum-distance rules |
| Control | `ad_control.py`, `pid_controller.py` | Speed PID and steering PID with a speed-dependent look-ahead |
| DAgger authority | `ad_llm.py` (`DaggerArbiter`) | Switches between LLM control and expert takeover when the LLM output is unsafe or too slow (threshold 6 km/h) |
| Scenarios and runner | `run_test_cases.py`, `extra_scenarios.py`, `drive_in_carla.py`, `run_all_tests.py` | 100 test cases (TC_01 to TC_100), interactive tour mode, batch mode |
| Metrics and reports | `ad_metrics.py`, `leaderboard_metrics.py`, `analyze_all_csvs.py`, `benchmark_llm_comparison.py` | DS, RC, IS, collision and red-light classification, per-case CSV logs, Markdown comparison report |
| Offline tests | `offline_planner_tests.py`, `offline_test_runner.py` | 37 closed-loop planner tests with a kinematic simulator (no CARLA needed) |
| Hardware bridge (prototype) | `hardware_bridge.py` | Interface sketch for passing actuator commands to an embedded controller |

## Requirements

- Windows 10/11 (the launcher is PowerShell), Python 3.10 or newer.
- CARLA 0.9.16 simulator, installed separately (not part of this repository).
- Ollama, with a model pulled, for example `ollama pull qwen2.5:0.5b`.
- Laptop used for development: AMD Ryzen 7 7735HS, 16 GB RAM, NVIDIA RTX 3050 (6 GB VRAM). CARLA and the LLM share the GPU, so the LLM may fall back to CPU if VRAM is tight.
- Python packages: see `requirements.txt`.

## Quick start

```powershell
# Interactive demo: runs all test cases in order at real-time speed, with tour mode
.\start_drive.ps1 -Demo -Model qwen2.5:0.5b

# Batch run of all test cases (writes the CSV and Markdown reports)
.\start_drive.ps1 -Batch -Model qwen2.5:0.5b

# Run a range of test cases only
.\start_drive.ps1 -Batch -From 1 -To 20 -Model qwen2.5:0.5b

# Fast (non-real-time) mode for quick regression runs
.\start_drive.ps1 -Batch -Fast -Model qwen2.5:0.5b

# Planner tests without CARLA
python offline_planner_tests.py
```

`start_drive.ps1` starts Ollama before CARLA so that the model is loaded on the GPU first. Use `-OllamaCPU` only for debugging.

### Interactive keys (drive_in_carla.py)

| Key | Action |
|---|---|
| N / P | Next / previous test case |
| R | Restart the current test case |
| M or TAB | Toggle manual / AI mode |
| L | Toggle LLM on or off |
| C | Cycle the LLM authority (dagger, advisory, expert) |
| T | Toggle tour mode (auto-advance) |
| G | Pause or resume |
| B | Show or hide the HUD |
| Q | Quit |
| ESC | Quit the current run |

## Test cases

The 100 test cases cover: clear lanes and cruising; static obstacles and bypasses; cut-ins by cars, bicycles and motorcycles; leading vehicles that brake hard or stop and go; pedestrians (walking, running, standing, crossing in groups, at night and in rain); traffic lights (red stop, green pass, yellow, green-to-red, red-to-green, red with a queue, red during a lane change); junction turns; oncoming traffic and wrong-way drivers; road works, cones and barriers; weather and night; and a few mixed-rule validation cases. Each case has a scripted start, an expected outcome and a logged result in `results/ai/result_TC_<nn>_<NAME>_AI.csv` (LLM-assisted) or `results/manual/result_TC_<nn>_<NAME>_MANUAL.csv` (manual driving).

## Results (latest run, 100 test cases per model)

| Metric | qwen2.5:0.5b | qwen2.5:1.5b | DriveGPT4-V2 (paper, Longest6) |
|---|---|---|---|
| Driving Score (DS) | 97.4 | 93.6 | 70 |
| Route Completion (%) | 97.8 | 94.1 | 91 |
| Infraction Score (IS) | 1.00 | 0.99 | 0.77 |
| Pedestrian collisions per episode | 0.00 | 0.00 | 0.00 |
| Vehicle collisions per episode | 0.01 | 0.01 | 0.80 |
| Red-light violations per episode | 0.00 | 0.00 | 0.04 |
| LLM reply time (mean, replies received) | about 0.48 s | about 1.07 s | n/a |
| LLM share of control steps | 1.3 % | 1.9 % | n/a |

These are short scripted CARLA scenarios and are not directly comparable with the paper's 36 Longest6 routes. The one vehicle collision per model is test case TC_09 (wrong-way oncoming car), recorded before the latest brake-on-oncoming fix. Full tables are in `reports/LLM_Comparison_Report.md` and `reports/drivegpt4_v2_paper_metrics_v2.csv`.

Per-test-case numbers are in `reports/drivegpt4_v2_paper_metrics_v2.csv`; the aggregate comparison below is generated from that file.

### Aggregate comparison (generated from the same CSV)

| Model | Episodes | Mean DS | Mean RC % | Mean IS | Pedestrian coll. | Vehicle coll. | Red-light viol. | Blocked / timeout per 100 | Mean LLM reply (ms) |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 100 | 97.4 | 97.8 | 1.00 | 0 | 1 | 0 | 2 | 477 |
| qwen2.5:1.5b | 100 | 93.6 | 94.1 | 0.99 | 0 | 1 | 0 | 3 | 1074 |
| DriveGPT4-V2 (paper, Longest6) | 36 routes | 70 | 91 | 0.77 | 0.00 | 0.80 | 0.04 | 0.16 | n/a |

## Known limitations

- LLM latency (0.3 to 2 s per reply) is too slow for direct vehicle control, so the LLM works as an advisory layer and the planner drives most of the time.
- Route completion is lower on the rain-plus-obstacle overtake case because the planner declines some overtakes in traffic.
- Results come from a simulator on one laptop; no on-road testing has been done.

## Documentation

- `docs/presentation/Project_Seminar_Vivek_Chakali.pptx` - project seminar slides (Abstract Level Presentation).
- `docs/images/block_diagram.png` - system block diagram (slide 8 of the seminar).
- `docs/base_paper/DriveGPT4-V2_CVPR2025_Xu_et_al.pdf` - the base paper: Xu et al., "DriveGPT4-V2", CVPR 2025.
- `PROJECT_FIXES_v20.md` - change log (v20 to v26), base-paper parameter table and command cheat sheet.
- `reports/LLM_Comparison_Report.md` - latest model comparison and per-case table.

## License

Academic project. Add a license before reusing the code.
