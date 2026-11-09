# v20: Robust DriveGPT4-V2 closed-loop stack (CARLA 0.9.16 + Ollama)

Target machine: Ryzen 7 7735HS, 16 GB RAM, RTX 3050 Laptop 6 GB.
Backups of the old version: `drive_in_carla_v19_backup.py`, `start_drive_v19_backup.ps1`,
`config_v19_backup.py`, `LLM_Comparison_Report_v19.md`.

## 1. Why the old version collided, stopped and froze

| Symptom | Root cause found in v19 |
|---|---|
| LLM "no decision", 4–7 s latency | `start_drive.ps1` forced Ollama onto the **CPU** (`CUDA_VISIBLE_DEVICES=""`). Prompt ≈ 960 tokens, but Ollama only kept the **last 194 tokens** (`ollama_serve_err.log`: `truncating input prompt limit=194 prompt=964`), so the model never saw most of the scene. Many episodes logged no LLM reply at all (latency stuck at the 125 ms default). |
| Car freezes / never resumes | ~3,000 lines of stacked `fix_*.py` patches with latches (walker latch, red latch, scenario-specific `if TC_xx` blocks) that fight each other. 30+ test cases ended as *Agent Block* (DS 3–30). |
| Collisions in junctions / with props | Obstacles were tested in a corridor along the **car's heading**, not the road (wrong in curves and junctions). Props (box, barriers) were not obstacles at all. TC_65 logged 193 static-collision frames, TC_50–54 logged 80–119. |
| Frozen lights in later tests | `tl.freeze(True)` freezes **all** lights and was never undone at teardown. |
| Unreliable metrics | Collision type guessed from distances, every contact frame counted, RC forced to ≥ 95 %, FPS computed from LLM latency. |

## 2. New architecture (all old scenario files reused unchanged)

```
sensors (LiDAR 32ch, front radar, IMU, GNSS, collision) + object list
   -> ad_world.py    Perception in the ROUTE frame (s along lane, d across), weather, LiDAR/radar fusion
   -> ad_planner.py  Expert with privileged info: target speed + lateral path offset
                     (car following, crossing-pedestrian / cut-in / cross-traffic prediction,
                      red/yellow lights, nudge / lane change / opposite-lane bypass with gap checks,
                      queue & road-closure holds, wrong-way evasion, reverse recovery)
   -> ad_llm.py      LLM agent (async, compact ~250-token prompt, JSON) + DAgger arbiter
   -> ad_control.py  Speed PID + steering (target angle) PID with rate limit
   -> ad_metrics.py  DS / RC / IS, Ped/Veh/Stat/Red/Dev/TO/Block, 2 Hz trace, report
drive_in_carla.py    session, test cases, keys, HUD, spectator, batch evaluation
```

The car **never waits for the LLM**: the expert decision exists on every tick. The LLM drives
whenever it agrees with the expert (inside the expert's safety envelope). When it disagrees
by more than 6 km/h, or proposes a false stop or a missed yield, the expert takes over and the
sample is appended to `dagger_dataset.jsonl` (paper Stage 2, data aggregation).

## 3. Base-paper parameters → where they are in this project

| DriveGPT4-V2 (CVPR 2025) | Paper value | This project (v20) |
|---|---|---|
| Planner LLM | Qwen-0.5B / TinyLLaMA-1.1B (LLaVA-style) | Ollama `qwen2.5:1.5b` (default), `qwen2.5:0.5b`, `llama3.2:1b`, `gemma2:2b`, `llama3.2:3b` (L key) |
| Visual input | 3 front cameras 384×384, SigLIP + MV-VT | No cameras (6 GB VRAM). Scene is tokenised as text from LiDAR + radar + object list (`ad_llm.build_prompt`) |
| Vehicle-state input | ego speed, target point | `ego_speed`, `target_point s=40 d=…` in the prompt |
| Decision outputs | target speed, target angle, 8 waypoints (4 s), 10 route points | target speed (arbiter), target angle (steering PID), `Waypoints_8_Future_4s`, `Route_Points_10_Ahead` in `drivegpt4_v2_decihead_trace_v2.csv` |
| Low-level control | 2 PID controllers (speed, angle) | `SpeedPID`, `SteeringPID` (`ad_control.py`) |
| Expert | same model + privileged GT objects (30 m) + hazard info | Rule-based expert with GT objects (75 m) + light/hazard info (`ad_planner.py`) |
| Online imitation | DAgger: expert takes over above a threshold | Threshold `AD_DAGGER_THRESHOLD_KMH = 6.0`; samples saved in `dagger_dataset.jsonl` |
| Data rate | 2 Hz samples | 2 Hz trace and per-test-case logs |
| Loss | L1 on TS, TA, WP, RP | `L1_Speed_Error_kmh` metric (LLM vs expert) |
| Infraction penalties | Ped 0.50, Veh 0.60, Stat 0.65, Red 0.70, Stop 0.70 | Same (`ad_metrics.PENALTY`) |
| Metrics | DS = RC × IS, Ped, Veh, Stat, Red, Dev, TO, Block | Same, plus Sim FPS, LLM latency, LLM Hz, LLM control %, DAgger % |
| Benchmark | Longest6 (36 routes) | 100 test cases (TC_01–TC_100), 150 m route target, 70 s timeout |
| Inference speed | 8.1 FPS (0.5B, A800) | `LLM_Hz` and `LLM_Mean_Latency_ms` columns (expect ~0.2–0.6 s on the RTX 3050) |
| Training | lr 2e-5, 60 epochs, cosine, 16×A800 | Not trained here (zero-shot LLM). `dagger_dataset.jsonl` is ready for a later LoRA fine-tune |

## 4. How to run

```powershell
.\start_drive.ps1                                  # interactive, AI mode, qwen2.5:1.5b on GPU
.\start_drive.ps1 -Scenario TC_47_RED_LIGHT_STOP
.\start_drive.ps1 -Batch                           # all 100 test cases -> LLM_Comparison_Report.md
.\start_drive.ps1 -Batch -From 39 -To 54 -Model qwen2.5:0.5b
.\start_drive.ps1 -Batch -LLM off                  # expert-only baseline (for the comparison table)
.\start_drive.ps1 -OllamaCPU                       # only if CARLA runs out of video memory
python offline_planner_tests.py                    # 26 closed-loop planner tests, no CARLA needed
```

Keys: ESC quit · M/TAB AI↔Manual · N/P next/previous test case · R restart · L next LLM ·
C chase/driver view · T route preference · B AEB assist in manual · Q reverse gear (manual).

Outputs: `drivegpt4_v2_paper_metrics_v2.csv`, `drivegpt4_v2_decihead_trace_v2.csv`,
`LLM_Comparison_Report.md`, `results/result_<TC>_<MODE>.csv`, `dagger_dataset.jsonl`.

## 5. Performance settings for a 6 GB GPU

* CARLA `-quality-level=Low`, 1280×720 window, synchronous 20 Hz (`SIM_DELTA_SECONDS = 0.05`).
* No camera sensors, so no render-target VRAM. LiDAR 32 ch at 50 k pts/s plus one radar are CPU raycasts.
* On `_Opt` maps the layers ParkedVehicles, Particles and Foliage are unloaded. Map-baked parked cars are not
  actors, so no object list can see them; removing them stops "invisible" static crashes.
* Ollama: `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, flash attention, keep-alive 30 min,
  `num_ctx 1024`, `num_predict 48`, one request in flight, keep-alive HTTP connection.
* One `world.get_actors()` per tick, cached extents, lane geometry and lights indexed per lane.

## 6. Test cases whose correct result is "stop and wait"

`TC_10, TC_31, TC_47, TC_49, TC_52, TC_54, TC_64` (permanent red or yellow, a stationary queue, a closed lane).
For these, RC counts as 100 % when the car holds correctly with no infraction (`Expected_Hold_OK = 1`).
You can edit the list as `AD_EXPECT_HOLD` in `config.py`.

## 7. Known limits

* `TC_13_TIGHT_CORRIDOR`: if the two parked cars leave less than the ego width + 0.24 m, the car waits
  instead of scraping through. That is the safe and correct outcome.
* The LLM is zero-shot, so DAgger % is high at first. Fine-tune on `dagger_dataset.jsonl` to bring it down
  (the paper's Stage 2).

## v21 update (lane rules, reverse, demo length)

* **Lane priority (traffic rules):** a lateral move is chosen in this order: 0 = nudge inside our lane,
  1 = centre of one of OUR same-direction lanes (left or right, any of 2–3 lanes), 2 = straddling our lanes,
  3 = opposite lane. The opposite lane is used only when every same-direction option is blocked by
  static obstacles or road works (or none exists), and only if no oncoming vehicle is inside the manoeuvre
  window plus 30 m. If our lanes are busy with moving traffic, the car waits for a gap instead.
  A closed lane (road works) is passed via the opposite lane only after slowing to < 3.6 km/h and checking.
* **Reverse:** only when the obstacle is too close to steer around even at walking pace
  (gap < 1.0·lateral shift + 3 m) or when a slow steer-out gets stuck at a corner. Before and during
  reversing, the car checks objects and LiDAR returns behind it and any vehicle approaching from behind,
  and keeps at least 1 m free. With enough room it steers out slowly (6 km/h) instead.
* **No more stops where a lane change is possible:** a stopped vehicle that was moving is treated as
  blocking after 4 s (was 6 s). The car waits 10 m behind static obstacles so it has room to steer out.
  A parked queue in our lane triggers a lane change to a free same-direction lane.
* **Real-time pacing** is now the default (CARLA at Low quality ran ~5× faster than real time, so test
  cases looked short and the LLM rarely kept up). `-Demo` runs each test case for the full `-Duration`
  (default 180 s). `-Fast` gives the old quick metrics mode.
* Traffic lights also have a geometric fallback (any stop line across our lane ahead). The light state,
  its distance and the lateral offset are now logged in `results/result_*.csv`.

## Command cheat sheet (v22)

| Purpose | Command |
|---|---|
| Tour for the guide: all test cases in a loop, 3 min each, every key works | `.\start_drive.ps1 -Demo` |
| Tour, shorter cases, chosen range | `.\start_drive.ps1 -Demo -Duration 90 -From 27 -To 54` |
| Tour with a different LLM | `.\start_drive.ps1 -Demo -Model qwen2.5:0.5b` |
| Tour in slow motion | `.\start_drive.ps1 -Demo -Speed 0.5` |
| One test case, drive as long as you like | `.\start_drive.ps1 -Scenario TC_02_STATIC_OBSTACLE` |
| Manual driving | `.\start_drive.ps1 -Mode MANUAL` |
| Evaluation, real time (writes the report) | `.\start_drive.ps1 -Batch` |
| Evaluation, quick (faster than real time) | `.\start_drive.ps1 -Batch -Fast` |
| Evaluation, full length per case | `.\start_drive.ps1 -Batch -Demo -Duration 120` |
| Expert-only baseline | `.\start_drive.ps1 -Batch -LLM off` |
| LLM authority variants | `-Authority dagger` (default) / `advisory` / `expert` |
| Ollama on the CPU (only if VRAM runs out) | `-OllamaCPU` |
| CARLA already open | `-NoCarlaStart` |
| Offline planner tests (no CARLA) | `python offline_planner_tests.py` |

Keys: ESC quit · M/TAB AI↔Manual · N/P next/previous · R restart · L next LLM · C camera ·
T route at junction · G pause/resume tour · B AEB assist · Q reverse gear (manual) · W/A/S/D drive · SPACE hand brake.

## v23 update (from the -Batch -Fast run, 200 rows)

* **Traffic-light test cases (TC_47–54) never saw their light:** on this map the "junction" start had an
  unsignalised junction, and `extra_scenarios.py` set some other light red. Signal test cases now pick a start
  whose approach lane has a traffic light on it, and set exactly that light. Each load prints `[Light] ...`.
* **Pedestrian / animal in the free lane:** treated as temporary traffic, so the car waits and then uses its
  own lane. It no longer escapes into the opposite lane.
* **Road-work taper (TC_66) stuck:** cones are chained into one obstacle, and the side clearance is the same
  when checking the gap as when planning the path.
* **Wrong-way driver (TC_09) collision:** the car dodges into a free same-direction lane centre at up to 18 km/h, re-aims if
  the oncoming car drifts, and never stops in front of it.
* **No overtaking of moving vehicles** within 60 m of a junction or a non-green light (TC_79).
* **Pedestrian standing at the bumper for over 10 s:** the car backs off 2.5 m (if the rear is clear) so the person can pass.
* TC_13 stays blocked on purpose: the gap between the two parked cars is narrower than the car plus a safe
  margin, and both sides are blocked.


## v26 - Red light and lane changes
- A lane change / overtake / bypass is NOT started when a RED (or a YELLOW that cannot be stopped for) light is close ahead (within max(60 m, manoeuvre length)). Reason shown: `hold: no lane change, red light N m ahead`. The car follows/stops in its own lane instead.
- If the light turns red DURING a lane change, the planner finishes the lateral move before the stop line (shortening the transition if needed, creeping at <= 6 km/h), then stops behind the line in the new lane and resumes on green. If the move has barely begun (<40 %) it is cancelled back to the original lane.
- New offline tests: `red light blocks lane change`, `red during lane change` (37/37 pass). Mock 100-case batch: 0 collisions, 0 red-light violations.
