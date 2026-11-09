"""
benchmark_llm_comparison.py (Automated Multi-LLM Benchmark Runner for DriveGPT4-V2)
Evaluates multiple Ollama models across CARLA test cases with active Traffic Manager flow,
and generates drivegpt4_v2_paper_metrics.csv, llm_model_comparison_summary.csv, and LLM_Comparison_Report.md.
"""

import argparse
import csv
import os
import sys
import time

carla_python_path = "C:/Users/vivek/Documents/CARLA_0.9.16/PythonAPI/carla"
if carla_python_path not in sys.path:
    sys.path.append(carla_python_path)

import carla
import config
from main_closed_loop import setup_simulation, restore_simulation
from drive_in_carla import Session, SCENARIOS, BASE_SCENARIOS


class DummyKeyInput:
    def __init__(self):
        self.event_count = 0

    def update(self):
        return True

    def once(self, key):
        return False

    def down(self, key):
        return False


def evaluate_model_on_scenarios(client, world, model_name, scenarios, frames_per_scenario=140):
    config.OLLAMA_MODEL = model_name
    print(f"\n==========================================")
    print(f" Benchmarking LLM Model: {model_name}")
    print(f"==========================================")

    sess = Session(world, client=client)
    sess.set_mode("AI")
    dummy_keys = DummyKeyInput()

    for sc_name in scenarios:
        if sc_name not in SCENARIOS:
            print(f"[Skip] Scenario {sc_name} not found in SCENARIOS list.")
            continue
        idx = SCENARIOS.index(sc_name)
        print(f"\n[Run] Scenario: {sc_name} | Model: {model_name} ({frames_per_scenario} frames)")
        sess.load(idx)
        sess.set_mode("AI")

        for _ in range(frames_per_scenario):
            if sess.ego is None or not sess.ego.is_alive:
                break
            sess.step(dummy_keys)

        sess.paper_tracker.flush_episode()

    completed = list(sess.paper_tracker.completed_runs)
    sess.teardown()
    return completed


def write_combined_markdown_report(all_rows, report_path="LLM_Comparison_Report.md"):
    if not all_rows:
        return
    groups = {}
    for r in all_rows:
        groups.setdefault(r["Model"], []).append(r)

    md = [
        "# DriveGPT4-V2 Closed-Loop Evaluation & Multi-LLM Benchmark Report\n",
        "## 1. Aggregated CARLA Leaderboard Metrics by LLM Model (CVPR 2025 Table 2 & Table 3 Format)\n",
        "| LLM Model | Episodes | DS ↑ | RC (%) ↑ | IS ↑ | Ped ↓ | Veh ↓ | Stat ↓ | Red ↓ | Block ↓ | FPS ↑ | Latency (ms) ↓ | L1 Speed Err ↓ | DAgger Takeover (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, rows in groups.items():
        n = max(1, len(rows))
        md.append(
            f"| **{m_name}** | {n} | **{sum(r['Driving_Score_DS'] for r in rows)/n:.1f}** | "
            f"{sum(r['Route_Completion_RC_%'] for r in rows)/n:.1f}% | "
            f"{sum(r['Infraction_Score_IS'] for r in rows)/n:.2f} | "
            f"{sum(r['Pedestrian_Collisions'] for r in rows)/n:.2f} | "
            f"{sum(r['Vehicle_Collisions'] for r in rows)/n:.2f} | "
            f"{sum(r['Static_Collisions'] for r in rows)/n:.2f} | "
            f"{sum(r['Red_Light_Violations'] for r in rows)/n:.2f} | "
            f"{sum(r['Agent_Blocks'] for r in rows)/n:.2f} | "
            f"{sum(r['FPS'] for r in rows)/n:.1f} | "
            f"{sum(r['Mean_Latency_ms'] for r in rows)/n:.1f} | "
            f"{sum(r['L1_Speed_Error_kmh'] for r in rows)/n:.2f} km/h | "
            f"{sum(r['DAgger_Takeover_%'] for r in rows)/n:.1f}% |"
        )

    md.extend([
        "\n## 2. Per-Scenario Breakdown & Primary LLM Failure Modes\n",
        "| Test Case | Model | DS ↑ | RC (%) | IS | L1 Err (km/h) | DAgger Takeover (%) | Primary LLM Failure Mode |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for r in all_rows:
        md.append(
            f"| `{r['Test_Case']}` | `{r['Model']}` | **{r['Driving_Score_DS']}** | "
            f"{r['Route_Completion_RC_%']}% | {r['Infraction_Score_IS']} | "
            f"{r['L1_Speed_Error_kmh']} | {r['DAgger_Takeover_%']}% | `{r['Primary_Failure_Mode']}` |"
        )

    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:3b"],
        help="List of Ollama LLM models to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=BASE_SCENARIOS,
        help="List of test cases to run (defaults to all 26 BASE_SCENARIOS).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=140,
        help="Number of simulation frames per test case (140 frames = 7.0s simulated time).",
    )
    args = parser.parse_args()

    print("[1] Connecting to CARLA Simulator for Multi-LLM Benchmarking...")
    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT)
    world, original_settings = setup_simulation(client, map_name=config.CARLA_MAP, reload=False)

    all_summaries = []
    summary_csv = "llm_model_comparison_summary.csv"

    try:
        for m in args.models:
            rows = evaluate_model_on_scenarios(client, world, m, args.scenarios, args.frames)
            all_summaries.extend(rows)
    finally:
        restore_simulation(client, original_settings)

    if all_summaries:
        keys = list(all_summaries[0].keys())
        with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_summaries)

        write_combined_markdown_report(all_summaries, "LLM_Comparison_Report.md")
        print(f"\n[Complete] Multi-LLM Benchmark finished! Generated:")
        print(f"       1. {summary_csv}")
        print(f"       2. drivegpt4_v2_paper_metrics.csv")
        print(f"       3. drivegpt4_v2_decihead_trace.csv")
        print(f"       4. LLM_Comparison_Report.md")


if __name__ == "__main__":
    main()