import csv
import time
import os

class MetricsLogger:
    def __init__(self, filename="simulation_results.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.total_interventions = 0
        self.total_cycles = 0
        
        # Create CSV and write headers if it does not exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Elapsed_Time_s", "Speed_kmh", "LiDAR_Dist_m", 
                    "LLM_Latency_ms", "LLM_Intent", 
                    "Safety_Intervention_Active", "Ambiguity_Detected",
                    "Collision_Event"
                ])
    
    def log_cycle(
        self, speed, lidar_dist, llm_latency_ms, intent, safety_override,
        collision_detected=False, ambiguity_detected=False,
    ):
        self.total_cycles += 1
        if safety_override:
            self.total_interventions += 1
            
        elapsed = round(time.time() - self.start_time, 2)
        row = [
            elapsed, 
            round(speed, 2), 
            round(lidar_dist, 2), 
            round(llm_latency_ms, 1), 
            intent, 
            safety_override,
            ambiguity_detected,
            collision_detected
        ]
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def print_summary(self):
        intervention_rate = (self.total_interventions / max(self.total_cycles, 1)) * 100
        print("\n--- Simulation Evaluation Summary ---")
        print(f"Total Decision Cycles: {self.total_cycles}")
        print(f"Safety Interventions: {self.total_interventions}")
        print(f"Intervention Rate: {intervention_rate:.1f}%")
        print(f"Data saved to: {self.filename}")
        print("-------------------------------------")