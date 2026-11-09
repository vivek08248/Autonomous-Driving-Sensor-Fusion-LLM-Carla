import cv2
import time
from llm_planner import LLMAgent
from safety_filter import SafetyFilter
from prompt_builder import generate_multisensor_prompt
from hardware_bridge import PhysicalActuator

def run_laptop_test():
    print("Starting Laptop Vision & LLM Test...")
    
    # Connect to webcam and mock actuator
    cap = cv2.VideoCapture(0)
    motors = PhysicalActuator()
    llm = LLMAgent()
    safety = SafetyFilter()
    
    # Create a GUI window with a slider to fake a LiDAR sensor
    cv2.namedWindow("AI Vision (Press 'Q' to quit)")
    cv2.createTrackbar("Mock LiDAR (m)", "AI Vision (Press 'Q' to quit)", 20, 50, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Read mocked distance from your slider
        mock_lidar_dist = cv2.getTrackbarPos("Mock LiDAR (m)", "AI Vision (Press 'Q' to quit)")
        mock_speed_kmh = 20.0
        
        # 2. Draw data on the webcam feed
        cv2.putText(frame, f"Simulated Distance: {mock_lidar_dist}m", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AI Vision (Press 'Q' to quit)", frame)
        
        # 3. LLM & Safety Loop
        prompt = generate_multisensor_prompt(mock_speed_kmh, mock_lidar_dist, "Maintain Lane")
        llm_decision = llm.get_driving_decision(prompt)
        final_decision = safety.validate_intent(llm_decision, mock_lidar_dist, mock_speed_kmh)
        
        # 4. Print results to terminal
        print(f"LiDAR: {mock_lidar_dist}m | LLM Intent: {llm_decision.get('intent')} | Override: {final_decision.get('override_active')} -> Final: {final_decision.get('intent')}")
        
        # 5. Send to Mock Motors
        target_speed = final_decision.get("target_speed_kmh", 0.0)
        throttle = 25.0 if target_speed > 0 else 0.0
        motors.send_control(throttle, 0.0)
        
        # The window will freeze briefly while the LLM thinks; this is normal.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_laptop_test()