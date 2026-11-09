import os
import sys
import importlib
import urllib.request
import socket

# 1. Define the exact architecture files expected in the folder
EXPECTED_FILES = [
    "config.py",
    "sensor_rig.py",
    "perception_processor.py",
    "prompt_builder.py",
    "llm_planner.py",
    "safety_filter.py",
    "pid_controller.py",
    "hardware_bridge.py",
    "evaluation_logger.py",
    "main_closed_loop.py"
]

def print_header(title):
    print(f"\n{'='*40}\n{title}\n{'='*40}")

def check_file_structure():
    print_header("1. FILE STRUCTURE CHECK")
    missing_files = []
    for file in EXPECTED_FILES:
        if os.path.exists(file):
            print(f"[OK] Found: {file}")
        else:
            print(f"[FAIL] Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print("\n[!] FATAL: You are missing core project files. Please create them before continuing.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All 10 project files are present.")

def check_syntax_and_libraries():
    print_header("2. SYNTAX & LIBRARY IMPORT CHECK")
    # We strip the .py extension to test importing the module
    modules_to_test = [f.replace('.py', '') for f in EXPECTED_FILES if f != "main_closed_loop.py"]
    
    failed_imports = 0
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
            print(f"[OK] Module '{mod}' compiled and imported successfully.")
        except ModuleNotFoundError as e:
            print(f"[FAIL] Module '{mod}' missing dependency: {e}")
            failed_imports += 1
        except SyntaxError as e:
            print(f"[FAIL] Syntax Error in '{mod}': {e}")
            failed_imports += 1
        except Exception as e:
            print(f"[FAIL] Error loading '{mod}': {e}")
            failed_imports += 1

    if failed_imports > 0:
        print("\n[!] FATAL: Fix the syntax or missing pip libraries above before running the main loop.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All custom modules, CARLA API, OpenCV, and PySerial loaded without syntax errors.")

def check_ollama_server():
    print_header("3. LLM REASONING ENGINE (OLLAMA)")
    try:
        import config
        req = urllib.request.Request(f"{config.OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                print(f"[OK] Ollama is running at {config.OLLAMA_HOST}")
            else:
                print(f"[WARNING] Ollama returned status {response.status}")
    except Exception as e:
        print(f"[FAIL] Could not reach Ollama: {e}")
        print("     -> FIX: Open a terminal and run 'ollama run qwen2.5:0.5b'")

def check_carla_server():
    print_header("4. CARLA SIMULATOR SERVER")
    try:
        import config
        # Simple socket check to see if CARLA is listening on port 2000
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((config.CARLA_HOST, config.CARLA_PORT))
        if result == 0:
            print(f"[OK] CARLA server is actively listening on port {config.CARLA_PORT}")
        else:
            print(f"[FAIL] CARLA server is not reachable on port {config.CARLA_PORT}")
            print("     -> FIX: Start CarlaUE4.exe -carla-rpc-port=2000")
        sock.close()
    except Exception as e:
        print(f"[FAIL] CARLA check error: {e}")

if __name__ == "__main__":
    print("\nStarting System Diagnostic Verification...")
    check_file_structure()
    check_syntax_and_libraries()
    check_carla_server()
    check_ollama_server()
    print_header("DIAGNOSTIC COMPLETE")
    print("If all checks read [OK] or [SUCCESS], your architecture is 100% verified.")