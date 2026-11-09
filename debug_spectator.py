"""
debug_spectator.py

Run this standalone script WHILE CarlaUE4.exe is running to diagnose
why the spectator camera is not following the ego vehicle.

Usage:
    python debug_spectator.py
"""

import carla
import time
import math

HOST = "localhost"
PORT = 2000
TIMEOUT = 10.0

print("\n" + "=" * 60)
print("CARLA SPECTATOR DEBUG TOOL")
print("=" * 60)

# ── Connect ────────────────────────────────────────────────────
client = carla.Client(HOST, PORT)
client.set_timeout(TIMEOUT)
world = client.get_world()
print(f"[OK] Connected to CARLA  map={world.get_map().name}")

# ── Check current spectator position ───────────────────────────
spectator = world.get_spectator()
sp_t = spectator.get_transform()
print(f"\n[SPECTATOR] Current position:")
print(f"  Location : x={sp_t.location.x:.2f}  y={sp_t.location.y:.2f}  z={sp_t.location.z:.2f}")
print(f"  Rotation : pitch={sp_t.rotation.pitch:.2f}  yaw={sp_t.rotation.yaw:.2f}")

# ── List existing vehicles ─────────────────────────────────────
vehicles = list(world.get_actors().filter("vehicle.*"))
print(f"\n[VEHICLES] {len(vehicles)} vehicle(s) in world:")
for v in vehicles:
    vt = v.get_transform()
    print(f"  ID={v.id}  type={v.type_id}")
    print(f"    Location : x={vt.location.x:.2f}  y={vt.location.y:.2f}  z={vt.location.z:.2f}")
    print(f"    Rotation : yaw={vt.rotation.yaw:.2f}")

# ── Spawn a test vehicle if none exist ────────────────────────
ego_vehicle = None
spawned_by_us = False

if vehicles:
    ego_vehicle = vehicles[0]
    print(f"\n[OK] Using existing vehicle ID={ego_vehicle.id}")
else:
    print("\n[INFO] No vehicles found - spawning a test Tesla...")
    blueprint_lib = world.get_blueprint_library()
    bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    for sp in spawn_points:
        test_t = carla.Transform(
            sp.location + carla.Location(z=0.5),
            sp.rotation
        )
        ego_vehicle = world.try_spawn_actor(bp, test_t)
        if ego_vehicle:
            print(f"[OK] Spawned vehicle ID={ego_vehicle.id}")
            spawned_by_us = True
            world.tick()
            break
    if ego_vehicle is None:
        print("[ERROR] Could not spawn vehicle. All spawn points blocked.")
        exit(1)

# ── Show where the vehicle is ──────────────────────────────────
vt = ego_vehicle.get_transform()
print(f"\n[VEHICLE] Transform of ego vehicle ID={ego_vehicle.id}:")
print(f"  Location : x={vt.location.x:.2f}  y={vt.location.y:.2f}  z={vt.location.z:.2f}")
print(f"  Rotation : yaw={vt.rotation.yaw:.2f}")

# ── Compute where spectator SHOULD be ─────────────────────────
DISTANCE = 8.0
HEIGHT = 4.0
forward = vt.get_forward_vector()
cam_loc = vt.location - forward * DISTANCE + carla.Location(z=HEIGHT)
direction = vt.location - cam_loc
yaw = math.degrees(math.atan2(direction.y, direction.x))
hdist = math.sqrt(direction.x**2 + direction.y**2)
pitch = math.degrees(math.atan2(direction.z, hdist))

target_transform = carla.Transform(
    cam_loc,
    carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
)

print(f"\n[TARGET] Spectator should be moved to:")
print(f"  Location : x={cam_loc.x:.2f}  y={cam_loc.y:.2f}  z={cam_loc.z:.2f}")
print(f"  Rotation : pitch={pitch:.2f}  yaw={yaw:.2f}")

# ── Try set_transform and verify ──────────────────────────────
print("\n[TEST] Calling spectator.set_transform() ...")
spectator.set_transform(target_transform)

# Read back immediately (before tick)
sp_after = spectator.get_transform()
print(f"\n[SPECTATOR] After set_transform (before tick):")
print(f"  Location : x={sp_after.location.x:.2f}  y={sp_after.location.y:.2f}  z={sp_after.location.z:.2f}")
print(f"  Rotation : pitch={sp_after.rotation.pitch:.2f}  yaw={sp_after.rotation.yaw:.2f}")

# Check if transform was accepted
loc_diff = abs(sp_after.location.x - cam_loc.x) + abs(sp_after.location.y - cam_loc.y)
if loc_diff < 0.5:
    print("\n[OK] set_transform() WAS accepted by CARLA server (location matches).")
    print("     --> The CARLA window SHOULD now show the view behind the vehicle.")
    print("     --> If the window STILL shows an aerial view, check:")
    print("         1. Is the CARLA window the right one? (Title: CarlaUE4)")
    print("         2. Click on the CARLA window to focus it.")
    print("         3. The window might be in PAUSED state - try pressing SPACE in it.")
else:
    print(f"\n[WARNING] set_transform() was NOT fully applied!")
    print(f"  Expected: x={cam_loc.x:.2f}  Got: x={sp_after.location.x:.2f}")
    print(f"  Difference: {loc_diff:.2f}m")
    print("  --> CARLA might need a world.tick() before the transform registers.")

# ── Tick and read back again ──────────────────────────────────
print("\n[TEST] Calling world.tick() ...")
try:
    world.tick()
    sp_after_tick = spectator.get_transform()
    print(f"\n[SPECTATOR] After tick:")
    print(f"  Location : x={sp_after_tick.location.x:.2f}  y={sp_after_tick.location.y:.2f}  z={sp_after_tick.location.z:.2f}")
    loc_diff2 = abs(sp_after_tick.location.x - cam_loc.x) + abs(sp_after_tick.location.y - cam_loc.y)
    if loc_diff2 < 0.5:
        print("[OK] Spectator confirmed at correct position after tick.")
    else:
        print(f"[ERROR] Spectator is at WRONG position after tick! Diff={loc_diff2:.2f}m")
        print("        The CARLA server is RESETTING the spectator transform.")
        print("        This is a CARLA synchronous mode issue.")
except RuntimeError as e:
    print(f"[INFO] world.tick() failed (may be in async mode): {e}")
    print("       In async mode, spectator should update immediately without tick.")

# ── Loop: keep updating spectator for 30 seconds ─────────────
print("\n" + "=" * 60)
print("Entering 30-second continuous spectator update loop.")
print("Watch the CARLA window - the view should switch to behind-vehicle.")
print("=" * 60 + "\n")

try:
    for i in range(150):
        # Set spectator
        vt = ego_vehicle.get_transform()
        forward = vt.get_forward_vector()
        cam_loc = vt.location - forward * DISTANCE + carla.Location(z=HEIGHT)
        direction = vt.location - cam_loc
        yaw = math.degrees(math.atan2(direction.y, direction.x))
        hdist = math.sqrt(direction.x**2 + direction.y**2)
        pitch = math.degrees(math.atan2(direction.z, hdist))

        spectator.set_transform(carla.Transform(
            cam_loc,
            carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
        ))

        try:
            world.tick()
        except RuntimeError:
            pass

        if i % 10 == 0:
            sp_t = spectator.get_transform()
            print(f"[{i:03d}] Spectator: x={sp_t.location.x:.1f}  y={sp_t.location.y:.1f}  z={sp_t.location.z:.1f}")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n[STOP] Loop interrupted.")

finally:
    if spawned_by_us and ego_vehicle and ego_vehicle.is_alive:
        ego_vehicle.destroy()
        print("[Cleanup] Test vehicle destroyed.")

print("\n[DONE] Debug complete.")
