import carla
import time

print("Connecting to CARLA...")
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()

# Async mode so set_transform takes effect immediately
s = world.get_settings()
s.synchronous_mode = False
world.apply_settings(s)

# Spawn a Tesla at the first spawn point
bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
sp = world.get_map().get_spawn_points()[0]
sp.location.z += 0.5
car = world.try_spawn_actor(bp, sp)
print(f"Ego spawned. ID = {car.id}")
print(f"Ego location: {car.get_location()}")

# Wait for the world to settle
time.sleep(2)

# ---- Position A: high overhead ----
print("\n[A] Moving spectator to overhead view (z=200)...")
spec = world.get_spectator()
spec.set_transform(carla.Transform(
    carla.Location(x=car.get_location().x, y=car.get_location().y, z=200),
    carla.Rotation(pitch=-90, yaw=0, roll=0)
))
print("    >>> WATCH THE CARLA WINDOW. Do you see the city from above? <<<")
time.sleep(8)

# ---- Position B: ground level next to car ----
print("\n[B] Moving spectator to ground level next to car...")
loc = car.get_location()
spec.set_transform(carla.Transform(
    carla.Location(x=loc.x - 10, y=loc.y, z=2),
    carla.Rotation(pitch=0, yaw=0, roll=0)
))
print("    >>> WATCH THE CARLA WINDOW. Do you see a street-level view? <<<")
time.sleep(8)

# ---- Position C: chase view behind car ----
print("\n[C] Moving spectator to chase view behind car...")
vt = car.get_transform()
fwd = vt.get_forward_vector()
cam = vt.location - fwd * 8.0 + carla.Location(z=4.0)
import math
d = vt.location - cam
yaw = math.degrees(math.atan2(d.y, d.x))
pitch = math.degrees(math.atan2(d.z, math.sqrt(d.x**2 + d.y**2)))
spec.set_transform(carla.Transform(cam, carla.Rotation(pitch=pitch, yaw=yaw, roll=0)))
print("    >>> WATCH THE CARLA WINDOW. Do you see the car from behind? <<<")
time.sleep(8)

print("\nDone. Destroying ego...")
car.destroy()
print("Test complete.")