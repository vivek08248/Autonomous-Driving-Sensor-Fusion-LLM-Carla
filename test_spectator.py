import carla, time
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Async mode
s = world.get_settings(); s.synchronous_mode = False; world.apply_settings(s)

# Spawn a car
bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
sp = world.get_map().get_spawn_points()[0]; sp.location.z += 0.5
car = world.try_spawn_actor(bp, sp)
time.sleep(1.0)

# Chase camera loop
import math
spec = world.get_spectator()
for _ in range(200):
    vt = car.get_transform()
    fwd = vt.get_forward_vector()
    cam = vt.location - fwd * 8.0 + carla.Location(z=4.0)
    d = vt.location - cam
    yaw = math.degrees(math.atan2(d.y, d.x))
    horiz = math.sqrt(d.x**2 + d.y**2)
    pitch = math.degrees(math.atan2(d.z, horiz)) - 15.0
    spec.set_transform(carla.Transform(cam, carla.Rotation(pitch=pitch, yaw=yaw, roll=0)))
    time.sleep(0.05)

car.destroy()