import carla, time

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()

bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
sp = world.get_map().get_spawn_points()[0]
sp.location.z += 0.5
car = world.try_spawn_actor(bp, sp)
print(f"Spawned. Watching for 10s...")

# Drive forward
for i in range(200):
    car.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    time.sleep(0.05)

car.destroy()
print("Done.")