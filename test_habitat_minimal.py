import habitat_sim

scene = "/home/mario/tfg/data/custom_scenes/skokloster-castle/skokloster-castle.glb"

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene
sim_cfg.enable_physics = False

color_sensor = habitat_sim.CameraSensorSpec()
color_sensor.uuid = "color_sensor"
color_sensor.sensor_type = habitat_sim.SensorType.COLOR
color_sensor.resolution = [480, 640]
color_sensor.position = [0.0, 1.5, 0.0]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [color_sensor]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

print("Antes de crear Simulator")
sim = habitat_sim.Simulator(cfg)
print("Simulator creado")

agent = sim.initialize_agent(0)
print("Agent creado")

obs = sim.get_sensor_observations()
print("Observations OK:", obs.keys())

sim.close()
print("Todo OK")
