import gymnasium
import gymnasium_env_gaze
import numpy
import time
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation

env = gymnasium.make("gymnasium_env_gaze/Gaze-v0", render_mode = "human")
env = FlattenObservation(env)
obs, info = env.reset()

print(obs)
print(info)

while True:
    random_action = numpy.random.randint(1,32)
    time.sleep(0.5)
    observation, reward, terminated, truncated, info = env.step(random_action)
    # print(info)
    env.render()