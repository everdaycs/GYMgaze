from gymnasium.envs.registration import register

register(
    id="gymnasium_env_gaze/Gaze-v0",
    entry_point="gymnasium_env_gaze.envs.gaze_env:GazeEnv",
)
