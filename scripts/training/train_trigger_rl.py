import os
import sys
import multiprocessing

# 将项目根目录添加到 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.rl.trigger_env import SonarTriggerEnv

# 避免与 ROS 的 scripts 包冲突，直接将当前目录加入路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from benchmark_callback import BenchmarkEvalCallback

def train():
    # Number of parallel environments (use CPU cores, max 16)
    n_envs = min(16, os.cpu_count() or 1)
    print(f"Using {n_envs} parallel environments...")

    # 1. Create parallel environments
    # Use SubprocVecEnv to run each environment in a separate process
    env = make_vec_env(
        SonarTriggerEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv
    )
    # Wrap environment for normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Define model
    # Use MultiInputPolicy for Dict observation (CNN for maps, MLP for vectors)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,   # Steps collected per environment
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient to prevent strategy from "stalling" due to high penalties
        tensorboard_log="./logs/trigger_rl/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./checkpoints/trigger_rl/",
        name_prefix="ppo_sonar"
    )

    # 3.5 Benchmark Evaluation Callback
    # Calculate eval_freq based on n_envs to evaluate roughly every 50,000 global steps
    benchmark_callback = BenchmarkEvalCallback(
        eval_freq=3125,  # 3125 * 16 = 50,000 steps
        verbose=1
    )

    # 4. Start training
    print("🏋️‍♂️ Starting model training...")
    model.learn(
        total_timesteps=6600000, 
        callback=[checkpoint_callback, benchmark_callback],
        progress_bar=True
    )

    # 5. Save final model
    model_path = "checkpoints/trigger_rl/ppo_sonar_final"
    model.save(model_path)
    env.save("checkpoints/trigger_rl/vec_normalize.pkl")
    print(f"Training completed, model saved to: {model_path}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./checkpoints/trigger_rl/", exist_ok=True)
    os.makedirs("./logs/trigger_rl/", exist_ok=True)
    
    train()
