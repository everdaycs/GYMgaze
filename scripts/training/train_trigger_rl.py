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

def train():
    # 并行环境数量 (使用 CPU 核心数，最多 16 个)
    n_envs = min(16, os.cpu_count() or 1)
    print(f"🚀 使用 {n_envs} 个并行环境进行加速训练...")

    # 1. 创建并行环境
    # 使用 SubprocVecEnv 在独立进程中运行每个环境
    env = make_vec_env(
        SonarTriggerEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv
    )
    # 包装环境以支持归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. 定义模型
    # 使用 MultiInputPolicy 处理字典输入 (CNN 处理地图, MLP 处理向量)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,   # 每个环境收集的步数 (总 buffer = n_steps * n_envs)
        batch_size=512, # 增加批量大小以利用更多数据
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/trigger_rl/"
    )

    # 3. 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/trigger_rl/",
        name_prefix="ppo_sonar"
    )

    # 4. 开始训练
    print("🚀 开始训练 RL 传感器触发策略...")
    model.learn(
        total_timesteps=300000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 5. 保存最终模型
    model_path = "checkpoints/trigger_rl/ppo_sonar_final"
    model.save(model_path)
    env.save("checkpoints/trigger_rl/vec_normalize.pkl")
    print(f"✅ 训练完成！模型已保存至: {model_path}")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("./checkpoints/trigger_rl/", exist_ok=True)
    os.makedirs("./logs/trigger_rl/", exist_ok=True)
    
    train()
