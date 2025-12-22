import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.rl.trigger_env import SonarTriggerEnv

def train():
    # 1. 创建环境
    env = SonarTriggerEnv()
    # 包装环境以支持向量化和归一化
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. 定义模型
    # 使用 MultiInputPolicy 处理字典输入 (CNN 处理地图, MLP 处理向量)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
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
        total_timesteps=50000,
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
