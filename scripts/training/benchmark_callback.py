from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import sys

# 将传感器策略实验室路径添加到 sys.path
_STRATEGY_LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sensor_strategy_lab"))
if _STRATEGY_LAB_ROOT not in sys.path:
    sys.path.insert(0, _STRATEGY_LAB_ROOT)

from strategy_benchmark import run_benchmark_episode


class BenchmarkEvalCallback(BaseCallback):
    """
    在训练期间定期评估模型在标准 Benchmark 数据集上的表现
    """
    def __init__(self, eval_freq=10000, verbose=1):
        super(BenchmarkEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_coverage = -np.inf
        
        # Benchmark 配置 (使用 benchmark 中的子集进行快速评估)
        self.scenes = ["sparse", "simple", "corridor"]
        self.seeds = [10, 20] # 仅使用少量种子进行快速评估
        self.steps = 1024

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"🔄 Starting Benchmark Evaluation at step {self.num_timesteps}...")
            
            temp_model_path = "checkpoints/trigger_rl/ppo_sonar_final.zip"
            temp_stats_path = "checkpoints/trigger_rl/vec_normalize.pkl"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)

            # Backup existing files (if any) to prevent corruption during evaluation
            import shutil
            has_model_bak = False
            has_stats_bak = False
            if os.path.exists(temp_model_path):
                shutil.copy(temp_model_path, temp_model_path + ".tmp_eval_bak")
                has_model_bak = True
            if os.path.exists(temp_stats_path):
                shutil.copy(temp_stats_path, temp_stats_path + ".tmp_eval_bak")
                has_stats_bak = True
            
            # Save current training state to the standard paths that RLStrategy expects
            self.model.save(temp_model_path)
            if hasattr(self.training_env, 'save'):
                self.training_env.save(temp_stats_path)
            elif hasattr(self.model, 'get_vec_normalize_env') and self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(temp_stats_path)
                
            # Run benchmark episodes
            coverages = []
            crosstalk_rates = []
            
            for scene in self.scenes:
                for seed in self.seeds:
                    try:
                        # run_benchmark_episode uses TriggerMode.RL which loads from temp_model_path
                        res = run_benchmark_episode("rl", seed, scene, self.steps)
                        coverages.append(res['coverage'])
                        crosstalk_rates.append(res['crosstalk_rate'])
                    except Exception as e:
                        print(f"Error during benchmark episode (scene={scene}, seed={seed}): {e}")
            
            # Restore backups if they existed
            if has_model_bak:
                shutil.move(temp_model_path + ".tmp_eval_bak", temp_model_path)
            if has_stats_bak:
                shutil.move(temp_stats_path + ".tmp_eval_bak", temp_stats_path)
            
            # Compute metrics
            mean_cov = np.mean(coverages) if coverages else 0
            mean_xtalk = np.mean(crosstalk_rates) if crosstalk_rates else 0
            
            if self.verbose > 0:
                print(f"📈 Eval Result: Coverage={mean_cov:.2f}, Crosstalk={mean_xtalk*100:.2f}%")
            
            # Record to Tensorboard
            self.logger.record("benchmark/coverage", mean_cov)
            self.logger.record("benchmark/crosstalk_rate", mean_xtalk)
            
            # Track best performance
            if mean_cov > self.best_mean_coverage:
                self.best_mean_coverage = mean_cov
                if self.verbose > 0:
                    print(f"🔥 New Best Benchmark Model (Cov: {mean_cov:.4f})")
                
                best_model_path = "checkpoints/trigger_rl/best_benchmark_model"
                self.model.save(best_model_path)
                if hasattr(self.training_env, 'save'):
                    self.training_env.save(best_model_path + "_vec_normalize.pkl")
                
        return True
