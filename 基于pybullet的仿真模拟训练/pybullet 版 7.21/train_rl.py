import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from drone_rl_env import DronePathPlanningEnv, DroneDynamicsControlEnv
from utils import auto_load_model, save_model_by_steps

def find_latest_checkpoint(save_dir, model_name):
    """查找最新的检查点文件"""
    checkpoint_pattern = os.path.join(save_dir, f"{model_name}_*_steps.zip")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # 按文件名中的步数排序，返回最新的
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
    return checkpoints[-1]

def find_best_model(save_dir, model_name):
    """查找最佳模型文件"""
    best_model_path = os.path.join(save_dir, f"{model_name}_best_model.zip")
    if os.path.exists(best_model_path):
        return best_model_path
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='path', choices=['path', 'dynamics'], help='训练环境类型')
    parser.add_argument('--timesteps', type=int, default=100000, help='训练步数')
    parser.add_argument('--save_dir', type=str, default='./rl_models', help='模型保存目录')
    parser.add_argument('--continue_training', action='store_true', help='继续之前的训练')
    parser.add_argument('--force_new', action='store_true', help='强制重新开始训练')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 配置日志
    configure(args.save_dir, ["stdout", "csv"])

    if args.env == 'path':
        env = DronePathPlanningEnv()
        model_name = 'ppo_path_planning'
    else:
        env = DroneDynamicsControlEnv()
        model_name = 'ppo_dynamics_control'

    env = Monitor(env)
    
    # 创建回调函数
    eval_callback = EvalCallback(env, best_model_save_path=args.save_dir,
                                 log_path=args.save_dir, eval_freq=5000,
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=args.save_dir,
                                           name_prefix=model_name)

    print(f"开始训练 {model_name} 模型...")
    print(f"训练步数: {args.timesteps}")
    print(f"保存目录: {args.save_dir}")
    
    # 检查 GPU 是否可用
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 统一模型加载逻辑
    model = None
    if not args.force_new:
        model = auto_load_model(model_name, args.save_dir, env, device) if args.continue_training else None
    if model is None:
        print("🆕 创建新模型...")
        model = PPO('MlpPolicy', env, verbose=1, device=device)
    
    # 开始训练
    model.learn(total_timesteps=args.timesteps, 
                callback=[eval_callback, checkpoint_callback])
    
    # 保存最终模型
    save_model_by_steps(model, args.save_dir, model_name)
    
    # 显示训练完成信息
    print(f"\n🎉 训练完成!")
    print(f"最终模型: {os.path.join(args.save_dir, model_name + '_' + str(model.num_timesteps) + '.zip')}")
    print(f"最佳模型: {os.path.join(args.save_dir, model_name + '_best_model.zip')}")
    print(f"检查点文件: {args.save_dir}/") 