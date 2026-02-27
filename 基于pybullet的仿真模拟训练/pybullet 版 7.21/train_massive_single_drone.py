#!/usr/bin/env python3
"""大规模单无人机训练脚本 - 使用大量并行环境提高GPU利用率"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from massive_single_drone_env import create_massive_single_drone_envs, SingleDronePathPlanningEnv
from utils import auto_load_model, save_model_by_steps

def train_massive_single_drone(
    num_envs=32,
    num_cpu=8,
    timesteps=500000,
    batch_size=256,
    learning_rate=3e-4,
    save_interval=50000,
    eval_interval=25000,
    model_name="massive_single_drone_ppo"
):
    """训练大规模单无人机模型"""
    
    print("🚀 开始大规模单无人机训练...")
    print(f"📊 训练参数:")
    print(f"   环境数量: {num_envs}")
    print(f"   CPU进程数: {num_cpu}")
    print(f"   总时间步: {timesteps}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {learning_rate}")
    
    # 创建训练环境
    print("\n🔧 创建训练环境...")
    train_env = create_massive_single_drone_envs(num_envs=num_envs, num_cpu=num_cpu)
    
    # 创建评估环境（使用较少的进程）
    print("🔧 创建评估环境...")
    eval_env = create_massive_single_drone_envs(num_envs=8, num_cpu=4)
    
    # 创建模型目录
    os.makedirs("models", exist_ok=True)
    
    # 配置PPO参数
    model_kwargs = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": 2048,  # 每个环境收集的步数
        "n_epochs": 10,   # 每次更新的训练轮数
        "gamma": 0.99,    # 折扣因子
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,  # 熵系数，鼓励探索
        "vf_coef": 0.5,    # 价值函数系数
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "tensorboard_log": None,  # 避免tensorboard依赖问题
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),  # 更大的网络
        },
        "verbose": 1
    }
    
    # 创建PPO模型（断点续训）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = auto_load_model(model_name, "models", train_env, device)
    if model is None:
        print("🤖 创建PPO模型...")
        model = PPO(
            "MlpPolicy",
            train_env,
            **model_kwargs
        )
    
    # 创建回调函数
    callbacks = []
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{model_name}_best",
        log_path=f"models/{model_name}_eval",
        eval_freq=max(eval_interval // num_envs, 1),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_interval // num_envs, 1),
        save_path=f"models/{model_name}_checkpoints",
        name_prefix=model_name
    )
    callbacks.append(checkpoint_callback)
    
    # 开始训练
    print(f"\n🎯 开始训练 {timesteps} 时间步...")
    print("⏱️  预计训练时间: 根据环境数量和GPU性能而定")
    
    try:
        # 创建自定义回调列表，避免进度条
        from stable_baselines3.common.callbacks import CallbackList
        callback_list = CallbackList(callbacks)
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback_list,
            progress_bar=False
        )
        
        # 保存最终模型
        save_model_by_steps(model, "models", model_name)
        print(f"✅ 训练完成！最终模型保存至: models/{model_name}_{model.num_timesteps}.zip")
        
        # 打印训练统计
        print("\n📊 训练统计:")
        print(f"   总环境数: {num_envs}")
        print(f"   总时间步: {timesteps}")
        print(f"   实际训练步数: {model.num_timesteps}")
        
        return model
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        # 保存中断时的模型
        save_model_by_steps(model, "models", model_name, suffix="interrupted")
        print(f"💾 中断模型保存至: models/{model_name}_{model.num_timesteps}_interrupted.zip")
        return model
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trained_model(model_path, num_test_episodes=5):
    """测试训练好的模型"""
    print(f"\n🧪 测试模型: {model_path}")
    
    try:
        # 加载模型
        model = PPO.load(model_path)
        print("✅ 模型加载成功")
        
        # 创建测试环境
        test_env = SingleDronePathPlanningEnv(env_id=0)
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_test_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            if info.get('target_reached', False):
                success_count += 1
            
            print(f"   回合 {episode + 1}: 奖励={episode_reward:.2f}, 步数={steps}, 成功={info.get('target_reached', False)}")
        
        avg_reward = total_reward / num_test_episodes
        success_rate = success_count / num_test_episodes
        
        print(f"\n📊 测试结果:")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   成功率: {success_rate:.2%}")
        
        return avg_reward, success_rate
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="大规模单无人机训练")
    parser.add_argument("--num_envs", type=int, default=32, help="并行环境数量")
    parser.add_argument("--num_cpu", type=int, default=8, help="CPU进程数")
    parser.add_argument("--timesteps", type=int, default=500000, help="训练时间步")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--save_interval", type=int, default=50000, help="保存间隔")
    parser.add_argument("--eval_interval", type=int, default=25000, help="评估间隔")
    parser.add_argument("--model_name", type=str, default="massive_single_drone_ppo", help="模型名称")
    parser.add_argument("--test_only", action="store_true", help="仅测试模型")
    parser.add_argument("--model_path", type=str, help="要测试的模型路径")
    
    args = parser.parse_args()
    
    if args.test_only:
        if args.model_path:
            test_trained_model(args.model_path)
        else:
            print("❌ 测试模式需要指定模型路径 (--model_path)")
        return
    
    # 开始训练
    model = train_massive_single_drone(
        num_envs=args.num_envs,
        num_cpu=args.num_cpu,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        model_name=args.model_name
    )
    
    if model is not None:
        # 测试最终模型
        final_model_path = f"models/{args.model_name}_{model.num_timesteps}.zip"
        if os.path.exists(final_model_path):
            print(f"\n🧪 测试最终模型...")
            test_trained_model(final_model_path)

if __name__ == "__main__":
    main() 