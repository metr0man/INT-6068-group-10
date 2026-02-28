import os
import glob
from stable_baselines3 import PPO

def find_latest_by_steps(save_dir, model_name):
    """查找步数最大的模型文件，文件名格式 model_name_数字.zip"""
    pattern = os.path.join(save_dir, f"{model_name}_*.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    def extract_steps(path):
        fname = os.path.basename(path)
        try:
            return int(fname.replace(f"{model_name}_", "").replace(".zip", ""))
        except Exception:
            return -1
    candidates = [(extract_steps(p), p) for p in candidates]
    candidates = [c for c in candidates if c[0] >= 0]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]

def auto_load_model(model_name, save_dir, env, device):
    """加载步数最大的模型，没有就新建"""
    latest_model_path = find_latest_by_steps(save_dir, model_name)
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"加载步数最大模型: {latest_model_path}")
        return PPO.load(latest_model_path, env=env, device=device)
    print("未找到可用模型，将新建模型")
    return None

def save_model_by_steps(model, save_dir, model_name, suffix=None):
    """统一的模型保存函数，按步数命名，可选中断等后缀"""
    filename = f"{model_name}_{model.num_timesteps}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".zip"
    path = os.path.join(save_dir, filename)
    model.save(path)
    print(f"✅ 模型已保存到 {path}")
    return path 