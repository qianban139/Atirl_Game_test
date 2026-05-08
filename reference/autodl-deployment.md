# AutoDL 云端训练方案

## 1. 租用配置

| 配置项 | 推荐 | 说明 |
|--------|------|------|
| GPU | **RTX 5090 (32GB)** | 32GB 显存，当前配置 vs 标准配置对比见下表 |
| CPU | ≥ 16 核 | 24 并行 env 需要较多 CPU 推理 |
| 内存 | ≥ 32GB | 24 env × Atari 实例 + Buffer |
| 系统盘 | ≥ 30GB | 系统 + 依赖 + ROM |
| 数据盘 | ≥ 100GB | checkpoint + TensorBoard 日志 |

### 5090 vs 标准配置

| 参数 | 标准 (3080 10GB) | 5090 (32GB) | 提升 |
|------|------------------|-------------|------|
| num_envs | 8 | 24 | 3× |
| rollout_steps | 128 | 256 | 2× |
| batch_size | 1024 | 6144 | 6× |
| minibatch_size | 256 | 1024 | 4× |
| feature_dim | 256 | 512 | 2× |
| 预计训练时间 | ~5 天 | **~1-2 天** | — |

**省钱技巧**：使用 AutoDL 的**按量计费**或**可中断实例**（spot），不要包周。5090 跑 1-2 天按量比包周便宜很多。

---

## 2. 实例初始化

SSH 登录后，一次性粘贴执行：

```bash
# 确认 GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')"

# 克隆项目
cd /root/autodl-tmp
git clone <你的仓库地址> ATRI
cd ATRI

# 安装依赖（清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 ROM
pip install autorom -i https://pypi.tuna.tsinghua.edu.cn/simple
AutoROM --accept-license
```

> **注意**：如果 AutoROM 下载 ROM 失败，手动下载 `Roms.rar`，解压到 `/root/.gymnasium/roms/`。

### 2.1 GPU 冒烟测试

```bash
python -c "
from config_5090 import Config5090 as Config
from networks import CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics
import torch

config = Config()
device = config.device
print(f'Device: {device}')
print(f'Batch size: {config.batch_size}')

# 创建网络并前向传播，确认显存分配正常
encoder = CNNEncoder(feature_dim=config.feature_dim).to(device)
actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(device)

# 模拟 6144 条样本的前向传播
B = config.batch_size
x = torch.randn(B, 4, 84, 84, device=device)
phi = encoder(x)
logits = actor(phi)
print(f'Forward pass OK: {phi.shape} -> {logits.shape}')
print(f'VRAM used: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print('GPU test passed.')
"
```

---

## 3. 启动训练

### 3.1 tmux 持久化

```bash
# 创建会话
tmux new -s dk_train

# 进入项目
cd /root/autodl-tmp/ATRI

# 启动 5090 训练
python train_5090.py

# Ctrl+B → D 断开（训练继续）
# 重连：tmux attach -t dk_train
```

### 3.2 监控

```bash
# 终端 1: GPU 状态
watch -n 1 nvidia-smi

# 终端 2: TensorBoard
python -m tensorboard.main --logdir logs/ --port 6006 --bind_all
# AutoDL 通常提供端口映射，通过网页访问 TensorBoard

# 终端 3: 实时日志（tmux 内）
# 直接看 stdout 输出即可
```

---

## 4. 训练策略

| 阶段 | 代码 | 预计时间 |
|------|------|----------|
| 冒烟测试 | 快速验证脚本 | ~1 分钟 |
| 快速验证 | 1M steps | ~30 分钟 |
| 完整训练 | `python train_5090.py` | **~1-2 天** |

### 4.1 快速验证

```bash
python -c "
from config_5090 import Config5090 as Config
Config.total_timesteps = 5000
Config.num_envs = 4
Config.rollout_steps = 32
Config.log_interval = 1
Config.save_interval = 1000000
from train_5090 import train
train()
"
```

5K 步无报错后，恢复正式参数跑 1M 步确认 loss 趋势正常，然后正式启动 50M 训练。

---

## 5. 结果回传

训练完成后下载到本地：

```bash
# 在服务器上打包
cd /root/autodl-tmp/ATRI
tar -czf results_5090.tar.gz checkpoints/ logs/

# 在本地终端下载
scp root@<实例IP>:/root/autodl-tmp/ATRI/results_5090.tar.gz .
```

或者用 AutoDL 网页的文件管理直接下载。

---

## 6. 并行策略

5090 单卡跑得快，不用双卡。时间线：

```
第 1 天：跑 PPO+ICM (train_5090.py)   → 1-2 天完成
第 2 天：跑 消融 PPO only (train_ablation.py + 修改 config 为 5090 参数)
         如果同时租两台 5090，可以并行跑，半天收工
```

---

## 7. 常见问题

| 问题 | 解决 |
|------|------|
| ALE ROM 找不到 | `pip install autorom && AutoROM --accept-license` |
| CUDA out of memory (32GB 不应该发生) | 降 `num_envs` 到 16，降 `minibatch_size` 到 512 |
| 24 env 导致 CPU 负载过高 | 降 `num_envs` 到 16，5090 完全够用 |
| 训练太慢 | 检查 `torch.cuda.is_available()`，确认用的 GPU 不是 CPU |
| spot 实例被回收 | checkpoint 每 500 episode 保存，重启后从最近 ckpt 恢复 |
| 5090 风扇噪音/温度 | AutoDL 机房管散热，不用操心 |
