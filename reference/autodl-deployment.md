# AutoDL 云端训练方案

## 1. 租用配置建议

| 配置项 | 推荐 | 说明 |
|--------|------|------|
| GPU | RTX 3080 / 3080Ti / 4090 | 单卡即可，PPO 不需要多卡 |
| 显存 | ≥ 10GB | 8 并行 env + CNN + ICM 约占用 4-6GB |
| CPU | 8 核 | env 并行依赖 CPU 推理 |
| 内存 | ≥ 16GB | Replay Buffer + 8 个 Atari 实例 |
| 系统盘 | ≥ 30GB | 系统 + 依赖 + ROM |
| 数据盘 | ≥ 50GB | 模型 checkpoint 每个 ~15MB，日志 TensorBoard |

**省钱技巧**：使用 AutoDL 的**按量计费**或**可中断实例**（spot），比包周便宜 50-70%。但 spot 可能被回收，记得频繁保存 checkpoint。

---

## 2. 实例初始化

SSH 登录后，一次性执行以下命令：

```bash
# 确认 CUDA 和 GPU 可用
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 克隆项目代码
cd /root/autodl-tmp
git clone <你的仓库地址> ATRI
cd ATRI

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 AutoROM（首次需接受协议）
pip install autorom -i https://pypi.tuna.tsinghua.edu.cn/simple
AutoROM --accept-license
```

> **注意**：如果 AutoROM 下载 ROM 失败（网络问题），可以手动下载 `Roms.rar`，解压到 `/root/.gymnasium/roms/` 目录。

---

## 3. 启动训练

### 3.1 使用 tmux 保持会话（防止 SSH 断开导致训练中断）

```bash
# 创建 tmux 会话
tmux new -s dk_train

# 进入项目目录
cd /root/autodl-tmp/ATRI

# 启动训练
python train.py

# 按 Ctrl+B 然后按 D 断开 tmux（训练继续运行）
# 重新连接：tmux attach -t dk_train
```

### 3.2 监控训练

```bash
# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# 启动 TensorBoard（在另一个 tmux 窗口）
tmux new -s tb
tensorboard --logdir logs/ --bind_all
# AutoDL 通常会提供端口映射地址访问 TensorBoard

# 查看训练日志
tail -f nohup.out  # 如果用了 nohup
```

---

## 4. 训练策略

| 阶段 | 步数 | 预计时间 | 目的 |
|------|------|----------|------|
| 冒烟测试 | 5,000 | ~2 分钟 | 确认无报错 |
| 快速验证 | 1M | ~1 小时 | 确认 loss 下降趋势 |
| 完整训练 | 50M | 3-5 天 | 获取最终结果 |

### 4.1 快速验证流程

```bash
# 5K 步验证
python -c "
from config import Config
Config.total_timesteps = 5000
Config.num_envs = 2
Config.rollout_steps = 32
Config.log_interval = 1
Config.save_interval = 1000000
from train import train
train()
"
```

5K 步无报错 → 1M 步验证 loss 趋势正常 → 启动完整 50M 训练。

---

## 5. 结果回传

训练完成后，将权重和日志下载到本地：

```bash
# 打包结果
cd /root/autodl-tmp/ATRI
tar -czf results.tar.gz checkpoints/ logs/

# 下载到本地（在本地终端执行）
scp root@<AutoDL实例IP>:/root/autodl-tmp/ATRI/results.tar.gz .
```

或者使用 AutoDL 网页的文件管理功能直接下载。

---

## 6. 并行策略（可选）

如果想加速实验，可以同时在两台机器上跑：

| 机器 | 运行脚本 | 预计时间 |
|------|----------|----------|
| 实例 A | `python train.py`（PPO+ICM） | 3-5 天 |
| 实例 B | `python train_ablation.py`（PPO only） | 2-3 天 |

这样两台同时跑，数据采集时间缩短一半。两台的超参和 seed 都相同（seed=42），确保可比性。

---

## 7. 关键细节

| 问题 | 解决 |
|------|------|
| ALE ROM 找不到 | `pip install autorom && AutoROM --accept-license` |
| gymnasium 版本不兼容 | 固定版本：`pip install gymnasium==1.0.0` |
| AtariPreprocessing 报 `terminal_on_life_loss` 异常 | gymnasium >= 1.0.0 需要 `gymnasium[atari]` 安装 |
| CUDA out of memory | 减少 `num_envs` 到 4，或减小 `minibatch_size` 到 128 |
| 训练太慢 | 确保 `torch.cuda.is_available()` 返回 True，检查是否用了 CPU |
| tmux 窗口被杀 | AutoDL 不杀 tmux 进程，但 spot 实例可能被回收——checkpoint 是救命稻草 |
