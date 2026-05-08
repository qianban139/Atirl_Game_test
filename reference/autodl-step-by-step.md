# AutoDL 购买与配置全过程

## 1. 注册与充值

1. 打开 [AutoDL 官网](https://www.autodl.com)
2. 点右上角「注册」，手机号 + 验证码即可
3. 注册后进入控制台首页
4. 点右上角「充值」
   - 支付宝/微信均可
   - 先充 **¥100-200** 就够了（5090 按量 ~¥3-5/小时，跑 2 天 ≈ ¥150-240）
   - **别充太多**，用不完退款麻烦

---

## 2. 挑选 GPU 实例

控制台左侧点「GPU 市场」→ 进入算力市场页面：

### 2.1 筛选条件

| 筛选项 | 选择 |
|--------|------|
| GPU 型号 | **RTX 5090**（勾选） |
| 计费方式 | **按量计费**（不要选包周/包月） |
| 地区 | 选离你近的（北京/内蒙/广州 任选，速度快慢差异不大） |

> **注意**：如果 RTX 5090 没货，备选 RTX 4090（24GB），config 自动降级到 `config.py`。

### 2.2 查看实例列表

每个实例卡片显示：

```
┌─────────────────────────────────┐
│ 🟢 RTX 5090 | 32GB VRAM        │
│ CPU: 16核 | 内存: 32GB         │
│ 系统盘: 30GB | 数据盘: 50GB    │
│ 按量: ¥4.20/时                │
│ 可用: 3 台                     │
│           [租用]                │
└─────────────────────────────────┘
```

**选一台点「租用」**。

---

## 3. 配置实例

点租用后弹出配置面板：

### 3.1 基础配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU 数量 | **1 卡** | PPO 不支持多卡 |
| 数据盘 | **100GB** | 默认 50GB 可能不够（日志 + checkpoint） |
| 计费方式 | **按量计费** | 关机就停止计费（只收系统盘存储费） |

### 3.2 选择镜像（重要）

点「自定义镜像」或从框架列表选：

```
推荐镜像：
  框架: PyTorch 2.5.0
  Python: 3.12
  CUDA: 12.4
```

或者在搜索框直接搜 `PyTorch 2.5`，选一个预装了 PyTorch 的镜像。

> **省事镜像推荐**：选带 `miniconda` 的 PyTorch 镜像，环境已配好，只需 `pip install gymnasium`。

**如果你选的镜像不带 PyTorch**（裸 Ubuntu），也行，后续手动装：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3.3 确认创建

检查配置摘要，确认无误后点「**立即创建**」。

```
┌───────────────────────────────────────┐
│ 请确认以下配置                         │
│                                       │
│ GPU:  RTX 5090 × 1                   │
│ 镜像: PyTorch 2.5.0 + Python 3.12     │
│ 数据盘: 100GB                          │
│ 计费: 按量计费  ¥4.20/时              │
│                                       │
│        [取消]    [确认创建]             │
└───────────────────────────────────────┘
```

创建后等 10-30 秒，实例状态变为 **「运行中」**。

---

## 4. 连接实例

实例运行后，有多种连接方式：

### 方式 1：Jupyter Lab（最简单）

控制台点击实例的「**JupyterLab**」按钮 → 浏览器打开

```
→ 左侧文件管理器
→ 新建 Terminal（菜单：File → New → Terminal）
→ 在终端中执行后续命令
```

### 方式 2：SSH 终端（推荐，更稳定）

实例卡片上找 SSH 连接信息：

```
ssh -p 12345 root@region-1.autodl.com
密码：实例详情页可查看/重置
```

**Windows 用户**：推荐用 VS Code Remote-SSH 或 MobaXterm 连接。

1. 打开 VS Code
2. 安装插件 `Remote - SSH`
3. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`
4. 输入 `ssh -p 12345 root@region-1.autodl.com`
5. 输入密码

### 方式 3：AutoDL 内置终端

控制台点击「**Web SSH**」→ 浏览器内终端，免配置。

---

## 5. 第一次登录后的操作

连接到实例后，按顺序执行：

```bash
# ====== 1. 验证 GPU ======
nvidia-smi
# 应显示：NVIDIA GeForce RTX 5090, 32GB

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0))"
# 输出: CUDA: True, NVIDIA GeForce RTX 5090

# ====== 2. 进入工作目录 ======
cd /root/autodl-tmp
# 这个目录在数据盘上，关机会保留。别放在 /root 下（系统盘，关机可能丢）

# ====== 3. 克隆项目 ======
git clone <你的仓库地址> ATRI
cd ATRI

# ====== 4. 安装依赖 ======
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ====== 5. 安装 ALE ROM ======
pip install autorom -i https://pypi.tuna.tsinghua.edu.cn/simple
AutoROM --accept-license

# 验证 ROM 安装
python -c "import gymnasium as gym; gym.make('ALE/DonkeyKong-v5'); print('ROM OK')"

# ====== 6. GPU 冒烟测试 ======
python -c "
from config_5090 import Config5090 as Config
from networks import CNNEncoder, Actor
import torch

config = Config()
encoder = CNNEncoder(feature_dim=config.feature_dim).to(config.device)
actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
B = config.batch_size
x = torch.randn(B, 4, 84, 84, device=config.device)
phi = encoder(x)
logits = actor(phi)
print(f'Batch {B}: {phi.shape} -> {logits.shape}')
print(f'VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print('All OK.')
"
```

---

## 6. 启动训练

```bash
# 创建 tmux 会话（防 SSH 断开）
tmux new -s dk

# 启动训练
python train_5090.py

# 看到第一行日志输出后，按 Ctrl+B 然后 D 断开
# 重连：tmux attach -t dk
```

训练开始后先观察 5 分钟：
- 日志正常输出（每 100 episode 一行）
- `nvidia-smi` 显示 GPU 利用率 > 80%
- 显存占用 < 20GB（32GB 绰绰有余）

确认正常后 `Ctrl+B D` 断开 tmux，**关掉终端，训练不会停**。

---

## 7. 监控与收尾

### 7.1 中途检查

```bash
# 重新 SSH 登录
ssh -p 12345 root@region-1.autodl.com

# 重连 tmux
tmux attach -t dk

# 看 GPU
nvidia-smi
```

### 7.2 下载结果

训练结束后（或你想提前下载 checkpoint）：

```bash
# 在服务器上打包
cd /root/autodl-tmp/ATRI
tar -czf results.tar.gz checkpoints/ logs/

# 在本地电脑上执行（不是服务器！）
scp -P 12345 root@region-1.autodl.com:/root/autodl-tmp/ATRI/results.tar.gz .
```

或者：AutoDL 网页控制台 → 实例详情 →「文件管理」→ 选中文件 →「下载」。

### 7.3 关机（重要！）

**训练完成后一定要关机**，否则持续计费：

```
AutoDL 控制台 → 实例列表 → 找到该实例 → 点「关机」
```

关机后：
- 数据盘（/root/autodl-tmp）内容保留
- 系统盘不收费
- 下次开机继续用

确认不需要了再点「**销毁**」（销毁后数据盘也清空）。

---

## 8. 费用预估

| 项目 | 计算 | 金额 |
|------|------|------|
| 5090 训练 2 天 | 48h × ¥4.2 | **≈ ¥200** |
| 消融实验 1.5 天 | 36h × ¥4.2 | ≈ ¥150 |
| 数据盘 100GB/天 | ¥0.0035/GB/天 × 100GB | ≈ ¥0.35/天 |
| **合计（2 台并行）** | | **≈ ¥350** |
| **合计（1 台串行）** | | **≈ ¥400**（时间长因为串行） |

> 如果你只租一台按量、跑完 PPO+ICM (2天) + 消融 (1.5天) = 3.5天内完成 = **¥350 左右**

---

## 9. 故障速查

| 症状 | 检查 |
|------|------|
| 创建实例后无法连接 | 等 1-2 分钟初始化，刷新页面 |
| SSH 连接被拒 | 确认端口号（不是 22）、密码正确 |
| `import torch` 失败 | 选错镜像，手动 `pip install torch` |
| `gymnasium[atari]` 安装失败 | 先 `apt update && apt install -y cmake gcc`，再重试 |
| `AutoROM` 下载失败 | 手动传 ROM 文件或 `export AUTO_ROM_ACCEPT_LICENSE=1` |
| 训练到一半实例消失 | spot 实例被回收，checkpoint 是救命稻草；重建实例，从最近 checkpoint 恢复 |
| GPU 利用率 0% | 检查 `torch.cuda.is_available()`，确认没用 CPU |
