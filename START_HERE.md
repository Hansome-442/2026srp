# 从这里开始

> 给项目组成员。目标：**一台没装过的新电脑，1 小时内把检测系统跑起来**。
> 每一步都有「✅ 做完的标志」，卡住就看[文末排错表](#排错表)。
> 需要细节（训练参数怎么调、为什么这么设）再翻 [`docs/10_项目交接与上手指南.md`](docs/10_项目交接与上手指南.md)。

---

## 你会收到四样东西

| 东西 | 大小 | 从哪拿 | 放到哪 |
|---|---|---|---|
| `2026srp_交接包_日期.zip`<br>代码 + 文档 + 权重 + 演示视频 + git 存档 | 34 MB | 群文件 | 解压到 `D:\2026srp` |
| `2026srp_subset_日期.zip`<br>**最小可用子集，3400 张** | **455 MB** | 群文件 / 微信 | 解压到 `D:\2026srp` |
| `2026srp_dataset_日期_part01.zip`<br>完整数据集，46,760 张 | 5.3 GB | 百度网盘 | 解压到 `D:\2026srp\data` |
| `MD5SUMS.txt`、`解压说明.txt` | 几 KB | 跟完整数据集放一起 | 同上 |

> **不想等网盘的话，只用前两个就够了**：455 MB 的子集足够把环境、代码、训练链路全部跑通。
> 完整数据集等你真要训练时再下。

> 建议路径用 `D:\2026srp`——**不要放在中文名、带空格的深层目录、OneDrive 同步目录里**，
> 这三类路径会引发各种奇怪的读写错误。

---

## 第 1 步：装环境（约 20 分钟）

### 1.1 装 Miniconda

没有 conda 的话先去 <https://docs.conda.io/en/latest/miniconda.html> 下 Windows 64 位安装包。
装完打开 **Anaconda Prompt**（不是普通 cmd）。

### 1.2 创建环境

```bash
cd /d D:\2026srp
conda env create -f environment.yml
conda activate classroom
```

### 1.3 确认 torch 是 GPU 版（最关键的一步）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望看到 `2.7.1+cu128 True` 这样带 `+cuXXX` 且末尾为 `True` 的输出。

- 看到 `False` 或版本号里没有 `+cu` → 装成 CPU 版了，重装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

- **RTX 50 系（5060/5070/5080/5090）必须 cu128 及以上**，cu121 装不上会报 `no kernel image`

✅ **做完的标志**：`check_env.py` 最后一行打印「环境就绪」

```bash
python scripts/check_env.py
```

---

## 第 2 步：放代码

### 方式 A：拿到的是 zip（最简单）

直接解压到 `D:\2026srp`，跳到第 3 步。
缺点：以后改代码只能靠再传包。

### 方式 B：拿到的是 `.bundle`（推荐，有 git 历史）

```bash
cd /d D
git clone 2026srp_20261117.bundle 2026srp
cd 2026srp
```

以后负责人更新了，只给你一个几十 KB 的 `update.bundle`：

```bash
git pull D:\下载\update.bundle main
```

### 方式 C：从仓库 clone（如果已经建了远程）

```bash
git clone https://gitee.com/<账号>/2026srp.git     # 或 GitHub 地址
```

✅ **做完的标志**：`D:\2026srp` 下能看到 `app\ src\ scripts\ docs\ configs\` 这几个文件夹

---

## 第 3 步：放数据（两种，按需求选）

> **先想清楚：你现在要做什么？**
> 只是把环境和代码跑通 → 用**方式 A**（455 MB，群里直接发）。
> 要复现论文指标、正式训练 → 才需要**方式 B**（5.3 GB，百度网盘）。

| | 方式 A：最小可用子集 | 方式 B：完整数据集 |
|---|---|---|
| 大小 | **455 MB**（3400 张，8 类均衡） | 5.3 GB（46,760 张） |
| 怎么拿 | 群文件 / 微信直接发 | 百度网盘 |
| 能干什么 | 跑通环境、跑通训练链路、调界面 | 正式训练、复现指标、写论文 |
| 能训练出能用的模型吗 | 不能（样本太少） | 能 |

### 3A 最小可用子集（推荐先做这个）

1. 拿到 `2026srp_subset_20261117.zip`（455 MB），**解压到项目根目录** `D:\2026srp`
2. 它会生成 `data\subset\` 和 `data\subset.yaml`
3. 验证：

```bash
python scripts/check_env.py            # 看 data/subset 相关项
python scripts/train.py --data data/subset.yaml --epochs 10 --name subset_try --batch 8
```

跑完 `runs\train\subset_try\results.csv` 里 loss 在降，就说明**训练链路完全通了**。

> ⚠️ 子集只有 3400 张，训出来的模型精度没意义，**别拿它的指标写报告**。

### 3B 完整数据集（要正式训练时）

#### 3B.1 先校验，别急着解压

把 `MD5SUMS.txt` 和所有分卷放同一个文件夹，在该文件夹打开 Anaconda Prompt：

```bash
python D:\2026srp\scripts\pack_dataset.py --verify .
```

每一行都是 ✅ 才继续；有 ❌ 就只重新下载那一卷。

> 没有 MD5SUMS.txt 的话可以用系统命令逐个核对：
> `certutil -hashfile 2026srp_dataset_part01.zip MD5`

#### 3B.2 解压到 `data` 目录

**所有分卷解压到同一个 `D:\2026srp\data` 目录**（从 part01 开始，覆盖提示选「是」）。

解压完目录必须长这样：

```
D:\2026srp\data\
├─ dataset.yaml                    ← 训练入口，缺了就跑不了
├─ processed\
│  ├─ images\{train,val,test}\     ← 34,603 / 8,351 / 3,806 张
│  └─ labels\{train,val,test}\
└─ raw\                            ← 只有要重建数据集时才需要，可以先不放
```

✅ **做完的标志**：

```bash
python scripts/check_env.py
```

应该看到 `✅ data/processed/images/train    34603 个文件`

---

## 第 4 步：跑起来

按顺序跑，一步绿了再下一步：

```bash
# ① 端到端链路自检（模型→检测→跟踪→分析→预警→报表）
python scripts/smoke_test.py

# ② 训练链路冒烟（1 轮 / 小图，3~5 分钟）
python scripts/train.py --smoke

# ③ 打开界面
python app/desktop_app.py
```

✅ **做完的标志**：
- ① 打印「7 项全 PASS」
- ③ 弹出一个窗口，顶部有一排彩色类别图例；点「本地图片检测」选一张教室照片，能画出框

`runs/best.pt` 已经在包里，所以界面一打开用的就是训练好的 8 类模型，不是 COCO。

---

## 第 5 步：开始改代码

```bash
git checkout -b feature/你负责的部分     # 别直接在 main 上改
```

改完：

```bash
git add -A
git commit -m "feat: 说明你改了什么、为什么"
```

**不要提交**：`data/`、`runs/`、`*.pt`、`*.mp4`、`*.zip`（`.gitignore` 已经挡住，别用 `git add -f` 硬塞）。

你该做什么，看 [`docs/10` 第 6 节「下一步工作重心」](docs/10_项目交接与上手指南.md)。
优先级最高的是：**自采本校教室数据** 和 **按类阈值 + 预警跨帧确认（纯代码，半天）**。

---

## 排错表

| 现象 | 原因 | 解决 |
|---|---|---|
| `conda: 无法将"conda"项识别为…` | 用的是普通 cmd | 改用 Anaconda Prompt |
| `torch.cuda.is_available()` 是 False | 装成 CPU 版 | 按 1.3 重装 |
| `no kernel image is available` | torch 版本低于显卡架构 | 50 系必须 cu128+ |
| `CUDA out of memory` | batch 太大 | 训练加 `--batch 4`，或 `--imgsz 512` |
| `DataLoader worker exited` | Windows 多进程 | `configs\train.yaml` 里 `workers: 0` |
| 界面能开但一个框都没有 | 权重没放对，回退到 COCO | 确认 `runs\best.pt` 存在 |
| 中文标签显示成方块 | 缺微软雅黑 | Windows 自带；缺的话改 `src/detector.py` 字体候选 |
| `ModuleNotFoundError: src` | 不在项目根目录执行 | `cd /d D:\2026srp` 再跑 |
| 解压后 `check_env` 说图片是 0 | 解压到了错误层级 | 确认是 `data\processed\images`，不是 `data\processed\processed\images` |
| 训练 loss 是 nan | 标签异常或学习率 | 加 `--no-amp`，检查标签是否越界 |

还解决不了，把这三样发群里：**完整报错**、`python scripts/check_env.py` 的输出、你执行的命令。
