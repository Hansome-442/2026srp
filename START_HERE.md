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

> **解压到哪个盘都行**（D、E、F 都可以），下文用 `D:\2026srp` 只是举例。
> 但有四条要注意：
> 1. **全英文路径**，别带中文和空格（`D:\我的项目\2026srp` 容易出问题）
> 2. **不要放 OneDrive / 坚果云等同步目录**（训练时文件被锁会报错）
> 3. **别放太深**（`C:\Users\xxx\Desktop\课程\大三下\2026srp` 这种长度容易撞 Windows 路径长度限制）
> 4. **留够 15 GB**（环境 7.6 GB + 数据 5.3 GB）
>
> 代码里已经把路径都改成相对的了，换盘符不用改任何配置。

---

## 第 1 步：装环境（约 20 分钟）

### 1.0 你需要装哪些软件

| 软件 | 必须吗 | 干什么用 | 大概多大 |
|---|---|---|---|
| **Miniconda** | ✅ **必须** | 建一个隔离的 Python 环境，不污染系统 | 安装包 90 MB |
| Git | ⭕ 可选 | 从 GitHub 拉代码、提交改动 | 60 MB |
| PyCharm | ⭕ 可选 | 写代码方便；**只想运行程序的话不需要** | 500 MB+ |

**装完之后**：环境本身约 **7.6 GB**（其中 torch 占 5.4 GB），再加上数据集 5.3 GB，**C 盘或 D 盘至少要留 15 GB 空闲**。

### 1.1 装 Miniconda（Windows）

**① 下载**

打开 <https://docs.conda.io/en/latest/miniconda.html>

往下找 **Windows installers**，点 **Miniconda3 Windows 64-bit**（文件名类似 `Miniconda3-latest-Windows-x86_64.exe`）。

> 网页进不去的话，用清华镜像：<https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>
> 找 `Miniconda3-latest-Windows-x86_64.exe`

**② 双击安装，一路 Next，只有两处要改**

| 安装页面 | 选什么 |
|---|---|
| Install for | **Just Me (recommended)** ← 不要选 All Users，否则要管理员权限 |
| Advanced Options | **不要勾** "Add Miniconda3 to my PATH environment variable" |
| | **勾上** "Register Miniconda3 as my default Python 3.x" |

其他保持默认，点 Install，等 2~3 分钟。

**③ 打开命令窗口（这一步最容易搞错）**

装完后按 **Win 键 → 输入 `Anaconda`** → 点 **`Anaconda Prompt (miniconda3)`**。

打开后是一个黑窗口，**后面所有命令都在这个窗口里敲**。

> ⚠️ 不是普通 `cmd`，也不是 `PowerShell`，更不是 PyCharm 里的终端。
> 在普通窗口里敲 `conda` 会提示「不是内部或外部命令」，不是没装好，是窗口选错了。

**④ 验证**

在黑窗口里敲：

```bash
conda --version
```

显示 `conda 24.x.x` 之类的版本号，就说明装好了。

> **已经装过 Anaconda 的人**：不用再装 Miniconda，直接打开 `Anaconda Prompt` 就行，后面命令完全一样。

### 1.2 建环境并装依赖（先看自己有没有 NVIDIA 显卡）

**第 ② 步分两种，其余三步完全一样。**

```bash
cd /d D:\2026srp

# ① 建一个干净的 python 3.11 环境
conda create -n classroom python=3.11 -y
conda activate classroom

# ② 装 torch —— 按你的机器二选一
#   有 NVIDIA 显卡（要训练）：GPU 版，约 2.5 GB
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
#   没有 NVIDIA 显卡（只改代码/看界面）：CPU 版，约 200 MB
#   pip install torch torchvision

# ③ 再装其余依赖
pip install -r requirements.lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

| 你的机器 | 第 ② 步装 | 下载量 | 能做什么 |
|---|---|---|---|
| 有 NVIDIA 显卡（8 GB 以上） | cu128 版 | 2.5 GB | 全部，包括训练 |
| 没有 / 核显 / 显存 < 6 GB | CPU 版 | 200 MB | 改代码、开界面、图片检测；**不能训练** |

> **为什么不用 `conda env create -f environment.yml`？**
> 那个文件不含 GPU 版 torch，`ultralytics` 会装上 CPU 版，导致有显卡也用不上，
> 还白下 2.5 GB。按上面三步走最稳。

### 1.3 确认装对了

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

| 输出 | 含义 |
|---|---|
| `2.7.1+cu128 True` | ✅ 有显卡且装对了 |
| `2.7.x False`（没有 `+cu`） | ✅ 没显卡，装的是 CPU 版，**正常** |
| `2.7.1+cu128 False` | ❌ **有显卡却用不上**，重装下面这条 |

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

- **RTX 50 系（5060/5070/5080/5090）必须 cu128 及以上**，cu121 装不上会报 `no kernel image`
- 有显卡但一直 `False` → 多半是 NVIDIA 驱动太旧，去官网更新到最新版

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

> **用 PyCharm 的话**：关键是把解释器指到 `classroom` 环境，否则一跑就 `ModuleNotFoundError`。
> 具体步骤（含运行配置怎么填）见 [`docs/09_PyCharm运行配置.md`](docs/09_PyCharm运行配置.md)。
> 不想折腾 PyCharm 就双击 `run_app.bat`，它会自动找 python。

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
