# 从这里开始（只看这一篇）

> 全部 6 步，照着做就行。**你不需要看其他任何文档。**
> 卡住了，最下面有一张表告诉你翻哪篇。

---

## 0. 你会收到 2 个文件

| 文件 | 大小 | 干什么 |
|---|---|---|
| `2026srp_交接包_20261117.zip` | 34 MB | 代码 + 文档 + 训练好的模型 |
| `2026srp_subset_20261117.zip` | 455 MB | 小数据集，用来验证训练能不能跑 |

（完整数据集 5.3 GB 先别下，等分工定了再说）

---

## 1. 解压（2 分钟）

两个 zip **都解压到同一个目录**，比如 `D:\2026srp`。

三条要求：

- 路径**全英文**，别带中文和空格
- **别放 OneDrive / 坚果云**这类同步目录
- 磁盘**留 15 GB**（环境 7.6 GB + 数据 5.3 GB）

✅ **做完的标志**：`D:\2026srp` 里能看到 `app`、`src`、`scripts`、`configs` 四个文件夹。

---

## 2. 装 Miniconda（10 分钟）

这是**唯一必须装的软件**，用来给项目建一个独立的 Python 环境。

**下载**：<https://docs.conda.io/en/latest/miniconda.html>
进不去就用镜像：<https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>
选 **Miniconda3 Windows 64-bit**。

**双击安装，一路 Next，只改两处**：

| 安装页面 | 选什么 |
|---|---|
| Install for | **Just Me** |
| Advanced Options | **不要勾** Add to PATH<br>**要勾** Register as default Python |

**装完怎么打开命令窗口**：
按 Win 键 → 输入 `Anaconda` → 点 **`Anaconda Prompt (miniconda3)`**。

后面所有命令都在这个黑窗口里敲。
⚠️ 不是 cmd，不是 PowerShell —— 在那些里面敲 `conda` 会报「不是内部或外部命令」。

✅ **做完的标志**：黑窗口里敲 `conda --version`，显示版本号。

> 已经装过 Anaconda 的：不用重装，直接打开 `Anaconda Prompt` 就行。

---

## 3. 装环境（15 分钟）

在 Anaconda Prompt 里依次敲。**先看自己有没有 NVIDIA 显卡**，第 2 条不一样。

```bash
# 1) 建一个干净的 Python 3.11 环境
conda create -n classroom python=3.11 -y
conda activate classroom

# 2) 装 PyTorch —— 二选一
#    有 NVIDIA 显卡（要训练）：约 2.5 GB
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
#    没有显卡（只运行/改代码）：约 200 MB
#    pip install torch torchvision

# 3) 装其余依赖
cd /d D:\2026srp
pip install -r requirements.lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

✅ **做完的标志**：

```bash
python scripts/check_env.py
```

最后一行打印 **「环境就绪」**，并且失败项是 **0**。

> 没显卡的人会看到 `⚠️ 本机没有 NVIDIA 显卡`，**这是正常的**，不是错误。

---

## 4. 放数据（5 分钟）

把 `2026srp_subset_20261117.zip` 解压到 **`D:\2026srp`**（就是项目根目录，第 1 步那个地方）。

会得到 `data\subset\` 和 `data\subset.yaml`。

✅ **做完的标志**：再跑一次 `python scripts/check_env.py`，看到

```
✅ data/subset/images/train         2000 个文件
```

---

## 5. 跑起来

**双击 `run_app.bat`**（在 `D:\2026srp` 里）。

它会自动找到你装好的环境，不用先激活。

✅ **做完的标志**：弹出窗口，顶部有一排彩色方块（抬头听讲 / 阅读看书 / …）；
点「本地图片检测」，随便选一张教室照片，能画出框。

> 想用 PyCharm 的看 `docs\09_PyCharm运行配置.md`；不想折腾就双击 bat。

---

## 6. 跑通之后

在群里回一句 **「已通」**，并且发两个东西：

1. `check_env.py` 输出里以 **GPU** 开头的那一行（有卡会显示型号，没卡显示 ⚠️）
2. 你的 **GitHub 用户名**（拉你进代码仓库）

然后等分工。

---

## 出问题了翻这里

| 你看到的 | 去这篇的哪一节 |
|---|---|
| `conda` 不是内部或外部命令 | 本文「2. 装 Miniconda」最后一段 |
| `ModuleNotFoundError: No module named 'torch'` | 本文「3. 装环境」——环境没激活 |
| `torch.cuda.is_available()` 是 False | 本文「3. 装环境」第 2 条重装 |
| `FileNotFoundError: E:\2026srp\...` | `docs\14_路径问题排查.md` |
| `FileNotFoundError: configs/classes.yaml` | `docs\14_路径问题排查.md` 第 2 条 |
| 解压完 check_env 说图片 0 张 | `docs\11_数据获取与解压说明.md` 第 3 步 |
| PyCharm 里跑报 ModuleNotFound | `docs\09_PyCharm运行配置.md` |
| 界面能开但一个框都没有 | 画面不是课堂场景（别对着屏幕拍） |
| 其他 | 群里问，把完整报错发出来 |

---

## 其他文档（都不用看）

`docs\` 里还有 13 篇，是写报告、做实验、查问题用的资料，**平时不用打开**：

| 文档 | 什么时候看 |
|---|---|
| `09_PyCharm运行配置` | 要在 PyCharm 里跑 |
| `11_数据获取与解压` | 下完整数据集、解压报错 |
| `14_路径问题排查` | 报 FileNotFoundError |
| `12_代码协作_GitHub` | 要提交代码 |
| `10_项目交接与上手指南` | 想了解整体设计、下一步做什么 |
| `13_项目完整介绍` | 汇报/答辩要讲项目时 |
| 01~08 | 课程学习笔记、数据集规范、分工表、实验记录 |
