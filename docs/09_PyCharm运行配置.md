# 在 PyCharm 里运行本项目

> 前提：已经按 [`START_HERE.md`](../START_HERE.md) 第 1 步装好了 `classroom` 环境。
> 本文用 `D:\2026srp` 举例，换成你自己的路径即可。

---

## 一、核心原则：一定要选对解释器

PyCharm 默认可能用系统 Python 或 conda 的 base 环境，那些**都没装 torch/ultralytics**，一跑就是 `ModuleNotFoundError`。

**必须选 `classroom` 这个 conda 环境的 python.exe。**

---

## 二、四个步骤

### 1) 打开工程

`File → Open` → 选**项目根目录**（例如 `D:\2026srp`，能看到 `app` `src` `scripts` `configs` 的那一层）。

⚠️ 不要只打开 `app` 子目录，否则找不到 `configs/` 和 `runs/`。

### 2) 配置解释器

`File → Settings → Project: 2026srp → Python Interpreter`

| 情况 | 操作 |
|---|---|
| 下拉里已有 `Python 3.11 (classroom)` | 直接选中 |
| 没有 | 点齿轮 → `Add Interpreter` → `Add Local Interpreter` → 左侧选 `Conda Environment` → 选 `Use existing environment` → 在列表里找 `classroom` |

找不到 `classroom` 时，直接指到解释器文件：

```
<你的 conda 安装目录>\envs\classroom\python.exe
```

常见位置：`C:\Users\<你>\miniconda3\envs\classroom\python.exe`、`D:\anaconda3\envs\classroom\python.exe`

### 3) 验证解释器对不对（30 秒）

在 PyCharm 里打开 `scripts/check_env.py`，右键 → `Run 'check_env'`。

✅ 看到「环境就绪」就说明解释器配对了。
❌ 看到 `ModuleNotFoundError` → 解释器选错了，回第 2 步。

### 4) 建运行配置

`Run → Edit Configurations → + → Python`

**桌面端**

| 字段 | 值 |
|---|---|
| Script path | `<项目根>\app\desktop_app.py` |
| Parameters | 留空（可选 `--weights runs/best.pt`） |
| **Working directory** | **`<项目根>`（必填，否则找不到 configs/）** |
| Python interpreter | `classroom` |
| Emulate terminal in output console | 建议勾上（中文日志不乱码） |

**网页端**：把 `Module name` 设为 `streamlit`，Parameters 填 `run app/web_app.py --server.headless true`。

**只测模型**：Script path 选 `scripts/smoke_test.py`，Parameters 填 `--weights runs/best.pt --source test_images`。

---

## 三、常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: ultralytics` | 解释器选成了 base 或系统 Python | 改成 `classroom` |
| `FileNotFoundError: configs/classes.yaml` | Working directory 不对 | 设成项目根目录 |
| 摄像头打不开 | 被 QQ/微信/会议软件占用 | 先退出那些程序 |
| 界面能开但很卡 | 训练正占着 GPU | 等训练结束，或把 `configs/app.yaml` 的 `device` 改成 `cpu` |
| `Could not load the Qt platform plugin "windows"` | PyQt5 插件路径冲突 | Run 配置的 Environment variables 里加 `QT_PLUGIN_PATH=<你的env>\Library\plugins` |
| 中文显示成方块 | 缺微软雅黑 | Windows 自带；确认 `C:\Windows\Fonts\msyh.ttc` 存在 |
| 界面能开但没有任何框 | 画面不是课堂场景（比如对着屏幕拍） | 换成真实教室图片/视频；本模型只识别课堂场景 |

---

## 四、不想折腾 PyCharm 的话

在项目根目录打开 **Anaconda Prompt**：

```bash
conda activate classroom
python app\desktop_app.py
```

或者直接**双击 `run_app.bat`**（脚本会自动找 python）。
