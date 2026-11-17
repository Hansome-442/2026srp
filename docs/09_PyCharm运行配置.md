# PyCharm 里运行本项目的配置说明

## 一、能跑的环境（只有一个）

| 环境 | 解释器路径 | Python | torch | ultralytics | opencv | PyQt5 | 能跑吗 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **conda 环境 `pytorch`** | `E:\anaconda3\envs\pytorch\python.exe` | 3.11.15 | ✅ 2.7.1+cu128 | ✅ 8.4.178 | ✅ | ✅ | **✅ 可以** |
| anaconda base | `E:\anaconda3\python.exe` | 3.11.4 | ❌ | ❌ | ❌ | ✅ | ❌ 缺 torch/ultralytics/opencv |
| 系统 Python 3.8 | `C:\Users\123\AppData\Local\Programs\Python\Python38\python.exe` | 3.8.10 | ✅ 2.0 CPU | ❌ | ❌ | ❌ | ❌ 缺 ultralytics/opencv/PyQt5 |
| 系统 Python 3.13 | `...\Python313\python.exe` | 3.13.5 | ❌ | ❌ | ❌ | ❌ | ❌ |
| conda `py38_202530304481` | `E:\anaconda3\envs\py38_202530304481\python.exe` | 3.8.20 | ❌ | ❌ | ❌ | ❌ | ❌ |

**结论：只能用 conda 的 `pytorch` 环境。**
PyCharm 里它可能显示成两个名字（都指向同一个 python.exe）：
- `Python 3.10 (pytorch)` ← 名字标错了，实际是 3.11.15
- `pytorch`

## 二、PyCharm 配置步骤

### 1) 打开工程
`File → Open` → 选 `E:\2026srp`（不要只打开 `app` 子目录）

### 2) 选解释器
`File → Settings → Project: 2026srp → Python Interpreter`
- 下拉里直接选 **`Python 3.10 (pytorch)`** 或 **`pytorch`**；
- 如果下拉里没有：点 `Add Interpreter → Add Local Interpreter → Conda Environment → Use existing environment`，
  路径填 `E:\anaconda3\envs\pytorch\python.exe`。

### 3) 建运行配置（桌面端）
`Run → Edit Configurations → + → Python`
| 字段 | 值 |
| --- | --- |
| Script path | `E:\2026srp\app\desktop_app.py` |
| Parameters | 留空（可选 `--weights runs/best.pt`） |
| **Working directory** | **`E:\2026srp`**（必须，否则找不到 configs/ 和 runs/best.pt） |
| Python interpreter | `pytorch` |
| Emulate terminal in output console | 建议勾上（中文日志不乱码） |

### 4) 网页端（可选）
| 字段 | 值 |
| --- | --- |
| Module name | `streamlit`（选 "Module name" 而不是 script） |
| Parameters | `run app/web_app.py --server.headless true` |
| Working directory | `E:\2026srp` |

### 5) 只验证模型能不能用（最快）
Script path 选 `scripts/smoke_test.py`，Parameters 填 `--weights runs/best.pt --source test_images`，直接 Run。

## 三、常见坑

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: ultralytics` | 解释器选错了，选成 base/3.8；改成 `pytorch` |
| `FileNotFoundError: configs/classes.yaml` | Working directory 不是 `E:\2026srp` |
| 摄像头打不开 | QQ/微信/会议软件占用摄像头，先关掉 |
| 界面能开但很卡 | 训练在占 GPU，等训练结束或把 `configs/app.yaml` 的 `device` 改 `cpu` |
| `Could not load the Qt platform plugin "windows"` | 环境里 PyQt5 冲突，在 Run 配置的环境变量里加 `QT_PLUGIN_PATH=E:\anaconda3\envs\pytorch\Library\plugins` |
| 中文变方块 | 已内置字体回退（msyh.ttc），若仍异常检查 `C:\Windows\Fonts\msyh.ttc` 是否存在 |
