# 课堂行为检测系统（YOLOv11）· 2026SRP 项目代码

面向「多场景、多学段课堂行为数据集 + 可落地课堂行为检测系统」的完整工程：
数据集转换与体检 → YOLOv11 训练/验证/导出 → 可视化检测系统（图片/视频/摄像头 +
行为分析 + 异常预警 + 数据可视化 + 报表导出）。

> 对应预期成果：
> 1. 样本量 ≥ 5000、覆盖 8 类核心课堂行为的课堂行为数据集（7 类来自既有标注数据，第 8 类「玩手机」自采自标）；
> 2. 可落地的课堂行为检测系统（可视化），支持本地图片、视频、摄像头实时检测，含行为分析、异常预警、数据可视化。

> **新成员请先看 [`docs/10_项目交接与上手指南.md`](docs/10_项目交接与上手指南.md)**：
> 环境安装、数据放置、第一次训练、下一步分工、排错表都在里面。
> 下文命令里的 `python` 是原作者机器的路径，换成你自己的 python 即可。

---

## 一、目录结构

```
2026srp/
├─ configs/
│  ├─ classes.yaml          # 8 类行为定义（中英文名、颜色、异常等级、别名、预警规则）
│  ├─ train.yaml            # YOLOv11 训练超参（已按 RTX 5070 8GB 调优）
│  └─ app.yaml              # 检测系统配置（权重、阈值、跟踪、摄像头、报表）
├─ data/
│  ├─ raw/                  # 原始数据（VOC 或 YOLO 格式）放这里
│  ├─ to_label/             # 待标注图片（由视频抽帧得到）
│  ├─ assignments/          # 7 人分工任务清单
│  └─ processed/            # 转换 + 划分后的标准 YOLO 数据集
├─ scripts/
│  ├─ prepare_dataset.py    # VOC/XML 或 YOLO -> 标准数据集（分层划分 + 统计报告）
│  ├─ dataset_stats.py      # 数据集体检（类别分布、框数、缺失标注、尺寸分布）
│  ├─ train.py              # YOLOv11 训练
│  ├─ val.py                # 验证 + 每类指标报告
│  ├─ export.py             # 导出 ONNX/OpenVINO
│  └─ collect_helper.py     # 视频抽帧、7 人分工、标注校验、结果合并
├─ src/
│  ├─ detector.py           # YOLOv11 检测/跟踪封装 + 中文绘制
│  ├─ tracker.py            # 轻量 IOU 跟踪器（逐学生时长统计）
│  ├─ analytics.py          # 行为时长、抬头率、参与度、逐学生画像
│  ├─ alerts.py             # 异常预警（阈值 + 冷却 + jsonl 留档）
│  ├─ report.py             # CSV / Excel / Markdown / 图表导出
│  └─ utils.py              # 配置、中文路径读写、日志、字体
├─ app/
│  ├─ desktop_app.py        # PyQt5 桌面端（图片/视频/摄像头实时检测）
│  └─ web_app.py            # Streamlit 网页端（浏览器访问）
├─ docs/                    # 中文文档（搭建、数据规范、分工、使用、进度）
├─ runs/                    # 训练输出与权重
└─ reports/                 # 统计报告、预警日志、导出报表
```

---

## 二、5 分钟跑通（本机已验证）

```powershell
cd <项目目录>
conda activate classroom          # 环境名见 START_HERE.md

# 1) 用课程自带 VOC 数据先跑通数据流程（只取 person 一类做冒烟验证）
scripts\prepare_dataset.py --voc-dir <你本地 VOC 数据目录> --max-samples 400

# 2) 冒烟训练（1 轮、320 尺寸，验证环境与流程）
scripts\train.py --smoke

# 3) 验证 + 启动可视化系统
scripts\val.py --weights runs\best.pt
app\desktop_app.py                         # 桌面端
-m streamlit run app\web_app.py            # 网页端
```

拿到真正的课堂行为数据后：

```powershell
# 1) 数据 -> 标准数据集（自动分层划分、稀有类过采样、生成统计报告）
scripts\prepare_dataset.py --voc-dir data\raw\classroom_voc --oversample-min 300

# 2) 训练（读 configs/train.yaml；命令行可临时覆盖）
scripts\train.py
scripts\train.py --model yolo11n.pt --epochs 100 --batch 16 --name quick_try

# 3) 评估 / 导出 / 部署
scripts\val.py --weights runs\train\classroom_yolo11s\weights\best.pt --split test
scripts\export.py --weights runs\best.pt --format onnx
```

训练完成后把权重的实际路径填进 `configs/app.yaml` 的 `model.weights`，界面即用新模型。

---

## 三、8 类课堂行为定义

| id | 类别 | 说明 | 属性 |
| --- | --- | --- | --- |
| 0 | 抬头听讲 | 面向前方、抬头注视 | 正常 |
| 1 | 阅读看书 | 低头看书/资料 | 正常 |
| 2 | 书写记笔记 | 握笔书写 | 正常 |
| 3 | 举手发言 | 单/双臂上举 | 积极 |
| 4 | 站立回答 | 起立回答 | 积极 |
| 5 | 小组讨论 | 转头交流讨论 | 正常 |
| 6 | 趴桌睡觉 | 头伏桌面、长时间不动 | 异常（预警） |
| 7 | 玩手机 | **自采新增类**：双手持手机、屏幕光明显 | 异常（预警） |

类别名、颜色、别名、预警阈值都在 `configs/classes.yaml` 中集中管理，标注时使用中文类别名，
转换脚本会自动映射到 0~7 的类别 id。

---

## 四、系统功能

| 功能 | 桌面端 | 网页端 |
| --- | --- | --- |
| 本地图片检测 | ✅ | ✅ |
| 本地视频检测 | ✅ 逐帧 | ✅ 抽帧合成结果视频 |
| 摄像头实时检测 | ✅ 连续实时 | ✅ 拍照检测 |
| 目标跟踪（逐学生时长） | ✅ ByteTrack / 轻量 IOU | ✅ |
| 行为分析（抬头率、参与度、时长分布） | ✅ | ✅ |
| 异常预警（趴桌/玩手机） | ✅ 实时列表 + jsonl 日志 | ✅ |
| 数据可视化 | ✅ 柱状图 + 环形图 + 趋势图 | ✅ |
| 报表导出 | ✅ CSV / Excel / Markdown / PNG | ✅ Markdown + 下载 |

---

## 五、与其他开源课堂行为项目的差异点

参考调研（SCB-Dataset 系列、EduYOLO、各类 YOLOv8/v11 + PyQt5 课堂系统）显示：
公开中文项目绝大多数只做「逐帧检测 + 画框统计」，普遍**没有**目标跟踪、**没有**逐学生时间预算、
**没有**报表导出。本工程把这三项作为差异化功能实现：

1. ByteTrack / 轻量 IOU 跟踪 → 每个学生一个稳定 ID，累计各行为时长；
2. 逐学生行为画像（主要行为、异常时长）→ 可直接导出 Excel；
3. CSV / Excel / Markdown / 图表一键导出 + 预警 jsonl 留档。

---

## 六、进度对照

| 阶段 | 时间 | 本仓库对应产出 |
| --- | --- | --- |
| 学习 YOLOv11 / CNN / 深度学习 | 2026.04–06 | `docs/01_模型搭建与训练说明.md`、`configs/` |
| 数据采集标注 + 系统建设 + 训练部署 | 2026.07–12 | `scripts/collect_helper.py`、`scripts/prepare_dataset.py`、`scripts/train.py`、`app/` |
| 模型验证与算法调试 | 2027.01–02 | `scripts/val.py`、`scripts/dataset_stats.py`、`docs/04` |
| 结题报告 | 2027.02–03 | `reports/`（自动生成的统计数据、图表、评估报告） |

详见 `docs/05_项目进度与里程碑.md`。

---

## 七、本机已跑通的验证记录（RTX 5070 Laptop 8GB）

| 验证项 | 命令 | 结果 |
| --- | --- | --- |
| 环境自检 | `python -c "import torch,ultralytics"` | torch 2.7.1+cu128，CUDA 可用，ultralytics 8.4.178 |
| 代码编译 | `python -m compileall app src scripts` | 退出码 0，无语法错误 |
| 数据转换 | `prepare_dataset.py --voc-dir <你本地 VOC 数据目录> --max-samples 1200` | 578 张图 / 1498 个框，train:val:test = 386:125:67，生成 `data/demo_dataset.yaml` |
| 训练 | `train.py --model yolo11n.pt --epochs 5 --imgsz 416 --batch 16` | 5 轮完成；显存占用 1.29GB；mAP50 0.52（仅 5 轮 + 单类冒烟，非正式精度） |
| 验证 | `val.py --weights runs/best.pt` | 生成 `reports/val_report_val.md` |
| 系统链路自检 | `smoke_test.py --weights runs/best.pt` | 7 项全 PASS：推理 → 绘制 → 跟踪 → 行为分析 → 预警 → CSV/Markdown/图表导出 |
| 桌面端模块 | `import app.desktop_app` | 导入成功（窗口类与推理线程均可实例化） |

> 说明：以上用的是课程自带的 PASCAL VOC2007（person 类）做的**流程冒烟验证**，
> 目的是证明「数据 → 训练 → 系统」整条链路在本机可用；正式精度必须用课堂行为数据集
> （8 类）训练后重新评估。演示用配置见 `configs/classes_demo_person.yaml`。

### 拿到课堂行为数据后的三步

```powershell
# 1) 转换 + 划分 + 统计报告（自动识别 VOC / YOLO 两种格式）
scripts\prepare_dataset.py --voc-dir data\raw\classroom_voc --oversample-min 300

# 2) 正式训练（configs\train.yaml：yolo11s / 150 轮 / batch 8 / 640）
scripts\train.py

# 3) 评估并接入系统
scripts\val.py --weights runs\train\classroom_yolo11s\weights\best.pt --split test
# 把该权重路径填进 configs\app.yaml 的 model.weights，双击 run_app.bat 即可演示
```
