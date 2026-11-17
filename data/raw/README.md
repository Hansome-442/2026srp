# 原始数据目录

> **请先看 [00_数据放置说明.md](00_数据放置说明.md)** —— 里面有"放哪个目录、放什么格式、放完跑哪条命令"。
> 标注分工见 [01_标注分工_7人.md](01_标注分工_7人.md)，判定规则见 [02_八类定义与标注规范.md](02_八类定义与标注规范.md)。

把课堂行为的原始数据放在这里，支持两种格式：

## 1) VOC 格式（推荐）

```
data/raw/classroom_voc/
├─ Annotations/   000001.xml ...
└─ JPEGImages/    000001.jpg ...
```

也可以按「场景_学段」分子目录（统计报告会自动汇总覆盖度）：

```
data/raw/classroom_voc/
├─ 教室后排_大学/Annotations/*.xml
├─ 教室后排_大学/JPEGImages/*.jpg
├─ 小组讨论_初中/...
```

## 2) YOLO 格式（labelImg 直接导出 txt）

```
data/raw/self_collected/
├─ images/   000001.jpg ...
└─ labels/   000001.txt ...   # 每行：class_id cx cy w h（归一化）
```

## 转换为训练集

```powershell
& E:\anaconda3\envs\pytorch\python.exe scripts\prepare_dataset.py --voc-dir data\raw\classroom_voc --yolo-dir data\raw\self_collected --oversample-min 300
```

类别名请使用 `configs/classes.yaml` 中的中文类别名（抬头听讲 / 阅读看书 / 书写记笔记 /
举手发言 / 站立回答 / 小组讨论 / 趴桌睡觉 / 玩手机），别名表已覆盖常见简称。

详细的采集与标注规范见 `docs/02_数据集规范与8类定义.md`、`docs/03_采集与标注分工_7人.md`。
