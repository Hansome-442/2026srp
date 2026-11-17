新样本交这里
==============

组员补拍的图片，按下面格式打包发到群里，由负责人放到这个目录下合并：

  <批次名>/
    images/    xxx.jpg
    labels/    xxx.txt      ← 和图片同名，YOLO 格式

批次名建议：1125_小王_趴桌加拍
            1203_小李_手机类

合并命令（负责人执行）：
  python scripts/merge_new_data.py --src data/incoming/<批次名> --dry-run
  python scripts/merge_new_data.py --src data/incoming/<批次名>

合并后：
  python scripts/dataset_fingerprint.py     ← 版本号 +1
  python scripts/pack_dataset.py            ← 重新打包传网盘

注意：
  1. 新数据只合并进 train，val/test 保持不变（这样历史指标仍然可比）
  2. 标签必须和图片同名，且每行 5 个数（类别 x y w h）
  3. 重名文件会被跳过，不会覆盖已有数据
