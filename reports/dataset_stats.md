# 数据集统计报告

- 生成时间：2026-11-15 16:02:11
- 类别数：8
- 源目录：`data\raw\downloads\student_classroom_behavior + data\raw\downloads\yolov9_classroom + data\raw\downloads\scb3_phone_sleep\5k_HRW_yolo_Dataset_jpg\5k_HRW_yolo_Dataset_jpg + data\raw\downloads\scb3_phone_sleep\0.671k_university_yolo_Dataset\0.671k_university_yolo_Dataset + data\raw\downloads\hf_scb\extracted\SCB5-Discuss-2024-9-17 + data\raw\downloads\hf_scb\extracted\SCB5-Handrise-Read-write-2024-9-17 + data\raw\downloads\scb05\SCB-Dataset\SCB5-Talk-2024-9-17 + data\raw\downloads\scb05\SCB-Dataset\SCB5-Talk-Teacher-Behavior-2024-9-17 + data\raw\downloads\scb05\SCB-Dataset\SCB5-Stand-2024-9-17 + data\raw\downloads\scb4065`

## 一、划分概览

| 划分 | 图片数 | 标注框数 | 过采样图片 |
| --- | --- | --- | --- |
| train | 34603 | 386526 | 0 |
| val | 8351 | 46309 | 0 |
| test | 3806 | 16374 | 0 |

- 图片总数：**46760**（预期成果要求 >= 5000）
- 标注框总数：**449209**

### 去重与泄漏检查

- 感知哈希（dHash）聚类：**8132** 组，其中多图组共含 **48794** 张，最大组 1314 张
- 划分规则：**同一重复组只会进入同一个划分**（train/val/test 互斥），不存在跨划分数据泄漏

## 二、各类别实例数

| id | 类别 | 训练 | 验证 | 测试 | 合计 | 图片数 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 抬头听讲 | 149479 | 7186 | 2809 | 159474 | 9553 |
| 1 | 阅读看书 | 93299 | 13129 | 4196 | 110624 | 15205 |
| 2 | 书写记笔记 | 79969 | 9434 | 1893 | 91296 | 8074 |
| 3 | 举手发言 | 22978 | 6050 | 3195 | 32223 | 10041 |
| 4 | 站立回答 | 14566 | 4084 | 1662 | 20312 | 13163 |
| 5 | 小组讨论 | 15768 | 5828 | 2153 | 23749 | 11523 |
| 6 | 趴桌睡觉 | 2378 | 345 | 281 | 3004 | 1121 |
| 7 | 玩手机 | 8089 | 253 | 185 | 8527 | 1215 |

## 三、未识别标签（需检查标注或补充 aliases）

| 原始标签 | 出现次数 |
| --- | --- |
| IGNORE_bowhead | 37382 |
| IGNORE_book | 16985 |
| IGNORE_using_phone | 12912 |
| IGNORE_raisehead | 11615 |
| IGNORE_bend | 11487 |
| turn_head | 5339 |
| IGNORE_phone | 5070 |
| IGNORE_turnhead | 3361 |
| Eating | 2208 |
| IGNORE_sleep | 1627 |
| guide | 1432 |
| low_head | 947 |
| blackboard-writing | 934 |
| teacher | 680 |
| On-stage interaction | 578 |
| Fighting | 554 |

## 四、场景/学段覆盖（按子目录统计）

| 分组 | 图片数 |
| --- | --- |
| train | 39732 |
| val | 8709 |
| valid | 1183 |
| test | 384 |

## 五、来源数据集分布（每个划分都含全部来源）

| 来源 | train | val | test |
| --- | --- | --- | --- |
| 0.671k_university_yolo_Dataset | 671 | 0 | 0 |
| 5k_HRW_yolo_Dataset_jpg | 3501 | 1014 | 500 |
| SCB5-Discuss-2024-9-17 | 520 | 230 | 114 |
| SCB5-Handrise-Read-write-2024-9-17 | 4579 | 1507 | 778 |
| SCB5-Stand-2024-9-17 | 5344 | 1489 | 769 |
| SCB5-Talk-2024-9-17 | 3120 | 1001 | 418 |
| SCB5-Talk-Teacher-Behavior-2024-9-17 | 6061 | 2042 | 817 |
| scb4065 | 3220 | 470 | 375 |
| student_classroom_behavior | 7936 | 871 | 77 |
| yolov9_classroom | 1808 | 506 | 270 |
