"""课堂行为检测系统核心包。

模块划分：
    utils       通用工具（配置、中文路径读写、日志、绘图字体）
    detector    YOLOv11 检测/跟踪封装
    tracker     轻量 IOU 跟踪器（不依赖 ultralytics 跟踪器时使用）
    analytics   行为分析与统计（时长、抬头率、参与度）
    alerts      异常预警（规则、冷却、日志）
    report      报表与图表导出
"""

__version__ = "1.0.0"
