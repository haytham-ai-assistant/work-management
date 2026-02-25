# 真实传感器数据格式规范

## 概述

本文档定义了用于视触传感器(VBTS)力估计算法验证的真实传感器数据格式。该格式旨在支持从实际传感器系统收集的数据，包括图像、位移场、力测量和元数据。

## 数据目录结构

```
data/
├── raw/                          # 原始数据
│   ├── experiment_001/           # 实验1
│   │   ├── metadata.json         # 实验元数据
│   │   ├── calibration/          # 校准数据
│   │   │   ├── marker_pattern.png
│   │   │   └── calibration_params.json
│   │   ├── images/               # 原始图像序列
│   │   │   ├── frame_0000.png
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   ├── forces/               # 力测量数据
│   │   │   ├── force_measurements.csv
│   │   │   └── force_log.json
│   │   └── timestamps.csv        # 时间戳同步
│   └── experiment_002/
│       └── ...
├── processed/                    # 处理后的数据
│   ├── experiment_001/
│   │   ├── displacement_fields/  # 计算的位移场
│   │   │   ├── disp_0000.npy
│   │   │   ├── disp_0001.npy
│   │   │   └── ...
│   │   ├── marker_positions/     # 标记点位置
│   │   │   ├── markers_0000.npy
│   │   │   └── ...
│   │   ├── force_estimates/      # 力估计结果
│   │   │   └── estimates.csv
│   │   └── validation_results/   # 验证结果
│   │       └── validation_report.md
│   └── ...
└── datasets/                     # 整理好的数据集
    ├── contact_force_dataset_v1.0/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── ...
```

## 文件格式规范

### 1. 实验元数据 (metadata.json)

```json
{
  "experiment_id": "exp_001",
  "date": "2026-02-25",
  "sensor_type": "VBTS_Tac3D",
  "sensor_parameters": {
    "sensor_size_mm": [100, 100],
    "marker_radius_mm": 6.0,
    "marker_spacing_mm": 25.0,
    "pixel_to_mm": 0.1,
    "sensor_thickness_mm": 5.0,
    "material_youngs_modulus_pa": 2000000.0,
    "material_poissons_ratio": 0.49
  },
  "experiment_setup": {
    "force_sensor_type": "ATI_Nano17",
    "force_sensor_range_n": [0, 50],
    "indenter_type": "spherical",
    "indenter_radius_mm": 50.0,
    "loading_mechanism": "linear_actuator",
    "max_force_n": 40.0,
    "loading_rate_n_s": 5.0
  },
  "data_acquisition": {
    "camera_model": "FLIR_BFS-U3",
    "camera_resolution": [640, 480],
    "frame_rate_fps": 30,
    "exposure_time_ms": 10,
    "lighting_condition": "diffuse_led",
    "synchronization_method": "hardware_trigger"
  },
  "test_conditions": [
    {
      "condition_id": "cond_001",
      "description": "5N vertical load, center contact",
      "target_force_n": 5.0,
      "contact_position_mm": [50, 50],
      "duration_s": 2.0,
      "repetitions": 3
    },
    {
      "condition_id": "cond_002",
      "description": "10N vertical load, center contact",
      "target_force_n": 10.0,
      "contact_position_mm": [50, 50],
      "duration_s": 2.0,
      "repetitions": 3
    }
  ]
}
```

### 2. 图像数据

- **格式**: PNG (无损压缩)
- **命名**: `frame_XXXX.png` (4位数字序号)
- **分辨率**: 与metadata中camera_resolution一致
- **颜色空间**: 灰度或RGB，建议灰度以减少数据量
- **建议**: 包含未加载状态的基础帧 (`frame_0000.png` 作为参考帧)

### 3. 力测量数据 (force_measurements.csv)

```csv
frame_id,timestamp_s,force_x_n,force_y_n,force_z_n,torque_x_nm,torque_y_nm,torque_z_nm,condition_id
0,0.000,0.01,0.02,-0.03,0.001,0.002,0.003,cond_001
1,0.033,0.12,0.15,-1.25,0.012,0.015,0.125,cond_001
2,0.067,0.25,0.31,-2.78,0.025,0.031,0.278,cond_001
...
```

### 4. 时间戳文件 (timestamps.csv)

```csv
frame_id,image_timestamp_s,force_timestamp_s,sync_offset_s
0,0.000,0.001,0.001
1,0.033,0.034,0.001
2,0.067,0.068,0.001
...
```

### 5. 处理后的位移场数据 (.npy格式)

- **文件**: `disp_XXXX.npy`
- **格式**: NumPy数组，形状为 `(N, 2)`，其中N为标记点数量
- **单位**: 毫米 (mm)
- **内容**: 每个标记点的x和y方向位移

### 6. 标记点位置数据 (.npy格式)

- **文件**: `markers_XXXX.npy`
- **格式**: NumPy数组，形状为 `(N, 2)`
- **单位**: 毫米 (mm)
- **内容**: 每个标记点的x和y坐标

## 数据收集流程

### 阶段1: 传感器校准
1. 采集未加载状态的基础图像
2. 检测标记点模式
3. 计算像素到毫米转换系数
4. 记录传感器参数

### 阶段2: 实验数据采集
1. 设置实验条件 (力大小、位置、速度)
2. 同步启动图像采集和力测量
3. 按预定序列施加载荷
4. 记录所有传感器数据和时间戳
5. 重复多次以获得统计可靠性

### 阶段3: 数据处理
1. 时间戳对齐和同步
2. 从图像序列计算位移场
3. 提取力测量真值
4. 整理为算法可用的格式

### 阶段4: 验证与分析
1. 运行力估计算法
2. 比较估计力与测量力
3. 计算性能指标 (误差、相关性等)
4. 生成验证报告

## 最小数据集要求

对于算法验证，最小数据集应包含:

1. **3种力水平**: 5N, 20N, 40N (覆盖典型使用范围)
2. **3种接触位置**: 中心、边缘、角部 (测试空间变化)
3. **每种条件3次重复**: 统计可靠性
4. **总数据量**: 3×3×3 = 27个测试序列
5. **每个序列**: 至少2秒数据 (60帧 @ 30fps)

## 数据质量检查

采集后应检查:
1. 图像质量: 对比度、清晰度、照明均匀性
2. 力数据: 噪声水平、漂移、范围适当性
3. 同步精度: 时间戳对齐误差 < 1ms
4. 元数据完整性: 所有参数记录完整

## 工具与脚本

本仓库提供以下工具辅助数据收集:

1. `data_collection/collect_data.py` - 数据采集脚本模板
2. `data_collection/process_raw_data.py` - 原始数据处理
3. `data_collection/validate_dataset.py` - 数据集验证
4. `data_collection/create_synthetic_realistic.py` - 生成逼真合成数据

## 下一步

1. 根据本规范设置传感器系统
2. 运行校准程序
3. 采集最小数据集
4. 使用提供的工具处理数据
5. 运行端到端算法验证

## 参考文献

1. EASES Tac3D传感器数据手册
2. ATI Nano17力传感器用户指南
3. OpenCV相机校准教程
4. 相关学术论文中的实验方法部分