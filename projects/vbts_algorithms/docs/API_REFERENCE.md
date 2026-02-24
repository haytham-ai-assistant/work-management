# VBTS 算法库 API 参考

## 概述

视触传感器(Vision-Based Tactile Sensor)算法库，提供标记点检测和力估计的核心算法。该库设计为模块化、可扩展，支持合成数据生成、算法验证和性能评估。

## 目录结构

```
vbts_algorithms/
├── src/algorithms/
│   ├── marker_detection.py    # 标记点检测算法
│   └── force_estimation.py    # 力估计算法
├── examples/
│   └── vbts_pipeline_demo.py  # 完整流程演示
├── tests/                     # 单元测试
├── docs/                      # 文档
└── README.md                  # 项目说明
```

## 安装与依赖

### 核心依赖
- Python 3.8+
- NumPy (必需)
- Matplotlib (可选，用于可视化)
- SciPy (可选，用于高级功能)

### 环境设置
```bash
# 克隆项目后，添加src目录到Python路径
import sys
sys.path.insert(0, '/path/to/vbts_algorithms/src')
```

## 标记点检测算法 (MarkerDetection)

### 类定义
```python
from algorithms.marker_detection import MarkerDetection

detector = MarkerDetection(marker_radius=5.0, grid_spacing=20.0)
```

### 构造函数参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `marker_radius` | float | 5.0 | 标记点半径（像素） |
| `grid_spacing` | float | 20.0 | 标记点网格间距（像素） |

### 主要方法

#### `generate_synthetic_markers(image_shape, grid_offset)`
生成合成标记点网格。

**参数:**
- `image_shape`: (height, width) 图像尺寸
- `grid_offset`: (x_offset, y_offset) 网格偏移量

**返回:** `(N, 2)` numpy数组，标记点中心坐标 [x, y]

#### `simulate_deformation(markers, force_center, force_magnitude, deformation_radius)`
模拟外力作用下的标记点位移。

**参数:**
- `markers`: 原始标记点坐标 `(N, 2)`
- `force_center`: 力作用中心 `(x, y)`
- `force_magnitude`: 力大小（影响位移幅度）
- `deformation_radius`: 变形影响半径

**返回:** 变形后的标记点坐标 `(N, 2)`

#### `calculate_displacement_field(original_markers, deformed_markers)`
计算位移场。

**参数:**
- `original_markers`: 原始标记点坐标
- `deformed_markers`: 变形后标记点坐标

**返回:** 位移向量场 `(N, 2)` [dx, dy]

#### `generate_synthetic_image(markers, image_shape, marker_intensity, background_intensity, marker_std)`
生成包含标记点的合成图像。

**参数:**
- `markers`: 标记点坐标（如为None则自动生成）
- `image_shape`: 图像尺寸 `(height, width)`
- `marker_intensity`: 标记点亮度 (0-255)
- `background_intensity`: 背景亮度 (0-255)
- `marker_std`: 标记点高斯模糊标准差

**返回:** 合成图像 `(height, width)` uint8数组

#### `detect_markers_from_image(image, intensity_threshold, min_distance)`
从图像中检测标记点（简化版本，无需OpenCV）。

**参数:**
- `image`: 输入图像 (2D numpy数组)
- `intensity_threshold`: 强度阈值 (0-255)
- `min_distance`: 标记点之间的最小距离（像素）

**返回:** 检测到的标记点坐标 `(N, 2)`

#### `visualize_markers(original_markers, deformed_markers, displacement_field, save_path)`
可视化标记点和位移场。

**参数:**
- `original_markers`: 原始标记点
- `deformed_markers`: 变形后标记点（可选）
- `displacement_field`: 位移场（可选）
- `save_path`: 保存图像路径（可选）

**注意:** 需要Matplotlib，如不可用则提供文本输出。

### 使用示例
```python
# 创建检测器
detector = MarkerDetection(marker_radius=6.0, grid_spacing=25.0)

# 生成合成标记点
markers = detector.generate_synthetic_markers(image_shape=(480, 640))

# 模拟变形
deformed = detector.simulate_deformation(
    markers, 
    force_center=(320, 240),
    force_magnitude=15.0
)

# 计算位移场
displacement = detector.calculate_displacement_field(markers, deformed)

# 可视化
detector.visualize_markers(markers, deformed, displacement)
```

## 力估计算法 (ForceEstimation)

### 类定义
```python
from algorithms.force_estimation import ForceEstimation

estimator = ForceEstimation(
    youngs_modulus=2.0e6,   # 杨氏模量 (Pa)
    poissons_ratio=0.49,    # 泊松比
    sensor_thickness=5.0    # 传感器厚度 (mm)
)
```

### 构造函数参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `youngs_modulus` | float | 1.0e6 | 杨氏模量 (Pa) |
| `poissons_ratio` | float | 0.49 | 泊松比 |
| `sensor_thickness` | float | 5.0 | 传感器厚度 (mm) |

### 主要方法

#### `estimate_force_from_displacement(displacement_field, positions, method)`
从位移场估计力分布。

**参数:**
- `displacement_field`: 位移场 `(N, 2)`
- `positions`: 测量点位置 `(N, 2)`
- `method`: 力估计方法 (`'hertz'`, `'boussinesq'`, `'fem'`)

**返回:** 包含力估计结果的字典

**结果字典结构:**
```python
{
    'method': str,           # 使用的方法
    'total_force': float,    # 总接触力 (N)
    'force_distribution': np.ndarray,  # 力分布
    'max_force': float,      # 最大局部力 (N)
    'force_center': np.ndarray  # 力中心位置
}
```

#### `calibrate_from_data(displacement_fields, known_forces, initial_params, method)`
从已知力-位移数据校准材料参数。

**参数:**
- `displacement_fields`: 位移场列表
- `known_forces`: 已知力列表
- `initial_params`: 初始参数字典
- `method`: 使用的力估计方法

**返回:** 包含校准参数和评估指标的字典

#### `validate_estimation(displacement_fields, known_forces, method)`
验证力估计准确性。

**参数:**
- `displacement_fields`: 位移场列表
- `known_forces`: 已知力列表
- `method`: 力估计方法

**返回:** 包含验证指标的字典

#### `cross_validate_methods(displacement_fields, known_forces, methods)`
交叉验证不同力估计方法。

**参数:**
- `displacement_fields`: 位移场列表
- `known_forces`: 已知力列表
- `methods`: 要比较的方法列表

**返回:** 包含各方法性能比较的字典

#### `visualize_force_distribution(force_result, positions, save_path)`
可视化力分布。

**参数:**
- `force_result`: `estimate_force_from_displacement` 返回的结果
- `positions`: 测量点位置（可选）
- `save_path`: 保存图像路径（可选）

**注意:** 需要Matplotlib，如不可用则提供文本输出。

### 力估计方法说明

#### 1. Hertz接触理论
- **原理**: 弹性球体与平面的接触模型
- **适用场景**: 球形接触器，小变形
- **参数**: 接触半径、球体半径
- **输出**: 总接触力和压力分布

#### 2. Boussinesq解
- **原理**: 弹性半空间表面受集中力作用的位移场反问题
- **适用场景**: 任意接触形状，均匀材料
- **输出**: 表面力分布网格

#### 3. 有限元反问题 (FEM)
- **原理**: 从位移场反推节点力的有限元方法
- **适用场景**: 复杂几何形状，非均匀材料
- **输出**: 节点力向量

### 使用示例
```python
# 创建力估计器
estimator = ForceEstimation(youngs_modulus=2.0e6)

# 生成示例位移场
displacement_field = np.random.randn(100, 2) * 0.1
positions = np.random.rand(100, 2) * 100

# 估计力
result = estimator.estimate_force_from_displacement(
    displacement_field, 
    positions, 
    method='hertz'
)

print(f"总接触力: {result['total_force']:.3f} N")

# 校准参数
calibration = estimator.calibrate_from_data(
    displacement_fields=[displacement_field],
    known_forces=[5.0],  # 已知为5N
    method='hertz'
)

# 可视化
estimator.visualize_force_distribution(result, positions)
```

## 实验验证框架

### 标记点检测验证
```bash
python scripts/analysis/marker_detection_validation.py
```

### 力估计验证
```bash
python scripts/analysis/force_estimation_validation.py
```

### 验证输出
每个验证脚本生成:
1. **JSON结果文件**: 详细测试结果
2. **CSV文件**: 便于数据分析
3. **Markdown报告**: 实验总结和分析
4. **可视化图表** (如Matplotlib可用)

## 单元测试

运行所有单元测试:
```bash
python -m unittest discover tests/
```

运行特定测试:
```bash
python -m unittest tests.test_marker_detection
python -m unittest tests.test_force_estimation
```

## 故障排除

### 常见问题

#### 1. 导入错误
**症状**: `ModuleNotFoundError: No module named 'algorithms'`
**解决方案**: 确保已添加src目录到Python路径
```python
import sys
sys.path.insert(0, '/path/to/vbts_algorithms/src')
```

#### 2. Matplotlib不可用
**症状**: 可视化函数失败
**解决方案**: 安装matplotlib或使用文本输出模式
```bash
pip install matplotlib
```

#### 3. SciPy不可用
**症状**: 有限元方法使用简化网格
**解决方案**: 安装scipy以获得更准确的网格生成
```bash
pip install scipy
```

#### 4. 力估计误差大
**症状**: 估计力与真实力差异巨大
**解决方案**: 
- 校准材料参数
- 检查位移场单位（应为米，不是像素）
- 调整力估计算法参数

### 性能优化建议
1. **位移场预处理**: 去除噪声，平滑数据
2. **参数校准**: 针对特定传感器材料进行校准
3. **方法选择**: 根据应用场景选择最合适的方法
4. **并行处理**: 对大量数据使用并行计算

## 扩展开发

### 添加新的力估计方法
1. 在`ForceEstimation`类中添加新方法
2. 在`estimate_force_from_displacement`中集成新方法
3. 更新验证脚本以包含新方法

### 集成真实传感器数据
1. 实现图像读取接口
2. 添加数据预处理模块
3. 创建传感器特定的校准程序

### 性能优化
1. 使用Numba加速数值计算
2. 实现GPU加速版本
3. 添加缓存机制

## 许可证与引用

本项目基于MIT许可证开源。如使用本代码于学术研究，请引用相关论文。

## 支持与贡献

- **问题报告**: 通过GitHub Issues提交
- **功能请求**: 通过GitHub Discussions提出
- **代码贡献**: 提交Pull Request

---

*文档版本: 1.0*  
*最后更新: 2026年2月24日*  
*维护者: VBTS算法开发团队*