# 视触传感器实验验证项目框架

基于视触技术要点学习路线图建立的实验验证框架，用于系统化验证VBTS（Vision-Based Tactile Sensor）算法和硬件设计。

## 项目结构

```
experimental_validation/
├── experiments/           # 实验设计文件
│   ├── 01_marker_detection/      # 标记点检测实验
│   ├── 02_displacement_calibration/ # 位移标定实验
│   ├── 03_force_estimation/      # 力估计实验
│   └── templates/                # 实验模板
├── data/                  # 实验数据
│   ├── raw/              # 原始数据（图像、视频、传感器读数）
│   └── processed/        # 处理后的数据
├── results/              # 实验结果
│   ├── figures/          # 图表和可视化
│   └── tables/           # 数据表格
├── scripts/              # 分析脚本
│   ├── data_processing/  # 数据预处理脚本
│   ├── analysis/         # 分析脚本
│   └── visualization/    # 可视化脚本
├── notebooks/            # Jupyter notebooks（探索性分析）
└── docs/                 # 文档和报告
    ├── protocols/        # 实验协议
    └── reports/          # 实验报告
```

## 实验路线图

### 阶段1：标记点检测验证
1. **合成数据验证** - 使用`vbts_algorithms`项目中的合成数据测试标记点检测算法
2. **实际图像测试** - 使用真实视触传感器图像测试检测精度
3. **性能基准测试** - 比较不同检测算法（Hough圆、特征检测、深度学习）

### 阶段2：位移场标定验证
1. **已知位移生成** - 使用精密位移台产生已知位移，验证算法精度
2. **三维重建验证** - 验证光度立体法、立体视觉等三维重建方法
3. **误差分析** - 分析位移测量误差来源（相机标定、镜头畸变、光照变化）

### 阶段3：力估计验证
1. **标准力传感器比对** - 使用标准力传感器（如ATI Mini40）验证力估计精度
2. **材料参数校准** - 校准硅胶材料的杨氏模量、泊松比等参数
3. **动态力测试** - 验证动态力跟踪能力

### 阶段4：系统集成验证
1. **实时性测试** - 验证算法实时处理能力（帧率、延迟）
2. **鲁棒性测试** - 在不同光照、温度、湿度条件下的性能
3. **长期稳定性测试** - 长期运行稳定性验证

## 实验模板

每个实验目录应包含：
- `experiment_design.md` - 实验设计（目的、假设、变量、步骤）
- `protocol.md` - 详细实验协议
- `data_collection.py` - 数据采集脚本（如果适用）
- `analysis.ipynb` - 数据分析notebook
- `results_summary.md` - 实验结果总结

## 数据管理规范

### 原始数据
- 使用有意义的文件名：`YYYYMMDD_experiment_sample_condition.ext`
- 包含元数据文件：`YYYYMMDD_experiment_metadata.json`
- 原始数据不可修改，所有处理在副本上进行

### 处理后的数据
- 保存为开放格式（`.npz`, `.h5`, `.csv`）
- 包含处理参数记录
- 版本控制处理脚本

## 分析工具

项目依赖`vbts_algorithms`项目中的算法实现。确保在分析脚本中添加路径：

```python
import sys
sys.path.append('/path/to/vbts_algorithms/src')
from algorithms import marker_detection, force_estimation
```

## 快速开始

1. **设置环境**：
   ```bash
   cd /workspace/工作/projects/experimental_validation
   # 创建Python虚拟环境（推荐）
   python -m venv venv
   source venv/bin/activate
   pip install numpy matplotlib scipy opencv-python jupyter
   ```

2. **运行示例实验**：
   ```bash
   # 运行标记点检测验证实验
   python scripts/analysis/marker_detection_validation.py
   ```

3. **查看结果**：
   - 结果图表保存在`results/figures/`
   - 数据表格保存在`results/tables/`

## 注意事项

1. **可重复性**：所有实验应确保可重复，记录所有参数和环境条件
2. **版本控制**：实验代码、数据和结果应进行版本控制
3. **安全第一**：涉及硬件实验时注意电气安全和机械安全
4. **伦理考虑**：涉及人体或动物实验时遵循相关伦理规范

## 参考资源

- [视触技术要点学习路线图](../data/视触技术要点/技术学习路线图.md)
- [视触传感器算法文档](../data/视触技术要点/concepts/视触传感器算法与处理.md)
- [VBTS算法实现](../vbts_algorithms/)

---

*最后更新：2026年2月24日*  
*项目负责人：[您的姓名]*  
*Git仓库：[https://github.com/haytham-ai-assistant/work-management](https://github.com/haytham-ai-assistant/work-management)*