#!/usr/bin/env python3
"""
系统验证脚本 - 验证视触传感器学习系统的核心功能
"""

import sys
import os
import numpy as np

print("=" * 70)
print("视触传感器学习系统 - 核心功能验证")
print("=" * 70)

# 1. 检查目录结构
print("\n1. 检查目录结构...")
required_dirs = [
    "/workspace/工作/data/视触技术要点",
    "/workspace/工作/projects/vbts_algorithms",
    "/workspace/工作/projects/experimental_validation",
    "/workspace/工作/market_report"
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} (缺失)")

# 2. 检查核心算法文件
print("\n2. 检查核心算法文件...")
required_files = [
    "/workspace/工作/projects/vbts_algorithms/src/algorithms/marker_detection.py",
    "/workspace/工作/projects/vbts_algorithms/src/algorithms/force_estimation.py",
    "/workspace/工作/projects/vbts_algorithms/tests/test_marker_detection.py",
    "/workspace/工作/projects/vbts_algorithms/tests/test_force_estimation.py"
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"   ✓ {os.path.basename(file_path)}")
    else:
        print(f"   ✗ {os.path.basename(file_path)} (缺失)")

# 3. 检查市场报告
print("\n3. 检查市场报告文件...")
market_files = [
    "/workspace/工作/market_report/视触传感器市场与技术分析报告.md",
    "/workspace/工作/market_report/视触传感器市场与技术分析报告.docx",
    "/workspace/工作/market_report/视触传感器市场与技术分析报告.pptx"
]

for file_path in market_files:
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        print(f"   ✓ {os.path.basename(file_path)} ({size_kb:.1f} KB)")
    else:
        print(f"   ✗ {os.path.basename(file_path)} (缺失)")

# 4. 测试算法导入
print("\n4. 测试算法导入...")
sys.path.insert(0, "/workspace/工作/projects/vbts_algorithms/src")

try:
    from algorithms.marker_detection import MarkerDetection
    from algorithms.force_estimation import ForceEstimation
    print("   ✓ 成功导入算法模块")
    
    # 创建算法实例
    md = MarkerDetection()
    fe = ForceEstimation()
    print("   ✓ 成功创建算法实例")
    
    # 生成测试数据
    print("\n5. 生成测试数据...")
    positions = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
    displacement_field = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    
    # 测试力估计
    print("6. 测试力估计方法...")
    try:
        # Hertz方法 - 需要单个位移向量，使用最大位移点
        max_disp_idx = np.argmax(np.linalg.norm(displacement_field, axis=1))
        single_displacement = displacement_field[max_disp_idx]
        # 添加z轴位移（假设为合成数据）
        if len(single_displacement) == 2:
            single_displacement = np.array([single_displacement[0], single_displacement[1], 1.0])
        force_h, force_dist = fe.hertz_contact_force(single_displacement)
        print(f"   ✓ Hertz方法: {force_h:.3f} N (基于最大位移点)")
    except Exception as e:
        print(f"   ✗ Hertz方法失败: {e}")
    
    try:
        # Boussinesq方法 - 返回力分布网格
        force_grid = fe.boussinesq_solution(displacement_field, positions)
        total_force_b = np.sum(force_grid) * (20.0**2) / 1000.0  # 网格分辨率20mm，转换为N
        print(f"   ✓ Boussinesq方法: 力分布网格 {force_grid.shape}, 总力估计 {total_force_b:.3f} N")
    except Exception as e:
        print(f"   ✗ Boussinesq方法失败: {e}")
    
    try:
        # 自适应方法选择 - 返回字典
        result = fe.adaptive_force_estimation(displacement_field, positions)
        force_a = result.get('total_force', 0)
        method_used = result.get('method_selected', 'unknown')
        print(f"   ✓ 自适应方法选择: {force_a:.3f} N (使用{method_used})")
    except Exception as e:
        print(f"   ✗ 自适应方法失败: {e}")
    
    print("\n7. 测试位移场滤波...")
    try:
        filtered_field = fe.filter_displacement_field(
            displacement_field, positions, filter_type='gaussian'
        )
        print(f"   ✓ 位移场滤波成功")
    except Exception as e:
        print(f"   ✗ 位移场滤波失败: {e}")
        
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
except Exception as e:
    print(f"   ✗ 测试过程中出错: {e}")

# 8. 检查Git状态
print("\n8. 检查Git状态...")
try:
    import subprocess
    result = subprocess.run(
        ["git", "-C", "/workspace/工作", "status", "--short"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        if result.stdout.strip() == "":
            print("   ✓ Git工作目录干净")
        else:
            print("   ⚠ Git有未提交的更改")
            print(f"     输出: {result.stdout[:100]}...")
    else:
        print(f"   ⚠ Git命令失败: {result.stderr[:100]}")
except Exception as e:
    print(f"   ⚠ 检查Git状态失败: {e}")

# 9. 检查项目打包文件
print("\n9. 检查项目打包文件...")
zip_path = "/workspace/vbts_project.zip"
if os.path.exists(zip_path):
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   ✓ 项目打包文件: {size_mb:.1f} MB")
else:
    print(f"   ✗ 项目打包文件缺失")

print("\n" + "=" * 70)
print("验证完成!")
print("=" * 70)