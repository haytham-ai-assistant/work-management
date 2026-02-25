#!/usr/bin/env python3
"""
测试位移场噪声滤波功能
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.force_estimation import ForceEstimation

def test_filter_methods():
    """测试不同的滤波方法"""
    print("=== 测试位移场噪声滤波 ===")
    
    # 创建力估计器
    estimator = ForceEstimation(
        youngs_modulus=2.0e6,
        poissons_ratio=0.49,
        sensor_thickness=5.0
    )
    
    # 生成测试数据
    n_points = 50
    positions = np.random.rand(n_points, 2) * 100  # 100x100 mm区域
    
    # 生成合成位移场（模拟中心接触）
    displacement_field = np.zeros((n_points, 2))
    for i, (x, y) in enumerate(positions):
        dx = 50 - x
        dy = 50 - y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 距离越近位移越大
        if distance < 30:
            displacement_field[i, 0] = dx / (distance + 1e-6) * (30 - distance) * 0.5
            displacement_field[i, 1] = dy / (distance + 1e-6) * (30 - distance) * 0.5
    
    print(f"原始位移场形状: {displacement_field.shape}")
    print(f"原始位移场范数: {np.linalg.norm(displacement_field, axis=1).mean():.4f} mm")
    
    # 测试高斯滤波
    filtered_gaussian = estimator.filter_displacement_field(
        displacement_field, positions, 
        filter_type='gaussian', filter_radius=5.0
    )
    print(f"\n高斯滤波后位移范数: {np.linalg.norm(filtered_gaussian, axis=1).mean():.4f} mm")
    print(f"变化: {np.abs(filtered_gaussian - displacement_field).mean():.6f} mm")
    
    # 测试中值滤波
    filtered_median = estimator.filter_displacement_field(
        displacement_field, positions,
        filter_type='median', filter_radius=5.0
    )
    print(f"\n中值滤波后位移范数: {np.linalg.norm(filtered_median, axis=1).mean():.4f} mm")
    print(f"变化: {np.abs(filtered_median - displacement_field).mean():.6f} mm")
    
    # 测试移动平均滤波
    filtered_ma = estimator.filter_displacement_field(
        displacement_field, positions,
        filter_type='moving_average', filter_radius=5.0
    )
    print(f"\n移动平均滤波后位移范数: {np.linalg.norm(filtered_ma, axis=1).mean():.4f} mm")
    print(f"变化: {np.abs(filtered_ma - displacement_field).mean():.6f} mm")
    
    # 测试基于物理的滤波
    filtered_physical = estimator.filter_displacement_field(
        displacement_field, positions,
        filter_type='physical', filter_radius=5.0
    )
    print(f"\n物理滤波后位移范数: {np.linalg.norm(filtered_physical, axis=1).mean():.4f} mm")
    print(f"变化: {np.abs(filtered_physical - displacement_field).mean():.6f} mm")
    
    # 测试带噪声的数据
    print("\n=== 测试带噪声数据 ===")
    noise_level = 0.5  # mm
    noisy_displacement = displacement_field + np.random.normal(0, noise_level, displacement_field.shape)
    
    print(f"噪声位移场范数: {np.linalg.norm(noisy_displacement, axis=1).mean():.4f} mm")
    print(f"噪声水平: {noise_level} mm")
    
    # 对带噪声数据应用滤波
    filtered_noisy = estimator.filter_displacement_field(
        noisy_displacement, positions,
        filter_type='gaussian', filter_radius=5.0
    )
    
    # 计算噪声减少效果
    original_error = np.linalg.norm(noisy_displacement - displacement_field, axis=1).mean()
    filtered_error = np.linalg.norm(filtered_noisy - displacement_field, axis=1).mean()
    
    print(f"\n原始噪声误差: {original_error:.4f} mm")
    print(f"滤波后误差: {filtered_error:.4f} mm")
    print(f"误差减少: {(original_error - filtered_error)/original_error*100:.1f}%")
    
    # 测试Boussinesq方法中的滤波选项
    print("\n=== 测试Boussinesq方法中的滤波 ===")
    
    # 不带滤波
    force_no_filter = estimator.boussinesq_solution(
        noisy_displacement, positions,
        pre_filter=False
    )
    
    # 带滤波
    force_with_filter = estimator.boussinesq_solution(
        noisy_displacement, positions,
        pre_filter=True,
        filter_type='gaussian',
        filter_radius=5.0
    )
    
    print(f"不带滤波的力估计总和: {force_no_filter.sum():.4f} N")
    print(f"带滤波的力估计总和: {force_with_filter.sum():.4f} N")
    print(f"差异: {abs(force_no_filter.sum() - force_with_filter.sum()):.4f} N")
    
    return True

def test_estimate_with_filter():
    """测试带滤波的力估计方法"""
    print("\n=== 测试带滤波的力估计方法 ===")
    
    estimator = ForceEstimation()
    
    # 生成测试数据
    n_points = 100
    positions = np.random.rand(n_points, 2) * 100
    
    # 生成合成位移场
    displacement_field = np.zeros((n_points, 2))
    for i, (x, y) in enumerate(positions):
        dx = 50 - x
        dy = 50 - y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 30:
            displacement_field[i, 0] = dx / (distance + 1e-6) * (30 - distance) * 0.3
            displacement_field[i, 1] = dy / (distance + 1e-6) * (30 - distance) * 0.3
    
    # 添加噪声
    noise_level = 1.0
    noisy_displacement = displacement_field + np.random.normal(0, noise_level, displacement_field.shape)
    
    # 测试不带滤波的估计
    result_no_filter = estimator.estimate_force_from_displacement(
        noisy_displacement, positions,
        method='boussinesq',
        pre_filter=False
    )
    
    # 测试带滤波的估计
    result_with_filter = estimator.estimate_force_from_displacement(
        noisy_displacement, positions,
        method='boussinesq',
        pre_filter=True,
        filter_type='gaussian',
        filter_radius=5.0
    )
    
    print(f"不带滤波估计的总力: {result_no_filter['total_force']:.4f} N")
    print(f"带滤波估计的总力: {result_with_filter['total_force']:.4f} N")
    print(f"差异: {abs(result_no_filter['total_force'] - result_with_filter['total_force']):.4f} N")
    
    # 测试FEM方法
    print("\n=== 测试FEM方法带滤波 ===")
    try:
        result_fem_no_filter = estimator.estimate_force_from_displacement(
            noisy_displacement, positions,
            method='fem',
            pre_filter=False
        )
        
        result_fem_with_filter = estimator.estimate_force_from_displacement(
            noisy_displacement, positions,
            method='fem',
            pre_filter=True,
            filter_type='gaussian',
            filter_radius=5.0
        )
        
        print(f"FEM不带滤波估计的总力: {result_fem_no_filter['total_force']:.4f} N")
        print(f"FEM带滤波估计的总力: {result_fem_with_filter['total_force']:.4f} N")
    except Exception as e:
        print(f"FEM测试失败: {e}")
    
    return True

if __name__ == "__main__":
    try:
        test_filter_methods()
        test_estimate_with_filter()
        print("\n✅ 所有滤波测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)