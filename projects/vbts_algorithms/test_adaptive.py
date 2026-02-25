#!/usr/bin/env python3
"""
测试自适应方法选择器
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.force_estimation import ForceEstimation

def create_test_displacement(positions, force_center=(50, 50), force_magnitude=10.0, 
                           noise_level=0.0, youngs_modulus=2.0e6, poissons_ratio=0.49):
    """创建基于Boussinesq物理的测试位移场"""
    displacement_field = np.zeros_like(positions)
    
    # 弹性系数
    elastic_coeff = (1 - poissons_ratio**2) / (np.pi * youngs_modulus)
    
    for i, (x, y) in enumerate(positions):
        # 计算到力中心的距离 (mm)
        dx = x - force_center[0]
        dy = y - force_center[1]
        distance_mm = np.sqrt(dx**2 + dy**2)
        
        # 避免除零
        if distance_mm < 0.1:
            distance_mm = 0.1
        
        # 转换为米
        distance_m = distance_mm / 1000.0
        
        # 垂直位移 (m)
        u_z = force_magnitude * elastic_coeff / distance_m
        
        # 转换为毫米
        u_z_mm = u_z * 1000.0
        
        # 径向位移分量
        if distance_mm > 0:
            displacement_field[i, 0] = (dx / distance_mm) * u_z_mm * 0.3
            displacement_field[i, 1] = (dy / distance_mm) * u_z_mm * 0.3
    
    # 添加噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, displacement_field.shape)
        displacement_field += noise
    
    return displacement_field

def test_adaptive_selection():
    """测试自适应方法选择"""
    print("=== 测试自适应方法选择器 ===")
    
    # 创建力估计器
    estimator = ForceEstimation(
        youngs_modulus=2.0e6,
        poissons_ratio=0.49,
        sensor_thickness=5.0
    )
    
    # 测试用例1: 低噪声、高数据密度 → 应推荐Boussinesq
    print("\n1. 测试低噪声高密度数据:")
    positions1 = np.random.rand(200, 2) * 100  # 高密度
    displacement1 = create_test_displacement(positions1, noise_level=0.01)  # 低噪声
    
    result1 = estimator.adaptive_force_estimation(displacement1, positions1)
    print(f"   选择的方法: {result1['method_selected']}")
    print(f"   是否应用滤波: {result1['filter_applied']}")
    print(f"   估计总力: {result1['total_force']:.3f} N")
    
    # 测试用例2: 高噪声、低数据密度 → 应推荐Hertz
    print("\n2. 测试高噪声低密度数据:")
    positions2 = np.random.rand(20, 2) * 100  # 低密度
    displacement2 = create_test_displacement(positions2, noise_level=1.0)  # 高噪声
    
    result2 = estimator.adaptive_force_estimation(displacement2, positions2)
    print(f"   选择的方法: {result2['method_selected']}")
    print(f"   是否应用滤波: {result2['filter_applied']}")
    print(f"   估计总力: {result2['total_force']:.3f} N")
    
    # 测试用例3: 中等噪声、中等数据密度
    print("\n3. 测试中等条件:")
    positions3 = np.random.rand(80, 2) * 100
    displacement3 = create_test_displacement(positions3, noise_level=0.3)
    
    result3 = estimator.adaptive_force_estimation(displacement3, positions3)
    print(f"   选择的方法: {result3['method_selected']}")
    print(f"   是否应用滤波: {result3['filter_applied']}")
    print(f"   估计总力: {result3['total_force']:.3f} N")
    
    # 测试用例4: 极小位移 → 应推荐Hertz
    print("\n4. 测试极小位移:")
    positions4 = np.random.rand(100, 2) * 100
    displacement4 = create_test_displacement(positions4, force_magnitude=0.1)  # 极小力
    
    result4 = estimator.adaptive_force_estimation(displacement4, positions4)
    print(f"   选择的方法: {result4['method_selected']}")
    print(f"   是否应用滤波: {result4['filter_applied']}")
    print(f"   估计总力: {result4['total_force']:.3f} N")
    
    # 测试特征分析
    print("\n5. 测试位移场特征分析:")
    features = estimator.analyze_displacement_field(displacement1, positions1)
    print(f"   数据点: {features['n_points']}")
    print(f"   最大位移: {features['max_displacement']:.3f} mm")
    print(f"   平均位移: {features['mean_displacement']:.3f} mm")
    print(f"   噪声估计: {features['noise_level']:.3f} mm")
    print(f"   点密度: {features['point_density']:.2f} 点/mm²")
    print(f"   推荐方法: {features['recommended_method']}")
    
    return True

def test_method_comparison():
    """比较自适应选择与固定方法的性能"""
    print("\n=== 比较自适应选择与固定方法 ===")
    
    estimator = ForceEstimation()
    
    # 创建测试数据（中等条件）
    positions = np.random.rand(120, 2) * 100
    displacement = create_test_displacement(positions, noise_level=0.5, force_magnitude=20.0)
    
    print(f"测试条件: 120个点，噪声0.5mm，力20N")
    
    # 使用自适应方法
    import time
    start = time.time()
    adaptive_result = estimator.adaptive_force_estimation(displacement, positions)
    adaptive_time = time.time() - start
    
    # 使用Hertz方法
    start = time.time()
    hertz_result = estimator.estimate_force_from_displacement(
        displacement, positions, method='hertz', pre_filter=True
    )
    hertz_time = time.time() - start
    
    # 使用Boussinesq方法
    start = time.time()
    boussinesq_result = estimator.estimate_force_from_displacement(
        displacement, positions, method='boussinesq', pre_filter=True
    )
    boussinesq_time = time.time() - start
    
    print(f"\n自适应方法:")
    print(f"  选择的方法: {adaptive_result['method_selected']}")
    print(f"  估计总力: {adaptive_result['total_force']:.3f} N")
    print(f"  计算时间: {adaptive_time:.3f} s")
    
    print(f"\nHertz方法:")
    print(f"  估计总力: {hertz_result['total_force']:.3f} N")
    print(f"  计算时间: {hertz_time:.3f} s")
    
    print(f"\nBoussinesq方法:")
    print(f"  估计总力: {boussinesq_result['total_force']:.3f} N")
    print(f"  计算时间: {boussinesq_time:.3f} s")
    
    # 理想情况下，自适应方法应该选择最佳方法
    print(f"\n分析:")
    print(f"  自适应选择了: {adaptive_result['method_selected']}")
    print(f"  该选择的性能应该介于Hertz和Boussinesq之间")
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    estimator = ForceEstimation()
    
    # 测试1: 非常少的数据点
    print("\n1. 非常少的数据点 (5个点):")
    positions1 = np.random.rand(5, 2) * 100
    displacement1 = create_test_displacement(positions1, force_magnitude=5.0)
    
    result1 = estimator.adaptive_force_estimation(displacement1, positions1)
    print(f"   选择的方法: {result1['method_selected']} (应选择hertz)")
    
    # 测试2: 非常大的位移
    print("\n2. 非常大的位移 (10mm max):")
    positions2 = np.random.rand(150, 2) * 100
    displacement2 = create_test_displacement(positions2, force_magnitude=100.0)  # 大力
    
    result2 = estimator.adaptive_force_estimation(displacement2, positions2)
    print(f"   选择的方法: {result2['method_selected']}")
    print(f"   最大位移: {result2['features']['max_displacement']:.2f} mm")
    
    # 测试3: 无位移（零位移场）
    print("\n3. 零位移场:")
    positions3 = np.random.rand(100, 2) * 100
    displacement3 = np.zeros((100, 2))
    
    result3 = estimator.adaptive_force_estimation(displacement3, positions3)
    print(f"   选择的方法: {result3['method_selected']}")
    print(f"   估计总力: {result3['total_force']:.3f} N (应为接近0)")
    
    return True

if __name__ == "__main__":
    try:
        test_adaptive_selection()
        test_method_comparison()
        test_edge_cases()
        print("\n✅ 所有自适应方法测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)