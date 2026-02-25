#!/usr/bin/env python3
"""
测试FEM改进效果
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

def test_fem_improvement():
    """测试FEM改进效果"""
    print("=== 测试FEM改进效果 ===")
    
    # 创建力估计器
    estimator = ForceEstimation(
        youngs_modulus=2.0e6,
        poissons_ratio=0.49,
        sensor_thickness=5.0
    )
    
    # 生成测试数据
    n_points = 50
    positions = np.random.rand(n_points, 2) * 100
    
    # 生成合成位移场 (10N力)
    displacement = create_test_displacement(positions, force_magnitude=10.0, noise_level=0.1)
    
    print(f"测试条件: {n_points}个点，10N力，噪声0.1mm")
    print(f"最大位移: {np.linalg.norm(displacement, axis=1).max():.4f} mm")
    print(f"平均位移: {np.linalg.norm(displacement, axis=1).mean():.4f} mm")
    
    # 使用FEM方法
    print("\n使用FEM方法估计力:")
    
    # 生成三角形网格连接
    # 使用内部方法创建网格连接
    connectivity = estimator._create_grid_connectivity(positions)
    print(f"创建了 {len(connectivity)} 个三角形单元")
    
    # 直接调用finite_element_inverse方法
    node_forces = estimator.finite_element_inverse(
        displacement, positions, connectivity,
        regularization=1e-6, check_condition=True
    )
    
    # 计算总力（节点力范数的和）
    total_force_fem = np.linalg.norm(node_forces, axis=1).sum()
    print(f"FEM估计总力: {total_force_fem:.3f} N")
    print(f"期望总力: 10.0 N")
    print(f"相对误差: {abs(total_force_fem - 10.0)/10.0*100:.1f}%")
    
    # 与Hertz方法比较
    print("\n与Hertz方法比较:")
    result_hertz = estimator.estimate_force_from_displacement(
        displacement, positions, method='hertz'
    )
    print(f"Hertz估计总力: {result_hertz['total_force']:.3f} N")
    print(f"Hertz相对误差: {abs(result_hertz['total_force'] - 10.0)/10.0*100:.1f}%")
    
    # 与Boussinesq方法比较
    print("\n与Boussinesq方法比较:")
    result_boussinesq = estimator.estimate_force_from_displacement(
        displacement, positions, method='boussinesq'
    )
    print(f"Boussinesq估计总力: {result_boussinesq['total_force']:.3f} N")
    print(f"Boussinesq相对误差: {abs(result_boussinesq['total_force'] - 10.0)/10.0*100:.1f}%")
    
    # 测试不同正则化参数
    print("\n=== 测试不同正则化参数 ===")
    reg_params = [1e-8, 1e-6, 1e-4, 1e-2]
    
    for reg in reg_params:
        node_forces_reg = estimator.finite_element_inverse(
            displacement, positions, connectivity,
            regularization=reg, check_condition=True
        )
        total_force_reg = np.linalg.norm(node_forces_reg, axis=1).sum()
        print(f"正则化={reg:.1e}: 总力={total_force_reg:.3f} N, 误差={abs(total_force_reg - 10.0)/10.0*100:.1f}%")
    
    return total_force_fem

if __name__ == "__main__":
    try:
        test_fem_improvement()
        print("\n✅ FEM改进测试完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)