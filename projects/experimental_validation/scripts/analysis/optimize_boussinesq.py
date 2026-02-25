#!/usr/bin/env python3
"""
Boussinesq参数优化脚本

通过网格搜索寻找最优的正则化参数和网格分辨率，以最小化力估计误差。
"""

import sys
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

# 添加vbts_algorithms到路径
project_root = Path(__file__).parent.parent.parent.parent
vbts_algorithms_path = project_root / "vbts_algorithms" / "src"
if vbts_algorithms_path.exists():
    sys.path.insert(0, str(vbts_algorithms_path))

try:
    from algorithms.force_estimation import ForceEstimation
    from algorithms.marker_detection import MarkerDetection
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入算法模块: {e}")
    ALGORITHMS_AVAILABLE = False


def generate_test_displacement(force_magnitude=10.0, youngs_modulus=2.0e6, noise_level=0.0):
    """
    生成测试位移场（复制自验证脚本）
    
    基于Boussinesq解生成真实的位移场，用于测试力估计算法。
    """
    # 像素到毫米转换系数
    pixel_to_mm = 0.1
    
    # 初始化标记点检测器（用于生成合成数据）
    marker_detector = MarkerDetection(
        marker_radius=6.0,
        grid_spacing=25.0
    )
    
    # 生成标记点网格 (像素坐标)
    markers_pixels = marker_detector.generate_synthetic_markers(
        image_shape=(480, 640),
        grid_offset=(30, 30)
    )
    
    # 转换为毫米坐标
    markers = markers_pixels * pixel_to_mm
    
    # 力作用中心（图像中心，转换为毫米）
    force_center_pixels = np.array([320, 240])
    force_center = force_center_pixels * pixel_to_mm
    
    # Boussinesq解: 弹性半空间表面点力引起的垂直位移
    poissons_ratio = 0.49
    elastic_coeff = (1 - poissons_ratio**2) / (np.pi * youngs_modulus)
    
    displacement_field = np.zeros_like(markers)
    
    for i, (x, y) in enumerate(markers):
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
        
        # 假设位移方向为径向向外
        if distance_mm > 0:
            displacement_field[i, 0] = (dx / distance_mm) * u_z_mm * 0.3  # 比例因子
            displacement_field[i, 1] = (dy / distance_mm) * u_z_mm * 0.3
    
    # 添加噪声 (单位: mm)
    if noise_level > 0:
        noise_mm = noise_level * pixel_to_mm
        noise = np.random.normal(0, noise_mm, displacement_field.shape)
        displacement_field += noise
    
    return displacement_field, markers, force_magnitude


def evaluate_parameters(force_estimator, displacement_field, positions, 
                        ground_truth_force, grid_resolution, regularization):
    """评估特定参数下的力估计误差"""
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 直接调用Boussinesq解方法
        force_grid = force_estimator.boussinesq_solution(
            displacement_field, positions,
            grid_resolution=grid_resolution,
            regularization=regularization
        )
        
        # 计算总力
        total_force = np.sum(force_grid)
        
        # 计算误差
        absolute_error = abs(total_force - ground_truth_force)
        relative_error = absolute_error / ground_truth_force if ground_truth_force > 0 else float('inf')
        
        # 计算计算时间
        computation_time = time.time() - start_time
        
        return {
            'total_force': total_force,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'computation_time': computation_time,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_parameter_sweep(force_estimator, test_cases, grid_resolutions, regularizations):
    """运行参数扫描"""
    results = []
    
    total_combinations = len(test_cases) * len(grid_resolutions) * len(regularizations)
    print(f"总参数组合数: {total_combinations}")
    
    completed = 0
    for case_idx, test_case in enumerate(test_cases):
        displacement_field, positions, ground_truth = test_case
        
        for grid_res in grid_resolutions:
            for reg in regularizations:
                completed += 1
                
                # 进度显示
                if completed % 10 == 0:
                    progress = (completed / total_combinations) * 100
                    print(f"进度: {completed}/{total_combinations} ({progress:.1f}%)")
                
                # 评估参数
                eval_result = evaluate_parameters(
                    force_estimator, displacement_field, positions,
                    ground_truth, grid_res, reg
                )
                
                if eval_result['success']:
                    result = {
                        'test_case_id': case_idx,
                        'grid_resolution': grid_res,
                        'regularization': reg,
                        'estimated_force': eval_result['total_force'],
                        'ground_truth_force': ground_truth,
                        'absolute_error': eval_result['absolute_error'],
                        'relative_error': eval_result['relative_error'],
                        'computation_time': eval_result['computation_time']
                    }
                    results.append(result)
                else:
                    print(f"参数评估失败: grid={grid_res}, reg={reg}, error={eval_result.get('error', 'unknown')}")
    
    return results


def analyze_optimization_results(results):
    """分析优化结果，找出最佳参数"""
    if not results:
        print("没有有效结果可分析")
        return None
    
    # 按参数分组
    param_results = defaultdict(list)
    for result in results:
        key = (result['grid_resolution'], result['regularization'])
        param_results[key].append(result)
    
    # 计算每个参数组合的平均性能
    param_performance = []
    for (grid_res, reg), result_list in param_results.items():
        avg_rel_error = np.mean([r['relative_error'] for r in result_list]) * 100
        avg_abs_error = np.mean([r['absolute_error'] for r in result_list])
        avg_time = np.mean([r['computation_time'] for r in result_list])
        num_cases = len(result_list)
        
        param_performance.append({
            'grid_resolution': grid_res,
            'regularization': reg,
            'avg_relative_error': avg_rel_error,
            'avg_absolute_error': avg_abs_error,
            'avg_computation_time': avg_time,
            'num_test_cases': num_cases
        })
    
    # 按平均相对误差排序
    param_performance.sort(key=lambda x: x['avg_relative_error'])
    
    return param_performance


def save_results(results, param_performance, output_dir):
    """保存优化结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    detailed_path = output_dir / "boussinesq_optimization_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存参数性能总结
    summary_path = output_dir / "boussinesq_optimization_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Boussinesq参数优化结果\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 最佳参数组合\n\n")
        if param_performance:
            best = param_performance[0]
            f.write(f"- **网格分辨率**: {best['grid_resolution']} mm\n")
            f.write(f"- **正则化参数**: {best['regularization']}\n")
            f.write(f"- **平均相对误差**: {best['avg_relative_error']:.1f}%\n")
            f.write(f"- **平均绝对误差**: {best['avg_absolute_error']:.2f} N\n")
            f.write(f"- **平均计算时间**: {best['avg_computation_time']:.3f} s\n")
            f.write(f"- **测试用例数**: {best['num_test_cases']}\n\n")
        
        f.write("## 所有参数组合性能排序\n\n")
        f.write("| 排名 | 网格分辨率 (mm) | 正则化参数 | 平均相对误差 (%) | 平均绝对误差 (N) | 平均计算时间 (s) |\n")
        f.write("|------|-----------------|------------|------------------|------------------|------------------|\n")
        
        for i, perf in enumerate(param_performance[:20], 1):  # 显示前20名
            f.write(f"| {i} | {perf['grid_resolution']} | {perf['regularization']:.1e} | "
                   f"{perf['avg_relative_error']:.1f} | {perf['avg_absolute_error']:.2f} | "
                   f"{perf['avg_computation_time']:.3f} |\n")
        
        # 添加建议
        f.write("\n## 优化建议\n\n")
        if param_performance:
            best = param_performance[0]
            second_best = param_performance[1] if len(param_performance) > 1 else None
            
            f.write("1. **推荐参数**:\n")
            f.write(f"   - 网格分辨率: **{best['grid_resolution']} mm**\n")
            f.write(f"   - 正则化参数: **{best['regularization']:.1e}**\n")
            f.write(f"   - 预期相对误差: **{best['avg_relative_error']:.1f}%**\n\n")
            
            if second_best:
                f.write("2. **备选参数** (如果计算时间过长):\n")
                f.write(f"   - 网格分辨率: {second_best['grid_resolution']} mm\n")
                f.write(f"   - 正则化参数: {second_best['regularization']:.1e}\n")
                f.write(f"   - 预期相对误差: {second_best['avg_relative_error']:.1f}%\n\n")
            
            # 分析趋势
            grid_res_errors = defaultdict(list)
            reg_errors = defaultdict(list)
            
            for perf in param_performance:
                grid_res_errors[perf['grid_resolution']].append(perf['avg_relative_error'])
                reg_errors[perf['regularization']].append(perf['avg_relative_error'])
            
            f.write("3. **参数趋势分析**:\n")
            f.write("   - 网格分辨率影响:\n")
            for grid_res, errors in sorted(grid_res_errors.items()):
                avg_error = np.mean(errors)
                f.write(f"     - {grid_res} mm: {avg_error:.1f}% 平均误差\n")
            
            f.write("   - 正则化参数影响:\n")
            for reg, errors in sorted(reg_errors.items()):
                avg_error = np.mean(errors)
                f.write(f"     - {reg:.1e}: {avg_error:.1f}% 平均误差\n")
    
    print(f"详细结果保存至: {detailed_path}")
    print(f"优化总结保存至: {summary_path}")
    
    return detailed_path, summary_path


def main():
    """主函数"""
    if not ALGORITHMS_AVAILABLE:
        print("错误: 算法模块不可用，无法运行优化")
        return
    
    print("=" * 80)
    print("BOUSSINESQ参数优化")
    print("=" * 80)
    
    # 初始化力估计器（使用默认参数）
    estimator = ForceEstimation(
        youngs_modulus=2.0e6,
        poissons_ratio=0.49,
        sensor_thickness=5.0
    )
    
    # 生成测试用例（选择代表性的测试场景，限制数量以加速）
    print("\n1. 生成测试用例...")
    test_cases = []
    
    # 基于验证结果分析，选择有代表性的测试场景
    # 只使用2个测试用例以加速优化
    test_scenarios = [
        (10.0, 2.0e6, 0.0),   # 基准场景
        (20.0, 2.0e6, 0.0),   # 较大力
    ]
    
    for force, youngs, noise in test_scenarios:
        print(f"  生成: 力={force}N, 杨氏模量={youngs/1e6}MPa, 噪声={noise}")
        displacement, positions, ground_truth = generate_test_displacement(
            force_magnitude=force,
            youngs_modulus=youngs,
            noise_level=noise
        )
        test_cases.append((displacement, positions, ground_truth))
    
    print(f"  生成 {len(test_cases)} 个测试用例 (为加速优化限制数量)")
    
    # 参数搜索范围（限制以减少计算时间）
    print("\n2. 设置参数搜索范围...")
    grid_resolutions = [5.0, 10.0, 20.0]  # mm (较粗的网格以加速)
    regularizations = [1e-8, 1e-6, 1e-4, 1e-2]  # 正则化参数
    
    print(f"   网格分辨率: {grid_resolutions} mm (使用较粗网格加速)")
    print(f"   正则化参数: {regularizations}")
    print(f"   总参数组合: {len(grid_resolutions) * len(regularizations)}")
    
    # 运行参数扫描
    print("\n3. 运行参数扫描...")
    print("   这可能需要一些时间...")
    
    results = run_parameter_sweep(
        estimator, test_cases, grid_resolutions, regularizations
    )
    
    print(f"   完成 {len(results)} 个参数评估")
    
    # 分析结果
    print("\n4. 分析优化结果...")
    param_performance = analyze_optimization_results(results)
    
    if param_performance:
        best = param_performance[0]
        print(f"\n   最佳参数组合:")
        print(f"   - 网格分辨率: {best['grid_resolution']} mm")
        print(f"   - 正则化参数: {best['regularization']:.1e}")
        print(f"   - 平均相对误差: {best['avg_relative_error']:.1f}%")
        print(f"   - 平均计算时间: {best['avg_computation_time']:.3f} s")
        
        # 显示前5名
        print(f"\n   前5名参数组合:")
        for i, perf in enumerate(param_performance[:5], 1):
            print(f"   {i}. grid={perf['grid_resolution']}mm, reg={perf['regularization']:.1e}, "
                 f"error={perf['avg_relative_error']:.1f}%, time={perf['avg_computation_time']:.3f}s")
    
    # 保存结果
    print("\n5. 保存结果...")
    output_dir = Path(__file__).parent.parent.parent / "results" / "boussinesq_optimization"
    detailed_path, summary_path = save_results(results, param_performance, output_dir)
    
    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)
    print(f"\n优化总结保存至: {summary_path}")
    print(f"详细结果保存至: {detailed_path}")
    
    if param_performance:
        best = param_performance[0]
        baseline_error = 153.8  # 验证报告中的平均误差
        improvement = ((baseline_error - best['avg_relative_error']) / baseline_error) * 100
        print(f"\n与基线相比改进: {improvement:.1f}% (基线: {baseline_error:.1f}% → 优化后: {best['avg_relative_error']:.1f}%)")


if __name__ == "__main__":
    main()